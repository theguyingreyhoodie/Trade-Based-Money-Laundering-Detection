"""EVM backend: the ``EvidenceRegistry`` contract on a local Hardhat node, called through web3.py.

Account 0 of the node is the registry admin. Each organisation is assigned one
of the node's other accounts and writes with it, so the contract's allowlist is
what decides who may register evidence.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from web3 import Web3
from web3.exceptions import ContractLogicError
from web3.logs import DISCARD

from tbml.ledger.base import EvidenceLedger, EvidenceRecord, Journal, LedgerError, PermissionDenied, Receipt, normalise_digest, storage_slots
from tbml.utils import get_logger, load_json, save_json

log = get_logger(__name__)


class EvmLedger(EvidenceLedger):
    backend = "evm"

    def __init__(self, rpc_url: str, artifact: str | Path, state_dir: str | Path, max_batch: int = 100, fresh: bool = False) -> None:
        self.w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 60}))
        if not self.w3.is_connected():
            raise LedgerError(f"no EVM node at {rpc_url}")
        self.rpc_url = rpc_url
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.journal = Journal(self.state_dir / "journal.jsonl")
        self.max_batch = int(max_batch)
        self.accounts = list(self.w3.eth.accounts)
        self.admin = self.accounts[0]
        self._artifact = json.loads(Path(artifact).read_text())
        self._orgs: dict[str, str] = {}
        self._names: dict[str, str] = {}
        self._gas = 0
        self._slots = 0
        self.restarted = False
        self._deploy_path = self.state_dir / "deployment.json"
        self._attach_or_deploy(fresh)

    # ------------------------------------------------------------------ deployment
    def _attach_or_deploy(self, fresh: bool) -> None:
        info = load_json(self._deploy_path) if self._deploy_path.is_file() and not fresh else None
        if info and info.get("chain_id") == self.w3.eth.chain_id and len(self.w3.eth.get_code(info["address"])) > 0:
            self.contract = self.w3.eth.contract(address=info["address"], abi=self._artifact["abi"])
            self.deploy_block = int(info["deploy_block"])
            self._orgs = dict(info.get("organisations", {}))
            self._names = {address: name for name, address in self._orgs.items()}
            self._gas = int(info.get("gas_used", 0))
            self._slots = int(info.get("storage_slots", 0))
            return
        self.restarted = info is not None and not fresh  # a deployment was on record but the chain no longer has it
        factory = self.w3.eth.contract(abi=self._artifact["abi"], bytecode=self._artifact["bytecode"])
        receipt = self.w3.eth.wait_for_transaction_receipt(factory.constructor().transact({"from": self.admin}), poll_latency=0.01)
        self.contract = self.w3.eth.contract(address=receipt.contractAddress, abi=self._artifact["abi"])
        self.deploy_block = int(receipt.blockNumber)
        self.deploy_gas = int(receipt.gasUsed)
        self._orgs, self._names, self._gas, self._slots = {}, {}, 0, 0
        self._save()
        log.info("EvidenceRegistry deployed at %s (block %d, %d gas)", receipt.contractAddress, self.deploy_block, receipt.gasUsed)

    def _save(self) -> None:
        save_json({"address": self.contract.address, "chain_id": self.w3.eth.chain_id, "deploy_block": self.deploy_block, "admin": self.admin,
                   "rpc_url": self.rpc_url, "organisations": self._orgs, "gas_used": self._gas, "storage_slots": self._slots}, self._deploy_path)

    # ------------------------------------------------------------------ organisations
    def register_organisation(self, name: str) -> str:
        if name in self._orgs:
            return self._orgs[name]
        used = set(self._orgs.values()) | {self.admin}
        free = [a for a in self.accounts if a not in used]
        if not free:
            raise LedgerError("the node has no unused account left for another organisation")
        address = free[0]
        self._send(self.contract.functions.registerOrganisation(address, name), self.admin)
        self._orgs[name] = address
        self._names[address] = name
        self._save()
        return address

    def revoke_organisation(self, name: str) -> None:
        if name in self._orgs:
            self._send(self.contract.functions.revokeOrganisation(self._orgs[name]), self.admin)
            del self._orgs[name]
            self._save()

    def organisations(self) -> dict[str, str]:
        return dict(self._orgs)

    # ------------------------------------------------------------------ writes
    def _send(self, call: Any, sender: str) -> Any:
        try:
            tx = call.transact({"from": sender})
        except ContractLogicError as exc:
            text = str(exc)
            if "NotRegisteredOrganisation" in text or "NotAdmin" in text:
                raise PermissionDenied(text) from exc
            raise LedgerError(text) from exc
        receipt = self.w3.eth.wait_for_transaction_receipt(tx, poll_latency=0.005)
        if receipt.status != 1:
            raise LedgerError(f"transaction {tx.hex()} reverted")
        self._gas += int(receipt.gasUsed)
        return receipt

    def _sender(self, org: str) -> str:
        """Account of ``org``. An unknown name maps to an unused account, so the contract itself refuses it."""
        if org in self._orgs:
            return self._orgs[org]
        used = set(self._orgs.values()) | {self.admin}
        outsiders = [a for a in self.accounts if a not in used]
        return outsiders[-1] if outsiders else self.accounts[-1]

    def register(self, digest: str, doc_type: str, ref_id: str, org: str, ts: int | None = None) -> Receipt:
        d = normalise_digest(digest)
        ts = self.now() if ts is None else int(ts)
        start = time.perf_counter()
        existing = self.verify(d) if org in self._orgs else None
        if existing is not None and existing.exists:
            return Receipt(digest=d, doc_type=doc_type, ref_id=ref_id, org=org, status="duplicate", tx_id="", block=0, client_ts=ts, cost=0,
                           latency_s=time.perf_counter() - start)
        receipt = self._send(self.contract.functions.registerEvidence(bytes.fromhex(d), doc_type, ref_id, ts), self._sender(org))
        self._slots += storage_slots(doc_type, ref_id)
        out = Receipt(digest=d, doc_type=doc_type, ref_id=ref_id, org=org, status="registered", tx_id="0x" + receipt.transactionHash.hex().removeprefix("0x"),
                      block=int(receipt.blockNumber), client_ts=ts, cost=int(receipt.gasUsed), latency_s=time.perf_counter() - start)
        self.journal.append(out)
        self._save()
        return out

    def register_batch(self, digests: list[str], doc_type: str, ref_ids: list[str], org: str, ts: int | None = None) -> list[Receipt]:
        if len(digests) != len(ref_ids):
            raise LedgerError("digests and ref_ids differ in length")
        ts = self.now() if ts is None else int(ts)
        clean = [normalise_digest(d) for d in digests]
        receipts: list[Receipt] = []
        for start_idx in range(0, len(clean), self.max_batch):
            chunk = clean[start_idx : start_idx + self.max_batch]
            refs = ref_ids[start_idx : start_idx + self.max_batch]
            start = time.perf_counter()
            receipt = self._send(self.contract.functions.registerEvidenceBatch([bytes.fromhex(d) for d in chunk], doc_type, refs, ts), self._sender(org))
            latency = time.perf_counter() - start
            # the receipt may also hold DuplicateEvidence logs; only EvidenceRegistered ones are decoded here
            stored = {log_["args"]["hash"].hex() for log_ in self.contract.events.EvidenceRegistered().process_receipt(receipt, errors=DISCARD)}
            n_new = max(len(stored), 1)
            for d, ref in zip(chunk, refs):
                new = d in stored
                stored.discard(d)
                if new:
                    self._slots += storage_slots(doc_type, ref)
                out = Receipt(digest=d, doc_type=doc_type, ref_id=ref, org=org, status="registered" if new else "duplicate",
                              tx_id="0x" + receipt.transactionHash.hex().removeprefix("0x"), block=int(receipt.blockNumber), client_ts=ts,
                              cost=int(receipt.gasUsed / n_new) if new else 0, latency_s=latency / len(chunk))
                self.journal.append(out)
                receipts.append(out)
        self._save()
        return receipts

    # ------------------------------------------------------------------ reads
    def verify(self, digest: str) -> EvidenceRecord:
        d = normalise_digest(digest)
        exists, submitter, doc_type, ref_id, client_ts, block_ts = self.contract.functions.verify(bytes.fromhex(d)).call()
        if not exists:
            return EvidenceRecord(digest=d, exists=False)
        name = self._names.get(submitter) or self.contract.functions.organisationName(submitter).call()
        return EvidenceRecord(digest=d, exists=True, org=name, submitter=submitter, doc_type=doc_type, ref_id=ref_id, client_ts=int(client_ts), block_ts=int(block_ts))

    def audit_trail(self, doc_type: str | None = None) -> list[dict[str, Any]]:
        logs = self.contract.events.EvidenceRegistered().get_logs(from_block=self.deploy_block)
        trail = []
        for entry in logs:
            args = entry["args"]
            if doc_type and args["docType"] != doc_type:
                continue
            trail.append({"digest": args["hash"].hex(), "ref_id": args["refId"], "doc_type": args["docType"],
                          "org": self._names.get(args["submitter"], ""), "submitter": args["submitter"], "client_ts": int(args["clientTs"]),
                          "block_ts": int(args["blockTs"]), "block": int(entry["blockNumber"]), "tx_id": "0x" + entry["transactionHash"].hex().removeprefix("0x")})
        return trail

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "contract": self.contract.address,
            "chain_id": self.w3.eth.chain_id,
            "block_number": int(self.w3.eth.block_number),
            "n_records": int(self.contract.functions.evidenceCount().call()),
            "n_organisations": int(self.contract.functions.organisationCount().call()),
            "storage_bytes": self._slots * 32,
            "storage_note": "contract storage slots x 32 bytes; state-trie overhead not included",
            "contract_code_bytes": len(self.w3.eth.get_code(self.contract.address)),
            "total_cost": self._gas,
            "cost_unit": "gas",
        }

    # ------------------------------------------------------------------ node controls used by the benchmark
    def set_mining(self, interval_ms: int | None) -> None:
        """``None`` / 0 = mine every transaction at once; otherwise one block every ``interval_ms``."""
        if interval_ms:
            self.w3.provider.make_request("evm_setAutomine", [False])
            self.w3.provider.make_request("evm_setIntervalMining", [int(interval_ms)])
        else:
            self.w3.provider.make_request("evm_setIntervalMining", [0])
            self.w3.provider.make_request("evm_setAutomine", [True])
