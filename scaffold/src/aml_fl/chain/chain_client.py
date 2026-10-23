"""web3.py client for the FLAudit contract (hardhat/contracts/FLAudit.sol).

Hashes are SHA-256 hex digests (64 hex characters, e.g. from fl.strategy.hash_parameters) and
are stored on-chain as bytes32. A run is identified by a name such as "fl_hybrid_s42"; the
contract stores keccak256 of that name.

The client sends transactions from accounts the node manages (eth_accounts), which is what a
local Hardhat node provides. Account 0 deploys the contract and is the operator (the FL
server); the following accounts play the banks.

Needs the compiled contract: `npx hardhat build` in hardhat/.
"""
from __future__ import annotations

import json
from pathlib import Path

from web3 import Web3

HARDHAT_DIR = Path(__file__).resolve().parent / "hardhat"
ARTIFACT = HARDHAT_DIR / "artifacts" / "contracts" / "FLAudit.sol" / "FLAudit.json"
DEFAULT_RPC = "http://127.0.0.1:8545"


def load_artifact(path: str | Path = ARTIFACT) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Compile the contract: cd {HARDHAT_DIR} && npx hardhat build")
    return json.loads(path.read_text())


def run_key(run_id: str) -> bytes:
    return Web3.keccak(text=run_id)


def to_bytes32(digest: str) -> bytes:
    """'ab12...' (64 hex characters, optional 0x) -> 32 bytes."""
    raw = bytes.fromhex(digest.removeprefix("0x"))
    if len(raw) != 32:
        raise ValueError(f"Expected a 32-byte hex digest, got {len(raw)} bytes")
    return raw


def connect(rpc_url: str = DEFAULT_RPC) -> Web3:
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        raise ConnectionError(f"No Ethereum node at {rpc_url}. Start one: cd {HARDHAT_DIR} && npx hardhat node")
    return w3


class AuditChain:
    def __init__(self, w3: Web3, address: str):
        self.w3 = w3
        self.contract = w3.eth.contract(address=Web3.to_checksum_address(address), abi=load_artifact()["abi"])
        self.address = self.contract.address
        self.operator = self.contract.functions.operator().call()

    @classmethod
    def deploy(cls, w3: Web3, operator: str | None = None) -> "AuditChain":
        art = load_artifact()
        factory = w3.eth.contract(abi=art["abi"], bytecode=art["bytecode"])
        tx = factory.constructor().transact({"from": operator or w3.eth.accounts[0]})
        receipt = w3.eth.wait_for_transaction_receipt(tx)
        if receipt.status != 1:
            raise RuntimeError("FLAudit deployment failed")
        return cls(w3, receipt.contractAddress)

    @staticmethod
    def is_deployed(w3: Web3, address: str) -> bool:
        """False if nothing is at that address (e.g. the local node was restarted)."""
        return len(w3.eth.get_code(Web3.to_checksum_address(address))) > 0

    def _send(self, call, sender: str):
        receipt = self.w3.eth.wait_for_transaction_receipt(call.transact({"from": sender}))
        if receipt.status != 1:
            raise RuntimeError(f"Transaction reverted: {receipt.transactionHash.hex()}")
        return receipt

    # --- writes ------------------------------------------------------------------------
    def register_bank(self, bank: str) -> None:
        if not self.is_bank(bank):
            self._send(self.contract.functions.registerBank(bank), self.operator)

    def record_global_model(self, run_id: str, rnd: int, digest: str) -> None:
        self._send(self.contract.functions.recordGlobalModel(run_key(run_id), rnd, to_bytes32(digest)),
                   self.operator)

    def record_run(self, run_id: str, audit_log: list[dict], banks: dict[str, str] | None = None) -> int:
        """Record every hash of an audit.json that is not on-chain yet: the global model of each
        round (as operator) and, if `banks` maps bank names to addresses, each bank's update (sent
        from that bank's address). Returns how many hashes were written. Raises if one is already
        recorded with a different value."""
        written = 0
        for entry in audit_log:
            rnd = entry["round"]
            onchain = self.global_model_hash(run_id, rnd)
            if onchain is None:
                self.record_global_model(run_id, rnd, entry["hash"])
                written += 1
            elif onchain != entry["hash"]:
                raise ValueError(f"{run_id} round {rnd} is already recorded with a different hash")
            for bank, digest in entry.get("updates", {}).items() if banks else ():
                onchain = self.update_hash(run_id, rnd, banks[bank])
                if onchain is None:
                    self.record_update(run_id, rnd, digest, banks[bank])
                    written += 1
                elif onchain != digest:
                    raise ValueError(f"{run_id} round {rnd}: {bank}'s update is already recorded with a different hash")
        return written

    def record_update(self, run_id: str, rnd: int, digest: str, bank: str) -> None:
        self._send(self.contract.functions.recordUpdate(run_key(run_id), rnd, to_bytes32(digest)), bank)

    def log_alert(self, digest: str, bank: str) -> None:
        self._send(self.contract.functions.logAlert(to_bytes32(digest)), bank)

    # --- reads -------------------------------------------------------------------------
    def is_bank(self, address: str) -> bool:
        return self.contract.functions.isBank(address).call()

    def global_model_hash(self, run_id: str, rnd: int) -> str | None:
        raw = self.contract.functions.globalModelHash(run_key(run_id), rnd).call()
        return None if raw == bytes(32) else raw.hex()

    def verify_global_model(self, run_id: str, rnd: int, digest: str) -> bool:
        return self.contract.functions.verifyGlobalModel(run_key(run_id), rnd, to_bytes32(digest)).call()

    def update_hash(self, run_id: str, rnd: int, bank: str) -> str | None:
        raw = self.contract.functions.updateHash(run_key(run_id), rnd, bank).call()
        return None if raw == bytes(32) else raw.hex()

    def verify_run(self, run_id: str, audit_log: list[dict], banks: dict[str, str] | None = None) -> list[int]:
        """Rounds of audit_log that do not match the chain: the global model hash, or (if `banks`
        is given) any bank's update hash. Empty list: everything matches."""
        bad = []
        for e in audit_log:
            ok = self.verify_global_model(run_id, e["round"], e["hash"])
            if banks:
                ok = ok and all(self.update_hash(run_id, e["round"], banks[b]) == h
                                for b, h in e.get("updates", {}).items())
            if not ok:
                bad.append(e["round"])
        return bad

    def alert_bank(self, digest: str) -> str | None:
        bank = self.contract.functions.alertBank(to_bytes32(digest)).call()
        return None if int(bank, 16) == 0 else bank
