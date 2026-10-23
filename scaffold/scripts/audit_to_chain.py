"""Phase 4: write the FL audit hashes (experiments/results/<run>/audit.json) to the FLAudit
contract, then check them against the chain.

Start a local chain first (and compile the contract once):

    cd src/aml_fl/chain/hardhat && npm install && npx hardhat build && npx hardhat node

Then, in another terminal:

    python scripts/audit_to_chain.py fl_hybrid_s42 fl_zero_filled_s42   # record + verify
    python scripts/audit_to_chain.py fl_hybrid_s42 --verify-only        # only check

The contract is deployed on first use and its address kept in
src/aml_fl/chain/hardhat/deployments/localhost.json; it is redeployed if the node was
restarted. Account 0 is the operator (FL server), accounts 1..n are registered as the banks.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aml_fl.chain.chain_client import DEFAULT_RPC, HARDHAT_DIR, AuditChain, connect  # noqa: E402
from aml_fl.utils import load_config, setup_logging  # noqa: E402

log = logging.getLogger("audit_to_chain")
DEPLOYMENT = HARDHAT_DIR / "deployments" / "localhost.json"


def get_chain(rpc_url: str, num_banks: int, deployment: Path = DEPLOYMENT) -> AuditChain:
    """The deployed contract for this node (deployed and banks registered if needed)."""
    w3 = connect(rpc_url)
    saved = json.loads(deployment.read_text()) if deployment.exists() else {}
    if saved.get("rpc") == rpc_url and AuditChain.is_deployed(w3, saved["address"]):
        chain = AuditChain(w3, saved["address"])
    else:
        chain = AuditChain.deploy(w3)
        deployment.parent.mkdir(parents=True, exist_ok=True)
        deployment.write_text(json.dumps({"rpc": rpc_url, "chain_id": w3.eth.chain_id,
                                          "address": chain.address}, indent=2))
        log.info("Deployed FLAudit at %s (operator %s)", chain.address, chain.operator)
    for bank in w3.eth.accounts[1:1 + num_banks]:
        chain.register_bank(bank)
    return chain


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run folders in experiments/results, e.g. fl_hybrid_s42")
    ap.add_argument("--rpc", default=DEFAULT_RPC)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--verify-only", action="store_true", help="do not write, only compare with the chain")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(ROOT / args.config)

    chain = get_chain(args.rpc, cfg["partition"]["num_clients"])
    failed = False
    for run in args.runs:
        path = ROOT / "experiments" / "results" / run / "audit.json"
        if not path.exists():
            sys.exit(f"{path} not found. Run scripts/run_fl.py first.")
        audit = json.loads(path.read_text())
        if not args.verify_only:
            written = chain.record_run(run, audit)
            log.info("%s: recorded %d new rounds (%d already on-chain)", run, written, len(audit) - written)
        bad = chain.verify_run(run, audit)
        if bad:
            failed = True
            log.error("%s: %d of %d rounds do NOT match the chain: %s", run, len(bad), len(audit), bad[:10])
        else:
            log.info("%s: all %d rounds match the chain", run, len(audit))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
