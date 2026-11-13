#!/usr/bin/env python
"""Phase 5a: register the trade documents of the primary seed on the evidence ledger.

Every invoice, shipment, payment and customs declaration is hashed (SHA-256 of
canonical JSON). The hashes are anchored as Merkle batches by the organisation
that owns the documents. Raw documents never leave their owner.

Starts a fresh registry (a new contract on the EVM backend), so it is the first
ledger step of the pipeline. Writes ``results/ledger/registration.json``.
"""

from __future__ import annotations

from tbml.data.generator import load_dataset
from tbml.ledger.evidence import EvidenceStore, register_documents, store_path
from tbml.ledger.factory import open_ledger
from tbml.utils import get_logger, parse_cli, save_json

log = get_logger("ledger_register")


def main() -> None:
    args, cfg = parse_cli(__doc__, lambda p: p.add_argument("--limit", type=int, default=None, help="register at most this many documents per table"))
    seed = int(cfg.project.seed)
    ds = load_dataset(cfg.seed_dir("data_dir", seed))
    ledger = open_ledger(cfg, fresh=True)
    store = EvidenceStore(store_path(cfg))
    store.clear()
    summary = register_documents(ledger, store, ds.tables, cfg, run_id=f"seed{seed}", limit=args.limit)
    summary.update({"seed": seed, "organisations": ledger.organisations(), "ledger": ledger.stats(), "index": store.counts()})
    save_json(summary, cfg.path("results_dir", "ledger", "registration.json"))
    store.close()
    ledger.close()


if __name__ == "__main__":
    main()
