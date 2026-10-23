"""Phase 5: alerts with SHAP explanations from a finished FL run, optionally logged on-chain.

For every bank: load its final model (<run>/models/, written by run_fl.py), score its test
transactions, flag those at or above its own threshold, and explain each flag with SHAP on the
bank's own data. Each alert gets a hash (explain.alert_hash); with --chain that hash is logged
on the FLAudit contract from the bank's own account.

Writes experiments/results/<run>/alerts.json. "label" and "pattern" are ground truth from the
dataset, for evaluating the demo only; a real bank would not know them.

Usage:
    python scripts/build_alerts.py fl_hybrid_s42
    python scripts/build_alerts.py fl_hybrid_s42 --chain     # also log alert hashes (needs a node)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aml_fl.utils import load_config, load_table, set_seed, setup_logging  # noqa: E402

log = logging.getLogger("build_alerts")
TXN_FIELDS = ["timestamp", "src_key", "dst_key", "src_bank", "dst_bank", "payment_format",
              "payment_currency", "receiving_currency"]


def bank_alerts(run_dir: Path, part_dir: Path, client: str, model_version: str, pattern_of: pd.Series | None,
                background: int, top: int, seed: int) -> list[dict]:
    import torch

    from aml_fl.explain import alert_hash, group_by_column, shap_values, top_features
    from aml_fl.fl.client_app import load_bank_data, new_model
    from aml_fl.models import predict_proba

    meta = json.loads((run_dir / "models" / f"{client}.json").read_text())
    d = load_bank_data(meta["mode"], part_dir / client, meta.get("global_columns"))
    if d.feature_names != meta["feature_names"]:
        raise ValueError(f"{client}: data layout differs from the one the model was trained on")
    model = new_model(meta["mode"], d)
    model.load_state_dict(torch.load(run_dir / "models" / f"{client}.pt"))

    scores = predict_proba(model, d.X["test"])
    flagged = np.flatnonzero(scores >= meta["threshold"])
    rng = np.random.default_rng(seed)
    bg = d.X["train"][rng.choice(len(d.X["train"]), min(background, len(d.X["train"])), replace=False)]
    contrib, base = shap_values(model, bg, d.X["test"][flagged])
    columns, grouped = group_by_column(contrib, d.feature_names)

    raw = load_table(part_dir / client / "transactions").set_index("txn_id")
    raw = raw.loc[d.txn_ids["test"][flagged]]
    alerts = []
    for k, i in enumerate(flagged):
        txn = raw.iloc[k]
        values = {c: txn[c] for c in columns if c in txn.index}
        alert = {
            "bank": client,
            "txn_id": int(d.txn_ids["test"][i]),
            **{f: (str(txn[f]) if f == "timestamp" else txn[f].item() if hasattr(txn[f], "item") else txn[f])
               for f in TXN_FIELDS if f in txn.index},
            "amount_usd": round(float(np.expm1(txn["log_amount"])), 2),
            "score": round(float(scores[i]), 6),
            "threshold": round(float(meta["threshold"]), 6),
            "base_value": round(base, 6),
            "top_features": top_features(columns, grouped[k], values, top),
            "model_version": model_version,
            "label": int(txn["label"]),
            "pattern": str(pattern_of.get(int(d.txn_ids["test"][i]), "none")) if pattern_of is not None else None,
        }
        alert["alert_hash"] = alert_hash(alert)
        alerts.append(alert)
    log.info("%s: %d of %d test transactions flagged (%d true laundering)", client, len(alerts),
             len(scores), sum(a["label"] for a in alerts))
    return alerts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="FL run folder in experiments/results, e.g. fl_hybrid_s42")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--background", type=int, default=256, help="training rows used as the SHAP background")
    ap.add_argument("--top", type=int, default=5, help="features kept per alert")
    ap.add_argument("--chain", action="store_true", help="log every alert hash on the FLAudit contract")
    ap.add_argument("--rpc", default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(ROOT / args.config)
    set_seed(cfg["seed"])

    run_dir = ROOT / "experiments" / "results" / args.run
    if not (run_dir / "models").exists():
        sys.exit(f"No saved models in {run_dir}. Rerun scripts/run_fl.py (models are saved after the last round).")
    audit = json.loads((run_dir / "audit.json").read_text())
    model_version = audit[-1]["hash"]   # the final global model this run's banks score with
    try:
        pattern_of = load_table(ROOT / cfg["data"]["processed_dir"] / "patterns").set_index("txn_id")["pattern"]
    except FileNotFoundError:
        pattern_of = None

    part_dir = ROOT / cfg["data"]["partitions_dir"]
    clients = sorted(p.stem for p in (run_dir / "models").glob("client_*.json"))
    alerts = [a for c in clients for a in bank_alerts(run_dir, part_dir, c, model_version, pattern_of,
                                                      args.background, args.top, cfg["seed"])]
    out = run_dir / "alerts.json"
    out.write_text(json.dumps({"run": args.run, "model_version": model_version, "alerts": alerts}, default=str))
    log.info("Saved %d alerts to %s", len(alerts), out)

    if args.chain:
        sys.path.insert(0, str(ROOT / "scripts"))
        from audit_to_chain import bank_accounts, get_chain

        from aml_fl.chain.chain_client import DEFAULT_RPC

        chain = get_chain(args.rpc or DEFAULT_RPC, cfg["partition"]["num_clients"])
        banks = bank_accounts(chain, cfg["partition"]["num_clients"])
        logged = 0
        for a in alerts:
            if chain.alert_bank(a["alert_hash"]) is None:
                chain.log_alert(a["alert_hash"], banks[a["bank"]])
                logged += 1
        log.info("Logged %d alert hashes on-chain (%d already there)", logged, len(alerts) - logged)


if __name__ == "__main__":
    main()
