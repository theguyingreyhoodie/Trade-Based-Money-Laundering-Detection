"""Phase 3: federated learning simulation with Flower (one simulated node per client bank).

    naive        : every bank uses only the core features, the whole BankModel is FedAvg'd
    encoder_head : every bank uses its full own schema with a private encoder; only the
                   SharedHead is FedAvg'd
    hybrid       : a core encoder (core features) and the head are FedAvg'd; a private
                   encoder handles each bank's own features (our method)

Needs the partitions from prepare_data.py. Results use the same columns as baselines.csv:

    experiments/results/fl_<mode>.csv              per-client and pooled test metrics
    experiments/results/fl_<mode>_by_pattern.csv   recall per laundering pattern
    experiments/results/fl_<mode>/history.csv      per-round train loss / val metrics
    experiments/results/fl_<mode>/audit.json       SHA-256 of the global model each round

Usage:
    python scripts/run_fl.py --config configs/default.yaml
    python scripts/run_fl.py --rounds 3          # quick check
    python scripts/run_fl.py --seed 1            # writes fl_<mode>_s1.csv, fl_<mode>_s1/ instead
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aml_fl.features import CATEGORICAL, SHARED_VOCAB  # noqa: E402
from aml_fl.metrics import evaluate, pooled_evaluate, recall_by_pattern  # noqa: E402
from aml_fl.models import BankModel, HybridModel  # noqa: E402
from aml_fl.utils import load_config, load_table, set_seed, setup_logging  # noqa: E402

log = logging.getLogger("run_fl")


def core_input_dim(core: list[str]) -> int:
    """Width of the core feature vector: one per numeric column, one-hot (+ <other>) per categorical."""
    return sum(len(SHARED_VOCAB[c]) + 1 if c in CATEGORICAL else 1 for c in core)


def summarize(scores: pd.DataFrame, pattern_of: pd.Series | None, setup: str) -> tuple[list[dict], list[dict]]:
    """Per-client and pooled metrics (each client uses its own val-tuned threshold)."""
    rows, by_pattern = [], []
    for client, s in scores.groupby("client", sort=True):
        rows.append({"setup": setup, "client": client, "model": "mlp",
                     **evaluate(s["label"].to_numpy(), s["score"].to_numpy(), float(s["threshold"].iloc[0]))})
    pooled = pooled_evaluate(scores["label"].to_numpy(), scores["score"].to_numpy(), scores["threshold"].to_numpy())
    rows.append({"setup": setup, "client": "pooled", "model": "mlp", **pooled})
    if pattern_of is not None:
        pats = pattern_of.reindex(scores["txn_id"]).fillna("none").to_numpy()
        per = recall_by_pattern(pats, (scores["score"] - scores["threshold"]).to_numpy(), 0.0)
        by_pattern = [{"setup": setup, "client": "pooled", "model": "mlp", **r} for r in per.to_dict("records")]
    return rows, by_pattern


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mode", default=None, help="override fl.mode")
    ap.add_argument("--rounds", type=int, default=None, help="override fl.num_rounds")
    ap.add_argument("--seed", type=int, default=None,
                    help="override seed; outputs get a _s<seed> suffix so repeated runs do not overwrite")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(ROOT / args.config)
    fl, mlp = cfg["fl"], cfg.get("mlp", {})
    mode = args.mode or fl["mode"]
    rounds = args.rounds or fl["num_rounds"]
    seed = cfg["seed"] if args.seed is None else args.seed
    name = f"fl_{mode}" if args.seed is None else f"fl_{mode}_s{seed}"
    set_seed(seed)
    if " " in sys.executable:
        # Ray (Flower's simulation backend) cannot start workers from such a path; it hangs
        # for minutes and then fails with "worker startup repeatedly failed".
        sys.exit(f"Python path contains spaces: {sys.executable}\n"
                 "Flower's simulation needs a venv outside such a path, e.g.\n"
                 "  python3.12 -m venv ~/.venvs/aml-fl && ~/.venvs/aml-fl/bin/pip install -r requirements.txt")

    # Imported here so the rest of the project does not need Flower installed.
    from flwr.app import ArrayRecord
    from flwr.simulation import run_simulation

    from aml_fl.fl.client_app import make_client_app, shared_state
    from aml_fl.fl.server_app import make_server_app

    part_dir = ROOT / cfg["data"]["partitions_dir"]
    n_clients = len(list(part_dir.glob("client_*")))
    if n_clients == 0:
        sys.exit(f"No partitions in {part_dir}. Run scripts/prepare_data.py first.")
    out_dir = ROOT / "experiments" / "results" / name
    shutil.rmtree(out_dir, ignore_errors=True)

    # The server only ever holds the shared parameters (naive: the whole model; encoder_head:
    # the head; hybrid: core encoder + head). Private encoder sizes do not matter here.
    core_dim = core_input_dim(cfg["core_features"])
    template = HybridModel(core_dim, 1) if mode == "hybrid" else BankModel(input_dim=core_dim)
    initial = ArrayRecord(shared_state(template, mode))
    client_app = make_client_app(part_dir, out_dir, mode=mode, lr=mlp.get("lr", 1e-3),
                                 batch_size=mlp.get("batch_size", 512),
                                 local_epochs=fl.get("local_epochs", 1), seed=seed)
    server_app = make_server_app(initial, n_clients, rounds, out_dir)
    log.info("FL mode=%s, %d clients, %d rounds, seed %d", mode, n_clients, rounds, seed)
    # Ray starts (CPUs / num_cpus) workers; size it so there is about one worker per bank.
    cpus_per_client = max(1, (os.cpu_count() or 1) // n_clients)
    run_simulation(server_app=server_app, client_app=client_app, num_supernodes=n_clients,
                   backend_config={"client_resources": {"num_cpus": cpus_per_client, "num_gpus": 0.0}})

    score_files = sorted((out_dir / "scores").glob("client_*.parquet"))
    if len(score_files) != n_clients:
        sys.exit(f"Expected {n_clients} score files in {out_dir / 'scores'}, found {len(score_files)}")
    scores = pd.concat([pd.read_parquet(p).assign(client=p.stem) for p in score_files], ignore_index=True)
    try:
        pattern_of = load_table(ROOT / cfg["data"]["processed_dir"] / "patterns").set_index("txn_id")["pattern"]
    except FileNotFoundError:
        pattern_of = None
    rows, by_pattern = summarize(scores, pattern_of, f"fl_{mode}")
    for r in rows + by_pattern:
        r["seed"] = seed

    res = ROOT / "experiments" / "results"
    pd.DataFrame(rows).to_csv(res / f"{name}.csv", index=False)
    if by_pattern:
        pd.DataFrame(by_pattern).to_csv(res / f"{name}_by_pattern.csv", index=False)
    pooled = rows[-1]
    log.info("pooled test: pr_auc=%.3f f1=%.3f recall=%.3f fpr=%.4f", pooled["pr_auc"], pooled["f1"],
             pooled["recall"], pooled["fpr"])
    log.info("Saved %s", res / f"{name}.csv")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
