"""Mean and spread of results over seeds (runs made with --seed).

Reads experiments/results/baselines_s<seed>.csv and fl_<mode>_s<seed>.csv and writes

    experiments/results/seeds_summary.csv          headline rows (centralized, pooled local-only,
                                                   pooled FL): mean / std / min / max per metric
    experiments/results/seeds_per_client.csv       test PR-AUC per client: mean / std

Usage:
    python scripts/summarize_seeds.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
METRICS = ["pr_auc", "roc_auc", "precision", "recall", "f1", "fpr"]


def load_runs(results: Path = RESULTS) -> pd.DataFrame:
    files = sorted(results.glob("baselines_s*.csv")) + sorted(results.glob("fl_*_s*.csv"))
    files = [f for f in files if not f.stem.endswith("_by_pattern")]
    if not files:
        sys.exit(f"No per-seed results in {results}. Run run_baselines.py / run_fl.py with --seed.")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def summarize(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    headline = runs[runs["client"].isin(["all", "pooled"])]
    agg = headline.groupby(["setup", "client", "model"])[METRICS].agg(["mean", "std", "min", "max"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg.insert(0, "seeds", headline.groupby(["setup", "client", "model"])["seed"].nunique())
    agg = agg.reset_index().sort_values("pr_auc_mean", ascending=False)

    per_client = runs[~runs["client"].isin(["all", "pooled"])]
    pc = per_client.groupby(["setup", "model", "client"])["pr_auc"].agg(["mean", "std"]).unstack("client")
    pc.columns = [f"{c}_{s}" for s, c in pc.columns]
    return agg, pc.reset_index()


def main() -> None:
    agg, pc = summarize(load_runs())
    agg.to_csv(RESULTS / "seeds_summary.csv", index=False)
    pc.to_csv(RESULTS / "seeds_per_client.csv", index=False)
    print("| Setup | Model | Seeds | Test PR-AUC (mean ± std) | F1 | Recall | FPR |")
    print("|---|---|---|---|---|---|---|")
    for r in agg.itertuples():
        setup = r.setup if r.client == "all" else f"{r.setup} (pooled)"
        print(f"| {setup} | {r.model} | {r.seeds} | {r.pr_auc_mean:.3f} ± {r.pr_auc_std:.3f} | "
              f"{r.f1_mean:.3f} | {r.recall_mean:.3f} | {r.fpr_mean:.4f} |")
    print(f"\nSaved {RESULTS / 'seeds_summary.csv'} and {RESULTS / 'seeds_per_client.csv'}")


if __name__ == "__main__":
    main()
