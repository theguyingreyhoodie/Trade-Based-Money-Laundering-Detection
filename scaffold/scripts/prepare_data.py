"""Phase 1 pipeline: raw IBM CSV -> features -> time split -> profiles -> client partitions.

Usage:
    python scripts/prepare_data.py --config configs/default.yaml
    python scripts/prepare_data.py --config configs/default.yaml --nrows 500000   # quick dev run

Outputs:
    data/processed/all_transactions.parquet   every feature + every profile column (for
                                               centralized baselines)
    data/processed/summary.json               dataset statistics, USD rates, bank->client map
    data/processed/patterns.parquet           txn_id -> laundering pattern (evaluation only;
                                               needs *_Patterns.txt next to the raw CSV)
    data/partitions/client_i/transactions.parquet   that bank's own columns only
    data/partitions/client_i/schema.json
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aml_fl import data, features, partition, patterns, profiles, split  # noqa: E402
from aml_fl.utils import load_config, save_json, save_table, set_seed, setup_logging  # noqa: E402

log = logging.getLogger("prepare_data")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--nrows", type=int, default=None, help="override data.nrows")
    ap.add_argument("--raw", default=None, help="override data.raw_csv")
    ap.add_argument("--patterns", default=None,
                    help="override data.patterns_txt (default: *_Patterns.txt next to the raw CSV)")
    ap.add_argument("--label-signal", type=float, default=None, help="override profiles.label_signal")
    ap.add_argument("--no-profiles", action="store_true", help="skip synthetic profiles (ablation)")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(ROOT / args.config)
    set_seed(cfg["seed"])
    raw_csv = ROOT / (args.raw or cfg["data"]["raw_csv"])
    nrows = args.nrows if args.nrows is not None else cfg["data"]["nrows"]
    use_profiles = cfg["profiles"]["enabled"] and not args.no_profiles
    label_signal = cfg["profiles"]["label_signal"] if args.label_signal is None else args.label_signal

    if not raw_csv.exists():
        sys.exit(f"Raw data not found at {raw_csv}. See README -> 'Get the data'.")

    log.info("Loading %s (nrows=%s)", raw_csv, nrows)
    df, rates = data.clean(data.load_raw(raw_csv, nrows=nrows))
    summary = data.summarize(df)
    log.info("Rows=%d, laundering=%d (%.4f%%), banks=%d", summary["rows"], summary["laundering_rows"],
             100 * summary["laundering_rate"], summary["num_src_banks"])

    fcfg = cfg["features"]
    df = features.build_features(df, fcfg["burst_window_hours"], fcfg["round_amount_min"])

    scfg = cfg["split"]
    df["split"] = split.time_split(df, scfg["train_frac"], scfg["val_frac"])
    for s in ("train", "val", "test"):
        part = df[df["split"] == s]
        log.info("%-5s %8d txns, %6d laundering", s, len(part), int(part["label"].sum()))

    if use_profiles:
        df = profiles.attach_profiles(df, df["split"], label_signal, cfg["seed"])
        log.info("Attached synthetic profiles (label_signal=%.2f)", label_signal)

    out_proc = ROOT / cfg["data"]["processed_dir"]
    patterns_txt = args.patterns or cfg["data"].get("patterns_txt")
    patterns_txt = ROOT / patterns_txt if patterns_txt else patterns.default_path(raw_csv)
    pattern_counts = None
    if patterns_txt.exists():
        df["pattern"] = patterns.tag_transactions(df, patterns.parse_patterns(patterns_txt))
        save_table(df[["txn_id", "split", "pattern"]], out_proc / "patterns")
        pattern_counts = {s: df.loc[df["split"] == s, "pattern"].value_counts().drop(patterns.NONE, errors="ignore")
                          .to_dict() for s in ("train", "val", "test")}
        log.info("Tagged laundering patterns from %s: %s", patterns_txt, pattern_counts["test"])
    else:
        log.info("No patterns file at %s; skipping per-pattern tags", patterns_txt)

    keep =partition.ID_COLUMNS + features.TXN_FEATURES + (list(profiles.PROFILE_COLUMNS) if use_profiles else [])
    path = save_table(df[keep], out_proc / "all_transactions")
    log.info("Saved %s", path)

    pcfg = cfg["partition"]
    clients, bank_map = partition.partition(df, pcfg["num_clients"], pcfg["mode"], cfg["schemas"], use_profiles)
    out_part = ROOT / cfg["data"]["partitions_dir"]
    client_stats = {}
    for name, cdf in clients.items():
        cols = [c for c in cdf.columns if c not in partition.ID_COLUMNS]
        save_table(cdf, out_part / name / "transactions")
        save_json({"client": name, "columns": cols}, out_part / name / "schema.json")
        client_stats[name] = {
            "rows": int(len(cdf)),
            "laundering": int(cdf["label"].sum()),
            "per_split": {s: [int((cdf["split"] == s).sum()), int(cdf.loc[cdf["split"] == s, "label"].sum())]
                          for s in ("train", "val", "test")},
            "num_columns": len(cols),
        }

    summary.update({
        "usd_rates": rates,
        "label_signal": label_signal if use_profiles else None,
        "partition_mode": pcfg["mode"],
        "patterns_per_split": pattern_counts,
        "clients": client_stats,
        "bank_to_client": {str(k): v for k, v in bank_map.items()},
    })
    save_json(summary, out_proc / "summary.json")
    log.info("Done. Partitions in %s", out_part)


if __name__ == "__main__":
    main()
