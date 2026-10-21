"""Generate a small fake CSV in the exact IBM AML format, for tests and for trying the
pipeline before the real dataset is downloaded. NOT for results — use the real data.

Also writes a matching '*_Patterns.txt' (same format as the IBM one) listing the injected
laundering attempts: fan-in, fan-out, cycle and scatter-gather.

Usage:
    python scripts/make_toy_data.py --out data/raw/toy_Trans.csv --rows 20000
    # -> data/raw/toy_Trans.csv and data/raw/toy_Patterns.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CURRENCIES = {"US Dollar": 1.0, "Euro": 0.85, "Yuan": 6.7, "Rupee": 74.0, "UK Pound": 0.75}
FORMATS = ["Cheque", "Credit Card", "ACH", "Cash", "Wire", "Reinvestment"]
PATTERN_SIZE = 12  # transactions per injected attempt


def make_toy_with_patterns(rows: int = 20000, n_banks: int = 12, n_accounts: int = 1500,
                           seed: int = 0) -> tuple[pd.DataFrame, list[tuple[str, np.ndarray]]]:
    """Return the transactions and the injected attempts as [(pattern name, row indices)]."""
    rng = np.random.default_rng(seed)
    acct_bank = rng.integers(1, n_banks + 1, size=n_accounts)
    acct_id = np.array([f"{rng.integers(16**8, 16**9):X}" for _ in range(n_accounts)])
    start = pd.Timestamp("2022-09-01")

    src = rng.integers(0, n_accounts, size=rows)
    dst = rng.integers(0, n_accounts, size=rows)
    minutes = np.sort(rng.integers(0, 10 * 24 * 60, size=rows))
    cur_pay = rng.choice(list(CURRENCIES), size=rows, p=[0.6, 0.15, 0.1, 0.1, 0.05])
    cur_recv = np.where(rng.random(rows) < 0.9, cur_pay, rng.choice(list(CURRENCIES), size=rows))
    usd = rng.lognormal(np.log(2000), 1.2, size=rows)
    label = np.zeros(rows, dtype=int)

    # Inject laundering attempts. Each one takes PATTERN_SIZE rows (time-ordered, because
    # `minutes` is sorted) and rewrites who pays whom. Rows are never reused across attempts.
    n_each = max(1, rows // 8000)
    free = rng.permutation(rows)
    attempts: list[tuple[str, np.ndarray]] = []
    for name in ["FAN-IN", "FAN-OUT", "CYCLE", "SCATTER-GATHER"]:
        for _ in range(n_each):
            idx, free = np.sort(free[:PATTERN_SIZE]), free[PATTERN_SIZE:]
            accts = rng.choice(n_accounts, size=PATTERN_SIZE + 2, replace=False)
            hub, sink, others = accts[0], accts[1], accts[2:]
            if name == "FAN-IN":      # many accounts -> one mule
                src[idx], dst[idx] = others, hub
            elif name == "FAN-OUT":   # one account -> many accounts
                src[idx], dst[idx] = hub, others
            elif name == "CYCLE":     # a0 -> a1 -> ... -> a11 -> a0
                src[idx], dst[idx] = others, np.roll(others, -1)
            else:                     # hub -> 6 intermediates -> sink
                half = PATTERN_SIZE // 2
                src[idx[:half]], dst[idx[:half]] = hub, others[:half]
                src[idx[half:]], dst[idx[half:]] = others[:half], sink
            usd[idx] = rng.uniform(8000, 9900, size=PATTERN_SIZE)  # just under a reporting threshold
            label[idx] = 1
            attempts.append((name, idx))

    paid = usd * np.array([CURRENCIES[c] for c in cur_pay])
    recv = usd * np.array([CURRENCIES[c] for c in cur_recv])
    ts = start + pd.to_timedelta(minutes, unit="m")
    df = pd.DataFrame({
        "Timestamp": ts.strftime("%Y/%m/%d %H:%M"),
        "From Bank": acct_bank[src],
        "Account": acct_id[src],
        "To Bank": acct_bank[dst],
        "Account.1": acct_id[dst],
        "Amount Received": recv.round(2),
        "Receiving Currency": cur_recv,
        "Amount Paid": paid.round(2),
        "Payment Currency": cur_pay,
        "Payment Format": rng.choice(FORMATS, size=rows),
        "Is Laundering": label,
    })
    return df, attempts


def make_toy(rows: int = 20000, n_banks: int = 12, n_accounts: int = 1500, seed: int = 0) -> pd.DataFrame:
    return make_toy_with_patterns(rows, n_banks, n_accounts, seed)[0]


def write_ibm_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write with the original header (two columns literally named 'Account')."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [c if c != "Account.1" else "Account" for c in df.columns]
    df.to_csv(path, index=False, header=header)


def patterns_path_for(trans_csv: str | Path) -> Path:
    """'x/toy_Trans.csv' -> 'x/toy_Patterns.txt' (the IBM naming)."""
    p = Path(trans_csv)
    return p.with_name(p.name.replace("_Trans.csv", "_Patterns.txt")) if p.name.endswith("_Trans.csv") \
        else p.with_suffix(".patterns.txt")


def write_patterns_txt(df: pd.DataFrame, attempts: list[tuple[str, np.ndarray]], path: str | Path) -> None:
    """Write the attempts in the IBM '*_Patterns.txt' format (banks zero-padded, as in the original)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["From Bank"] = out["From Bank"].map("{:03d}".format)
    out["To Bank"] = out["To Bank"].map("{:03d}".format)
    with open(path, "w", encoding="utf-8") as f:
        for name, idx in attempts:
            f.write(f"BEGIN LAUNDERING ATTEMPT - {name}:  Toy {name.title()}\n")
            out.iloc[idx].to_csv(f, index=False, header=False)
            f.write(f"END LAUNDERING ATTEMPT - {name}\n\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/toy_Trans.csv")
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    trans, toy_attempts = make_toy_with_patterns(a.rows, seed=a.seed)
    write_ibm_csv(trans, a.out)
    patterns_out = patterns_path_for(a.out)
    write_patterns_txt(trans, toy_attempts, patterns_out)
    print(f"Wrote {a.out} and {patterns_out}")
