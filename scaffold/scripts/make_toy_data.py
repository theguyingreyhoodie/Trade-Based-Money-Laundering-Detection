"""Generate a small fake CSV in the exact IBM AML format, for tests and for trying the
pipeline before the real dataset is downloaded. NOT for results — use the real data.

Usage:
    python scripts/make_toy_data.py --out data/raw/toy_Trans.csv --rows 20000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CURRENCIES = {"US Dollar": 1.0, "Euro": 0.85, "Yuan": 6.7, "Rupee": 74.0, "UK Pound": 0.75}
FORMATS = ["Cheque", "Credit Card", "ACH", "Cash", "Wire", "Reinvestment"]


def make_toy(rows: int = 20000, n_banks: int = 12, n_accounts: int = 1500, seed: int = 0) -> pd.DataFrame:
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

    # Inject fan-in patterns: a "mule" account receives from many accounts in a short time.
    n_mules = max(1, rows // 2000)
    for _ in range(n_mules):
        mule = rng.integers(0, n_accounts)
        idx = rng.choice(rows, size=12, replace=False)
        dst[idx] = mule
        usd[idx] = rng.uniform(8000, 9900, size=12)  # just under a reporting threshold
        label[idx] = 1

    paid = usd * np.array([CURRENCIES[c] for c in cur_pay])
    recv = usd * np.array([CURRENCIES[c] for c in cur_recv])
    ts = start + pd.to_timedelta(minutes, unit="m")
    return pd.DataFrame({
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


def write_ibm_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write with the original header (two columns literally named 'Account')."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [c if c != "Account.1" else "Account" for c in df.columns]
    df.to_csv(path, index=False, header=header)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/toy_Trans.csv")
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    write_ibm_csv(make_toy(a.rows, seed=a.seed), a.out)
    print(f"Wrote {a.out}")
