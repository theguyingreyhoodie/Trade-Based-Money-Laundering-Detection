"""Load and clean the IBM 'Transactions for Anti Money Laundering' dataset (HI-Small).

Raw CSV columns (Kaggle: ealtman2019/ibm-transactions-for-anti-money-laundering-aml):
    Timestamp, From Bank, Account, To Bank, Account.1, Amount Received,
    Receiving Currency, Amount Paid, Payment Currency, Payment Format, Is Laundering

Note: the file has two columns literally named "Account"; pandas renames the second one
to "Account.1".
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

RENAME = {
    "Timestamp": "timestamp",
    "From Bank": "src_bank",
    "Account": "src_account",
    "To Bank": "dst_bank",
    "Account.1": "dst_account",
    "Amount Received": "amount_received",
    "Receiving Currency": "receiving_currency",
    "Amount Paid": "amount_paid",
    "Payment Currency": "payment_currency",
    "Payment Format": "payment_format",
    "Is Laundering": "label",
}

USD = "US Dollar"


def load_raw(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Read the raw CSV and give columns clean snake_case names."""
    df = pd.read_csv(path, nrows=nrows, dtype={"Account": str, "Account.1": str})
    missing = set(RENAME) - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected CSV format, missing columns: {sorted(missing)}")
    return df.rename(columns=RENAME)


def estimate_usd_rates(df: pd.DataFrame) -> dict[str, float]:
    """Estimate 'units of currency X per 1 USD' from the data itself.

    Cross-currency transactions record both the amount paid and the amount received, so
    for payments made in USD and received in currency X, received/paid is the exchange
    rate. We take the median over all such transactions (and the reverse direction).
    Currencies with no USD pair fall back to 1.0 with a warning.
    """
    rates: dict[str, float] = {USD: 1.0}
    paid_usd = df[(df.payment_currency == USD) & (df.receiving_currency != USD) & (df.amount_paid > 0)]
    recv_usd = df[(df.receiving_currency == USD) & (df.payment_currency != USD) & (df.amount_received > 0)]
    fwd = (paid_usd.amount_received / paid_usd.amount_paid).groupby(paid_usd.receiving_currency).median()
    rev = (recv_usd.amount_paid / recv_usd.amount_received).groupby(recv_usd.payment_currency).median()
    rates.update(rev.to_dict())
    rates.update(fwd.to_dict())  # prefer the forward estimate when both exist
    for cur in set(df.payment_currency) | set(df.receiving_currency):
        if cur not in rates:
            log.warning("No USD exchange pair found for %r; assuming rate 1.0", cur)
            rates[cur] = 1.0
    return rates


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Parse types, build global account keys, add USD amounts, sort by time.

    Returns the cleaned frame and the estimated USD exchange rates.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M")
    df["src_bank"] = df["src_bank"].astype(int)
    df["dst_bank"] = df["dst_bank"].astype(int)
    df["label"] = df["label"].astype(np.int8)

    # Account IDs are only guaranteed unique together with their bank.
    df["src_key"] = df["src_bank"].astype(str) + "_" + df["src_account"].astype(str)
    df["dst_key"] = df["dst_bank"].astype(str) + "_" + df["dst_account"].astype(str)

    rates = estimate_usd_rates(df)
    df["amount_usd"] = df["amount_paid"] / df["payment_currency"].map(rates)

    before = len(df)
    df = df[df["amount_paid"] > 0].drop_duplicates()
    if len(df) < before:
        log.info("Dropped %d zero-amount or duplicate rows", before - len(df))

    # Stable sort keeps the original file order for transactions in the same minute.
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    df.insert(0, "txn_id", np.arange(len(df), dtype=np.int64))
    return df, rates


def trim(df: pd.DataFrame, end_time: str | None) -> pd.DataFrame:
    """Keep transactions strictly before `end_time` (None = keep everything).

    HI-Small's normal activity stops on 2022-09-10. The last days (Sept 11-18) hold ~1,100
    transactions of which ~59% are laundering (unfinished attempts), all of which would land
    in the test split and inflate test scores. The default config cuts them off.
    """
    if end_time is None:
        return df
    keep = df["timestamp"] < pd.Timestamp(end_time)
    log.info("Trimmed %d transactions (%d laundering) at or after %s",
             int((~keep).sum()), int(df.loc[~keep, "label"].sum()), end_time)
    return df[keep].reset_index(drop=True)


def summarize(df: pd.DataFrame) -> dict:
    """Quick statistics worth checking (and quoting in the report)."""
    return {
        "rows": int(len(df)),
        "laundering_rows": int(df["label"].sum()),
        "laundering_rate": float(df["label"].mean()),
        "time_start": str(df["timestamp"].min()),
        "time_end": str(df["timestamp"].max()),
        "num_src_banks": int(df["src_bank"].nunique()),
        "num_accounts": int(pd.concat([df["src_key"], df["dst_key"]]).nunique()),
        "payment_formats": df["payment_format"].value_counts().to_dict(),
        "currencies": sorted(set(df["payment_currency"]) | set(df["receiving_currency"])),
    }
