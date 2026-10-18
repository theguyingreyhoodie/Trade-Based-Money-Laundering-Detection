"""Feature engineering for transaction-level AML detection.

Every account-level feature uses only transactions that happened BEFORE the current one
(for the same account), so no information leaks from the future. The label is never used.

Feature pool (column names):
    Transaction : log_amount, payment_format, payment_currency, receiving_currency,
                  cross_currency, cross_bank, is_round, hour, dayofweek
    Sender      : src_txn_count, src_amount_mean, src_amount_std, src_fan_out,
                  src_num_currencies, src_secs_since_prev, src_burst
    Receiver    : dst_fan_in, dst_received_volume, dst_burst

Caveat: the raw timestamps have minute resolution. Transactions in the same minute are
ordered by file order, so "previous" means "earlier in the file" within a minute.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CATEGORICAL = ["payment_format", "payment_currency", "receiving_currency"]
NUMERIC = [
    "log_amount", "cross_currency", "cross_bank", "is_round", "hour", "dayofweek",
    "src_txn_count", "src_amount_mean", "src_amount_std", "src_fan_out",
    "src_num_currencies", "src_secs_since_prev", "src_burst",
    "dst_fan_in", "dst_received_volume", "dst_burst",
]
TXN_FEATURES = CATEGORICAL + NUMERIC

# Value used for "no previous transaction" in time-gap features (30 days, in seconds).
NO_PREV_SECS = 30 * 24 * 3600.0


def _prior_distinct_count(df: pd.DataFrame, key: str, other: str) -> pd.Series:
    """For each row: how many distinct `other` values `key` had in earlier rows."""
    is_new = ~df.duplicated([key, other])
    return is_new.groupby(df[key]).cumsum() - is_new.astype(int)


def _prior_window_count(df: pd.DataFrame, key: str, hours: int) -> pd.Series:
    """For each row: number of earlier rows for the same `key` within the last `hours`."""
    codes = pd.factorize(df[key])[0].astype(np.int64)
    minutes = ((df["timestamp"] - df["timestamp"].min()).dt.total_seconds() // 60).astype(np.int64).to_numpy()
    big = int(minutes.max()) + hours * 60 + 1
    # Rows are time-sorted, so a stable sort by key keeps time order inside each key.
    order = np.argsort(codes, kind="stable")
    composite = codes[order] * big + minutes[order]          # sorted ascending
    pos = np.arange(len(order))
    # first earlier row of the same key that is still inside (t - window, t]
    lower = np.searchsorted(composite, composite - hours * 60, side="right")
    out = np.empty(len(order), dtype=np.float32)
    out[order] = pos - lower
    return pd.Series(out, index=df.index)


def build_features(df: pd.DataFrame, burst_window_hours: int = 24,
                   round_amount_min: float = 100.0) -> pd.DataFrame:
    """Add all pool features to a cleaned, time-sorted frame (see data.clean)."""
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Input must be sorted by timestamp (use data.clean first).")
    out = df.copy()
    amt = out["amount_usd"]

    # --- transaction-level -------------------------------------------------------
    out["log_amount"] = np.log1p(amt).astype(np.float32)
    out["cross_currency"] = (out["payment_currency"] != out["receiving_currency"]).astype(np.int8)
    out["cross_bank"] = (out["src_bank"] != out["dst_bank"]).astype(np.int8)
    out["is_round"] = ((out["amount_paid"] >= round_amount_min)
                       & (np.isclose(out["amount_paid"] % 100, 0))).astype(np.int8)
    out["hour"] = out["timestamp"].dt.hour.astype(np.int8)
    out["dayofweek"] = out["timestamp"].dt.dayofweek.astype(np.int8)

    # --- sender history (past only) ---------------------------------------------
    g = out.groupby("src_key")
    out["src_txn_count"] = g.cumcount().astype(np.float32)
    prev_sum = g["log_amount"].cumsum() - out["log_amount"]
    prev_sq = (out["log_amount"] ** 2).groupby(out["src_key"]).cumsum() - out["log_amount"] ** 2
    n = out["src_txn_count"].replace(0, np.nan)
    mean = prev_sum / n
    var = (prev_sq / n - mean ** 2).clip(lower=0)
    out["src_amount_mean"] = mean.fillna(0).astype(np.float32)
    out["src_amount_std"] = np.sqrt(var).fillna(0).astype(np.float32)
    out["src_fan_out"] = _prior_distinct_count(out, "src_key", "dst_key").astype(np.float32)
    out["src_num_currencies"] = _prior_distinct_count(out, "src_key", "payment_currency").astype(np.float32)
    gap = g["timestamp"].diff().dt.total_seconds()
    out["src_secs_since_prev"] = gap.fillna(NO_PREV_SECS).astype(np.float32)
    out["src_burst"] = _prior_window_count(out, "src_key", burst_window_hours)

    # --- receiver history (past only) -------------------------------------------
    out["dst_fan_in"] = _prior_distinct_count(out, "dst_key", "src_key").astype(np.float32)
    recv = out.groupby("dst_key")["amount_usd"].cumsum() - amt
    out["dst_received_volume"] = np.log1p(recv.clip(lower=0)).astype(np.float32)
    out["dst_burst"] = _prior_window_count(out, "dst_key", burst_window_hours)

    log.info("Built %d features for %d transactions", len(TXN_FEATURES), len(out))
    return out
