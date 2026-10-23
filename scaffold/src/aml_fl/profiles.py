"""Synthetic bank-specific customer profiles.

The IBM dataset has transactions only (no KYC / customer data). Real banks hold very
different customer attributes, which is the reason our FL design needs per-bank encoders.
To make the schemas genuinely different, we generate customer attributes per sender
account; each client bank later keeps only its own subset of these columns (see the
`schemas` section of the config).

Circularity control
-------------------
Attributes are mostly random. With probability `label_signal`, an attribute of an
account that sent laundering money *in the training period* is drawn from a "riskier"
distribution (newer account, higher KYC risk, ...). Only training-period labels are used,
so validation/test labels never influence the features. Run with label_signal = 0.0 for
the ablation that shows results do not depend on this synthetic signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PROFILE_COLUMNS = {
    # name: kind
    "account_age_days": "numeric",
    "customer_type": "categorical",
    "kyc_risk_level": "categorical",
    "occupation_category": "categorical",
    "declared_annual_income": "numeric",
    "branch_region": "categorical",
    "account_type": "categorical",
    "num_linked_accounts": "numeric",
    "onboarding_channel": "categorical",
}

_CUSTOMER_TYPES = ["individual", "business"]
_KYC = ["low", "medium", "high"]
_OCCUPATIONS = ["salaried", "self_employed", "student", "retired", "trader", "unemployed"]
_REGIONS = ["north", "south", "east", "west", "central"]
_ACCOUNT_TYPES = ["savings", "current", "nri", "salary"]
_CHANNELS = ["branch", "online", "agent"]
# Every value a categorical profile column can take (used when banks agree on one layout).
PROFILE_VOCAB = {"customer_type": _CUSTOMER_TYPES, "kyc_risk_level": _KYC, "occupation_category": _OCCUPATIONS,
                 "branch_region": _REGIONS, "account_type": _ACCOUNT_TYPES, "onboarding_channel": _CHANNELS}


def generate_profiles(accounts: pd.Series, risky: pd.Series, label_signal: float,
                      seed: int) -> pd.DataFrame:
    """Generate one profile row per account.

    accounts: unique account keys.
    risky:    boolean Series aligned with `accounts`; True if the account sent laundering
              money in the TRAINING period.
    """
    if not 0.0 <= label_signal <= 1.0:
        raise ValueError("label_signal must be in [0, 1]")
    rng = np.random.default_rng(seed)
    n = len(accounts)
    risky = np.asarray(risky, dtype=bool)

    def shifted() -> np.ndarray:
        """Independent per-attribute draw: does this risky account get a risky value?"""
        return risky & (rng.random(n) < label_signal)

    # account_age_days: log-normal around ~3 years; risky -> under 6 months
    age = rng.lognormal(mean=np.log(1100), sigma=0.8, size=n)
    s = shifted()
    age[s] = rng.uniform(5, 180, size=s.sum())

    customer_type = rng.choice(_CUSTOMER_TYPES, size=n, p=[0.8, 0.2])
    s = shifted()
    customer_type[s] = "business"

    kyc = rng.choice(_KYC, size=n, p=[0.7, 0.25, 0.05])
    s = shifted()
    kyc[s] = rng.choice(["medium", "high"], size=s.sum(), p=[0.4, 0.6])

    occupation = rng.choice(_OCCUPATIONS, size=n, p=[0.45, 0.2, 0.1, 0.1, 0.1, 0.05])
    s = shifted()
    occupation[s] = rng.choice(["student", "unemployed", "trader"], size=s.sum())

    income = rng.lognormal(mean=np.log(600_000), sigma=0.7, size=n)
    s = shifted()
    income[s] = rng.lognormal(mean=np.log(150_000), sigma=0.5, size=s.sum())

    region = rng.choice(_REGIONS, size=n)  # pure noise on purpose

    account_type = rng.choice(_ACCOUNT_TYPES, size=n, p=[0.55, 0.3, 0.05, 0.1])
    s = shifted()
    account_type[s] = "current"

    linked = rng.poisson(1.2, size=n).astype(float)
    s = shifted()
    linked[s] = rng.poisson(5, size=s.sum())

    channel = rng.choice(_CHANNELS, size=n, p=[0.5, 0.45, 0.05])
    s = shifted()
    channel[s] = rng.choice(["online", "agent"], size=s.sum(), p=[0.6, 0.4])

    return pd.DataFrame({
        "account_key": np.asarray(accounts),
        "account_age_days": age.round().astype(np.float32),
        "customer_type": customer_type,
        "kyc_risk_level": kyc,
        "occupation_category": occupation,
        "declared_annual_income": income.round(-3).astype(np.float32),
        "branch_region": region,
        "account_type": account_type,
        "num_linked_accounts": linked.astype(np.float32),
        "onboarding_channel": channel,
    })


def attach_profiles(df: pd.DataFrame, split: pd.Series, label_signal: float, seed: int) -> pd.DataFrame:
    """Generate profiles for every sender account and join them onto the transactions."""
    accounts = pd.Series(df["src_key"].unique())
    train_launderers = set(df.loc[(split == "train") & (df["label"] == 1), "src_key"])
    risky = accounts.isin(train_launderers)
    prof = generate_profiles(accounts, risky, label_signal, seed)
    out = df.merge(prof, left_on="src_key", right_on="account_key", how="left")
    return out.drop(columns="account_key")
