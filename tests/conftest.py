"""Shared pytest fixtures.

The suite runs on the ``test`` profile (a few thousand trades) so that the whole
pipeline can be exercised in seconds. Expensive objects are session-scoped.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tbml.config import Config, load_config
from tbml.data.generator import TradeDataset, generate_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg(tmp_path_factory: pytest.TempPathFactory) -> Config:
    """Config on the ``test`` profile, writing to a temporary directory."""
    base = tmp_path_factory.mktemp("tbml")
    config = load_config(REPO_ROOT / "configs" / "default.yaml", profile="test")
    config.set_path("paths.data_dir", str(base / "data"))
    config.set_path("paths.artifacts_dir", str(base / "artifacts"))
    config.set_path("paths.results_dir", str(base / "results"))
    config.set_path("ledger.backend", "python")
    return config


@pytest.fixture(scope="session")
def dataset(cfg: Config) -> TradeDataset:
    """One small generated dataset (seed 0) shared by the whole suite."""
    return generate_dataset(cfg, 0)


@pytest.fixture(scope="session")
def linked(dataset: TradeDataset):
    """Linked trades (one row per invoice) of the shared dataset."""
    from tbml.features.link import link_dataset

    return link_dataset(dataset)


@pytest.fixture(scope="session")
def features(cfg: Config, dataset: TradeDataset, linked):
    """Centralised feature set of the shared dataset."""
    from tbml.features.build import Reference, build_features

    ref = Reference.from_tables(dataset.tables)
    return build_features(linked, dataset.tables["kyc_entities"], dataset.tables["banks"], ref, cfg, dataset.tables["labels"])
