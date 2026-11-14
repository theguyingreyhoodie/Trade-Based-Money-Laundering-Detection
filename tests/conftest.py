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


@pytest.fixture(scope="session")
def pipeline_run(tmp_path_factory: pytest.TempPathFactory):
    """Run the command-line pipeline end to end on the test profile, in a temporary directory.

    Covers the same scripts as ``scripts/run_all.sh`` except the experiment sweep
    and the stretch experiments (see ``test_pipeline.py`` for those). The
    pure-Python ledger is used so the test needs no Node.js.
    """
    import runpy
    import sys

    root = tmp_path_factory.mktemp("pipeline")
    common = [
        "--profile", "test",
        "--set", f"paths.data_dir={root / 'data'}",
        "--set", f"paths.artifacts_dir={root / 'artifacts'}",
        "--set", f"paths.results_dir={root / 'results'}",
        "--set", "ledger.backend=python",
        "--set", "ledger.auto_start_node=false",
        "--set", "fl.experiments.n_clients=[3]",
        "--set", "fl.experiments.alphas=[0.5]",
        "--set", "ledger.benchmark.n_single=20",
        "--set", "ledger.benchmark.n_batched_items=100",
    ]

    def run(script: str, *extra: str) -> None:
        saved = sys.argv
        sys.argv = [script, *common, *extra]
        try:
            runpy.run_path(str(REPO_ROOT / "scripts" / script), run_name="__main__")
        finally:
            sys.argv = saved

    for script in ["generate_data.py", "build_features.py", "run_baselines.py", "ledger_register.py", "run_fl.py", "generate_alerts.py",
                   "ledger_tamper_demo.py", "ledger_benchmark.py", "evaluate.py"]:
        run(script)
    overrides = [common[i + 1] for i in range(2, len(common), 2)]
    return {"root": root, "results": root / "results", "artifacts": root / "artifacts", "run": run, "overrides": overrides}
