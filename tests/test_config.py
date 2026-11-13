"""Config loader: profiles, overrides, paths, seeding helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tbml.config import Config, load_config, parse_override
from tbml.utils import derive_seed, save_json, load_json, seed_everything, to_jsonable

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "default.yaml"


def test_default_config_has_all_sections() -> None:
    cfg = load_config(CONFIG)
    for section in ["project", "paths", "data", "features", "split", "rules", "models", "fl", "ledger", "scoring"]:
        assert section in cfg, section
    assert 0.02 <= cfg.data.illicit_rate <= 0.05
    assert abs(sum(cfg.data.typology_mix.values()) - 1.0) < 1e-9
    assert len(cfg.data.typology_mix) == 10
    assert len(cfg.project.seeds) >= 3


def test_profile_and_overrides_precedence() -> None:
    base = load_config(CONFIG)
    quick = load_config(CONFIG, profile="quick")
    assert quick.data.n_trades < base.data.n_trades
    over = load_config(CONFIG, profile="quick", overrides=["data.n_trades=777", "fl.primary.alpha=0.1"], seed=9)
    assert over.data.n_trades == 777
    assert over.fl.primary.alpha == 0.1
    assert over.project.seed == 9
    with pytest.raises(KeyError):
        load_config(CONFIG, profile="does-not-exist")


def test_override_types() -> None:
    assert parse_override("a.b=3") == ("a.b", 3)
    assert parse_override("a=[1, 2]") == ("a", [1, 2])
    assert parse_override("x=true") == ("x", True)
    assert parse_override("name=fedprox") == ("name", "fedprox")
    with pytest.raises(ValueError):
        parse_override("novalue")


def test_paths_resolve_under_root() -> None:
    cfg = load_config(CONFIG)
    assert cfg.root == REPO_ROOT
    assert cfg.path("results_dir") == REPO_ROOT / "results"
    assert cfg.seed_dir("data_dir", 2, "raw") == REPO_ROOT / "data" / "seed_2" / "raw"


def test_config_is_picklable_and_copyable() -> None:
    import copy
    import pickle

    cfg = load_config(CONFIG)
    clone = pickle.loads(pickle.dumps(cfg))
    assert clone.to_dict() == cfg.to_dict()
    deep = copy.deepcopy(cfg)
    deep.set_path("data.n_trades", 1)
    assert cfg.data.n_trades != 1
    assert isinstance(cfg.data, Config)


def test_seed_helpers(tmp_path: Path) -> None:
    seed_everything(5)
    a = np.random.rand(3)
    seed_everything(5)
    assert np.allclose(a, np.random.rand(3))
    assert derive_seed(0, "x") == derive_seed(0, "x")
    assert derive_seed(0, "x") != derive_seed(1, "x")
    assert derive_seed(0, "x") != derive_seed(0, "y")

    payload = {"a": np.float32(1.5), "b": np.arange(3), "c": float("nan"), "p": tmp_path}
    path = save_json(payload, tmp_path / "out" / "x.json")
    assert load_json(path) == {"a": 1.5, "b": [0, 1, 2], "c": None, "p": str(tmp_path)}
    assert to_jsonable({"k": np.bool_(True)}) == {"k": True}
