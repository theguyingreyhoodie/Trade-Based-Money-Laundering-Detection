"""Configuration loading.

All settings live in one YAML file (``configs/default.yaml``). This module loads
it into a :class:`Config` (a dict with attribute access), applies an optional
named profile and ``key=value`` overrides, and resolves project paths.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import yaml

DEFAULT_CONFIG_NAME = "configs/default.yaml"


class Config(Mapping[str, Any]):
    """Read-mostly nested mapping with attribute access (``cfg.data.n_trades``)."""

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        object.__setattr__(self, "_data", {})
        for key, value in (data or {}).items():
            self._data[key] = _wrap(value)

    # --- mapping protocol -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError as exc:
            raise AttributeError(f"config has no key '{key}'") from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self._data[key] = _wrap(value)

    def __repr__(self) -> str:
        return f"Config({self.to_dict()!r})"

    def __deepcopy__(self, memo: dict) -> "Config":
        return Config(copy.deepcopy(self.to_dict(), memo))

    def __getstate__(self) -> dict:
        return self.to_dict()

    def __setstate__(self, state: dict) -> None:
        object.__setattr__(self, "_data", {})
        for key, value in state.items():
            self._data[key] = _wrap(value)

    # --- helpers ----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a plain (deep) ``dict`` copy."""
        return {k: _unwrap(v) for k, v in self._data.items()}

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Look up ``a.b.c``; return ``default`` if any part is missing."""
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, Config) and part in node:
                node = node[part]
            else:
                return default
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        """Set ``a.b.c = value``, creating intermediate sections as needed."""
        parts = dotted.split(".")
        node = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], Config):
                node._data[part] = Config()
            node = node[part]
        node._data[parts[-1]] = _wrap(value)

    # --- project paths ----------------------------------------------------
    @property
    def root(self) -> Path:
        """Repository root (directory that contains ``configs/``)."""
        return Path(self._data["_root"])

    def path(self, name: str, *parts: str, create: bool = False) -> Path:
        """Resolve ``paths.<name>`` against the repository root."""
        base = Path(self.paths[name])
        full = base if base.is_absolute() else self.root / base
        full = full.joinpath(*parts)
        if create:
            target = full if not full.suffix else full.parent
            target.mkdir(parents=True, exist_ok=True)
        return full

    def seed_dir(self, name: str, seed: int | None = None, *parts: str, create: bool = False) -> Path:
        """Per-seed directory, e.g. ``data/seed_0/...``."""
        seed = self.project.seed if seed is None else seed
        return self.path(name, f"seed_{seed}", *parts, create=create)


def _wrap(value: Any) -> Any:
    if isinstance(value, Config):
        return value
    if isinstance(value, Mapping):
        return Config(value)
    return value


def _unwrap(value: Any) -> Any:
    if isinstance(value, Config):
        return value.to_dict()
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    return value


def find_root(start: Path | None = None) -> Path:
    """Locate the repository root.

    Order: ``TBML_ROOT`` environment variable, then the first parent of
    ``start`` (default: current directory) holding ``configs/default.yaml``,
    then the source checkout this file lives in.
    """
    env = os.environ.get("TBML_ROOT")
    if env:
        return Path(env).resolve()
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / DEFAULT_CONFIG_NAME).is_file():
            return candidate
    return Path(__file__).resolve().parents[2]


def parse_override(text: str) -> tuple[str, Any]:
    """Parse ``a.b=value``; the value is read as YAML so types are inferred."""
    if "=" not in text:
        raise ValueError(f"override must look like key=value, got '{text}'")
    key, raw = text.split("=", 1)
    return key.strip(), yaml.safe_load(raw)


def load_config(
    path: str | Path | None = None,
    overrides: Iterable[str] | None = None,
    profile: str | None = None,
    seed: int | None = None,
) -> Config:
    """Load the YAML config, then apply a profile, CLI overrides and a seed.

    Precedence (lowest to highest): file < profile < ``overrides`` < ``seed``.
    """
    root = find_root()
    cfg_path = Path(path) if path else root / DEFAULT_CONFIG_NAME
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path) if (Path.cwd() / cfg_path).is_file() else root / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    profiles = raw.pop("profiles", {}) or {}
    cfg = Config(raw)
    cfg._data["_root"] = str(root)
    cfg._data["_config_path"] = str(cfg_path)
    cfg._data["_profile"] = profile or ""

    if profile:
        if profile not in profiles:
            raise KeyError(f"unknown profile '{profile}' (available: {sorted(profiles)})")
        for key, value in profiles[profile].items():
            cfg.set_path(key, value)
    for item in overrides or []:
        key, value = parse_override(item)
        cfg.set_path(key, value)
    if seed is not None:
        cfg.set_path("project.seed", int(seed))
    return cfg
