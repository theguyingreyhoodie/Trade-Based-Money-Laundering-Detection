"""Small shared helpers: logging, seeding, JSON output, timing, CLI plumbing."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from tbml.config import Config, load_config

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    """Return a logger; the root handler is configured once per process."""
    global _configured
    if not _configured:
        logging.basicConfig(
            level=os.environ.get("TBML_LOG_LEVEL", level or "INFO").upper(),
            format=_LOG_FORMAT,
            datefmt="%H:%M:%S",
        )
        for noisy in ("alembic", "flwr", "ray", "urllib3", "web3", "matplotlib", "PIL", "shap", "numba", "filelock"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        _configured = True
    logger = logging.getLogger(name)
    if level:
        logger.setLevel(level.upper())
    return logger


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and (if it is importable) PyTorch."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
    except ImportError:  # torch is not needed for the data / ledger steps
        pass


def derive_seed(seed: int, *labels: Any) -> int:
    """Stable child seed from a parent seed and labels (independent streams)."""
    import hashlib

    text = "|".join([str(seed), *[str(label) for label in labels]])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def to_jsonable(obj: Any) -> Any:
    """Convert NumPy / pandas / Path values into plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Config):
        return to_jsonable(obj.to_dict())
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def save_json(obj: Any, path: str | Path) -> Path:
    """Write ``obj`` as pretty JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(obj), handle, indent=2, sort_keys=False)
        handle.write("\n")
    return path


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@contextmanager
def timed(label: str, logger: logging.Logger | None = None) -> Iterator[dict[str, float]]:
    """Context manager that measures wall time; yields a dict filled on exit."""
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start
        if logger:
            logger.info("%s took %.2fs", label, result["seconds"])


def build_parser(description: str) -> argparse.ArgumentParser:
    """Argument parser with the options every script shares."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=None, help="path to the YAML config (default: configs/default.yaml)")
    parser.add_argument("--profile", default=os.environ.get("TBML_PROFILE"), help="named profile from the config, e.g. quick")
    parser.add_argument("--seed", type=int, default=None, help="override project.seed")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="override any config value, e.g. --set fl.rounds=10 (repeatable)")
    return parser


def parse_cli(
    description: str,
    add_arguments: Callable[[argparse.ArgumentParser], None] | None = None,
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, Config]:
    """Parse the shared CLI options and return ``(args, config)``."""
    parser = build_parser(description)
    if add_arguments:
        add_arguments(parser)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, overrides=args.overrides, profile=args.profile, seed=args.seed)
    get_logger("tbml", cfg.get_path("logging.level", "INFO"))
    return args, cfg
