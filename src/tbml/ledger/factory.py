"""Choose and open the evidence-ledger backend."""

from __future__ import annotations

from pathlib import Path

from tbml.config import Config
from tbml.ledger.base import EvidenceLedger
from tbml.ledger.node import artifact_path, hardhat_installed, rpc_alive, start_node
from tbml.utils import get_logger

log = get_logger(__name__)


def ledger_dir(cfg: Config) -> Path:
    return cfg.path("artifacts_dir", "ledger", create=True)


def resolve_backend(cfg: Config, backend: str | None = None) -> str:
    """``evm`` if a node answers or can be started, else ``python`` (when the setting is ``auto``)."""
    choice = str(backend or cfg.ledger.backend)
    if choice in ("evm", "python"):
        return choice
    if choice != "auto":
        raise ValueError(f"unknown ledger backend '{choice}' (auto | evm | python)")
    if rpc_alive(str(cfg.ledger.rpc_url)):
        return "evm"
    if bool(cfg.ledger.auto_start_node) and hardhat_installed(cfg) and artifact_path(cfg).is_file() and start_node(cfg):
        return "evm"
    log.warning("no EVM node available (Node.js / Hardhat missing or not startable): using the pure-Python ledger")
    return "python"


def open_ledger(cfg: Config, backend: str | None = None, fresh: bool = False, state_dir: Path | None = None) -> EvidenceLedger:
    """Open the ledger, register the configured organisations, and re-anchor if needed.

    ``fresh=True`` starts an empty registry (new contract, or an emptied Python
    ledger). If the EVM chain was restarted since the last run (a development
    node keeps its state in memory), the records in the off-chain journal are
    registered again so that earlier evidence stays verifiable.
    """
    choice = resolve_backend(cfg, backend)
    base = Path(state_dir) if state_dir else ledger_dir(cfg)
    if choice == "evm":
        from tbml.ledger.evm import EvmLedger

        if not rpc_alive(str(cfg.ledger.rpc_url)) and not (bool(cfg.ledger.auto_start_node) and start_node(cfg)):
            raise RuntimeError(f"no EVM node at {cfg.ledger.rpc_url}; start one with `make chain-up`")
        ledger: EvidenceLedger = EvmLedger(str(cfg.ledger.rpc_url), artifact_path(cfg), base / "evm", max_batch=int(cfg.ledger.max_batch_items), fresh=fresh)
        if fresh:
            ledger.journal.clear()
    else:
        from tbml.ledger.pyledger import PyLedger

        directory = base / "python"
        if fresh and directory.is_dir():
            for name in ("chain.jsonl", "journal.jsonl"):
                (directory / name).unlink(missing_ok=True)
        ledger = PyLedger(directory)

    for name in cfg.ledger.organisations:
        ledger.register_organisation(str(name))

    if getattr(ledger, "restarted", False):
        entries = ledger.journal.entries()
        ledger.journal.clear()
        if entries:
            count = ledger.replay(entries)
            log.warning("the development chain was restarted: re-anchored %d journal records on the new contract", count)
    return ledger
