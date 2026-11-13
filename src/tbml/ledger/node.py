"""Start, stop and probe the local Hardhat node (the permissioned-chain prototype)."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from tbml.config import Config
from tbml.utils import get_logger

log = get_logger(__name__)


def rpc_alive(url: str, timeout: float = 1.0) -> bool:
    """True if a JSON-RPC endpoint answers ``eth_chainId`` at ``url``."""
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "eth_chainId", "params": []}).encode()
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return "result" in json.loads(response.read())
    except Exception:
        return False


def chain_dir(cfg: Config) -> Path:
    return cfg.root / str(cfg.ledger.chain_dir)


def artifact_path(cfg: Config) -> Path:
    return chain_dir(cfg) / "artifacts" / "contracts" / "EvidenceRegistry.sol" / "EvidenceRegistry.json"


def hardhat_installed(cfg: Config) -> bool:
    """Node.js is on the PATH and the Hardhat project has been installed (``make chain-install``)."""
    return shutil.which("node") is not None and (chain_dir(cfg) / "node_modules" / ".bin" / "hardhat").exists()


def _pid_file(cfg: Config) -> Path:
    return cfg.root / ".chain.pid"


def start_node(cfg: Config, wait_seconds: float = 40.0) -> bool:
    """Start ``hardhat node`` in the background; returns True once the RPC answers."""
    url = str(cfg.ledger.rpc_url)
    if rpc_alive(url):
        return True
    if not hardhat_installed(cfg):
        return False
    parsed = urlparse(url)
    log_file = open(cfg.root / ".chain.log", "w", encoding="utf-8")
    process = subprocess.Popen(
        [str(chain_dir(cfg) / "node_modules" / ".bin" / "hardhat"), "node", "--hostname", parsed.hostname or "127.0.0.1", "--port", str(parsed.port or 8545)],
        cwd=chain_dir(cfg), stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True,
    )
    _pid_file(cfg).write_text(str(process.pid))
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if rpc_alive(url):
            log.info("Hardhat node started (pid %d) at %s", process.pid, url)
            return True
        if process.poll() is not None:
            log.warning("Hardhat node exited with code %s (see .chain.log)", process.returncode)
            return False
        time.sleep(0.3)
    log.warning("Hardhat node did not answer within %.0fs", wait_seconds)
    return False


def stop_node(cfg: Config) -> bool:
    """Stop the node started by :func:`start_node`; returns True if a process was signalled."""
    pid_file = _pid_file(cfg)
    if not pid_file.is_file():
        return False
    try:
        pid = int(pid_file.read_text().strip())
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        stopped = True
    except (ValueError, ProcessLookupError, PermissionError):
        stopped = False
    pid_file.unlink(missing_ok=True)
    return stopped
