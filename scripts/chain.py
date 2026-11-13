#!/usr/bin/env python
"""Manage the local Hardhat node (the permissioned-chain prototype).

    python scripts/chain.py up        start the node in the background (http://127.0.0.1:8545)
    python scripts/chain.py down      stop the node started by `up`
    python scripts/chain.py status    is a node answering, and what does the registry hold
    python scripts/chain.py restore   re-anchor the off-chain journal after the node was restarted

A Hardhat node keeps its state in memory, so stopping it discards the chain.
The pipeline keeps an off-chain journal of everything it registered; `restore`
(also done automatically when a ledger is opened) registers those records again
on a fresh contract. On a real permissioned network this step does not exist.
"""

from __future__ import annotations

from tbml.ledger.node import hardhat_installed, rpc_alive, start_node, stop_node
from tbml.utils import get_logger, parse_cli

log = get_logger("chain")


def main() -> None:
    args, cfg = parse_cli(__doc__, lambda p: p.add_argument("command", choices=["up", "down", "status", "restore"]))
    url = str(cfg.ledger.rpc_url)
    if args.command == "up":
        if not hardhat_installed(cfg):
            raise SystemExit("Hardhat is not installed. Run `make chain-install` (needs Node.js), or set ledger.backend=python.")
        raise SystemExit(0 if start_node(cfg) else 1)
    if args.command == "down":
        log.info("node stopped" if stop_node(cfg) else "no node was started by this project")
        return
    alive = rpc_alive(url)
    log.info("node at %s: %s", url, "running" if alive else "not running")
    if args.command == "status" and not alive:
        return
    from tbml.ledger.factory import open_ledger

    ledger = open_ledger(cfg)
    stats = ledger.stats()
    log.info("%s ledger: %d records, %d organisations, %d storage bytes, total cost %d %s", stats["backend"], stats["n_records"],
             stats["n_organisations"], stats["storage_bytes"], stats["total_cost"], stats["cost_unit"])
    ledger.close()


if __name__ == "__main__":
    main()
