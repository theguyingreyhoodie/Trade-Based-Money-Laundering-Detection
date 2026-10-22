"""Flower ServerApp: runs AuditedFedAvg and saves the training history and audit hashes."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from ..utils import save_json
from .strategy import AuditedFedAvg


def make_server_app(initial_arrays: ArrayRecord, num_clients: int, num_rounds: int,
                    out_dir: str | Path) -> ServerApp:
    """Writes <out_dir>/history.csv (per-round metrics) and <out_dir>/audit.json (hashes)."""
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        strategy = AuditedFedAvg(min_train_nodes=num_clients, min_evaluate_nodes=num_clients,
                                 min_available_nodes=num_clients)
        result = strategy.start(grid=grid, initial_arrays=initial_arrays, num_rounds=num_rounds,
                                evaluate_config=ConfigRecord({"num-rounds": num_rounds}))
        rows: dict[int, dict] = {}
        for kind, per_round in (("train", result.train_metrics_clientapp),
                                ("eval", result.evaluate_metrics_clientapp)):
            for rnd, rec in per_round.items():
                rows.setdefault(rnd, {"round": rnd}).update(
                    {k: v for k, v in rec.items() if k != "num-examples"})
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(sorted(rows.values(), key=lambda r: r["round"])).to_csv(out / "history.csv", index=False)
        save_json(strategy.audit_log, out / "audit.json")

    return app
