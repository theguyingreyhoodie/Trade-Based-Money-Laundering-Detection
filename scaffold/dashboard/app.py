"""Phase 5 dashboard: a FastAPI app serving the results, the explained alerts and live checks
against the FLAudit contract, plus the static front end in dashboard/static/.

    python dashboard/app.py                      # http://127.0.0.1:8000, alerts of fl_hybrid_s42
    python dashboard/app.py --run fl_hybrid_s43 --port 8001

Needs experiments/results/ (run_baselines.py / run_fl.py --seed / summarize_seeds.py) and
<run>/alerts.json (build_alerts.py). Chain checks need a running node with the contract
(audit_to_chain.py); without one the dashboard still works and reports the chain as offline.
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from aml_fl.explain import alert_hash  # noqa: E402
from aml_fl.utils import load_config  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
STATIC = Path(__file__).resolve().parent / "static"
MODES = ["hybrid", "zero_filled", "encoder_head", "naive"]
ALERT_LIST_FIELDS = ["txn_id", "bank", "timestamp", "amount_usd", "payment_format", "payment_currency",
                     "receiving_currency", "score", "logit", "threshold", "label", "pattern"]


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def create_app(run: str = "fl_hybrid_s42", rpc: str | None = None, results: Path = RESULTS) -> FastAPI:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    num_banks = cfg["partition"]["num_clients"]
    app = FastAPI(title="AML FL dashboard")

    @lru_cache(maxsize=1)
    def alerts() -> dict:
        path = results / run / "alerts.json"
        if not path.exists():
            raise HTTPException(404, f"No alerts for {run}. Run: python scripts/build_alerts.py {run}")
        data = json.loads(path.read_text())
        data["by_txn"] = {a["txn_id"]: a for a in data["alerts"]}
        return data

    def chain():
        """(AuditChain, bank name -> address), or None when no node / contract is reachable."""
        from audit_to_chain import DEPLOYMENT, bank_accounts

        from aml_fl.chain.chain_client import DEFAULT_RPC, AuditChain, connect

        try:
            dep = json.loads(DEPLOYMENT.read_text())
            w3 = connect(rpc or dep.get("rpc") or DEFAULT_RPC)
            if not AuditChain.is_deployed(w3, dep["address"]):
                return None
            c = AuditChain(w3, dep["address"])
            return c, bank_accounts(c, num_banks)
        except Exception:
            return None

    @app.get("/api/overview")
    def overview() -> dict:
        summary = json.loads((ROOT / cfg["data"]["processed_dir"] / "summary.json").read_text())
        seeds = pd.read_csv(results / "seeds_summary.csv")
        per_client = pd.read_csv(results / "seeds_per_client.csv")
        dataset = {k: summary[k] for k in ("rows", "laundering_rows", "laundering_rate", "time_start", "time_end",
                                           "num_src_banks", "num_accounts")}
        dataset["banks"] = {name: {k: c[k] for k in ("rows", "laundering", "num_columns", "per_split")}
                            for name, c in summary["clients"].items()}
        return {"run": run, "dataset": dataset, "setups": _records(seeds), "per_client": _records(per_client)}

    @app.get("/api/history")
    def history(seed: int = 42) -> dict:
        out = {}
        for mode in MODES:
            path = results / f"fl_{mode}_s{seed}" / "history.csv"
            if path.exists():
                out[mode] = _records(pd.read_csv(path)[["round", "train_loss", "val_pr_auc", "val_f1"]])
        return {"seed": seed, "modes": out}

    @app.get("/api/patterns")
    def patterns() -> dict:
        frames = [pd.read_csv(p) for p in results.glob("*_by_pattern.csv") if "_s" in p.stem]
        if not frames:
            return {"rows": []}
        df = pd.concat(frames, ignore_index=True)
        df = df[df["client"].isin(["pooled", "all"])]
        agg = (df.groupby(["setup", "model", "pattern"])
               .agg(recall=("recall", "mean"), positives=("positives", "mean"), seeds=("seed", "nunique"))
               .reset_index())
        return {"rows": _records(agg)}

    @app.get("/api/alerts")
    def list_alerts(bank: str | None = None, label: int | None = None, limit: int = 200,
                    offset: int = 0) -> dict:
        data = alerts()
        rows = [a for a in data["alerts"] if (bank is None or a["bank"] == bank)
                and (label is None or a["label"] == label)]
        rows.sort(key=lambda a: -a.get("logit", a["score"]))   # scores near 1 tie when rounded; log-odds do not
        counts = {}
        for a in data["alerts"]:
            c = counts.setdefault(a["bank"], {"alerts": 0, "laundering": 0})
            c["alerts"] += 1
            c["laundering"] += a["label"]
        return {"run": data["run"], "model_version": data["model_version"], "total": len(rows),
                "per_bank": counts,
                "alerts": [{k: a.get(k) for k in ALERT_LIST_FIELDS} for a in rows[offset:offset + limit]]}

    @app.get("/api/alerts/{txn_id}")
    def get_alert(txn_id: int) -> dict:
        a = alerts()["by_txn"].get(txn_id)
        if a is None:
            raise HTTPException(404, f"No alert for transaction {txn_id}")
        return a

    @app.get("/api/runs")
    def runs() -> dict:
        """FL runs that have an audit.json, newest mode first in MODES order."""
        out = []
        for path in sorted(results.glob("fl_*/audit.json")):
            audit = json.loads(path.read_text())
            out.append({"run": path.parent.name, "rounds": len(audit),
                        "bank_hashes": any("updates" in e for e in audit),
                        "alerts": (path.parent / "alerts.json").exists()})
        order = {f"fl_{m}": i for i, m in enumerate(MODES)}
        out.sort(key=lambda r: (order.get(r["run"].rsplit("_s", 1)[0], 99), r["run"]))
        return {"default": run, "runs": out, "bank_names": cfg["partition"].get("bank_names", {})}

    @app.get("/api/chain/status")
    def chain_status() -> dict:
        found = chain()
        if found is None:
            return {"online": False}
        c, banks = found
        return {"online": True, "address": c.address, "operator": c.operator, "chain_id": c.w3.eth.chain_id,
                "block": c.w3.eth.block_number, "counts": c.event_counts(),
                "banks": [{"bank": b, "address": addr, "registered": c.is_bank(addr)} for b, addr in banks.items()]}

    @app.get("/api/chain/runs/{name}")
    def verify_run(name: str) -> dict:
        path = results / name / "audit.json"
        if not path.exists():
            raise HTTPException(404, f"No audit.json for {name}")
        audit = json.loads(path.read_text())
        found = chain()
        if found is None:
            return {"online": False, "run": name, "rounds": len(audit)}
        c, banks = found
        rounds = []
        for e in audit:
            onchain = c.global_model_hash(name, e["round"])
            updates = {b: c.update_hash(name, e["round"], banks[b]) == h for b, h in e.get("updates", {}).items()}
            rounds.append({"round": e["round"], "hash": e["hash"], "recorded": onchain is not None,
                           "match": onchain == e["hash"], "updates": updates})
        return {"online": True, "run": name, "rounds": rounds,
                "all_match": all(r["match"] and all(r["updates"].values()) for r in rounds)}

    @app.get("/api/chain/alerts/{txn_id}")
    def verify_alert(txn_id: int) -> dict:
        a = get_alert(txn_id)
        recomputed = alert_hash(a)
        result = {"txn_id": txn_id, "stored_hash": a["alert_hash"], "recomputed_hash": recomputed,
                  "intact": recomputed == a["alert_hash"]}
        found = chain()
        if found is None:
            return {**result, "online": False}
        c, banks = found
        logged_by = c.alert_bank(recomputed)
        return {**result, "online": True, "logged": logged_by is not None,
                "logged_by_bank": next((b for b, addr in banks.items() if addr == logged_by), None),
                "verified": result["intact"] and logged_by == banks.get(a["bank"])}

    if STATIC.exists():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(STATIC / "index.html")

    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="fl_hybrid_s42", help="FL run whose alerts are shown")
    ap.add_argument("--rpc", default=None, help="Ethereum node (default: the one the contract was deployed on)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(create_app(args.run, args.rpc), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
