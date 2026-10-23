"""Dashboard API (dashboard/app.py) on a small results folder, with no chain reachable."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dashboard"))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from aml_fl.explain import alert_hash  # noqa: E402

NO_CHAIN = "http://127.0.0.1:9"   # nothing listens here


def _alert(txn, bank, score, logit, label=0):
    a = {"bank": bank, "txn_id": txn, "timestamp": "2022-09-09 14:51:00", "src_key": "1_A", "dst_key": "2_B",
         "payment_format": "ACH", "payment_currency": "US Dollar", "receiving_currency": "Euro", "amount_usd": 120.5,
         "score": score, "threshold": 0.5, "base_value": -1.0, "logit": logit, "label": label, "pattern": "none",
         "top_features": [{"feature": "log_amount", "value": 4.8, "contribution": 0.7}], "model_version": "ab" * 32}
    a["alert_hash"] = alert_hash(a)
    return a


@pytest.fixture
def results(tmp_path):
    run = tmp_path / "fl_hybrid_s42"
    run.mkdir()
    (run / "audit.json").write_text(json.dumps([{"round": 1, "hash": "cd" * 32, "updates": {"client_0": "ef" * 32}}]))
    alerts = [_alert(10, "client_0", 0.9999, 9.5, label=1), _alert(11, "client_1", 0.9999, 12.0), _alert(12, "client_0", 0.7, 0.85)]
    (run / "alerts.json").write_text(json.dumps({"run": "fl_hybrid_s42", "model_version": "ab" * 32, "alerts": alerts}))
    (run / "history.csv").write_text("round,train_loss,val_pr_auc,val_f1\n1,0.5,0.1,0.2\n2,0.4,0.15,0.25\n")
    return tmp_path


@pytest.fixture
def client(results):
    from app import create_app

    return TestClient(create_app("fl_hybrid_s42", rpc=NO_CHAIN, results=results))


def test_alert_queue_is_sorted_by_log_odds_and_filtered(client):
    data = client.get("/api/alerts").json()
    assert [a["txn_id"] for a in data["alerts"]] == [11, 10, 12]   # equal scores, ordered by logit
    assert data["per_bank"] == {"client_0": {"alerts": 2, "laundering": 1}, "client_1": {"alerts": 1, "laundering": 0}}
    assert [a["txn_id"] for a in client.get("/api/alerts?bank=client_0").json()["alerts"]] == [10, 12]
    assert "top_features" not in data["alerts"][0]                  # the list stays light
    assert client.get("/api/alerts/12").json()["top_features"][0]["feature"] == "log_amount"
    assert client.get("/api/alerts/99").status_code == 404


def test_alert_check_detects_a_changed_alert_without_a_chain(client, results):
    ok = client.get("/api/chain/alerts/10").json()
    assert ok["intact"] and ok["online"] is False
    path = results / "fl_hybrid_s42" / "alerts.json"
    data = json.loads(path.read_text())
    data["alerts"][0]["score"] = 0.42                                 # edited after it was hashed
    path.write_text(json.dumps(data))
    from app import create_app

    fresh = TestClient(create_app("fl_hybrid_s42", rpc=NO_CHAIN, results=results))
    assert fresh.get("/api/chain/alerts/10").json()["intact"] is False


def test_runs_history_and_offline_chain(client):
    runs = client.get("/api/runs").json()
    assert runs["runs"] == [{"run": "fl_hybrid_s42", "rounds": 1, "bank_hashes": True, "alerts": True}]
    assert len(client.get("/api/history?seed=42").json()["modes"]["hybrid"]) == 2
    assert client.get("/api/chain/status").json() == {"online": False}
    assert client.get("/api/chain/runs/fl_hybrid_s42").json() == {"online": False, "run": "fl_hybrid_s42", "rounds": 1}
