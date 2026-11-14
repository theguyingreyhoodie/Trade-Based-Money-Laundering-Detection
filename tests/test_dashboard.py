"""Investigator dashboard, driven headlessly with Streamlit's AppTest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


@pytest.fixture
def app(pipeline_run, monkeypatch) -> AppTest:
    monkeypatch.setenv("TBML_PROFILE", "test")
    monkeypatch.setenv("TBML_DASHBOARD_OVERRIDES", json.dumps(pipeline_run["overrides"]))
    st.cache_resource.clear()
    st.cache_data.clear()
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    yield at
    st.cache_resource.clear()
    st.cache_data.clear()


@pytest.mark.slow
def test_alert_list_filters_and_detail(app, pipeline_run) -> None:
    alerts = json.loads((pipeline_run["results"] / "alerts" / "alerts.json").read_text())["alerts"]
    assert app.title[0].value == "Alerts"
    assert app.metric[0].value == f"{len(alerts):,}"
    assert app.subheader[0].value.startswith(alerts[0]["alert_id"]), "the highest-scoring alert is shown first"
    text = " ".join(m.value for m in app.markdown)
    assert "Why the model flagged it" in text and "Rules fired" in text and "Suspected typology" in text and "Evidence" in text
    assert alerts[0]["ledger"]["sha256"] in text and alerts[0]["trade_id"] in text

    bands = app.multiselect[0]
    assert bands.label == "Band" and set(bands.options) == {"Critical", "High", "Medium"}
    bands.set_value(["Medium"]).run()
    assert not app.exception
    medium = [a for a in alerts if a["risk_band"] == "Medium"]
    if medium:
        assert app.subheader[0].value.startswith(medium[0]["alert_id"])
        assert "Medium" in app.subheader[0].value
    typology = app.multiselect[1]
    assert typology.label == "Suspected typology"
    app.slider[0].set_value(100).run()
    assert not app.exception


@pytest.mark.slow
def test_verify_on_chain_button(app, pipeline_run) -> None:
    button = next(b for b in app.button if b.label == "Verify on-chain")
    button.click().run()
    assert not app.exception
    assert app.success and "Verified" in app.success[0].value and "fl_client_" in app.success[0].value
    assert not app.error

    # the same alert with its score lowered no longer matches the ledger
    box = next(c for c in app.checkbox if c.label.startswith("Simulate tampering"))
    box.check().run()
    next(b for b in app.button if b.label == "Verify on-chain").click().run()
    assert not app.exception
    assert app.error and "Verification failed" in app.error[0].value
    assert not app.success


@pytest.mark.slow
def test_training_and_ledger_views(app, pipeline_run) -> None:
    primary = json.loads((pipeline_run["results"] / "fl" / "primary.json").read_text())
    app.radio[0].set_value("Federated training").run()
    assert not app.exception and app.title[0].value == "Federated training"
    assert app.metric[0].value == f"{primary['best_round']} of {primary['rounds']}"
    assert app.metric[1].value == f"{primary['final']['f1']:.3f}"
    next(b for b in app.button if b.label.startswith("Verify the saved global model")).click().run()
    assert not app.exception
    assert app.success and "fl_aggregator" in app.success[0].value

    app.radio[0].set_value("Evidence ledger").run()
    assert not app.exception and app.title[0].value == "Evidence ledger"
    assert app.metric[0].value == "Python"
    assert int(app.metric[1].value.replace(",", "")) > primary["rounds"] * (primary["n_clients"] + 1)
    digest = primary["model_hashes"][0]["sha256"]
    app.text_input[0].set_value(digest).run()
    assert app.success and "model_update" in app.success[0].value
    app.text_input[0].set_value("ab" * 32).run()
    assert app.error and "not on the ledger" in app.error[0].value


@pytest.mark.slow
def test_dashboard_without_results(tmp_path, monkeypatch) -> None:
    """Before the pipeline has run, every view explains what to run instead of failing."""
    overrides = [f"paths.results_dir={tmp_path / 'results'}", f"paths.artifacts_dir={tmp_path / 'artifacts'}", f"paths.data_dir={tmp_path / 'data'}",
                 "ledger.backend=python", "ledger.auto_start_node=false"]
    monkeypatch.setenv("TBML_PROFILE", "test")
    monkeypatch.setenv("TBML_DASHBOARD_OVERRIDES", json.dumps(overrides))
    st.cache_resource.clear()
    st.cache_data.clear()
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.info and "generate_alerts.py" in at.info[0].value
    at.radio[0].set_value("Federated training").run()
    assert not at.exception and "run_fl.py" in at.info[0].value
    at.radio[0].set_value("Evidence ledger").run()
    assert not at.exception
    st.cache_resource.clear()
    st.cache_data.clear()
