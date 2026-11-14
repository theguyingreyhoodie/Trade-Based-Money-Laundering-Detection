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


def page_html(at: AppTest) -> str:
    """Everything the app rendered with st.html (headers, the open alert, the evidence chain)."""
    return " ".join(h.value for h in at.html)


def open_alert(at: AppTest) -> str:
    """The header of the alert shown in the detail panel."""
    return next(h.value for h in at.html if "tb-alert-head" in h.value)


@pytest.mark.slow
def test_alert_list_filters_and_detail(app, pipeline_run) -> None:
    alerts = json.loads((pipeline_run["results"] / "alerts" / "alerts.json").read_text())["alerts"]
    assert app.title[0].value == "Alerts"
    text = page_html(app)
    assert f"{len(alerts):,} alerts" in text
    assert alerts[0]["alert_id"] in open_alert(app), "the highest-scoring alert is open first"
    assert "Why the model flagged it" in text and "Rules fired" in text and "Score" in text and "Evidence" in text
    assert alerts[0]["ledger"]["sha256"][:12] in text and alerts[0]["trade_id"] in text
    assert "Not checked" in text, "nothing is claimed verified before the check runs"
    assert not any(ch in text for ch in "🟢🟡🟠🔴🔎"), "no emoji in the interface"

    # AppTest lists a radio's options as displayed: the band followed by its count
    band = app.radio(key="band")
    assert [o.split(" ")[0] for o in band.options] == ["All", "Critical", "High", "Medium"]
    assert band.options[0] == f"All :gray[{len(alerts):,}]"
    band.set_value(next(o for o in band.options if o.startswith("Medium"))).run()
    assert not app.exception
    medium = [a for a in alerts if a["risk_band"] == "Medium"]
    if medium:
        head = open_alert(app)
        assert medium[0]["alert_id"] in head and "Medium" in head

        # search narrows the list to one alert
        app.text_input(key="search").set_value(medium[-1]["alert_id"]).run()
        assert not app.exception
        assert medium[-1]["alert_id"] in open_alert(app)
        assert "1 of " in page_html(app)
        app.text_input(key="search").set_value("no-such-alert").run()
        assert app.info and "No alert matches" in app.info[0].value
        app.text_input(key="search").set_value("").run()

    typology = app.selectbox(key="typology")
    assert typology.options[0] == "All typologies"
    chosen = typology.options[1]
    typology.set_value(chosen).run()
    assert not app.exception


@pytest.mark.slow
def test_institution_scope(app, pipeline_run) -> None:
    alerts = json.loads((pipeline_run["results"] / "alerts" / "alerts.json").read_text())["alerts"]
    scope = app.selectbox(key="scope")
    assert scope.options[0] == "All institutions"
    institution = scope.options[1]
    scope.set_value(institution).run()
    assert not app.exception
    own = [a for a in alerts if a["institution"] == institution]
    assert f"{len(own):,} alerts</b> raised by {institution}" in page_html(app)
    assert own[0]["alert_id"] in open_alert(app)


@pytest.mark.slow
def test_verify_on_chain_button(app, pipeline_run) -> None:
    button = next(b for b in app.button if b.label == "Verify on-chain")
    button.click().run()
    assert not app.exception
    assert app.success and "Verified" in app.success[0].value and "fl_client_" in app.success[0].value
    assert not app.error
    assert page_html(app).count("On the ledger") == 2, "both links of the evidence chain are confirmed"

    # the same alert with its score lowered no longer matches the ledger
    box = next(c for c in app.checkbox if c.label.startswith("Simulate tampering"))
    box.check().run()
    next(b for b in app.button if b.label == "Verify on-chain").click().run()
    assert not app.exception
    assert app.error and "Verification failed" in app.error[0].value
    assert not app.success
    text = page_html(app)
    assert "Not on the ledger" in text and "On the ledger" in text, "the model link still verifies; the altered alert does not"


@pytest.mark.slow
def test_training_and_ledger_views(app, pipeline_run) -> None:
    primary = json.loads((pipeline_run["results"] / "fl" / "primary.json").read_text())
    app.radio(key="nav").set_value("Federated training").run()
    assert not app.exception and app.title[0].value == "Federated training"
    assert app.metric[0].label == "F1, test period" and app.metric[0].value == f"{primary['final']['f1']:.3f}"
    assert app.metric[2].value == f"{primary['best_round']} of {primary['rounds']}"
    assert app.metric[4].value == str(primary["boundary"]["replies_checked"])
    next(b for b in app.button if b.label.startswith("Verify the saved global model")).click().run()
    assert not app.exception
    assert app.success and "fl_aggregator" in app.success[0].value

    app.radio(key="nav").set_value("Evidence ledger").run()
    assert not app.exception and app.title[0].value == "Evidence ledger"
    assert app.metric[4].value == "Python"
    assert int(app.metric[0].value.replace(",", "")) > primary["rounds"] * (primary["n_clients"] + 1)
    digest = primary["model_hashes"][0]["sha256"]
    app.text_input(key="hash-check").set_value(digest).run()
    assert app.success and "model_update" in app.success[0].value
    app.text_input(key="hash-check").set_value("ab" * 32).run()
    assert app.error and "not on the ledger" in app.error[0].value
    doctype = app.radio(key="doctype")
    assert doctype.options[0] == "All" and "model update" in doctype.options
    doctype.set_value("model update").run()
    assert not app.exception


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
    at.radio(key="nav").set_value("Federated training").run()
    assert not at.exception and "run_fl.py" in at.info[0].value
    at.radio(key="nav").set_value("Evidence ledger").run()
    assert not at.exception
    st.cache_resource.clear()
    st.cache_data.clear()
