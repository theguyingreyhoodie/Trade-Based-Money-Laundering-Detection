"""Investigator dashboard (Phase 8, optional).

    make dashboard            # or: .venv/bin/python -m streamlit run dashboard/app.py

Reads the outputs of the pipeline (``results/alerts``, ``results/fl``,
``results/ledger``) and talks to the evidence ledger for live verification.
Views: alerts (filter, inspect, verify on-chain), federated training progress,
and the ledger's audit trail.

Alerts are prompts for human review. Nothing shown here is a finding of
money laundering.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from tbml.config import Config, load_config
from tbml.explain.alerts import verify_alert
from tbml.explain.scoring import BANDS
from tbml.fl.experiments import primary_scenario
from tbml.fl.simulation import model_dir
from tbml.ledger.audit import verify_model

BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
INK_2 = "#52514e"
BAND_MARK = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}

st.set_page_config(page_title="TBML investigator dashboard", page_icon="🔎", layout="wide")


# --------------------------------------------------------------------------- data access


@st.cache_resource
def get_config() -> Config:
    """Project config; tests and demos can redirect paths with TBML_DASHBOARD_OVERRIDES (a JSON list of key=value)."""
    overrides = json.loads(os.environ.get("TBML_DASHBOARD_OVERRIDES", "[]"))
    return load_config(profile=os.environ.get("TBML_PROFILE") or None, overrides=overrides)


@st.cache_data
def read_json(path: str, modified: float) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load(cfg: Config, *parts: str) -> Any:
    path = cfg.path("results_dir", *parts)
    return read_json(str(path), path.stat().st_mtime) if path.is_file() else None


@st.cache_resource
def get_ledger(_cfg: Config) -> Any:
    """Open the evidence ledger once per session; returns ``(ledger, error message)``."""
    try:
        from tbml.ledger.factory import open_ledger

        return open_ledger(_cfg), ""
    except Exception as exc:  # the dashboard stays usable without a ledger
        return None, f"{type(exc).__name__}: {exc}"


def when(ts: int) -> str:
    return dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC") if ts else "-"


def times(event_ts: int, block_ts: int) -> str:
    """Both timestamps a record carries: the submitter's clock and the chain's block time."""
    return f"event time {when(event_ts)} (submitter's clock), block time {when(block_ts)} (chain clock)"


def short(digest: str, n: int = 10) -> str:
    return f"{digest[:n]}…{digest[-6:]}" if digest and len(digest) > n + 8 else digest


# --------------------------------------------------------------------------- alerts view


def alerts_frame(alerts: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Alert": a["alert_id"], "Score": a["risk_score"], "Band": f"{BAND_MARK[a['risk_band']]} {a['risk_band']}",
            "Suspected typology": a["suspected_typology"].replace("_", " "), "Institution": a["institution"], "Invoiced": a["trade"]["invoice_date"],
            "Goods": a["trade"]["goods"], "Value (USD)": a["trade"]["invoice_value_usd"], "Exporter": a["trade"]["exporter_id"],
            "Importer": a["trade"]["importer_id"], "Rules fired": len(a["fired_rules"]),
        }
        for a in alerts
    ])


def _wrap(text: str, width: int = 44) -> list[str]:
    """Break a phrase into lines so it fits beside its bar."""
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return [*lines, line]


def _label_expr(df: pd.DataFrame) -> str:
    """Vega expression that turns a bar's rank into its (multi-line) phrase."""
    table = {str(rank): lines for rank, lines in zip(df["rank"], df["label"])}
    return f"({json.dumps(table)})[datum.value]"


def contribution_chart(alert: dict[str, Any]) -> alt.Chart:
    """SHAP contributions: bars to the right raise the risk, bars to the left lower it."""
    df = pd.DataFrame(alert["top_features"])
    df["label"] = df["text"].map(_wrap)
    order = list(range(len(df)))
    df["rank"] = order
    return (
        alt.Chart(df)
        .mark_bar(size=16, cornerRadiusEnd=4)
        .encode(
            x=alt.X("shap:Q", title="contribution to the log-odds", axis=alt.Axis(grid=True, gridColor="#e1e0d9", domain=False, ticks=False, labelColor=INK_2, titleColor=INK_2)),
            y=alt.Y("rank:O", sort=order, title=None,
                    axis=alt.Axis(labelExpr=_label_expr(df), labelLimit=330, labelColor="#0b0b0b", labelFontSize=12, ticks=False, domain=False, labelPadding=10)),
            color=alt.Color("direction:N", scale=alt.Scale(domain=["raises risk", "lowers risk"], range=[RED, BLUE]),
                            legend=alt.Legend(title=None, orient="bottom", direction="horizontal", labelColor=INK_2)),
            tooltip=[alt.Tooltip("text:N", title="indicator"), alt.Tooltip("shap:Q", title="contribution", format="+.2f"),
                     alt.Tooltip("value:Q", title="indicator value", format=".3f"), alt.Tooltip("direction:N", title="effect")],
        )
        .properties(height=alt.Step(50))
        .configure_view(strokeWidth=0)
        .configure_axisY(labelLineHeight=15)
    )


def show_alert(alert: dict[str, Any], ledger: Any, ledger_error: str) -> None:
    t = alert["trade"]
    st.subheader(f"{alert['alert_id']}  ·  {BAND_MARK[alert['risk_band']]} {alert['risk_band']}  ·  risk score {alert['risk_score']:.0f} / 100")
    st.caption(alert["disclaimer"])
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown(
            f"**Trade** `{alert['trade_id']}`, invoiced {t['invoice_date']}, held by `{alert['institution']}`  \n"
            f"**Parties** exporter `{t['exporter_id']}` → importer `{t['importer_id']}`; paid by `{t['payer_id']}`  \n"
            f"**Goods** {t['goods']} (HS {t['hs_code']}), {t['quantity']:,} {t['unit']}, USD {t['invoice_value_usd']:,.0f}  \n"
            f"**Route** {t['route']}"
        )
        c = alert["components"]
        a, b, d = st.columns(3)
        a.metric("Model probability", f"{c['model_probability']:.2f}")
        b.metric("Rules probability", f"{c['rules_probability']:.2f}")
        d.metric("Model weight", f"{c['model_weight']:.1f}")
        if alert.get("raised_by_rule_floor"):
            st.info("Raised to the Medium band by the rule floor: the rule evidence alone is strong although the model scores this trade low.")
        scores = alert["typology_scores"]
        st.markdown("**Suspected typology** " + alert["suspected_typology"].replace("_", " ")
                    + "  \n" + ", ".join(f"{k.replace('_', ' ')} {v:.2f}" for k, v in scores.items()))
        st.markdown("**Rules fired**")
        if alert["fired_rules"]:
            st.markdown("\n".join(f"- `{r['rule_id']}` {r['name']} ({r['typology'].replace('_', ' ')}, severity {r['severity']:.2f})" for r in alert["fired_rules"]))
        else:
            st.write("None. The score comes from the model alone.")
    with right:
        st.markdown("**Why the model flagged it**")
        st.altair_chart(contribution_chart(alert), use_container_width=True)
        st.caption("SHAP contributions of the five most influential indicators, computed against the institution's own training data.")

    st.markdown("**Evidence**")
    led = alert.get("ledger", {})
    st.markdown(f"SHA-256 `{led.get('sha256', '-')}`  \nTransaction `{led.get('tx_id', '-')}` (block {led.get('block', '-')}), registered by `{led.get('registered_by', '-')}` "
                f"on the {led.get('backend', '-')} ledger  \nScored by global model of round {alert['model']['round']}, SHA-256 `{short(alert['model']['global_model_sha256'], 16)}`")
    one, two, _ = st.columns([2, 3, 5])
    verify = one.button("Verify on-chain", type="primary", key=f"verify-{alert['alert_id']}", disabled=ledger is None)
    tamper = two.checkbox("Simulate tampering (lower the score by 30)", key=f"tamper-{alert['alert_id']}")
    if ledger is None:
        st.warning(f"No evidence ledger available, so live verification is off. {ledger_error}")
    elif verify:
        candidate = copy.deepcopy(alert)
        if tamper:
            candidate["risk_score"] = round(max(0.0, candidate["risk_score"] - 30.0), 1)
        result = verify_alert(ledger, candidate)
        if result["on_ledger"] and result["matches_alert_id"]:
            st.success(f"Verified. The alert's hash is on the ledger, registered by `{result['registered_by']}`; "
                       f"{times(result['event_time'], result['registered_at'])}. The alert has not been altered since it was raised.")
        else:
            st.error(f"Verification failed. The hash of this alert as presented (`{short(result['sha256'], 16)}`) is not on the ledger: "
                     "its content differs from what was registered.")


def view_alerts(cfg: Config, ledger: Any, ledger_error: str) -> None:
    st.title("Alerts")
    data = load(cfg, "alerts", "alerts.json")
    if not data or not data["alerts"]:
        st.info("No alerts yet. Run `python scripts/generate_alerts.py` (or `make all`).")
        return
    alerts = data["alerts"]
    summary = load(cfg, "alerts", "summary.json")
    counts = {b: sum(a["risk_band"] == b for a in alerts) for b in BANDS[1:]}
    cols = st.columns(5)
    cols[0].metric("Alerts", f"{len(alerts):,}")
    for col, band in zip(cols[1:4], ["Critical", "High", "Medium"]):
        col.metric(f"{BAND_MARK[band]} {band}", f"{counts[band]:,}")
    if summary:
        cols[4].metric("On illicit trades (evaluation)", f"{100 * summary['evaluation']['alerts']['precision']:.0f}%")
    st.caption("Alerts are prompts for human review of the test period, raised by each institution on its own trades. They are not findings of laundering.")

    f1, f2, f3, f4 = st.columns([2, 3, 2, 2])
    bands = f1.multiselect("Band", BANDS[1:][::-1], default=BANDS[1:][::-1])
    typologies = sorted({a["suspected_typology"] for a in alerts})
    chosen = f2.multiselect("Suspected typology", typologies, default=[], format_func=lambda t: t.replace("_", " "), placeholder="all typologies")
    institutions = sorted({a["institution"] for a in alerts})
    orgs = f3.multiselect("Institution", institutions, default=[], placeholder="all institutions")
    minimum = f4.slider("Minimum score", 0, 100, int(cfg.scoring.alert_min_score))
    shown = [a for a in alerts if a["risk_band"] in bands and (not chosen or a["suspected_typology"] in chosen) and (not orgs or a["institution"] in orgs)
             and a["risk_score"] >= minimum]
    st.write(f"{len(shown):,} of {len(alerts):,} alerts")
    if not shown:
        st.info("No alert matches these filters.")
        return
    event = st.dataframe(
        alerts_frame(shown), hide_index=True, use_container_width=True, height=330, on_select="rerun", selection_mode="single-row", key="alert-table",
        column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
                       "Value (USD)": st.column_config.NumberColumn(format="%,.0f")},
    )
    rows = event.selection.rows if event and event.selection else []
    position = rows[0] if rows else 0
    if not rows:
        st.caption("Select a row to inspect an alert. Showing the first one.")
    st.divider()
    show_alert(shown[min(position, len(shown) - 1)], ledger, ledger_error)


# --------------------------------------------------------------------------- federated training view


def view_training(cfg: Config, ledger: Any) -> None:
    st.title("Federated training")
    run = load(cfg, "fl", "primary.json")
    if not run:
        st.info("No primary federated run yet. Run `python scripts/run_fl.py`.")
        return
    final, comm = run["final"], run["communication"]
    st.caption(f"{run['scenario']['label']}, {run['strategy']}, seed {run['seed']}. Each institution trains on its own trades; the server receives model "
               "parameters and fixed-size aggregates only.")
    cols = st.columns(5)
    cols[0].metric("Selected round", f"{run['best_round']} of {run['rounds']}")
    cols[1].metric("F1 (test)", f"{final['f1']:.3f}")
    cols[2].metric("PR-AUC (test)", f"{final['pr_auc']:.3f}")
    cols[3].metric("Traffic, whole run", f"{comm['total_bytes'] / 1e6:.1f} MB")
    cols[4].metric(f"Client replies checked ({run['boundary']['violations']} violations)", f"{run['boundary']['replies_checked']}")

    history = pd.DataFrame(run["history"])
    long = history.melt(id_vars="round", value_vars=["test_f1", "val_f1"], var_name="series", value_name="F1")
    long["series"] = long["series"].map({"test_f1": "test period", "val_f1": "validation period"})
    hover = alt.selection_point(fields=["round"], nearest=True, on="pointerover", empty=False)
    base = alt.Chart(long).encode(
        x=alt.X("round:Q", title="round", axis=alt.Axis(grid=False, tickMinStep=1)),
        y=alt.Y("F1:Q", title="F1 at the validation-tuned threshold", scale=alt.Scale(zero=False), axis=alt.Axis(gridColor="#e1e0d9", domain=False, ticks=False)),
        color=alt.Color("series:N", scale=alt.Scale(domain=["test period", "validation period"], range=[BLUE, ORANGE]), legend=alt.Legend(title=None, orient="top")),
    )
    chart = alt.layer(
        base.mark_line(strokeWidth=2),
        base.mark_point(size=70, filled=True, stroke="#fcfcfb", strokeWidth=2).encode(
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            tooltip=[alt.Tooltip("round:Q"), alt.Tooltip("series:N"), alt.Tooltip("F1:Q", format=".3f")]).add_params(hover),
        alt.Chart(history).mark_rule(color="#c3c2b7").encode(x="round:Q").transform_filter(hover),
    ).properties(height=320, title="Global model quality by round").configure_view(strokeWidth=0).configure_title(anchor="start", fontSize=14)
    st.altair_chart(chart, use_container_width=True)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Institutions** (global model, global threshold, own test trades)")
        sizes = {c["client"]: c for c in run["partition"]["clients"]}
        st.dataframe(pd.DataFrame([{"Institution": f"fl_client_{c['client']}", "Trades": sizes[c["client"]]["n"],
                                    "Positive rate": f"{100 * sizes[c['client']]['illicit_rate']:.1f}%", "F1": round(c["f1"], 3),
                                    "Precision": round(c["precision"], 3), "Recall": round(c["recall"], 3)} for c in run["per_client"]]),
                     hide_index=True, use_container_width=True)
    with right:
        st.markdown("**Communication per institution per round**")
        per = comm["per_client_per_round"]
        st.dataframe(pd.DataFrame([{"Message": "model download (train)", "kB": round(per["train_download"] / 1e3, 1)},
                                   {"Message": "model upload (train)", "kB": round(per["train_upload"] / 1e3, 1)},
                                   {"Message": "model download (evaluate)", "kB": round(per["eval_download"] / 1e3, 1)},
                                   {"Message": "score histograms upload (evaluate)", "kB": round(per["eval_upload"] / 1e3, 1)}]),
                     hide_index=True, use_container_width=True)

    st.markdown("**Model hashes on the evidence ledger**")
    hashes = pd.DataFrame([{"Round": h["round"], "What": "global model" if h["kind"] == "global_model" else f"update of fl_client_{h['client']}",
                            "SHA-256": h["sha256"], "Transaction": h.get("tx_id", "-"), "Block": h.get("block", "-")} for h in run["model_hashes"]])
    st.dataframe(hashes, hide_index=True, use_container_width=True, height=260)
    path = model_dir(cfg, int(run["seed"]), primary_scenario(cfg), str(run["strategy"]), "primary") / "global_model.npz"
    if st.button("Verify the saved global model on-chain", disabled=ledger is None or not path.is_file()):
        with np.load(path) as z:
            names = list(z.files)
            digest, record = verify_model(ledger, names, [z[n] for n in names])
        if record.exists:
            st.success(f"Verified. The model file hashes to `{short(digest, 16)}`, registered as `{record.ref_id}` by `{record.org}`; "
                       f"{times(record.client_ts, record.block_ts)}.")
        else:
            st.error(f"Verification failed. The model file hashes to `{short(digest, 16)}`, which is not on the ledger.")


# --------------------------------------------------------------------------- ledger view


def view_ledger(cfg: Config, ledger: Any, ledger_error: str) -> None:
    st.title("Evidence ledger")
    if ledger is None:
        st.warning(f"No evidence ledger available. {ledger_error}")
    else:
        stats = ledger.stats()
        cols = st.columns(5)
        cols[0].metric("Backend", "EVM" if stats["backend"] == "evm" else "Python")
        cols[1].metric("Records", f"{stats['n_records']:,}")
        cols[2].metric("Organisations", f"{stats['n_organisations']}")
        cols[3].metric("Ledger storage", f"{stats['storage_bytes'] / 1e3:,.1f} kB")
        cols[4].metric(f"Total cost ({stats['cost_unit']})", f"{stats['total_cost']:,}")
        st.caption("Only hashes, document types, opaque reference ids and timestamps are on the ledger. Trade documents, model weights and alert contents stay off-chain.")
        trail = pd.DataFrame(ledger.audit_trail())
        if len(trail):
            kinds = sorted(trail["doc_type"].unique())
            chosen = st.multiselect("Document type", kinds, default=[], placeholder="all document types")
            if chosen:
                trail = trail[trail["doc_type"].isin(chosen)]
            trail = trail.sort_values("block", ascending=False).head(500)
            st.write(f"Audit trail, newest first ({len(trail):,} rows shown)")
            st.dataframe(pd.DataFrame({"Block": trail["block"], "Type": trail["doc_type"], "Reference": trail["ref_id"], "Registered by": trail["org"],
                                       "Event time (submitter)": trail["client_ts"].map(when), "Block time (chain)": trail["block_ts"].map(when),
                                       "SHA-256": trail["digest"], "Transaction": trail["tx_id"]}),
                         hide_index=True, use_container_width=True, height=380)
        st.markdown("**Check any hash**")
        digest = st.text_input("SHA-256 (hex)", placeholder="paste a 64-character hash")
        if digest:
            try:
                record = ledger.verify(digest)
                if record.exists:
                    st.success(f"On the ledger: `{record.doc_type}` `{record.ref_id}`, registered by `{record.org}`; {times(record.client_ts, record.block_ts)}.")
                else:
                    st.error("This hash is not on the ledger.")
            except Exception as exc:
                st.error(str(exc))
    bench = load(cfg, "ledger", "benchmark.json")
    if bench:
        st.markdown("**Benchmark**")
        rows = []
        for backend, b in bench.items():
            rows.append({"Backend": backend, "Latency p50 (ms)": round(b["single"]["latency"]["p50_ms"], 1), "Latency p95 (ms)": round(b["single"]["latency"]["p95_ms"], 1),
                         "Records/s, single": round(b["single"]["records_per_second"]), "Records/s, largest batch": round(b["batched"][-1]["records_per_second"]),
                         "Documents/s, Merkle": round(b["merkle"]["records_per_second"]), "Cost per record": round(b["single"]["cost_per_record"]),
                         "Cost per document, Merkle": round(b["merkle"]["cost_per_record"], 1), "Cost unit": b["cost_unit"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# --------------------------------------------------------------------------- app


def main() -> None:
    cfg = get_config()
    ledger, ledger_error = get_ledger(cfg)
    with st.sidebar:
        st.header("TBML detection")
        view = st.radio("View", ["Alerts", "Federated training", "Evidence ledger"], label_visibility="collapsed")
        st.divider()
        if ledger is None:
            st.caption("Ledger: not connected")
        else:
            st.caption(f"Ledger: {'EVM (Hardhat node)' if ledger.backend == 'evm' else 'pure-Python fallback'}")
        st.caption("Synthetic data. Prototype for a B.E. final-year project. Alerts support human review and decide nothing.")
    if view == "Alerts":
        view_alerts(cfg, ledger, ledger_error)
    elif view == "Federated training":
        view_training(cfg, ledger)
    else:
        view_ledger(cfg, ledger, ledger_error)


main()
