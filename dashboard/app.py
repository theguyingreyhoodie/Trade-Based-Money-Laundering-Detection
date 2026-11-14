"""Investigator dashboard (Phase 8, optional).

    make dashboard            # or: .venv/bin/python -m streamlit run dashboard/app.py

Reads the outputs of the pipeline (``results/alerts``, ``results/fl``,
``results/ledger``) and talks to the evidence ledger for live verification.
Screens: the alert queue (filter, inspect, verify the evidence on-chain), the
federated training run, and the ledger's audit trail.

Alerts are prompts for human review. Nothing shown here is a finding of
money laundering.

Look: the chrome carries no colour (ink text, a near-black primary action,
white panels on a blue-grey ground); the one data hue marks the main data
element of each screen. Theme tokens live in ``.streamlit/config.toml`` and
in ``CSS`` below.
"""

from __future__ import annotations

import base64
import copy
import datetime as dt
import html
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

HERE = Path(__file__).resolve().parent
HUE = "#1d5fbf"  # the one data hue, used on each screen's main data element
INK, MUTED, FAINT, RULE = "#0a0a0a", "#5f6368", "#a3a7ad", "#eceef1"
BAND_ORDER = list(BANDS[1:])[::-1]  # Critical, High, Medium (Low raises no alert)
BAND_PILL = {"Critical": "negative", "High": "warning", "Medium": "neutral", "Low": "neutral"}
VIEWS = ["Alerts", "Federated training", "Evidence ledger"]
ALL_INSTITUTIONS = "All institutions"
ALL_TYPOLOGIES = "All typologies"
DOC_LABEL = {"model_update": "model update", "global_model": "global model", "alert": "alert", "doc_batch:invoice": "invoice batch",
             "doc_batch:shipment": "shipment batch", "doc_batch:payment": "payment batch", "doc_batch:customs_declaration": "customs batch"}

st.set_page_config(page_title="TBML Detection", page_icon=str(HERE / "mark.png"), layout="wide")

CSS = """
<style>
:root {
  --ground: #f2f4f7; --surface: #ffffff; --active: #e3e6eb; --hover: #e9ecf0; --border: #e3e5e8; --rule: #eceef1;
  --text: #0a0a0a; --muted: #5f6368; --hue: #1d5fbf;
  --mono: "Geist Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
}
.stApp { font-variant-numeric: tabular-nums; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { padding: 36px 32px 72px; max-width: 1440px; }
section[data-testid="stSidebar"] { width: 256px !important; min-width: 256px !important; }
[data-testid="stSidebarUserContent"] { padding: 0 12px 24px; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted); }
h1, h2, h3 { letter-spacing: -0.01em; }

/* sidebar: product mark, navigation as a grey-pill list */
.tb-brand { display: flex; align-items: center; gap: 10px; padding: 0 10px 4px; }
.tb-brand svg { flex: none; }
.tb-brand b { display: block; font-size: 14px; font-weight: 600; line-height: 18px; }
.tb-brand span { display: block; font-size: 12px; color: var(--muted); line-height: 16px; }
.tb-side-label { font-size: 12px; color: var(--muted); padding: 0 10px; margin: 0; }
.tb-side-note { font-size: 12px; line-height: 17px; color: var(--muted); padding: 0 10px; margin: 0; }
.tb-side-note b { color: var(--text); font-weight: 500; }
.tb-dot { display: inline-block; width: 6px; height: 6px; border-radius: 3px; margin-right: 6px; vertical-align: 1px; background: #2e7d32; }
.tb-dot.off { background: #c4242a; }
.st-key-nav [data-testid="stRadio"], .st-key-nav [data-testid="stRadio"] > div, .st-key-nav [role="radiogroup"] > * { width: 100%; }
.st-key-nav [role="radiogroup"] { gap: 2px; width: 100%; }
.st-key-nav label[data-testid="stRadioOption"] { width: 100%; min-height: 32px; padding: 6px 10px; margin: 0; border-radius: 6px; align-items: center; }
.st-key-nav label[data-testid="stRadioOption"] > div > div:not([data-testid]) { display: none; }
.st-key-nav label[data-testid="stRadioOption"]:hover { background: var(--hover); }
.st-key-nav label[data-testid="stRadioOption"]:has(input:checked) { background: var(--active); }
.st-key-nav label p { font-size: 14px; }
.st-key-nav label:has(input:checked) p { font-weight: 500; }

/* underline tabs (band filter, document type) */
.st-key-band [role="radiogroup"], .st-key-doctype [role="radiogroup"] { gap: 24px; flex-wrap: nowrap; overflow-x: auto; border-bottom: 1px solid var(--border); }
.st-key-band label[data-testid="stRadioOption"], .st-key-doctype label[data-testid="stRadioOption"] {
  padding: 4px 0 10px; margin: 0 0 -1px; border-bottom: 2px solid transparent; border-radius: 0; white-space: nowrap; }
.st-key-band label[data-testid="stRadioOption"] > div > div:not([data-testid]), .st-key-doctype label[data-testid="stRadioOption"] > div > div:not([data-testid]) { display: none; }
.st-key-band label:has(input:checked), .st-key-doctype label:has(input:checked) { border-bottom-color: var(--text); }
.st-key-band label p, .st-key-doctype label p { font-size: 14px; color: var(--muted); }
.st-key-band label:has(input:checked) p, .st-key-doctype label:has(input:checked) p { color: var(--text); font-weight: 500; }

/* panels */
.st-key-detail, .st-key-panel-check, .st-key-panel-hashes { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 24px; }
.st-key-detail hr { margin: 8px 0; border-color: var(--rule); }

/* KPI strip: one divided container, the lead figure first and larger */
.st-key-kpis [data-testid="stHorizontalBlock"] { gap: 0; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; }
.st-key-kpis [data-testid="stColumn"] { padding: 16px 20px; }
.st-key-kpis [data-testid="stColumn"] + [data-testid="stColumn"] { border-left: 1px solid var(--rule); }
[data-testid="stMetricLabel"] p { font-size: 12px; font-weight: 500; color: var(--muted); }
[data-testid="stMetricValue"] { font-size: 22px; font-weight: 500; line-height: 30px; }
.st-key-kpis [data-testid="stColumn"]:first-child [data-testid="stMetricValue"] { font-size: 30px; line-height: 36px; }
[data-testid="stMetricDelta"] { font-size: 12px; }
[data-testid="stMetricDelta"] svg { display: none; }

/* page header */
.tb-head { display: flex; justify-content: space-between; align-items: flex-end; gap: 8px 24px; flex-wrap: wrap; margin-top: -8px; }
.tb-finding { font-size: 15px; line-height: 22px; margin: 0; max-width: 76ch; }
.tb-stamp { font-size: 12px; line-height: 18px; color: var(--muted); text-align: right; margin: 0; }
.tb-count { font-size: 13px; line-height: 18px; color: var(--muted); text-align: right; margin: 0; }
.tb-h { font-size: 16px; font-weight: 600; line-height: 22px; margin: 0; }
.tb-h small { font-size: 12px; font-weight: 400; color: var(--muted); margin-left: 8px; }
.tb-sub { font-size: 12px; line-height: 17px; color: var(--muted); margin: 0; }

/* shared bits */
.mono { font-family: var(--mono); font-size: 0.93em; letter-spacing: -0.01em; }
.pill { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 12px; font-weight: 500; line-height: 18px; white-space: nowrap; }
.pill.negative { background: #ffedea; color: #831c1e; }
.pill.warning { background: #fffbeb; color: #92400e; }
.pill.neutral { background: #eceef1; color: #3d3d3d; }
.pill.positive { background: #dcfae5; color: #04250e; }
.kv, .rules, .chain, .why { margin: 0 !important; padding: 0 !important; }
.kv { display: grid; grid-template-columns: max-content minmax(0, 1fr); column-gap: 0; }
.kv dt, .kv dd { margin: 0; padding: 6px 0; border-top: 1px solid var(--rule); font-size: 13px; line-height: 18px; }
.kv dt { color: var(--muted); }
.kv dd { text-align: right; padding-left: 16px; margin-left: 0 !important; overflow-wrap: anywhere; }
.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 6px 0 14px; vertical-align: -1px; }

/* alert detail */
.tb-alert-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
.tb-alert-id { font-size: 16px; font-weight: 600; line-height: 24px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.tb-meta { font-size: 13px; line-height: 19px; color: var(--muted); margin: 2px 0 0; }
.tb-score { text-align: right; white-space: nowrap; }
.tb-score .fig { font-size: 30px; font-weight: 500; line-height: 32px; }
.tb-score .unit { font-size: 13px; color: var(--muted); }
.tb-score .lbl { font-size: 12px; color: var(--muted); }
.tb-note { font-size: 12px; line-height: 17px; color: var(--muted); margin: 0; }
.why { list-style: none; }
.why li { margin: 0; padding: 0 0 12px; }
.why-row { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; font-size: 13px; line-height: 18px; }
.why-val { color: var(--muted); font-size: 12px; white-space: nowrap; }
.why-track { position: relative; height: 8px; margin-top: 6px; background: #f4f5f7; border-radius: 4px; }
.why-track b { position: absolute; top: 0; height: 8px; border-radius: 4px; }
.why-track b.up { background: var(--hue); }
.why-track b.down { background: #a3a7ad; }
.why-track i { position: absolute; top: -3px; bottom: -3px; width: 1px; background: #5f6368; }
.rules { list-style: none; margin: 0; padding: 0; }
.rules li { margin: 0; padding: 7px 0; border-top: 1px solid var(--rule); font-size: 13px; line-height: 18px; }
.rules li span { color: var(--muted); }
.rules li .mono { color: var(--text); margin-right: 6px; }

/* the evidence chain: each record a block, linked to the next */
.chain { list-style: none; margin: 0; padding: 0; }
.chain li { position: relative; padding: 0 0 14px 26px; }
.chain li::before { content: ""; position: absolute; left: 4px; top: 6px; width: 8px; height: 8px; border: 1.5px solid var(--text); border-radius: 2px; background: var(--surface); }
.chain li:last-child::before { background: var(--text); }
.chain li:not(:last-child)::after { content: ""; position: absolute; left: 8.75px; top: 17px; bottom: 1px; width: 1.5px; background: var(--text); }
.chain .row { display: flex; justify-content: space-between; align-items: center; gap: 8px; font-size: 13px; font-weight: 500; line-height: 20px; }
.chain .meta { font-size: 12px; line-height: 17px; color: var(--muted); overflow-wrap: anywhere; }

/* simple tables */
.tb-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.tb-table th { font-size: 12px; font-weight: 500; color: var(--muted); text-align: left; padding: 0 12px 8px 0; }
.tb-table td { padding: 9px 12px 9px 0; border-top: 1px solid var(--rule); }
.tb-table .n { text-align: right; }
.tb-table tr > :last-child { padding-right: 0; }

/* inbox: list and open item side by side on wide screens, stacked below 1180px */
@media (max-width: 1180px) {
  .st-key-inbox [data-testid="stHorizontalBlock"] { flex-direction: column; gap: 16px; }
  .st-key-inbox [data-testid="stColumn"] { width: 100% !important; flex: 1 1 100% !important; min-width: 0; }
}
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"] { padding: 24px 16px 56px; }
  .st-key-kpis [data-testid="stColumn"] + [data-testid="stColumn"] { border-left: 0; border-top: 1px solid var(--rule); }
  .st-key-detail, .st-key-panel-check, .st-key-panel-hashes { padding: 16px; }
  .tb-stamp, .tb-count { text-align: left; }
}
</style>
"""

MARK_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28">'
    '<path d="M8 14h4M16 14h4" stroke="#141414" stroke-width="1.6"/>'
    '<rect x="1.8" y="9.8" width="6.4" height="8.4" rx="1.6" fill="#fff" stroke="#141414" stroke-width="1.6"/>'
    '<rect x="11.8" y="9.8" width="4.4" height="8.4" rx="1.6" fill="#fff" stroke="#141414" stroke-width="1.6"/>'
    '<rect x="19.8" y="9.8" width="6.4" height="8.4" rx="1.6" fill="#141414" stroke="#141414" stroke-width="1.6"/></svg>'
)
MARK = f'<img src="data:image/svg+xml;base64,{base64.b64encode(MARK_SVG.encode()).decode()}" width="28" height="28" alt="">'


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


# --------------------------------------------------------------------------- formatting


def esc(value: Any) -> str:
    return html.escape(str(value))


def when(ts: int) -> str:
    return dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC") if ts else "-"


def day(iso: str) -> str:
    """'2024-08-08' -> '8 Aug 2024'."""
    try:
        d = dt.date.fromisoformat(str(iso)[:10])
    except ValueError:
        return str(iso)
    return f"{d.day} {d:%b %Y}"


def times(event_ts: int, block_ts: int) -> str:
    """Both timestamps a record carries: the submitter's clock and the chain's block time."""
    return f"event time {when(event_ts)} (submitter's clock), block time {when(block_ts)} (chain clock)"


def short(digest: str, n: int = 10) -> str:
    return f"{digest[:n]}…{digest[-6:]}" if digest and len(digest) > n + 8 else digest


def typology(name: str) -> str:
    return str(name).replace("_", " ")


def pill(text: str, kind: str) -> str:
    return f'<span class="pill {kind}">{esc(text)}</span>'


def kv(rows: list[tuple[str, str]]) -> str:
    """Key-value rows; values are trusted HTML built by the caller (escape data first)."""
    return '<dl class="kv">' + "".join(f"<dt>{esc(k)}</dt><dd>{v}</dd>" for k, v in rows) + "</dl>"


def page_header(title: str, finding: str, stamp: str = "") -> None:
    st.title(title, anchor=False)
    st.html(f'<div class="tb-head"><p class="tb-finding">{finding}</p>' + (f'<p class="tb-stamp">{stamp}</p>' if stamp else "") + "</div>")


def heading(text: str, note: str = "") -> None:
    st.html(f'<p class="tb-h">{esc(text)}' + (f"<small>{note}</small>" if note else "") + "</p>")


# --------------------------------------------------------------------------- alerts


def alerts_frame(alerts: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([
        {"Alert": a["alert_id"], "Risk": int(round(a["risk_score"])), "Band": a["risk_band"], "Suspected typology": typology(a["suspected_typology"])}
        for a in alerts
    ])


def matches(alert: dict[str, Any], query: str) -> bool:
    t = alert["trade"]
    haystack = " ".join(str(v) for v in (alert["alert_id"], alert["trade_id"], t["exporter_id"], t["importer_id"], t["payer_id"], t["goods"], t["hs_code"]))
    return query.lower() in haystack.lower()


def contributions_html(alert: dict[str, Any]) -> str:
    """SHAP contributions as a bar list: bars to the right of zero raise the risk (the data hue), to the left lower it (grey)."""
    feats = alert["top_features"]
    pos = max([f["shap"] for f in feats if f["shap"] > 0], default=0.0)
    neg = max([-f["shap"] for f in feats if f["shap"] < 0], default=0.0)
    span = (pos + neg) or 1.0
    zero = 100.0 * neg / span
    rows = []
    for f in feats:
        width = 100.0 * abs(f["shap"]) / span
        left = zero if f["shap"] >= 0 else zero - width
        kind = "up" if f["shap"] >= 0 else "down"
        zero_line = f'<i style="left:{zero:.2f}%"></i>' if neg else ""
        rows.append(
            f'<li><div class="why-row"><span>{esc(f["text"])}</span><span class="why-val">{f["shap"]:+.2f}</span></div>'
            f'<div class="why-track">{zero_line}'
            f'<b class="{kind}" style="left:{left:.2f}%;width:{max(width, 0.8):.2f}%"></b></div></li>'
        )
    return '<ol class="why">' + "".join(rows) + "</ol>"


def chain_html(alert: dict[str, Any], status: dict[str, bool | None]) -> str:
    """The alert's evidence as a chain of ledger records: the model that scored it, then the alert itself."""
    mark = {None: pill("Not checked", "neutral"), True: pill("On the ledger", "positive"), False: pill("Not on the ledger", "negative")}
    model, led = alert["model"], alert.get("ledger", {})
    links = [
        (f"Global model, round {model['round']}", status["model"],
         f"registered by fl_aggregator · <span class=\"mono\">{esc(short(model['global_model_sha256'], 12))}</span>"),
        ("Alert as raised", status["alert"],
         f"registered by {esc(led.get('registered_by', '-'))} · block {esc(led.get('block', '-'))} · "
         f"<span class=\"mono\">{esc(short(led.get('sha256', ''), 12))}</span>"),
    ]
    return '<ol class="chain">' + "".join(
        f'<li><div class="row"><span>{esc(name)}</span>{mark[state]}</div><div class="meta">{meta}</div></li>' for name, state, meta in links
    ) + "</ol>"


def show_alert(alert: dict[str, Any], ledger: Any, ledger_error: str) -> None:
    t, c = alert["trade"], alert["components"]
    band = alert["risk_band"]
    st.html(
        '<div class="tb-alert-head"><div>'
        f'<div class="tb-alert-id"><span class="mono">{esc(alert["alert_id"])}</span>{pill(band, BAND_PILL[band])}</div>'
        f'<p class="tb-meta">Trade <span class="mono">{esc(alert["trade_id"])}</span> · {esc(alert["institution"])} · invoiced {esc(day(t["invoice_date"]))}</p></div>'
        f'<div class="tb-score"><span class="fig">{alert["risk_score"]:.0f}</span><span class="unit"> /100</span><div class="lbl">risk score</div></div></div>'
        f'<p class="tb-note">{esc(alert["disclaimer"])}</p>'
    )
    st.html('<p class="tb-h">Why the model flagged it'
            f'<small><span class="swatch" style="background:{HUE};margin-left:0"></span>raises risk'
            f'<span class="swatch" style="background:{FAINT}"></span>lowers risk</small></p>')
    st.html(contributions_html(alert))
    st.caption("SHAP contributions to the log-odds of the model score: the five most influential indicators, "
               "computed against the institution's own training data.")

    left, right = st.columns(2, gap="large")
    with left:
        heading("Rules fired", f"{len(alert['fired_rules'])} of the rules engine")
        if alert["fired_rules"]:
            st.html('<ul class="rules">' + "".join(
                f'<li><span class="mono">{esc(r["rule_id"])}</span>{esc(r["name"])}<br><span>{esc(typology(r["typology"]))} · severity {r["severity"]:.2f}</span></li>'
                for r in alert["fired_rules"]) + "</ul>")
        else:
            st.html('<p class="tb-sub">None. The score comes from the model alone.</p>')
    with right:
        heading("Score")
        rows = [(f"Model probability, weight {c['model_weight']:.1f}", f"{c['model_probability']:.2f}"),
                (f"Rules probability, weight {1 - c['model_weight']:.1f}", f"{c['rules_probability']:.2f}")]
        rows += [(f"Typology: {typology(k)}", f"{v:.2f}") for k, v in alert["typology_scores"].items()]
        st.html(kv(rows))
        note = "Calibrated blend of the two probabilities, on a 0 to 100 scale."
        if alert.get("raised_by_rule_floor"):
            note = "Raised to the Medium band by the rule floor: the rule evidence alone is strong although the model scores this trade low."
        st.html(f'<p class="tb-sub" style="margin-top:8px">{esc(note)}</p>')

    st.divider()
    left, right = st.columns([5, 6], gap="large")
    with left:
        heading("Trade")
        st.html(kv([
            ("Goods", f"{esc(t['goods'])} <span class=\"tb-sub\">HS {esc(t['hs_code'])}</span>"),
            ("Quantity", f"{t['quantity']:,} {esc(t['unit'])}"),
            ("Invoice value", f"USD {t['invoice_value_usd']:,.0f}"),
            ("Exporter", f'<span class="mono">{esc(t["exporter_id"])}</span>'),
            ("Importer", f'<span class="mono">{esc(t["importer_id"])}</span>'),
            ("Paid by", f'<span class="mono">{esc(t["payer_id"])}</span>'),
            ("Route", f'<span class="mono">{esc(str(t["route"]).replace(" > ", " → "))}</span>'),
        ]))
    with right:
        backend = alert.get("ledger", {}).get("backend", "")
        heading("Evidence", {"evm": "EVM ledger", "python": "pure-Python ledger"}.get(backend, "ledger"))
        chain_slot = st.empty()
        tamper = st.checkbox("Simulate tampering (lower the score by 30)", key=f"tamper-{alert['alert_id']}")
        verify = st.button("Verify on-chain", type="primary", key=f"verify-{alert['alert_id']}", disabled=ledger is None)
        status: dict[str, bool | None] = {"model": None, "alert": None}
        if ledger is None:
            st.warning(f"No evidence ledger available, so live verification is off. {ledger_error}")
        elif verify:
            candidate = copy.deepcopy(alert)
            if tamper:
                candidate["risk_score"] = round(max(0.0, candidate["risk_score"] - 30.0), 1)
            result = verify_alert(ledger, candidate)
            model = ledger.verify(alert["model"]["global_model_sha256"])
            status = {"model": bool(model.exists), "alert": bool(result["on_ledger"] and result["matches_alert_id"])}
            if status["alert"] and status["model"]:
                st.success(f"Verified. The alert and the model that scored it are on the ledger, unchanged since they were registered. "
                           f"Alert registered by `{result['registered_by']}`: {times(result['event_time'], result['registered_at'])}.")
            elif not status["alert"]:
                st.error(f"Verification failed. The hash of this alert as presented (`{short(result['sha256'], 16)}`) is not on the ledger: "
                         "its content differs from what was registered.")
            else:
                st.error("Verification failed. The alert is on the ledger, but the global model it names is not.")
        chain_slot.html(chain_html(alert, status))


def view_alerts(cfg: Config, ledger: Any, ledger_error: str, scope: str) -> None:
    data = load(cfg, "alerts", "alerts.json")
    if not data or not data["alerts"]:
        page_header("Alerts", "Nothing to review yet.")
        st.info("No alerts yet. Run `python scripts/generate_alerts.py` (or `make all`).")
        return
    alerts = [a for a in data["alerts"] if scope == ALL_INSTITUTIONS or a["institution"] == scope]
    summary = load(cfg, "alerts", "summary.json")
    model = data["alerts"][0]["model"]
    stamp = f"Raised {esc(day(max(a['created_at'] for a in data['alerts'])))} · global model round {esc(model['round'])} · seed {esc(data.get('seed', '-'))}"
    if scope == ALL_INSTITUTIONS:
        finding = f"<b>{len(alerts):,} alerts</b> to review from the test period, raised by each institution on its own trades."
        if summary:
            finding += f" In the synthetic ground truth, {100 * summary['evaluation']['alerts']['precision']:.0f}% of them are on illicit trades."
    else:
        finding = f"<b>{len(alerts):,} alerts</b> raised by {esc(scope)} on its own trades in the test period."
    page_header("Alerts", finding, stamp)

    counts = {b: sum(a["risk_band"] == b for a in alerts) for b in BAND_ORDER}
    band = st.radio("Band", ["All", *BAND_ORDER], horizontal=True, key="band", label_visibility="collapsed",
                    format_func=lambda b: f"{b} :gray[{len(alerts) if b == 'All' else counts[b]:,}]")
    search_col, typ_col, count_col = st.columns([5, 3, 2], vertical_alignment="center")
    query = search_col.text_input("Search", key="search", placeholder="Search alerts, trades, entities", label_visibility="collapsed",
                                  icon=":material/search:").strip()
    typologies = sorted({a["suspected_typology"] for a in alerts})
    chosen = typ_col.selectbox("Suspected typology", [ALL_TYPOLOGIES, *typologies], key="typology", label_visibility="collapsed",
                               format_func=lambda x: x if x == ALL_TYPOLOGIES else typology(x))
    shown = [a for a in alerts if (band == "All" or a["risk_band"] == band) and (chosen == ALL_TYPOLOGIES or a["suspected_typology"] == chosen)
             and (not query or matches(a, query))]
    count_col.html(f'<p class="tb-count">{len(shown):,} of {len(alerts):,} alerts</p>')
    if not shown:
        st.info("No alert matches these filters.")
        return

    inbox = st.container(key="inbox")
    with inbox:
        listing, detail = inbox.columns([5, 7], gap="medium")
        with listing:
            event = st.dataframe(
                alerts_frame(shown), hide_index=True, width="stretch", height=min(1080, 38 + 36 * len(shown)), on_select="rerun", selection_mode="single-row",
                key=f"alert-table-{band}-{chosen}-{query}", row_height=36, selection_default={"selection": {"rows": [0]}},
                column_config={"Alert": st.column_config.TextColumn("Alert", width=120),
                               "Risk": st.column_config.NumberColumn("Risk", format="%d", width=52),
                               "Band": st.column_config.TextColumn("Band", width=72),
                               "Suspected typology": st.column_config.TextColumn("Suspected typology")},
            )
            rows = event.selection.rows if event and event.selection else []
            st.caption("Highest risk first. Select a row to open it.")
        with detail:
            with st.container(key="detail"):
                show_alert(shown[min(rows[0] if rows else 0, len(shown) - 1)], ledger, ledger_error)


# --------------------------------------------------------------------------- federated training


def f1_chart(run: dict[str, Any], reference: float | None) -> alt.LayerChart:
    """Global-model F1 by round: the test period in the data hue, validation in grey, the centralised upper bound dashed."""
    history = pd.DataFrame(run["history"])
    long = history.melt(id_vars="round", value_vars=["test_f1", "val_f1"], var_name="series", value_name="F1")
    long["series"] = long["series"].map({"test_f1": "test period", "val_f1": "validation period"})
    values = [*long["F1"].tolist(), *([reference] if reference is not None else [])]
    scale = alt.Scale(domain=[max(0.0, np.floor(min(values) * 20) / 20), min(1.0, np.ceil(max(values) * 20) / 20 + 0.05)], nice=False)
    hover = alt.selection_point(fields=["round"], nearest=True, on="pointerover", empty=False)
    x = alt.X("round:Q", title="round", scale=alt.Scale(domain=[1, int(run["rounds"])], nice=False),
              axis=alt.Axis(grid=False, tickMinStep=1, labelColor=MUTED, titleColor=MUTED, titleFontWeight=400, titleFontSize=12, domainColor=RULE, tickColor=RULE))
    y = alt.Y("F1:Q", title="F1 at the validation-tuned threshold", scale=scale,
              axis=alt.Axis(gridColor=RULE, domain=False, ticks=False, labelColor=MUTED, titleColor=MUTED, titleFontWeight=400, titleFontSize=12, tickCount=5, format=".2f"))
    color = alt.Color("series:N", scale=alt.Scale(domain=["test period", "validation period"], range=[HUE, FAINT]), legend=None)
    base = alt.Chart(long).encode(x=x, y=y, color=color)
    best = history[history["round"] == run["best_round"]].assign(label=lambda d: [f"selected round {run['best_round']} · F1 {v:.3f}" for v in d["test_f1"]])
    layers = [
        base.mark_line(strokeWidth=2),
        alt.Chart(history).mark_rule(color="#c9cdd3").encode(x="round:Q").transform_filter(hover),
        base.mark_point(size=64, filled=True, stroke="#ffffff", strokeWidth=2).encode(
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            tooltip=[alt.Tooltip("round:Q"), alt.Tooltip("series:N"), alt.Tooltip("F1:Q", format=".3f")]).add_params(hover),
        alt.Chart(best).mark_point(size=80, filled=True, color=HUE, stroke="#ffffff", strokeWidth=2).encode(x="round:Q", y=alt.Y("test_f1:Q", scale=scale)),
        alt.Chart(best).mark_text(align="right", dx=-8, dy=-14, fontSize=12, color=INK).encode(x="round:Q", y=alt.Y("test_f1:Q", scale=scale), text="label:N"),
    ]
    if reference is not None:
        ref = pd.DataFrame({"F1": [reference], "label": [f"centralised MLP on pooled raw data, {reference:.3f}"], "round": [1]})
        layers += [alt.Chart(ref).mark_rule(strokeDash=[4, 4], color=MUTED, strokeWidth=1).encode(y=alt.Y("F1:Q", scale=scale)),
                   alt.Chart(ref).mark_text(align="left", dy=-8, dx=2, fontSize=12, color=MUTED).encode(x="round:Q", y=alt.Y("F1:Q", scale=scale), text="label:N")]
    return alt.layer(*layers).properties(height=320, background="#ffffff", padding={"left": 8, "right": 16, "top": 12, "bottom": 8}).configure_view(strokeWidth=0)


def view_training(cfg: Config, ledger: Any) -> None:
    run = load(cfg, "fl", "primary.json")
    if not run:
        page_header("Federated training", "No federated run yet.")
        st.info("No primary federated run yet. Run `python scripts/run_fl.py`.")
        return
    final, comm, boundary = run["final"], run["communication"], run["boundary"]
    baseline = load(cfg, "baselines", f"seed_{run['seed']}.json")
    reference = float(baseline["methods"]["mlp"]["f1"]) if baseline and "mlp" in baseline.get("methods", {}) else None
    finding = f"The global model reaches <b>F1 {final['f1']:.3f}</b> on the test period while every trade stays with its institution"
    finding += f", {reference - final['f1']:.3f} below a centralised model trained on the pooled raw data." if reference is not None else "."
    page_header("Federated training", finding,
                f"{esc(run['scenario']['label'])} · {esc(str(run['strategy']).replace('fedavg', 'FedAvg').replace('fedprox', 'FedProx'))} · seed {esc(run['seed'])}")

    with st.container(key="kpis"):
        cols = st.columns([1.25, 1, 1, 1, 1.15], gap=None)
        cols[0].metric("F1, test period", f"{final['f1']:.3f}",
                       delta=f"{final['f1'] - reference:+.3f} vs centralised" if reference is not None else None, delta_color="off")
        cols[1].metric("PR-AUC, test period", f"{final['pr_auc']:.3f}")
        cols[2].metric("Selected round", f"{run['best_round']} of {run['rounds']}")
        cols[3].metric("Traffic, whole run", f"{comm['total_bytes'] / 1e6:.1f} MB")
        cols[4].metric("Client replies checked", f"{boundary['replies_checked']}", delta=f"{boundary['violations']} with raw data", delta_color="off")

    with st.container(key="panel-check"):
        st.html('<p class="tb-h">Global model quality by round'
                f'<small><span class="swatch" style="background:{HUE};margin-left:0"></span>test period'
                f'<span class="swatch" style="background:{FAINT}"></span>validation period'
                f'<span class="swatch" style="background:transparent;border-top:1.5px dashed {MUTED};border-radius:0;height:0;width:14px"></span>centralised upper bound</small></p>')
        st.altair_chart(f1_chart(run, reference), width="stretch")
        st.caption(f"The round is chosen on validation F1; the test period is never used for selection. {comm['n_parameters']:,} parameters, "
                   f"{comm['model_bytes'] / 1e3:.1f} kB per model.")

    left, right = st.columns([3, 2], gap="medium")
    with left:
        heading("Institutions", "global model, global threshold, own test trades")
        sizes = {c["client"]: c for c in run["partition"]["clients"]}
        body = "".join(
            f"<tr><td class=\"mono\">fl_client_{c['client']}</td><td class=\"n\">{sizes[c['client']]['n']:,}</td>"
            f"<td class=\"n\">{100 * sizes[c['client']]['illicit_rate']:.1f}%</td><td class=\"n\">{c['f1']:.3f}</td>"
            f"<td class=\"n\">{c['precision']:.3f}</td><td class=\"n\">{c['recall']:.3f}</td></tr>"
            for c in run["per_client"])
        st.html('<table class="tb-table"><thead><tr><th>Institution</th><th class="n">Trades</th><th class="n">Illicit rate</th><th class="n">F1</th>'
                f'<th class="n">Precision</th><th class="n">Recall</th></tr></thead><tbody>{body}</tbody></table>')
    with right:
        heading("Traffic per institution per round")
        per = comm["per_client_per_round"]
        st.html(kv([("Model download, train", f"{per['train_download'] / 1e3:.1f} kB"), ("Model upload, train", f"{per['train_upload'] / 1e3:.1f} kB"),
                    ("Model download, evaluate", f"{per['eval_download'] / 1e3:.1f} kB"), ("Score histograms upload", f"{per['eval_upload'] / 1e3:.1f} kB")]))

    st.html('<div style="height:8px"></div>')
    heading("Model hashes on the evidence ledger", f"{len(run['model_hashes']):,} records")
    hashes = pd.DataFrame([{"Round": h["round"], "Record": "global model" if h["kind"] == "global_model" else f"update of fl_client_{h['client']}",
                            "Registered by": h.get("ledger_org", "fl_aggregator" if h["kind"] == "global_model" else f"fl_client_{h['client']}"),
                            "SHA-256": h["sha256"], "Block": h.get("block")} for h in run["model_hashes"]]).sort_values(["Round", "Record"], ascending=[False, True])
    st.dataframe(hashes, hide_index=True, width="stretch", height=260, row_height=36,
                 column_config={"Round": st.column_config.NumberColumn("Round", format="%d", width="small"),
                                "Block": st.column_config.NumberColumn("Block", format="%d", width="small")})
    path = model_dir(cfg, int(run["seed"]), primary_scenario(cfg), str(run["strategy"]), "primary") / "global_model.npz"
    if st.button("Verify the saved global model on-chain", type="primary", disabled=ledger is None or not path.is_file()):
        with np.load(path) as z:
            names = list(z.files)
            digest, record = verify_model(ledger, names, [z[n] for n in names])
        if record.exists:
            st.success(f"Verified. The model file hashes to `{short(digest, 16)}`, registered as `{record.ref_id}` by `{record.org}`; "
                       f"{times(record.client_ts, record.block_ts)}.")
        else:
            st.error(f"Verification failed. The model file hashes to `{short(digest, 16)}`, which is not on the ledger.")


# --------------------------------------------------------------------------- evidence ledger


def view_ledger(cfg: Config, ledger: Any, ledger_error: str, scope: str) -> None:
    bench = load(cfg, "ledger", "benchmark.json")
    if ledger is None:
        page_header("Evidence ledger", "The ledger is not reachable.")
        st.warning(f"No evidence ledger available. {ledger_error}")
    else:
        stats = ledger.stats()
        trail = pd.DataFrame(ledger.audit_trail())
        orgs = int(stats["n_organisations"])
        page_header("Evidence ledger",
                    f"<b>{stats['n_records']:,} records</b> from {orgs} organisations. Only hashes, document types, opaque reference ids and timestamps "
                    "are on the ledger; documents, model weights and alerts stay with their owners.",
                    "EVM contract on a local Hardhat node" if stats["backend"] == "evm" else "Pure-Python append-only ledger")
        with st.container(key="kpis"):
            cols = st.columns([1.25, 1, 1, 1, 1], gap=None)
            cols[0].metric("Records", f"{stats['n_records']:,}")
            cols[1].metric("Organisations", f"{orgs}")
            cols[2].metric("Ledger storage", f"{stats['storage_bytes'] / 1e3:,.1f} kB")
            cols[3].metric(f"Total cost ({stats['cost_unit']})", f"{stats['total_cost']:,}")
            cols[4].metric("Backend", "EVM" if stats["backend"] == "evm" else "Python")

        with st.container(key="panel-hashes"):
            heading("Check a hash", "any SHA-256: a model update, a global model, an alert or a batch root")
            digest = st.text_input("SHA-256 (hex)", key="hash-check", placeholder="Paste a 64-character hash", label_visibility="collapsed").strip()
            if digest:
                try:
                    record = ledger.verify(digest)
                    if record.exists:
                        st.success(f"On the ledger: `{record.doc_type}` `{record.ref_id}`, registered by `{record.org}`; {times(record.client_ts, record.block_ts)}.")
                    else:
                        st.error("This hash is not on the ledger.")
                except Exception as exc:
                    st.error(str(exc))

        if len(trail):
            if scope != ALL_INSTITUTIONS:
                trail = trail[trail["org"] == scope]
            kinds = [k for k in DOC_LABEL if k in set(trail["doc_type"])] + sorted(set(trail["doc_type"]) - set(DOC_LABEL))
            st.html('<div style="height:8px"></div>')
            heading("Audit trail", "newest first" + (f" · registered by {esc(scope)}" if scope != ALL_INSTITUTIONS else ""))
            kind = st.radio("Document type", ["all", *kinds], horizontal=True, key="doctype", label_visibility="collapsed",
                            format_func=lambda k: "All" if k == "all" else DOC_LABEL.get(k, k))
            if kind != "all":
                trail = trail[trail["doc_type"] == kind]
            total = len(trail)
            trail = trail.sort_values("block", ascending=False).head(500)
            st.html(f'<p class="tb-count" style="text-align:left">{min(total, 500):,} of {total:,} records shown</p>')
            st.dataframe(pd.DataFrame({"Block": trail["block"], "Type": trail["doc_type"].map(lambda k: DOC_LABEL.get(k, k)), "Reference": trail["ref_id"],
                                       "Registered by": trail["org"], "Event time (submitter)": trail["client_ts"].map(when),
                                       "Block time (chain)": trail["block_ts"].map(when), "SHA-256": trail["digest"]}),
                         hide_index=True, width="stretch", height=380, row_height=36,
                         column_config={"Block": st.column_config.NumberColumn("Block", format="%d", width="small")})
            st.caption("Two clocks per record: the event time reported by the submitting organisation, and the block time set by the chain. "
                       "On a development node they drift apart, so read the event time.")
    if bench:
        st.html('<div style="height:8px"></div>')
        heading("Benchmark", "one laptop, local single-validator node; latency on an auto-mining node is a lower bound")
        body = ""
        for backend, b in bench.items():
            body += (f"<tr><td>{'EVM (Hardhat)' if backend == 'evm' else 'Pure Python'}</td>"
                     f"<td class=\"n\">{b['single']['latency']['p50_ms']:.1f} / {b['single']['latency']['p95_ms']:.1f}</td>"
                     f"<td class=\"n\">{b['single']['records_per_second']:,.0f}</td><td class=\"n\">{b['batched'][-1]['records_per_second']:,.0f}</td>"
                     f"<td class=\"n\">{b['merkle']['records_per_second']:,.0f}</td>"
                     f"<td class=\"n\">{b['single']['cost_per_record']:,.0f}</td><td class=\"n\">{b['merkle']['cost_per_record']:,.1f}</td><td>{esc(b['cost_unit'])}</td></tr>")
        st.html('<table class="tb-table"><thead><tr><th>Backend</th><th class="n">Latency p50 / p95, ms</th><th class="n">Records/s, single</th>'
                '<th class="n">Records/s, batch of 100</th><th class="n">Documents/s, Merkle</th><th class="n">Cost per record</th>'
                f'<th class="n">Cost per document, Merkle</th><th>Unit</th></tr></thead><tbody>{body}</tbody></table>')


# --------------------------------------------------------------------------- app


def sidebar(cfg: Config, ledger: Any) -> tuple[str, str]:
    alerts = load(cfg, "alerts", "alerts.json")
    institutions = sorted({a["institution"] for a in alerts["alerts"]}) if alerts and alerts.get("alerts") else []
    with st.sidebar:
        st.html(f'<div class="tb-brand">{MARK}<div><b>TBML Detection</b><span>Investigator console</span></div></div>')
        view = st.radio("View", VIEWS, key="nav", label_visibility="collapsed")
        st.html('<div style="height:12px"></div><p class="tb-side-label">Institution</p>')
        scope = st.selectbox("Institution", [ALL_INSTITUTIONS, *institutions], key="scope", label_visibility="collapsed")
        st.html('<div style="height:20px"></div>')
        if ledger is None:
            st.html('<p class="tb-side-note"><span class="tb-dot off"></span><b>Ledger not connected</b></p>')
        else:
            name = "EVM, local Hardhat node" if ledger.backend == "evm" else "pure-Python fallback"
            st.html(f'<p class="tb-side-note"><span class="tb-dot"></span><b>Ledger connected</b><br>{esc(name)}</p>')
        st.html('<p class="tb-side-note" style="margin-top:12px">Synthetic data. A B.E. final-year prototype. '
                "Alerts support human review and decide nothing.</p>")
    return view, scope


def main() -> None:
    st.html(CSS)
    cfg = get_config()
    ledger, ledger_error = get_ledger(cfg)
    view, scope = sidebar(cfg, ledger)
    if view == "Alerts":
        view_alerts(cfg, ledger, ledger_error, scope)
    elif view == "Federated training":
        view_training(cfg, ledger)
    else:
        view_ledger(cfg, ledger, ledger_error, scope)


main()
