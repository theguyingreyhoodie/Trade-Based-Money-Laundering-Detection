"""Explainable alerts for human investigators (Step 05 / 06, objective 5).

For every trade of the test period, each institution scores its own trades with
the global FL model and the rules engine, turns the two into a calibrated 0-100
risk score, and raises an alert when the score reaches ``scoring.alert_min_score``.

An alert carries what an investigator needs to start a review: the score and
band, the indicators that drove the model's output (SHAP, in plain English),
the red-flag rules that fired, the typology the evidence points to, and the
hash and transaction id under which the alert was registered on the ledger.

An alert is a prompt for human review. It is not a finding of laundering.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tbml.config import Config
from tbml.data.typologies import TYPOLOGIES, TYPOLOGY_CLASSES
from tbml.eval.metrics import ScoreHist, evaluate_hists
from tbml.explain.scoring import BANDS, RiskScorer, calibration_report, fit_risk_scorer, weight_sensitivity
from tbml.explain.shap_explain import explain, top_contributions
from tbml.features.build import FeatureSet
from tbml.features.dictionary import FEATURE_BY_NAME
from tbml.ledger.audit import client_org
from tbml.ledger.base import EvidenceLedger
from tbml.ledger.hashing import hash_record
from tbml.models.mlp import predict_proba
from tbml.rules.engine import RULES, rule_hits, rule_score, rule_severities
from tbml.utils import derive_seed, get_logger

log = get_logger(__name__)

ALERT_DOC_TYPE = "alert"
#: fields of an alert that are not part of its hash (they are added after hashing)
UNHASHED_FIELDS = ("ledger",)


@dataclass
class ClientScores:
    """One institution's trades with model and rule outputs."""

    client: int
    fs: FeatureSet
    model_score: np.ndarray
    hits: np.ndarray
    rule_score: np.ndarray

    def part(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = self.fs.split == name
        return self.model_score[m], self.rule_score[m], self.fs.typology[m]


def score_clients(cfg: Config, model: Any, clients: list[FeatureSet]) -> list[ClientScores]:
    """Each institution applies the global model and the rules to its own trades."""
    out = []
    for k, fs in enumerate(clients):
        hits = rule_hits(fs.X, cfg, fs.feature_names)
        out.append(ClientScores(k, fs, predict_proba(model, fs.X).astype(np.float64), hits, rule_score(hits, cfg)))
    return out


def suspected_typology(shap_row: np.ndarray, names: list[str], hits_row: np.ndarray, severities: np.ndarray) -> tuple[str, dict[str, float]]:
    """Which typology does the evidence point to?

    Model evidence: positive SHAP mass of the indicators that target a typology
    (an indicator that targets several typologies is shared between them).
    Rule evidence: severities of the fired rules of that typology. Each side is
    normalised to sum to one and the two are averaged; if one side has no
    evidence the other decides.
    """
    model_part = dict.fromkeys(TYPOLOGIES, 0.0)
    for j, name in enumerate(names):
        if shap_row[j] > 0:
            targets = FEATURE_BY_NAME[name].typologies
            for t in targets:
                model_part[t] += float(shap_row[j]) / len(targets)
    rule_part = dict.fromkeys(TYPOLOGIES, 0.0)
    for j in np.flatnonzero(hits_row):
        rule_part[RULES[j].typology] += float(severities[j])
    m_total, r_total = sum(model_part.values()), sum(rule_part.values())
    scores = {}
    for t in TYPOLOGIES:
        m = model_part[t] / m_total if m_total > 0 else 0.0
        r = rule_part[t] / r_total if r_total > 0 else 0.0
        scores[t] = (m + r) / 2 if (m_total > 0 and r_total > 0) else (m if m_total > 0 else r)
    best = max(scores, key=scores.get) if (m_total > 0 or r_total > 0) else "undetermined"
    return best, {t: round(s, 4) for t, s in sorted(scores.items(), key=lambda kv: -kv[1])}


def alert_digest(alert: dict[str, Any]) -> str:
    """SHA-256 of an alert's content (everything except the ledger receipt added afterwards)."""
    return hash_record({k: v for k, v in alert.items() if k not in UNHASHED_FIELDS})


def _trade_facts(row: pd.Series) -> dict[str, Any]:
    stops = [p for p in str(row.get("transshipment_ports", "")).split("|") if p]
    route = " > ".join([str(row["origin_port"]), *stops, str(row["dest_port"])]) if row.get("has_shipment", 0) == 1 else "no shipment record"
    return {
        "invoice_date": str(pd.Timestamp(row["invoice_date"]).date()),
        "exporter_id": row["exporter_id"],
        "importer_id": row["importer_id"],
        "hs_code": row["hs_code"],
        "goods": row["goods_description"],
        "quantity": int(row["quantity"]),
        "unit": row["unit"],
        "invoice_value_usd": round(float(row["invoice_value_usd"]), 2),
        "currency": row["currency"],
        "route": route,
        "payer_id": row["payer_id"],
    }


def build_alerts(
    cfg: Config,
    model: Any,
    scorer: RiskScorer,
    scored: list[ClientScores],
    linked: pd.DataFrame,
    model_info: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Raise alerts on the test period of every institution; returns (alerts, explanation info)."""
    scfg = cfg.scoring
    k_top = int(scfg.top_k_features)
    severities = rule_severities(cfg)
    trades = linked.set_index("trade_id")
    created = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    alerts: list[dict[str, Any]] = []
    shap_info = {"method": [], "additivity_error": 0.0, "background_rows": int(scfg.shap_background), "permutations": int(scfg.shap_permutations)}
    for cs in scored:
        fs = cs.fs
        risk = scorer.score(cs.model_score, cs.rule_score)
        rows = np.flatnonzero((fs.split == "test") & (risk >= float(scfg.alert_min_score)))
        if not len(rows):
            continue
        train = np.flatnonzero(fs.split == "train")
        rng = np.random.default_rng(derive_seed(seed, "shap-background", cs.client))
        background = fs.X[rng.choice(train, size=min(int(scfg.shap_background), len(train)), replace=False)] if len(train) else fs.X[:1]
        result = explain(model, background, fs.X[rows], permutations=int(scfg.shap_permutations), seed=derive_seed(seed, "shap", cs.client))
        shap_info["method"].append(result["method"])
        shap_info["additivity_error"] = max(shap_info["additivity_error"], result["additivity_error"])
        p_model, p_rules = scorer.components(cs.model_score[rows], cs.rule_score[rows])
        bands = scorer.band(risk[rows])
        floored = scorer.floored(cs.model_score[rows], cs.rule_score[rows])
        for i, r in enumerate(rows):
            trade_id = str(fs.trade_id[r])
            typology, typology_scores = suspected_typology(result["values"][i], fs.feature_names, cs.hits[r], severities)
            alert = {
                "alert_id": f"ALERT-{trade_id.split('-')[-1]}",
                "trade_id": trade_id,
                "institution": client_org(cs.client),
                "created_at": created,
                "risk_score": round(float(risk[r]), 1),
                "risk_band": str(bands[i]),
                "raised_by_rule_floor": bool(floored[i]),
                "components": {
                    "model_probability": round(float(p_model[i]), 4),
                    "rules_probability": round(float(p_rules[i]), 4),
                    "model_weight": float(scorer.model_weight),
                    "model_raw_score": round(float(cs.model_score[r]), 6),
                    "rule_raw_score": round(float(cs.rule_score[r]), 6),
                },
                "suspected_typology": typology,
                "typology_scores": dict(list(typology_scores.items())[:3]),
                "top_features": top_contributions(result["values"][i], fs.X[r], fs.feature_names, k_top),
                "shap_base_value": round(float(result["base_value"]), 4),
                "fired_rules": [{"rule_id": RULES[j].rule_id, "name": RULES[j].name, "typology": RULES[j].typology, "severity": float(severities[j])}
                                for j in np.flatnonzero(cs.hits[r])],
                "trade": _trade_facts(trades.loc[trade_id]),
                "model": model_info,
                "disclaimer": "Flagged for human review. This is not a determination of money laundering.",
                # a random salt makes the public hash useless for testing guesses about the alert's content
                "nonce": hashlib.sha256(f"{seed}|{trade_id}|{created}|{rng.random()}".encode()).hexdigest()[:32],
            }
            alerts.append(alert)
    alerts.sort(key=lambda a: (-a["risk_score"], a["alert_id"]))
    return alerts, shap_info


def register_alerts(ledger: EvidenceLedger, alerts: list[dict[str, Any]]) -> dict[str, Any]:
    """Register every alert's hash under the institution that raised it; annotate the alerts."""
    latencies, costs = [], []
    for alert in alerts:
        org = alert["institution"]
        if org not in ledger.organisations():
            ledger.register_organisation(org)
        digest = alert_digest(alert)
        receipt = ledger.register(digest, ALERT_DOC_TYPE, alert["alert_id"], org)
        alert["ledger"] = {"sha256": digest, "tx_id": receipt.tx_id, "block": receipt.block, "registered_by": org, "status": receipt.status,
                           "backend": ledger.backend}
        latencies.append(receipt.latency_s)
        costs.append(receipt.cost)
    lat = np.array(latencies) if latencies else np.zeros(1)
    return {"backend": ledger.backend, "alerts_registered": len(alerts), "latency_ms_p50": float(np.percentile(lat, 50) * 1000),
            "latency_ms_p95": float(np.percentile(lat, 95) * 1000), "cost_mean": float(np.mean(costs)) if costs else 0.0,
            "cost_unit": ledger.stats()["cost_unit"]}


def verify_alert(ledger: EvidenceLedger, alert: dict[str, Any]) -> dict[str, Any]:
    """Recompute an alert's hash and look it up on the ledger."""
    digest = alert_digest(alert)
    record = ledger.verify(digest)
    return {"sha256": digest, "on_ledger": record.exists, "registered_by": record.org, "ref_id": record.ref_id, "registered_at": record.block_ts,
            "event_time": record.client_ts,
            "matches_alert_id": record.exists and record.ref_id == alert.get("alert_id"), "doc_type": record.doc_type}


# --------------------------------------------------------------------------- evaluation (uses ground truth; not part of an alert)


def evaluate_scoring(cfg: Config, scorer: RiskScorer, scored: list[ClientScores], alerts: list[dict[str, Any]], hists: dict[str, ScoreHist]) -> dict[str, Any]:
    """How good is the risk score? Bands, calibration, blend vs components, typology attribution."""
    n_bins = int(cfg.fl.hist_bins)
    metric = str(cfg.models.threshold_metric)
    val_parts = [cs.part("val") for cs in scored]
    test_parts = [cs.part("test") for cs in scored]

    def hist(parts, fn) -> ScoreHist:
        return sum((ScoreHist.from_scores(fn(m, r), t, n_bins) for m, r, t in parts), ScoreHist.empty(n_bins))

    variants = {
        "risk_score": lambda m, r: scorer.score(m, r) / 100.0,
        "model_only": lambda m, r: m,
        "rules_only": lambda m, r: r,
    }
    comparison = {name: evaluate_hists(hist(val_parts, fn), hist(test_parts, fn), metric) for name, fn in variants.items()}

    # bands on the test period
    risk = np.concatenate([scorer.score(m, r) for m, r, _ in test_parts])
    y = np.concatenate([t > 0 for _, _, t in test_parts])
    band = scorer.band(risk)
    bands = []
    for name in BANDS:
        m = band == name
        bands.append({"band": name, "trades": int(m.sum()), "illicit": int(y[m].sum()), "precision": float(y[m].mean()) if m.any() else None,
                      "share_of_all_illicit": float(y[m].sum() / max(y.sum(), 1)), "share_of_trades": float(m.mean())})

    # calibration on the test period, before and after
    test_model = hist(test_parts, lambda m, r: m)
    test_blend = hist(test_parts, lambda m, r: scorer.blend(m, r))
    calibration = {
        "model_raw": calibration_report(test_model),
        "model_calibrated": calibration_report(test_model, scorer.model_calibrator),
        "risk_score": calibration_report(test_blend, scorer.final_calibrator),  # the calibrated blend, before the rule floor
    }
    for part in calibration.values():
        part.pop("reliability", None)
    calibration["risk_score_reliability"] = calibration_report(test_blend, scorer.final_calibrator)["reliability"]

    # typology attribution among alerts on truly illicit trades
    truth: dict[str, str] = {}
    for cs in scored:
        for tid, t in zip(cs.fs.trade_id, cs.fs.typology):
            truth[str(tid)] = TYPOLOGY_CLASSES[int(t)]
    hit = [a for a in alerts if truth[a["trade_id"]] != "none"]
    top1 = [a["suspected_typology"] == truth[a["trade_id"]] for a in hit]
    top2 = [truth[a["trade_id"]] in list(a["typology_scores"])[:2] for a in hit]
    per_typology = {}
    for t in TYPOLOGIES:
        rows = [a for a in hit if truth[a["trade_id"]] == t]
        per_typology[t] = {"alerts": len(rows), "attributed_correctly": float(np.mean([a["suspected_typology"] == t for a in rows])) if rows else None}

    weights = [round(w, 2) for w in np.linspace(0.0, 1.0, 11)]
    sensitivity = weight_sensitivity(val_parts, scorer, weights, n_bins)
    best = max(sensitivity, key=lambda row: row["val_pr_auc"])
    return {
        "comparison": comparison,
        "bands": bands,
        "calibration": calibration,
        "typology_attribution": {"alerts_on_illicit_trades": len(hit), "top1_accuracy": float(np.mean(top1)) if hit else None,
                                 "top2_accuracy": float(np.mean(top2)) if hit else None, "per_typology": per_typology},
        "weight_sensitivity": sensitivity,
        "model_weight": scorer.model_weight,
        "best_weight_on_validation": best["model_weight"],
        "rule_floor": {"probability": scorer.rule_floor_probability, "alerts_raised_by_floor": sum(a["raised_by_rule_floor"] for a in alerts),
                       "of_which_illicit": sum(a["raised_by_rule_floor"] and truth[a["trade_id"]] != "none" for a in alerts)},
        "alerts": {"total": len(alerts), "on_illicit": len(hit), "precision": len(hit) / max(len(alerts), 1),
                   "by_band": {b: sum(a["risk_band"] == b for a in alerts) for b in BANDS[1:]},
                   "by_institution": {org: sum(a["institution"] == org for a in alerts) for org in sorted({a["institution"] for a in alerts})}},
    }


# --------------------------------------------------------------------------- readable report


def alert_markdown(alert: dict[str, Any], truth: str | None = None) -> str:
    t = alert["trade"]
    lines = [
        f"### {alert['alert_id']}  |  {alert['risk_band']}  |  risk score {alert['risk_score']:.0f} / 100",
        "",
        f"- **Trade** {alert['trade_id']}, invoiced {t['invoice_date']}, held by `{alert['institution']}`",
        f"- **Parties** exporter {t['exporter_id']} to importer {t['importer_id']}; paid by {t['payer_id']}",
        f"- **Goods** {t['goods']} (HS {t['hs_code']}), {t['quantity']:,} {t['unit']}, USD {t['invoice_value_usd']:,.0f}; route {t['route']}",
        f"- **Score** model probability {alert['components']['model_probability']:.2f}, rules probability {alert['components']['rules_probability']:.2f}"
        f" (model weight {alert['components']['model_weight']:.1f})"
        + ("; raised to Medium by the rule floor" if alert.get("raised_by_rule_floor") else ""),
        f"- **Suspected typology** {alert['suspected_typology'].replace('_', ' ')}"
        + (f" (next: {', '.join(k.replace('_', ' ') for k in list(alert['typology_scores'])[1:3])})" if len(alert["typology_scores"]) > 1 else ""),
        "",
        "Why the model flagged it (SHAP contributions to the log-odds):",
        "",
    ]
    for i, f in enumerate(alert["top_features"], start=1):
        lines.append(f"{i}. {f['text']} ({f['direction']}, {f['shap']:+.2f})")
    lines += ["", "Rules fired: " + ("; ".join(f"{r['rule_id']} {r['name']}" for r in alert["fired_rules"]) if alert["fired_rules"] else "none"), ""]
    if "ledger" in alert:
        led = alert["ledger"]
        lines.append(f"Evidence: SHA-256 `{led['sha256']}`, transaction `{led['tx_id']}` (block {led['block']}), registered by `{led['registered_by']}`")
    if truth is not None:
        verdict = "legitimate trade (false alert)" if truth == "none" else truth.replace("_", " ")
        lines += ["", f"_Ground truth, for evaluation only and never shown to an investigator: {verdict}._"]
    lines.append("")
    return "\n".join(lines)


def sample_alerts(alerts: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """A varied sample: the top alert of every suspected typology, then the rest by band."""
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    for typology in TYPOLOGIES:
        for a in alerts:
            if a["suspected_typology"] == typology:
                chosen.append(a)
                seen.add(a["alert_id"])
                break
    for band in ["Medium", "High", "Critical"]:
        for a in alerts:
            if a["risk_band"] == band and a["alert_id"] not in seen:
                chosen.append(a)
                seen.add(a["alert_id"])
                break
    for a in alerts:  # fill up with the highest remaining scores
        if len(chosen) >= n:
            break
        if a["alert_id"] not in seen:
            chosen.append(a)
            seen.add(a["alert_id"])
    return sorted(chosen[:n], key=lambda a: -a["risk_score"])


def report_markdown(alerts: list[dict[str, Any]], summary: dict[str, Any], truth: dict[str, str], n: int) -> str:
    ev = summary["evaluation"]
    lines = [
        "# Alert report (sample)",
        "",
        "Generated by `scripts/generate_alerts.py`. Alerts are raised on the test period by each institution on its own trades,",
        "using the global federated model and the rules engine. **An alert is a prompt for human review, not a finding of laundering.**",
        "",
        f"- Alerts raised: {ev['alerts']['total']} (" + ", ".join(f"{b}: {c}" for b, c in ev["alerts"]["by_band"].items()) + ")",
        f"- Share of alerts on truly illicit trades: {ev['alerts']['precision']:.0%}",
        f"- Suspected typology correct for {ev['typology_attribution']['top1_accuracy']:.0%} of alerts on illicit trades"
        f" (within the top two: {ev['typology_attribution']['top2_accuracy']:.0%})",
        f"- Evidence ledger: {summary['ledger']['backend']}; every alert below can be re-verified with `scripts/ledger_tamper_demo.py` or the dashboard",
        "",
        "| Band | Score | Trades | Illicit | Precision | Share of all illicit trades |",
        "|---|---|---|---|---|---|",
    ]
    limits = summary["scorer"]["bands"]
    ranges = {"Low": f"0 - {limits['medium']:.0f}", "Medium": f"{limits['medium']:.0f} - {limits['high']:.0f}",
              "High": f"{limits['high']:.0f} - {limits['critical']:.0f}", "Critical": f"{limits['critical']:.0f} - 100"}
    for row in ev["bands"]:
        precision = "-" if row["precision"] is None else f"{row['precision']:.1%}"
        lines.append(f"| {row['band']} | {ranges[row['band']]} | {row['trades']:,} | {row['illicit']:,} | {precision} | {row['share_of_all_illicit']:.1%} |")
    lines += ["", f"## {min(n, len(alerts))} sample alerts", ""]
    for alert in sample_alerts(alerts, n):
        lines.append(alert_markdown(alert, truth.get(alert["trade_id"])))
    return "\n".join(lines)


def run_alert_pipeline(cfg: Config, seed: int, model: Any, model_info: dict[str, Any], clients: list[FeatureSet], linked: pd.DataFrame,
                       ledger: EvidenceLedger | None) -> dict[str, Any]:
    """Score, calibrate, alert, explain, register. Returns everything the script writes out."""
    scfg = cfg.scoring
    scored = score_clients(cfg, model, clients)
    scorer, hists = fit_risk_scorer([cs.part("val") for cs in scored], float(scfg.model_weight), str(scfg.calibration),
                                    {k: float(v) for k, v in scfg.bands.items()}, int(cfg.fl.hist_bins), float(scfg.rule_floor_probability))
    alerts, shap_info = build_alerts(cfg, model, scorer, scored, linked, model_info, seed)
    ledger_summary = register_alerts(ledger, alerts) if ledger is not None else {"backend": "none", "alerts_registered": 0}
    evaluation = evaluate_scoring(cfg, scorer, scored, alerts, hists)
    truth = {str(tid): TYPOLOGY_CLASSES[int(t)] for cs in scored for tid, t in zip(cs.fs.trade_id, cs.fs.typology)}
    summary = {"seed": seed, "model": model_info, "scorer": scorer.to_dict(), "shap": shap_info, "ledger": ledger_summary, "evaluation": evaluation}
    scores = pd.DataFrame({
        "trade_id": np.concatenate([cs.fs.trade_id for cs in scored]),
        "institution": np.concatenate([[client_org(cs.client)] * len(cs.fs) for cs in scored]),
        "split": np.concatenate([cs.fs.split for cs in scored]),
        "risk_score": np.round(np.concatenate([scorer.score(cs.model_score, cs.rule_score) for cs in scored]), 2),
        "model_score": np.concatenate([cs.model_score for cs in scored]),
        "rule_score": np.concatenate([cs.rule_score for cs in scored]),
    })
    scores["risk_band"] = scorer.band(scores["risk_score"].to_numpy())
    log.info("%d alerts on %d test trades (%.0f%% on illicit trades); typology attribution %.0f%%; SHAP %s, additivity error %.4f", len(alerts),
             int((scores["split"] == "test").sum()), 100 * evaluation["alerts"]["precision"],
             100 * (evaluation["typology_attribution"]["top1_accuracy"] or 0), "/".join(sorted(set(shap_info["method"]))) or "n/a", shap_info["additivity_error"])
    return {"alerts": alerts, "summary": summary, "scores": scores, "truth": truth, "report": report_markdown(alerts, summary, truth, int(scfg.report_sample))}
