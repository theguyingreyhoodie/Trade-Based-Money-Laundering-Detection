"""Evaluation and audit report (Step 06, objective 6).

Collects every result file written by the pipeline, aggregates over seeds
(mean and standard deviation), writes the tables as CSV, draws the figures and
assembles ``results/RESULTS.md``. All numbers in the report are read from the
result files; nothing is typed in by hand.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tbml.config import Config
from tbml.data.typologies import TYPOLOGIES
from tbml.eval import plots
from tbml.fl.experiments import experiment_plan, primary_scenario
from tbml.fl.partition import Scenario
from tbml.models.baselines import METHOD_LABELS as BASELINE_LABELS
from tbml.utils import get_logger, load_json

log = get_logger(__name__)

METRICS = ["precision", "recall", "f1", "pr_auc", "roc_auc", "fpr"]
METRIC_HEAD = ["Precision", "Recall", "F1", "PR-AUC", "ROC-AUC", "FPR"]
FL_METHODS = ["local", "pooled_local_features", "fedavg", "fedavg_client_thresholds", "fedavg_finetuned", "fedprox"]
FL_LABELS = {
    "local": "Local-only",
    "pooled_local_features": "Pooled model on local features (diagnostic)",
    "fedavg": "FedAvg",
    "fedavg_client_thresholds": "FedAvg, client thresholds",
    "fedavg_finetuned": "FedAvg + local fine-tuning",
    "fedprox": "FedProx",
}


# --------------------------------------------------------------------------- loading


def _maybe(path: Path) -> Any:
    return load_json(path) if path.is_file() else None


def load_results(cfg: Config) -> dict[str, Any]:
    """Everything the pipeline has written under ``results/``."""
    root = cfg.path("results_dir")
    seeds = [int(s) for s in cfg.project.seeds]
    out: dict[str, Any] = {"seeds": seeds, "baselines": {}, "fl": {}, "data": {}, "features": {}}
    for seed in seeds:
        out["baselines"][seed] = _maybe(root / "baselines" / f"seed_{seed}.json")
        out["data"][seed] = _maybe(root / "data" / f"summary_seed_{seed}.json")
        out["features"][seed] = _maybe(root / "features" / f"report_seed_{seed}.json")
        runs: dict[str, dict[str, Any]] = {}
        for scenario, _ in experiment_plan(cfg):
            methods = {}
            for name, file in [("local", "local"), ("pooled_local_features", "pooled_local_features"), ("fedavg", "fedavg"), ("fedprox", "fedprox"),
                               ("fedavg_finetuned", "fedavg_finetuned")]:
                result = _maybe(root / "fl" / f"seed_{seed}" / f"{scenario.name}__{file}.json")
                if result is not None:
                    methods[name] = result
            runs[scenario.name] = methods
        out["fl"][seed] = runs
    out["primary"] = _maybe(root / "fl" / "primary.json")
    out["ledger"] = {name: _maybe(root / "ledger" / f"{name}.json") for name in ("registration", "benchmark", "tamper_demo")}
    out["alerts"] = _maybe(root / "alerts" / "summary.json")
    out["stretch"] = {name: _maybe(root / "stretch" / f"{name}.json") for name in ("secagg", "dp", "poison", "personalised", "vertical")}
    return out


# --------------------------------------------------------------------------- small helpers


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array([v for v in values if v is not None], dtype=float)
    return (float(arr.mean()), float(arr.std())) if len(arr) else (float("nan"), float("nan"))


def pm(values: list[float], digits: int = 3) -> str:
    m, s = mean_std(values)
    return "-" if np.isnan(m) else f"{m:.{digits}f} ± {s:.{digits}f}"


def md_table(header: list[str], rows: list[list[Any]], align: str | None = None) -> str:
    align = align or ("l" + "r" * (len(header) - 1))
    sep = ["---" if a == "l" else "---:" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def mb(n_bytes: float) -> str:
    return f"{n_bytes / 1e6:.2f} MB" if n_bytes >= 1e5 else f"{n_bytes / 1e3:.1f} kB"


def fl_final(res: dict[str, Any], seed: int, scenario: str, method: str) -> dict[str, Any] | None:
    """The reported metrics of one FL-family method for one seed and scenario."""
    runs = res["fl"][seed].get(scenario, {})
    if method == "fedavg_client_thresholds":
        return runs["fedavg"]["final_client_thresholds"] if "fedavg" in runs else None
    return runs[method]["final"] if method in runs else None


def fl_values(res: dict[str, Any], scenario: str, method: str, metric: str = "f1") -> list[float]:
    out = []
    for seed in res["seeds"]:
        final = fl_final(res, seed, scenario, method)
        if final is not None:
            out.append(final[metric])
    return out


def baseline_values(res: dict[str, Any], method: str, metric: str = "f1") -> list[float]:
    return [res["baselines"][s]["methods"][method][metric] for s in res["seeds"] if res["baselines"][s]]


def ordered_scenarios(cfg: Config) -> list[Scenario]:
    """Scenarios by client count, then IID, Dirichlet from weak to strong skew, quantity skew."""
    kind = {"iid": 0, "dirichlet": 1, "quantity": 2}
    return sorted((s for s, _ in experiment_plan(cfg)), key=lambda s: (s.n_clients, kind[s.partition], -(s.alpha or 0.0)))


def scenario_by_name(cfg: Config) -> dict[str, Scenario]:
    return {s.name: s for s, _ in experiment_plan(cfg)}


# --------------------------------------------------------------------------- tables


def table_detection(res: dict[str, Any], cfg: Config, scenarios: list[str]) -> tuple[list[str], list[list[Any]], list[list[Any]]]:
    """Precision / recall / F1 / PR-AUC / ROC-AUC / FPR and confusion counts for every method."""
    names = scenario_by_name(cfg)
    header = ["Method", "Data", *METRIC_HEAD]
    rows: list[list[Any]] = []
    confusion: list[list[Any]] = []

    def add(label: str, data: str, finals: list[dict[str, Any]]) -> None:
        if not finals:
            return
        rows.append([label, data, *[pm([f[m] for f in finals]) for m in METRICS]])
        total = {k: sum(f["confusion"][k] for f in finals) for k in ("tp", "fp", "fn", "tn")}
        confusion.append([label, data, total["tp"], total["fp"], total["fn"], total["tn"]])

    for method in ["rules", "logreg", "random_forest", "lightgbm", "mlp"]:
        add(BASELINE_LABELS[method], "pooled", [res["baselines"][s]["methods"][method] for s in res["seeds"] if res["baselines"][s]])
    for scenario in scenarios:
        if scenario not in names:
            continue
        for method in FL_METHODS:
            finals = [f for f in (fl_final(res, s, scenario, method) for s in res["seeds"]) if f is not None]
            add(FL_LABELS[method], names[scenario].label, finals)
    return header, rows, confusion


def table_scenarios(res: dict[str, Any], cfg: Config) -> tuple[list[str], list[list[Any]]]:
    header = ["Split", "Clients", "Largest / smallest client", "Positive rate range", *[FL_LABELS[m] for m in FL_METHODS]]
    rows = []
    for scenario in ordered_scenarios(cfg):
        parts = [res["fl"][s][scenario.name]["fedavg"]["partition"] for s in res["seeds"] if "fedavg" in res["fl"][s].get(scenario.name, {})]
        if not parts:
            continue
        imbalance = np.mean([p["size_imbalance"] for p in parts])
        low = np.mean([p["illicit_rate_range"][0] for p in parts])
        high = np.mean([p["illicit_rate_range"][1] for p in parts])
        kind = "IID" if scenario.is_iid else (f"Dirichlet α={scenario.alpha:g}" if scenario.partition == "dirichlet" else f"Quantity skew α={scenario.alpha:g}")
        rows.append([kind, scenario.n_clients, f"{imbalance:.1f}x", f"{100 * low:.1f}% - {100 * high:.1f}%",
                     *[pm(fl_values(res, scenario.name, m)) for m in FL_METHODS]])
    return header, rows


def table_typology(res: dict[str, Any], cfg: Config, scenario: str) -> tuple[list[str], list[list[Any]], list[str], np.ndarray]:
    """Recall per typology for the main methods (mean over seeds)."""
    entries: list[tuple[str, Callable[[int], dict[str, Any] | None]]] = [
        ("Rules engine", lambda s: res["baselines"][s]["methods"]["rules"] if res["baselines"][s] else None),
        ("LightGBM (centralised)", lambda s: res["baselines"][s]["methods"]["lightgbm"] if res["baselines"][s] else None),
        ("MLP (centralised)", lambda s: res["baselines"][s]["methods"]["mlp"] if res["baselines"][s] else None),
        ("Pooled model on local features", lambda s: fl_final(res, s, scenario, "pooled_local_features")),
        ("FedAvg", lambda s: fl_final(res, s, scenario, "fedavg")),
        ("FedAvg + local fine-tuning", lambda s: fl_final(res, s, scenario, "fedavg_finetuned")),
        ("Local-only", lambda s: fl_final(res, s, scenario, "local")),
    ]
    labels, matrix = [], []
    for label, getter in entries:
        finals = [f for f in (getter(s) for s in res["seeds"]) if f is not None]
        if not finals:
            continue
        labels.append(label)
        matrix.append([np.mean([f["recall_by_typology"][t] for f in finals if f["recall_by_typology"][t] is not None] or [np.nan]) for t in TYPOLOGIES])
    recall = np.array(matrix)
    header = ["Method", *[t.replace("_", " ") for t in TYPOLOGIES]]
    rows = [[label, *[f"{v:.2f}" for v in row]] for label, row in zip(labels, recall)]
    return header, rows, labels, recall


def table_communication(res: dict[str, Any], cfg: Config) -> tuple[list[str], list[list[Any]]]:
    header = ["Clients (IID)", "Model", "Train: down / up per client per round", "Evaluate: down / up per client per round", "All clients per round",
              "Whole run (30 rounds)", "To reach 90% of centralised F1", "To converge (95% of own best F1)"]
    rows = []
    for scenario, _ in experiment_plan(cfg):
        if not scenario.is_iid:
            continue
        runs = [res["fl"][s][scenario.name]["fedavg"] for s in res["seeds"] if "fedavg" in res["fl"][s].get(scenario.name, {})]
        if not runs:
            continue
        c = [r["communication"] for r in runs]
        per = {k: np.mean([x["per_client_per_round"][k] for x in c]) for k in c[0]["per_client_per_round"]}
        per_round = np.mean([x["per_round_all_clients"]["train"] + x["per_round_all_clients"]["evaluate"] for x in c])
        reached = [x for x in c if x["rounds_to_target"] is not None]
        target = (f"{np.mean([x['rounds_to_target'] for x in reached]):.0f} rounds, {mb(np.mean([x['bytes_to_target'] for x in reached]))}"
                  f" ({len(reached)} of {len(c)} seeds)") if reached else f"not reached (0 of {len(c)} seeds)"
        conv = f"{np.mean([x['convergence']['rounds'] for x in c]):.0f} rounds, {mb(np.mean([x['convergence']['bytes'] for x in c]))}"
        rows.append([scenario.n_clients, mb(c[0]["model_bytes"]), f"{mb(per['train_download'])} / {mb(per['train_upload'])}",
                     f"{mb(per['eval_download'])} / {mb(per['eval_upload'])}", mb(per_round), mb(np.mean([x["total_bytes"] for x in c])), target, conv])
    return header, rows


# --------------------------------------------------------------------------- figures


def make_figures(res: dict[str, Any], cfg: Config, out: Path) -> dict[str, Path]:
    figs: dict[str, Path] = {}
    names = scenario_by_name(cfg)
    primary = primary_scenario(cfg).name
    counts = [int(n) for n in cfg.fl.experiments.n_clients]
    base = counts[0]
    alphas = sorted([float(a) for a in cfg.fl.experiments.alphas], reverse=True)
    central = mean_std(baseline_values(res, "mlp"))[0]
    n_seeds = len(res["seeds"])
    have_fl = all(res["fl"][s].get(f"n{base}_iid", {}).get("fedavg") for s in res["seeds"])

    # ---- method comparison on the primary scenario
    if all(res["baselines"][s] for s in res["seeds"]):
        rows = [{"label": BASELINE_LABELS[m], "f1": mean_std(baseline_values(res, m))[0], "f1_std": mean_std(baseline_values(res, m))[1], "local_data": False}
                for m in ["rules", "logreg", "random_forest", "lightgbm", "mlp"]]
        if have_fl and primary in names:
            for m in ["local", "fedavg", "fedavg_finetuned"]:
                values = fl_values(res, primary, m)
                if values:
                    rows.append({"label": FL_LABELS[m], "f1": mean_std(values)[0], "f1_std": mean_std(values)[1], "local_data": True})
        figs["methods"] = plots.method_comparison(rows, out / "method_comparison.png", "Detection quality by method",
                                                  f"Federated and local methods on the primary split: {names[primary].label}" if primary in names else "", n_seeds)
    if not have_fl:
        return figs

    # ---- local-only against FedAvg on every scenario
    rows = []
    for scenario in ordered_scenarios(cfg):
        local, fed = fl_values(res, scenario.name, "local"), fl_values(res, scenario.name, "fedavg")
        if local and fed:
            rows.append({"label": scenario.label, "local": mean_std(local)[0], "fedavg": mean_std(fed)[0]})
    figs["dumbbell"] = plots.scenario_dumbbell(rows, central, out / "local_vs_fedavg.png", n_seeds)

    # ---- F1 per round for IID and every alpha
    panels = []
    for name in [f"n{base}_iid", *[f"n{base}_dir{a:g}" for a in alphas]]:
        curves = [np.array([h["test_f1"] for h in res["fl"][s][name]["fedavg"]["history"]]) for s in res["seeds"] if "fedavg" in res["fl"][s].get(name, {})]
        if curves:
            stack = np.vstack(curves)
            label = "IID" if name.endswith("iid") else f"Dirichlet α = {names[name].alpha:g}"
            panels.append({"label": label, "mean": stack.mean(axis=0), "std": stack.std(axis=0), "local": mean_std(fl_values(res, name, "local"))[0]})
    figs["rounds"] = plots.f1_vs_round(panels, central, out / "f1_vs_round.png", n_seeds)

    # ---- F1 against data skew
    order = [f"n{base}_iid", *[f"n{base}_dir{a:g}" for a in alphas]]
    labels = ["IID", *[f"α = {a:g}" for a in alphas]]
    series = {}
    for method in ["fedavg", "local", "pooled_local_features", "fedavg_finetuned"]:
        stats = [mean_std(fl_values(res, name, method)) for name in order]
        if all(not np.isnan(m) for m, _ in stats):
            series[method] = {"mean": np.array([m for m, _ in stats]), "std": np.array([s for _, s in stats])}
    figs["alpha"] = plots.lines_by_setting(labels, series, central, out / "f1_vs_alpha.png", f"F1 against data skew ({base} clients)",
                                           f"data split, from identical clients (IID) to the most different clients (Dirichlet α = {min(alphas):g})", n_seeds)

    # ---- F1 against the number of clients
    order = [f"n{n}_iid" for n in counts]
    series = {}
    for method in ["fedavg", "local", "pooled_local_features", "fedavg_finetuned"]:
        stats = [mean_std(fl_values(res, name, method)) for name in order]
        if all(not np.isnan(m) for m, _ in stats):
            series[method] = {"mean": np.array([m for m, _ in stats]), "std": np.array([s for _, s in stats])}
    figs["clients"] = plots.lines_by_setting([f"{n} clients" for n in counts], series, central, out / "f1_vs_clients.png",
                                             "F1 against the number of clients (IID split)", "number of clients", n_seeds)

    # ---- recall per typology
    _, _, methods, recall = table_typology(res, cfg, f"n{base}_iid")
    figs["typology"] = plots.typology_heatmap(methods, TYPOLOGIES, recall, out / "recall_by_typology.png", n_seeds)

    # ---- communication: F1 against traffic
    curves = []
    for n in counts:
        runs = [res["fl"][s][f"n{n}_iid"]["fedavg"] for s in res["seeds"] if "fedavg" in res["fl"][s].get(f"n{n}_iid", {})]
        if runs:
            curves.append({"label": f"{n} clients", "mb": np.mean([[h["bytes_cumulative"] / 1e6 for h in r["history"]] for r in runs], axis=0),
                           "f1": np.mean([[h["test_f1"] for h in r["history"]] for r in runs], axis=0)})
    figs["communication"] = plots.communication(curves, out / "communication.png", n_seeds)
    return figs


def make_ledger_figures(res: dict[str, Any], out: Path) -> dict[str, Path]:
    figs: dict[str, Path] = {}
    bench = res["ledger"]["benchmark"]
    if not bench or "evm" not in bench:
        return figs
    evm = bench["evm"]
    modes = ["one transaction per record", *[f"batch of {row['batch_size']}" for row in evm["batched"] if row["batch_size"] > 1],
             f"Merkle root per {evm['merkle']['batch_size']} documents"]
    throughput = [evm["single"]["records_per_second"], *[row["records_per_second"] for row in evm["batched"] if row["batch_size"] > 1], evm["merkle"]["records_per_second"]]
    gas = [evm["single"]["cost_per_record"], *[row["cost_per_record"] for row in evm["batched"] if row["batch_size"] > 1], evm["merkle"]["cost_per_record"]]
    figs["ledger_modes"] = plots.ledger_modes(modes, throughput, gas, out / "ledger_throughput_gas.png")
    registration = res["ledger"]["registration"]
    n_docs = registration["documents"] if registration else 250000
    records = np.linspace(0, n_docs, 50)
    figs["storage"] = plots.storage_growth(records, records * evm["storage_bytes_per_record"], records * evm["merkle"]["storage_bytes_per_record"],
                                           out / "ledger_storage.png")
    return figs


def make_scoring_figures(res: dict[str, Any], out: Path) -> dict[str, Path]:
    alerts = res["alerts"]
    if not alerts:
        return {}
    ev = alerts["evaluation"]
    return {"bands": plots.risk_bands(ev["bands"], ev["calibration"]["risk_score_reliability"], out / "risk_bands.png")}


def make_stretch_figures(res: dict[str, Any], out: Path) -> dict[str, Path]:
    figs: dict[str, Path] = {}
    dp = res["stretch"]["dp"]
    if dp:
        eps = [row["epsilon"] for row in dp["private"]]
        stats = [mean_std([r["f1"] for r in row["runs"]]) for row in dp["private"]]
        figs["dp"] = plots.privacy_utility(eps, np.array([m for m, _ in stats]), np.array([s for _, s in stats]),
                                           mean_std([r["f1"] for r in dp["non_private"]])[0], out / "privacy_utility.png", len(dp["seeds"]))
    poison = res["stretch"]["poison"]
    if poison:
        order = [("no_attack", "No attack", False), ("attack_no_defence", "Attack, no defence", True), ("attack_median_clipping", "Attack, median clipping", True),
                 ("no_attack_median_clipping", "No attack, median clipping", False)]
        stats = [mean_std([r.get("last_round_f1", r["f1"]) for r in poison["runs"][key]]) for key, _, _ in order]
        figs["poison"] = plots.condition_bars([label for _, label, _ in order], np.array([m for m, _ in stats]), np.array([s for _, s in stats]),
                                              [h for _, _, h in order], out / "poisoning.png", "One malicious client: F1 of the model after the last round",
                                              f"F1 on the test period, global model of the last round ({plots.spread_text(len(poison['seeds']))})")
    vertical = res["stretch"]["vertical"]
    if vertical:
        order = [("customs_alone", "Customs alone", False), ("bank_alone", "Bank alone", False), ("trade_alone", "Trade registry alone", False),
                 ("split_nn", "Split neural network (all three)", True), ("pooled_columns", "One model on all three parties' columns", False)]
        stats = [mean_std([r["methods"][key]["f1"] for r in vertical["runs"]]) for key, _, _ in order]
        figs["vertical"] = plots.condition_bars([label for _, label, _ in order], np.array([m for m, _ in stats]), np.array([s for _, s in stats]),
                                                [h for _, _, h in order], out / "vertical_fl.png", "Vertical FL: three parties with different columns of the same trades",
                                                f"F1 on the test period ({plots.spread_text(len(vertical['runs']))})")
    return figs
