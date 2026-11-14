"""End-to-end: the command-line scripts, run in order on the test profile."""

from __future__ import annotations

import json

import pytest

from tbml.data.typologies import TYPOLOGIES


def _load(path):
    return json.loads(path.read_text())


@pytest.mark.slow
def test_scripts_produce_every_output(pipeline_run) -> None:
    results, artifacts, root = pipeline_run["results"], pipeline_run["artifacts"], pipeline_run["root"]
    for rel in ["data/summary_seed_0.json", "features/report_seed_0.json", "baselines/seed_0.json", "ledger/registration.json", "fl/primary.json",
                "alerts/alerts.json", "alerts/summary.json", "alerts/report.md", "ledger/tamper_demo.json", "ledger/benchmark.json", "RESULTS.md",
                "tables/detection_performance.csv", "tables/rules.csv", "plots/method_comparison.png", "plots/risk_bands.png"]:
        assert (results / rel).is_file(), rel
    for rel in ["raw/bank/payments.csv", "raw/trade/invoices.csv", "raw/customs/declarations.csv", "ground_truth/joined_trades.csv"]:
        assert (root / "data" / "seed_0" / rel).is_file(), rel
    assert (artifacts / "seed_0" / "features" / "central.npz").is_file()
    assert (artifacts / "ledger" / "python" / "chain.jsonl").is_file()

    baselines = _load(results / "baselines" / "seed_0.json")["methods"]
    assert set(baselines) == {"rules", "logreg", "random_forest", "lightgbm", "mlp"}
    primary = _load(results / "fl" / "primary.json")
    assert primary["boundary"]["violations"] == 0 and primary["n_clients"] >= 3
    assert primary["ledger"]["registered"] == primary["rounds"] * (primary["n_clients"] + 1), "every update and global model hash is on the ledger"
    assert all(h["ledger_status"] == "registered" and h["tx_id"] for h in primary["model_hashes"])


@pytest.mark.slow
def test_evidence_is_verifiable_end_to_end(pipeline_run) -> None:
    demo = _load(pipeline_run["results"] / "ledger" / "tamper_demo.json")
    for part in ("trade_document", "model_update", "alert"):
        assert demo[part] is not None and demo[part]["as_expected"], part
    assert demo["trade_document"]["original"]["verified"] and not demo["trade_document"]["tampered"]["verified"]
    assert demo["model_update"]["original"]["on_ledger"] and not demo["model_update"]["one_weight_changed"]["on_ledger"]
    registration = _load(pipeline_run["results"] / "ledger" / "registration.json")
    assert registration["documents"] > 20000 and registration["batches"] < registration["documents"] / 100
    alerts = _load(pipeline_run["results"] / "alerts" / "alerts.json")
    assert alerts["n_alerts"] == len(alerts["alerts"]) > 0
    assert all(a["ledger"]["status"] == "registered" for a in alerts["alerts"])
    assert alerts["model"]["global_model_sha256"] in {h["sha256"] for h in _load(pipeline_run["results"] / "fl" / "primary.json")["model_hashes"]}


@pytest.mark.slow
def test_results_report_maps_every_objective(pipeline_run) -> None:
    report = (pipeline_run["results"] / "RESULTS.md").read_text()
    for heading in ["## 1. Summary", "Objective 1: risk indicators", "Objective 2: baselines", "Objective 3: federated learning", "## 6. Communication overhead",
                    "Objective 4: blockchain evidence layer", "Objective 5: explainable risk scores", "## 10. Confusion matrices", "## 11. Limitations"]:
        assert heading in report, heading
    assert "All data is synthetic" in report and "flags trades for human review" in report
    for typology in TYPOLOGIES:
        assert typology.replace("_", " ") in report
    assert "| Rules engine |" in report and "| LightGBM |" in report


@pytest.mark.slow
def test_experiment_sweep_and_full_report(pipeline_run) -> None:
    """The sweep, the cheaper stretch parts, and the report with the federated sections filled in."""
    run, results = pipeline_run["run"], pipeline_run["results"]
    run("run_fl_experiments.py")
    files = sorted(p.name for p in (results / "fl" / "seed_0").glob("*.json"))
    assert {"n3_iid__fedavg.json", "n3_iid__local.json", "n3_iid__pooled_local_features.json", "n3_dir0.5__fedavg.json", "n3_dir0.5__fedprox.json",
            "n3_qty0.5__fedavg.json"} <= set(files)
    run("run_stretch.py", "--only", "secagg,personalised,vertical")
    secagg = _load(results / "stretch" / "secagg.json")
    assert secagg["one_round"]["max_abs_weight_difference"] < 1e-5 and secagg["one_round"]["same_confusion_matrix"]
    assert _load(results / "stretch" / "vertical.json")["runs"][0]["methods"]["split_nn"]["pr_auc"] > 0
    assert (results / "fl" / "seed_0" / "n3_iid__fedavg_finetuned.json").is_file()

    run("evaluate.py")
    report = (results / "RESULTS.md").read_text()
    for text in ["### 5.1 Does federating help?", "### 5.3 F1 by round", "### 5.4 F1 against data skew", "### 5.5 F1 against the number of clients",
                 "### 5.7 Raw data stays local", "### 9.1 Secure aggregation", "### 9.4 Vertical FL demo", "FedProx"]:
        assert text in report, text
    for name in ["local_vs_fedavg.png", "f1_vs_round.png", "f1_vs_alpha.png", "f1_vs_clients.png", "recall_by_typology.png", "communication.png", "vertical_fl.png"]:
        assert (results / "plots" / name).stat().st_size > 5000, name
    for name in ["fl_scenarios.csv", "communication.csv", "recall_by_typology.csv", "confusion_matrices.csv"]:
        assert (results / "tables" / name).is_file(), name
    with pytest.raises(SystemExit):
        run("run_stretch.py", "--only", "nonsense")
