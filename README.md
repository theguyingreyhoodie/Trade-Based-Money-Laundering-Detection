# Privacy-Preserving TBML Detection with Federated Learning and Blockchain

B.E. final-year project prototype. Trade-Based Money Laundering (TBML) hides illicit money in trade
transactions. The evidence is split across organisations: banks hold payments, traders and
carriers hold invoices and shipment records, customs holds declarations. None of them can pool raw
records with the others.

**Research question.** Can Federated Learning and Blockchain support privacy-preserving TBML
detection across multiple organisations?

**What this repository does.** It simulates trade data with ten laundering typologies, builds risk
indicators, trains detectors without moving raw data between institutions (Flower, FedAvg),
registers hashes of documents, model updates and alerts on a permissioned chain (Solidity on
Hardhat), and raises calibrated, explainable alerts for human investigators.

> All data is synthetic. Results compare methods under the simulator's assumptions; they are not
> estimates of real-world performance. The system flags trades for human review. It does not
> decide that anything is laundering.

## Results at a glance

Mean of 3 seeds on the test period (full tables and figures: [`results/RESULTS.md`](results/RESULTS.md)).

| Method | Raw data pooled? | F1 | PR-AUC |
|---|---|---|---|
| Rules engine | yes | 0.431 | 0.407 |
| LightGBM, centralised | yes | 0.806 | 0.874 |
| MLP, centralised | yes | 0.753 | 0.820 |
| Local-only (3 institutions, IID) | no | 0.653 | 0.699 |
| **FedAvg (3 institutions, IID)** | **no** | **0.702** | **0.755** |
| Local-only (10 institutions, IID) | no | 0.519 | 0.531 |
| **FedAvg (10 institutions, IID)** | **no** | **0.628** | **0.676** |

- With IID data FedAvg beats training alone, and the gain grows with the number of institutions
  (+0.05 F1 at 3, +0.06 at 5, +0.11 at 10).
- FedAvg stays within 0.03 F1 of a model trained on pooled rows that uses each institution's
  locally computed indicators (+0.004 at 3 IID institutions). Most of the gap to the centralised
  model therefore comes from **information fragmentation** (indicators that need related trades
  from several institutions), not from parameter averaging.
- Under strong data skew (Dirichlet α = 0.1) local-only training beats FedAvg (0.801 vs 0.737);
  fine-tuning the global model locally matches it (0.804).
- The server receives model parameters and fixed-size aggregates only (enforced and tested).
- Evidence ledger: 6-9 ms per registration (p50; timings vary between runs), 103k gas; Merkle
  batching brings a trade document down to 206 gas. A changed invoice, model update or alert fails
  verification.
- Alerts: precision 31% / 59% / 96% in the Medium / High / Critical bands; each alert lists its top
  SHAP contributions in plain English, the rules that fired and a suspected typology.

## Quick start

Requirements: Python 3.10+ and, for the EVM chain, Node.js 18+. Everything runs on a laptop CPU.
Without Node.js the pipeline uses a pure-Python ledger with the same interface.

```bash
make setup          # creates .venv, installs Python dependencies and the Hardhat project
make all            # reproduces every result (about 22 minutes on an Apple M5 laptop)
make test           # test suite
make dashboard      # investigator dashboard at http://localhost:8501
```

`make quick` runs the whole pipeline on a small profile in about 3 minutes. Most of `make all` is
the federated sweep (39 simulated federations, plus local-only and diagnostic models); on macOS run it as `caffeinate -i make all` so the
machine does not go to sleep part-way (a sleeping laptop pauses the run but does not change the
results, which are bit-for-bit reproducible from the seeds).

**The local chain.** The ledger steps start a Hardhat node at `http://127.0.0.1:8545` when none is
running, deploy `EvidenceRegistry`, and leave the node running so the dashboard can verify
evidence live. To control it yourself:

```bash
make chain-up       # start the node in the background (log: .chain.log)
make chain-down     # stop it
.venv/bin/python scripts/chain.py status
```

A Hardhat node keeps its chain in memory. After a restart the client re-registers its off-chain
journal on a fresh contract (hashes and event times are kept; transaction ids are new).

### Step by step

| Step | Command | Output |
|---|---|---|
| 1. Synthetic data | `python scripts/generate_data.py --all-seeds` | `data/seed_<s>/` (per-owner raw tables, reference tables, ground truth) |
| 2. Features | `python scripts/build_features.py --all-seeds` | `artifacts/seed_<s>/features/`, `docs/features.md` |
| 3. Baselines | `python scripts/run_baselines.py --all-seeds` | `results/baselines/`, `docs/rules.md` |
| 4. Register documents | `python scripts/ledger_register.py` | `results/ledger/registration.json` |
| 5. Primary federated run | `python scripts/run_fl.py` | `results/fl/primary.json`, global model, model hashes on the ledger |
| 6. Federated sweep | `python scripts/run_fl_experiments.py` | `results/fl/seed_<s>/` |
| 7. Alerts | `python scripts/generate_alerts.py` | `results/alerts/` (JSON, readable report, summary) |
| 8. Ledger checks | `python scripts/ledger_tamper_demo.py`, `python scripts/ledger_benchmark.py` | `results/ledger/` |
| 9. Stretch experiments | `python scripts/run_stretch.py` | `results/stretch/` |
| 10. Report | `python scripts/evaluate.py` | `results/RESULTS.md`, `results/plots/`, `results/tables/` |

Every script takes `--config`, `--profile <name>`, `--seed <n>` and `--set key=value` (repeatable),
and saves its metrics as JSON under `results/`.

## The six objectives: where the code is and where the result is

| # | Objective | Code | Result |
|---|---|---|---|
| 1 | Risk indicators from invoice, shipment, payment and counterparty data | `src/tbml/features/` (`link.py` record linkage, `build.py` 81 indicators, `dictionary.py`) | [`docs/features.md`](docs/features.md); RESULTS section 3; `results/features/` |
| 2 | Baselines: rules engine and classical ML | `src/tbml/rules/engine.py` (20 rules), `src/tbml/models/` (logistic regression, random forest, LightGBM, MLP) | [`docs/rules.md`](docs/rules.md); RESULTS section 4; `results/baselines/` |
| 3 | Federated training that keeps raw data local | `src/tbml/fl/` (`partition.py`, `data.py`, `client.py`, `server.py`, `boundary.py`, `simulation.py`, `local.py`) | [`docs/fl_design.md`](docs/fl_design.md); RESULTS sections 5 and 6; `results/fl/`; `tests/test_fl.py::test_server_only_ever_receives_parameters` |
| 4 | Blockchain evidence integrity verification and auditability | `chain/contracts/EvidenceRegistry.sol`, `src/tbml/ledger/` (`base.py` interface, `evm.py`, `pyledger.py`, `merkle.py`, `evidence.py`, `audit.py`) | [`docs/ledger.md`](docs/ledger.md); RESULTS section 7; `results/ledger/`; tamper tests in `tests/test_ledger.py` |
| 5 | Explainable TBML risk scores | `src/tbml/explain/` (`scoring.py`, `shap_explain.py`, `phrases.py`, `alerts.py`) | [`docs/scoring.md`](docs/scoring.md); RESULTS section 8; [`results/alerts/report.md`](results/alerts/report.md) |
| 6 | Evaluation of detection performance, communication overhead and blockchain latency/storage | `src/tbml/eval/` (`metrics.py`, `report.py`, `plots.py`, `results_md.py`), `src/tbml/ledger/bench.py` | [`results/RESULTS.md`](results/RESULTS.md), `results/plots/`, `results/tables/` |

Architecture steps of the proposal map to the same code: Step 01 data preparation
(`src/tbml/data/`), Step 02 preprocessing and indicators (`features/`), Step 03 baselines
(`rules/`, `models/`), Step 04 federated learning (`fl/`), Step 05 risk scoring (`explain/`),
Step 06 evaluation and audit (`eval/`, `ledger/`).

## How it fits together

```
 data owners (simulated)           each FL client (institution)               server / aggregator
 ------------------------          ------------------------------            ---------------------
 bank:    payments, KYC     -->    link its own records                       FedAvg over model
 trade:   invoices, B/Ls    -->    81 risk indicators (local view)   <---->   parameters (Flower)
 customs: declarations      -->    local training, local evaluation           sums score histograms
                                   rules + global model -> risk score          picks threshold, metrics
                                   SHAP + rules -> alert
        |                                   |                                          |
        v                                   v                                          v
   hash of every document           hash of every update, every alert         hash of every global model
        \___________________________________|__________________________________________/
                                            v
                     EvidenceRegistry (permissioned: allowlisted organisations only)
                     stores SHA-256 hashes + metadata; raw records stay off-chain
```

## Design decisions worth knowing

- **Horizontal, not vertical, federated learning.** The proposal's picture (Bank A payments,
  Bank B trade records, Customs) is a vertical split; FedAvg is horizontal. The primary system
  treats each client as an institution holding complete linked records of its own customers'
  trades. A vertical split-network demo for the Bank / Trade / Customs picture is in
  `src/tbml/fl/vertical.py` (design note: `docs/fl_design.md`, section 9).
- **Clients compute their indicators from their own records only**, including the trade-graph
  ones. Federated results are therefore not flattered by pooled information.
- **Evaluation stays inside the protocol.** Clients return score histograms; the server sums
  them, tunes the threshold on validation and reads all metrics off the test histogram. The same
  code scores every method.
- **A client-boundary guard** validates every reply against a whitelist (model arrays of the
  template's shape, bounded aggregates, one hash) before the server uses it.
- **Merkle batching** for trade documents: one transaction per 500 documents.
- **Fictional jurisdictions** with FATF-style risk tiers, so no real country is labelled.
- **One YAML config** (`configs/default.yaml`) with profiles (`quick`, `test`) and fixed seeds.

## Stretch goals (all implemented, results in RESULTS section 9)

| Goal | Where |
|---|---|
| Secure aggregation by pairwise additive masking | `src/tbml/fl/extensions.py` |
| DP-SGD with Opacus, privacy / utility curve | `src/tbml/fl/dp.py` |
| Vertical FL (split neural network) demo | `src/tbml/fl/vertical.py` |
| Model poisoning demo and norm-clipping defence | `src/tbml/fl/extensions.py`, `src/tbml/fl/server.py` |
| Local fine-tuning of the global model | `src/tbml/fl/local.py` |
| Investigator dashboard (Streamlit) | `dashboard/app.py` |

## Repository layout

```
configs/default.yaml     every setting, plus profiles
src/tbml/
  data/        simulator: world, entities, legitimate trade, typologies, per-owner views
  features/    cleaning, record linkage, risk indicators, feature dictionary
  rules/       red-flag rules engine
  models/      time-aware split, baselines, the MLP
  fl/          partitioning, Flower client and server, boundary guard, experiments, stretch goals
  ledger/      EvidenceLedger interface, EVM and pure-Python backends, hashing, Merkle, benchmark
  explain/     calibration and risk score, SHAP, plain-English phrases, alerts
  eval/        metrics, tables, figures, report
chain/         Hardhat project and EvidenceRegistry.sol
scripts/       one command-line script per step, run_all.sh
dashboard/     Streamlit app
docs/          data_generator.md, features.md, rules.md, fl_design.md, ledger.md, scoring.md
results/       metrics (JSON), tables (CSV), figures, RESULTS.md, alerts
tests/         pytest suite
```

Generated data (`data/`) and models (`artifacts/`) are not committed; they are reproduced from the
seeds.

## Tests

```bash
make test                                   # everything
.venv/bin/python -m pytest -m "not slow"    # skip the Flower simulations and end-to-end runs
```

The suite covers the generator (schema, label rates, reproducibility, identifier leakage), the
indicators (no look-ahead, labels never used), every rule, the metrics (against scikit-learn),
hashing and both ledger backends (including the tampered invoice and tampered model update), a
real simulated FL round (the server only receives parameters), the stretch goals, the dashboard,
and the command-line pipeline end to end. EVM tests start their own Hardhat node on port 8546 and
are skipped when Node.js is not installed.

## Known limitations

Synthetic data with clean labels and a stationary adversary; horizontal FL cannot see patterns
that span institutions; model updates and aggregates leak some information unless secure
aggregation or DP is switched on; no defence against a determined poisoner; a single-node
development chain without consensus; three seeds. Details: RESULTS section 11 and the
"Limitations" sections of the documents in `docs/`.

Technical notes:

- Flower marks `run_simulation()` as deprecated in favour of the `flwr run` CLI. It works in the
  pinned version (`flwr<1.44`); moving to `flwr run` means packaging the client and server as a
  Flower app.
- On macOS, PyTorch and LightGBM each bring an OpenMP runtime. `src/tbml/models/gbm.py` imports
  torch first and limits LightGBM to one thread; import LightGBM only through that module.
- `make setup` uses Homebrew's `python3.12` when present, because PyTorch wheels lag behind the
  newest Python releases.
