# Privacy-Preserving Money Laundering Detection Across Heterogeneous Banks

B.E. Computer Engineering project, MES Wadia College of Engineering, Pune (2026–27).
**Team:** Anuj Nanhe, Aniket Shelke, Laiba Shaikh. **Guide:** Dr. S. R. Khonde.

Several banks train a money-laundering detection model together using **Federated Learning**,
without sharing raw data. Each bank has a **different data schema**, so it keeps a private
encoder, and only a shared head is aggregated with **FedAvg** (orchestrated by Flower).
A **blockchain** audit layer records hashes of model updates and alerts, which makes the
process tamper-evident.

**Scope.** The project detects money laundering through bank transfers between accounts.
Trade-based money laundering (invoices, goods, pricing) is future scope.

The full plan is in [`docs/AML_FL_Implementation_Plan.pdf`](docs/AML_FL_Implementation_Plan.pdf).

## Status

The whole team works on every phase together.

| Phase | What | Status |
|---|---|---|
| 0 | Setup & learning | in progress |
| 1 | Data pipeline: features, time split, profiles, bank partitions | **first version done** |
| 2 | Baselines (centralized / local-only) + recall per pattern | LR, XGBoost, MLP done (toy data) |
| 3 | Federated learning (Flower) | naive FedAvg + encoder/shared head done (toy data); zero-filled next — see `src/aml_fl/fl/README.md` |
| 4 | Blockchain audit layer | todo — see `src/aml_fl/chain/README.md` |
| 5 | SHAP + Streamlit dashboard | todo |
| 6 | Evaluation & report | todo |

## Setup

```bash
git clone https://github.com/theguyingreyhoodie/Trade-Based-Money-Laundering-Detection.git
cd Trade-Based-Money-Laundering-Detection/scaffold
python3.12 -m venv .venv          # Python 3.11/3.12; newer versions may lack torch wheels
source .venv/bin/activate          # Windows: .venv\Scripts\activate
brew install libomp                # macOS only: XGBoost needs it
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest -q                          # should pass without the real dataset
```

If the path to your clone contains **spaces**, create the venv somewhere else, e.g.
`python3.12 -m venv ~/.venvs/aml-fl`. Flower's simulation runs on Ray, and Ray cannot start
its workers from a Python path with spaces (it hangs, then fails with "worker startup
repeatedly failed"). `scripts/run_fl.py` stops early with a message if this is the case.

## Get the data

1. Download **IBM Transactions for Anti Money Laundering (AML)** from Kaggle:
   <https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml>
2. Put `HI-Small_Trans.csv` and `HI-Small_Patterns.txt` in `data/raw/`. The Patterns file
   lists the laundering attempts by type (fan-in, cycle, ...). It is used only to report
   recall per pattern, never as a feature. The pipeline still runs without it.

To download only these two files with the Kaggle CLI:

```bash
kaggle datasets download ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f HI-Small_Trans.csv -p data/raw
kaggle datasets download ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f HI-Small_Patterns.txt -p data/raw
```

The dataset is **not** committed to git, because it is too large. Everyone downloads their own copy.

## Run the pipeline

```bash
# Quick try without the real data:
python scripts/make_toy_data.py --out data/raw/toy_Trans.csv   # also writes toy_Patterns.txt
python scripts/prepare_data.py --raw data/raw/toy_Trans.csv

# Real data (use --nrows 500000 while developing, then drop it for final runs):
python scripts/prepare_data.py --config configs/default.yaml --nrows 500000

# Baselines (Logistic Regression, XGBoost and the PyTorch MLP; centralized and local-only):
python scripts/run_baselines.py

# Federated learning simulation (one simulated node per client bank):
python scripts/run_fl.py               # mode and rounds from the fl: section of the config
python scripts/run_fl.py --rounds 3    # quick check
python scripts/run_fl.py --mode encoder_head   # private encoders, shared head
python scripts/run_fl.py --mode hybrid         # shared core encoder + private encoder + shared head
```

Outputs:

- `data/processed/all_transactions.parquet` contains every feature and profile column, for the centralized baselines.
- `data/processed/summary.json` holds dataset statistics, the estimated USD exchange rates and the bank-to-client map.
- `data/partitions/client_i/` contains each bank's own transactions and its own columns (`schema.json`).
- `data/processed/patterns.parquet` maps `txn_id` to its laundering pattern (only if the Patterns file exists).
- `experiments/results/baselines.csv` holds the baseline results.
- `experiments/results/fl_<mode>.csv` holds the FL test results (per client and pooled), with
  per-round metrics in `fl_<mode>/history.csv` and the model hashes in `fl_<mode>/audit.json`.
- `experiments/results/baselines_by_pattern.csv` holds recall per laundering pattern for each baseline.

## How the data pipeline works

1. **Clean** (`src/aml_fl/data.py`). Parses timestamps and builds account keys (`bank_account`).
   It also converts amounts to USD, using exchange rates estimated from the
   cross-currency transactions in the data itself.
2. **Features** (`src/aml_fl/features.py`). Builds transaction features plus sender and receiver
   history: count, mean and std of amount, fan-out, fan-in, currencies used, time gaps and
   24-hour bursts. Each history feature uses **only earlier transactions**, so there's no
   future leakage, and labels are never used.
3. **Time split** (`src/aml_fl/split.py`). Splits by time, not randomly: 60% train, 20% val, 20% test.
4. **Synthetic profiles** (`src/aml_fl/profiles.py`). Generates customer attributes per account
   (account age, KYC risk, occupation, income and so on). Only **training-period** labels
   influence them, through `label_signal` (default 0.1). Run with `--label-signal 0` or
   `--no-profiles` for the ablation.
5. **Partition** (`src/aml_fl/partition.py`). Assigns banks to 5 clients by *From Bank*. Every
   client gets the 9 `core_features` (transaction fields every bank sees), plus the history
   features and profile columns listed for it under `schemas` in the config, so the schemas
   differ.
6. **Preprocess** (`src/aml_fl/preprocess.py`). Each client fits one-hot encoding and scaling on
   its **own** training data only.
7. **Pattern tags** (`src/aml_fl/patterns.py`). Reads `*_Patterns.txt` and tags each laundering
   transaction with its pattern, for evaluation only. The tags are saved in a separate file, so
   they can never become features. Our features only look one hop from an account, so expect
   good recall on fan-in and fan-out and weaker recall on cycles and stacks.

## Repository layout

```
configs/         experiment configs (default.yaml)
data/            raw/, processed/, partitions/  (git-ignored)
docs/            project plan
notebooks/       exploration notebooks
scripts/         prepare_data.py, run_baselines.py, make_toy_data.py
src/aml_fl/      library code
  data.py features.py split.py profiles.py partition.py preprocess.py patterns.py
  models.py      Encoder + SharedHead + BankModel (PyTorch)
  metrics.py     PR-AUC, F1, recall, FPR, threshold tuning, recall per pattern
  fl/            Flower client/server (Phase 3)
  chain/         Solidity contract + web3 client (Phase 4)
dashboard/       Streamlit app (Phase 5)
experiments/     results CSVs
tests/           pytest suite
report/          report drafts
```

## Working together

- `main` must always work. Do your work on a branch, for example `data/features`,
  `fl/naive-fedavg` or `chain/contract`, and merge through a pull request.
- Run `pytest -q` before you open a PR.
- Never commit data, model checkpoints or secrets such as private keys or `.env` files.
- Fix random seeds, and save the config you used next to every result.
