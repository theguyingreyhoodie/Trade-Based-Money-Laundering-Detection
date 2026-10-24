# Privacy-Preserving Money Laundering Detection

B.E. Computer Engineering project, MES Wadia College of Engineering, Pune (2026–27).
**Team:** Anuj Nanhe, Aniket Shelke, Laiba Shaikh. **Guide:** Dr. S. R. Khonde.

Several banks train one money-laundering detection model together with **Federated Learning**
(Flower, FedAvg), and none of them shares its raw data. Each bank stores different columns:
an encoder for the columns every bank has and the prediction head are averaged across banks,
while each bank keeps a private encoder for its own columns. Alerts are explained with SHAP. A
**blockchain** audit layer (Hardhat + Solidity) records hashes of model updates and alerts, so
any later change to them can be detected.

![The dashboard: a flagged transaction, why the model flagged it, and its on-chain record](scaffold/docs/images/dashboard-alerts.png)

**Scope.** The project detects money laundering through bank transfers between accounts.
Trade-based money laundering (invoices, goods, pricing) is future scope.

Data: [IBM Transactions for Anti Money Laundering (HI-Small)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml),
plus synthetic bank-specific customer profiles.

## Quick start

```bash
cd scaffold
python3.12 -m venv ~/.venvs/aml-fl && source ~/.venvs/aml-fl/bin/activate   # outside paths with spaces
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest -q
python dashboard/app.py          # http://127.0.0.1:8000
```

**What comes with a clone.** The code, the report figures (`scaffold/docs/figures/`) and the
small results of every experiment (`scaffold/experiments/results/`: summaries, training
histories, audit hashes, alerts), so the dashboard and `make_figures.py` work straight away.
The dataset does not: to rerun experiments, download it from Kaggle (see
[`scaffold/README.md`](scaffold/README.md#get-the-data)) and run `prepare_data.py`. The full
set of experiments takes about 4 hours.

Setup, the pipeline and the project status are in [`scaffold/README.md`](scaffold/README.md).
The full plan is in [`scaffold/docs/AML_FL_Implementation_Plan.pdf`](scaffold/docs/AML_FL_Implementation_Plan.pdf).

## License

[MIT](LICENSE)
