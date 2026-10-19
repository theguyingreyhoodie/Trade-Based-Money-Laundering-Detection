# Privacy-Preserving Money Laundering Detection

B.E. Computer Engineering project, MES Wadia College of Engineering, Pune (2026–27).
**Team:** Anuj Nanhe, Aniket Shelke, Laiba Shaikh. **Guide:** Dr. S. R. Khonde.

Several banks train one money-laundering detection model together with **Federated Learning**
(Flower, FedAvg), and none of them shares its raw data. Each bank stores different columns,
so it keeps a private encoder, and only a shared head is averaged across banks. A
**blockchain** audit layer (Hardhat + Solidity) records hashes of model updates and alerts, so
any later change to them can be detected.

Data: [IBM Transactions for Anti Money Laundering (HI-Small)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml),
plus synthetic bank-specific customer profiles.

## Quick start

```bash
cd scaffold
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest -q
```

Setup, the pipeline and the project status are in [`scaffold/README.md`](scaffold/README.md).
The full plan is in [`scaffold/docs/AML_FL_Implementation_Plan.pdf`](scaffold/docs/AML_FL_Implementation_Plan.pdf).

## License

[MIT](LICENSE)
