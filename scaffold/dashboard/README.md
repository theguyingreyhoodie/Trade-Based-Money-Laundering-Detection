# Dashboard (Phase 5) — owner: Laiba

`app.py` will be a Streamlit app with three pages:

1. **Alerts:** flagged transactions, their risk scores and the top SHAP reasons for each.
2. **Training:** FL training curves and the centralized / local / federated comparison.
3. **Audit:** blockchain records, with a "Verify" button that re-hashes a record and checks it against the chain.

Run it with `streamlit run dashboard/app.py`.
