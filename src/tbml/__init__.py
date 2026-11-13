"""Privacy-preserving TBML detection: federated learning + blockchain evidence layer.

Sub-packages follow the pipeline steps of the project proposal:

``data``      synthetic trade simulator and per-owner views        (Step 01)
``features``  cleaning, record linkage, risk indicators           (Step 02, objective 1)
``rules``     transparent red-flag rules engine                   (Step 03, objective 2)
``models``    split, classical ML baselines and the MLP           (Step 03, objective 2)
``fl``        federated training with Flower                      (Step 04, objective 3)
``ledger``    evidence registry (EVM or pure-Python backend)      (objective 4)
``explain``   calibrated risk score, SHAP, alerts                 (Step 05, objective 5)
``eval``      metrics, plots and the results report               (Step 06, objective 6)
"""

__version__ = "0.1.0"
