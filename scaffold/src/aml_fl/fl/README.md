# Federated learning (Phase 3) — owner: Aniket

Planned files:

- `client_app.py`: Flower `ClientApp`. Each client loads its own partition with
  `aml_fl.preprocess.load_client("data/partitions/client_i")`, builds
  `aml_fl.models.BankModel(input_dim=...)` and exchanges **only** the shared head:
  - `get_parameters` → `model.get_shared_weights()`
  - `set_parameters` → `model.set_shared_weights(...)`
  - `fit` → `undersample(...)`, then `train_epochs(...)` for a few local epochs
  - `evaluate` → `predict_proba` + `aml_fl.metrics.evaluate`
- `server_app.py`: Flower `ServerApp` with `FedAvg` (later `FedProx`).
- `strategy.py`: custom strategy that subclasses FedAvg and hashes the global model in
  `aggregate_fit` for the blockchain audit log (Phase 4).

Build these in two steps:

1. **Naive FedAvg.** Every client uses only the features common to all clients, and the
   whole model is averaged. This is a baseline.
2. **Encoder + shared head.** Each client uses its own schema, and only the shared head is
   averaged. This is our method.

Start from Flower's PyTorch quickstart (`flwr new`) and adapt it.
