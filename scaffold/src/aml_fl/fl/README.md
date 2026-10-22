# Federated learning (Phase 3)

Five client banks train one laundering detector with Flower, without sharing raw data.
Each client loads its own partition with
`aml_fl.preprocess.load_client("data/partitions/client_i")`. `schema.json` lists the client's
columns and the `core_columns` that every client has.

Run with `python scripts/run_fl.py` (see the main README). **Status:** naive FedAvg
(`--mode naive`) and the encoder + shared head (`--mode encoder_head`) are implemented;
zero-filled FedAvg comes next.

In `encoder_head` mode a bank's encoder is stored in its own `context.state` between rounds and
is never put in a reply. Each bank reports `encoder_rounds`, which must equal the round
number in `history.csv`; a lower value means the encoder was lost between rounds.

Whenever the whole model is averaged, every bank must produce the same input layout. The
category columns therefore use the public lists in `features.SHARED_VOCAB` (payment formats
and currencies) instead of categories learned from each bank's own data.

## Three setups to compare

| Setup | Model averaged by FedAvg | Features each bank uses | What it shows |
|---|---|---|---|
| **Naive FedAvg** (baseline) | whole `BankModel` | the 9 `core_columns` only | what banks get from only the columns they all have |
| **Zero-filled FedAvg** (baseline) | whole `BankModel` | the union of all clients' columns; columns a bank lacks are 0, plus a 0/1 "missing" flag per column | the standard workaround for different columns; banks must agree on one global column list |
| **Encoder + shared head** (ours) | `SharedHead` only (`get_shared_weights` / `set_shared_weights`) | the bank's full own schema | uses everything each bank has, and keeps the encoder and the bank's column list private |

All three must land between the **local-only** and **centralized** MLP results in
`experiments/results/baselines.csv`. The key comparison is ours versus zero-filled FedAvg.
If zero-filled FedAvg scores higher, report the gap as the accuracy cost of keeping the
schema private.

## Training

- **Rounds:** start with 30 rounds of FedAvg, with clients weighted by their number of
  training rows. Try FedProx afterwards.
- **Local training:** in `fit`, undersample the client's training data again each round with
  `undersample(..., seed=seed + round)`, then run `train_epochs` for 1–2 epochs.
- **Settings:** read `lr` and `batch_size` from the `mlp:` section of the config, so the FL
  numbers are comparable with the MLP baselines.
- **Imbalance:** use `pos_weight(y)` of the undersampled data in the loss, as the baselines do.

## Evaluation

- Each client tunes its own threshold on its own validation split (`best_f1_threshold`), then
  evaluates on its test split with `aml_fl.metrics.evaluate`.
- Report per-client results and one pooled result (all clients' test scores together).
- Report recall per pattern with `recall_by_pattern`, joining `data/processed/patterns.parquet`
  on the client's test `txn_ids`.

## Planned files

- `client_app.py`: Flower `ClientApp`. `get_parameters` / `set_parameters` exchange the
  whole model (baselines) or the shared head only (ours). `fit` and `evaluate` as above.
- `server_app.py`: Flower `ServerApp` with `FedAvg` (later `FedProx`).
- `strategy.py`: subclasses FedAvg and, in `aggregate_fit`, computes
  `hash_parameters(arrays) -> str` (SHA-256 of the aggregated weights) for the blockchain audit
  log (Phase 4).

## Known limitations

- Each bank scales its own features, which matches the privacy story. In the two baselines,
  the same feature can therefore be scaled differently at different banks.
- If xgboost and torch run in the same process on macOS, set `torch.set_num_threads(1)`
  (see `scripts/run_baselines.py`).
