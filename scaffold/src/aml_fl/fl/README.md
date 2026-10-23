# Federated learning (Phase 3)

Five client banks train one laundering detector with Flower, without sharing raw data.
Each simulated node is one bank and reads only `data/partitions/client_<partition-id>/`.
`schema.json` lists the client's columns and the `core_columns` that every client has.

Run with `python scripts/run_fl.py --mode <mode>` (see the main README).

## Setups

| Mode | Shared (FedAvg'd) | Private (stays at the bank) | Features each bank uses |
|---|---|---|---|
| `naive` (baseline) | whole `BankModel` | nothing | the 9 `core_columns` only |
| `zero_filled` (baseline) | whole `BankModel` | nothing | the agreed union of all banks' columns; missing ones are 0 plus a 0/1 "missing" flag |
| `encoder_head` | `SharedHead` | `Encoder` | the bank's full own schema |
| `hybrid` | core encoder + `SharedHead` | encoder of the bank's own columns | `[core | own]` |

`hybrid` (`models.HybridModel`) splits the input in two. The core columns use the public
category lists in `features.SHARED_VOCAB`, so their layout is identical at every bank and a
core encoder can be averaged. The bank's own history and profile columns go through a private
encoder. Both 16-dim outputs are concatenated and fed to the shared head. The bank's own column
list and that encoder never leave the bank.

One rule decides what is sent in every mode: a parameter whose name starts with
`client_app.PRIVATE_PREFIX[mode]` stays at the bank, everything else is sent
(`client_app.shared_state`). The private part is kept in the node's `context.state` between
rounds. Each bank reports `encoder_rounds`, which must equal the round number in
`history.csv`; a lower value means the private encoder was lost between rounds.

`zero_filled` (`preprocess.load_client_zero_filled`) needs every bank to agree on one column
list (`run_fl.agreed_columns`: the union of the banks' schemas) and on the category lists of
every categorical column (`client_app.ALL_VOCAB`). Agreeing on that list reveals which columns
each bank stores; the hybrid mode avoids this.

The key comparison is ours versus zero-filled FedAvg. If zero-filled scores higher, report the
gap as the accuracy cost of keeping the schema private. Every setup should land between the
local-only and centralized MLP results in `experiments/results/baselines.csv`.

## Results (HI-Small, 100 rounds, seeds 42/43/44)

Pooled test PR-AUC, mean ± std over 3 seeds (`scripts/summarize_seeds.py`):

| Setup | Pooled test PR-AUC | Per seed |
|---|---|---|
| Centralized XGBoost (upper bound) | 0.340 ± 0.007 | |
| Centralized MLP (upper bound) | 0.241 ± 0.008 | |
| Local-only XGBoost | 0.159 ± 0.006 | |
| **FL hybrid** | **0.140 ± 0.026** | 0.119 / 0.133 / 0.169 |
| FL zero_filled | 0.119 ± 0.005 | 0.124 / 0.113 / 0.120 |
| Local-only MLP (lower bound) | 0.105 ± 0.006 | |
| FL encoder_head | 0.105 ± 0.004 | 0.109 / 0.102 / 0.103 |
| FL naive | 0.053 ± 0.006 | 0.048 / 0.052 / 0.059 |

- Hybrid versus zero-filled FedAvg (the key comparison): hybrid has the higher mean test
  PR-AUC, the higher validation PR-AUC on every seed (0.19-0.21 vs 0.15-0.18) and the higher
  mean test PR-AUC at 4 of 5 banks, but zero-filled wins seed 42 on test. With 3 seeds the
  test gap (+0.021) is not statistically significant. Hybrid matches or beats zero-filled
  while keeping each bank's column list private.
- Hybrid beats encoder_head on every seed, and beats the local-only MLP for all 5 banks
  (mean per-bank test PR-AUC). Its spread over seeds is the largest.
- Sharing only the head (`encoder_head`) gives no gain over training alone; sharing the
  core encoder is what helps.
- Naive FedAvg flags 3-4x more legitimate transactions (FPR 0.0047 vs 0.0013), so its
  per-pattern recall is not comparable with the other setups.
- Hybrid is still below local-only XGBoost and well below centralized training.

## Training

- FedAvg (`strategy.AuditedFedAvg`), all 5 banks every round, weighted by training rows.
  100 rounds by default (`fl.num_rounds`).
- Local training: each round, undersample the bank's training data again
  (`seed + round`), then `fl.local_epochs` epochs with `pos_weight` in the loss. `lr` and
  `batch_size` come from the `mlp:` section so the results are comparable with the baselines.
- `AuditedFedAvg` stores the SHA-256 of every aggregated model in `audit.json`, for the
  blockchain audit log (Phase 4).

## Evaluation

- Each bank tunes its own threshold on its own validation split (`best_f1_threshold`). After
  the last round it writes its test scores to `<out_dir>/scores/` (for the experimenter only;
  nothing is sent to the server).
- `run_fl.py` reports per-client results and one pooled result (`metrics.pooled_evaluate`:
  every row flagged by its own bank's threshold), plus recall per pattern.

## Files

- `client_app.py`: Flower `ClientApp` (train / evaluate) for all modes.
- `server_app.py`: Flower `ServerApp`; writes `history.csv` and `audit.json`.
- `strategy.py`: `AuditedFedAvg` and `hash_parameters`.

## Known limitations

- Each bank scales its own features, which matches the privacy story. In shared parts of the
  model, the same feature can therefore be scaled differently at different banks.
- If xgboost and torch run in the same process on macOS, set `torch.set_num_threads(1)`
  (see `scripts/run_baselines.py`).
