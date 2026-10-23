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
| **FL hybrid** | **0.139 ± 0.027** | 0.116 / 0.133 / 0.169 |
| FL zero_filled | 0.119 ± 0.005 | 0.124 / 0.113 / 0.120 |
| Local-only MLP (lower bound) | 0.105 ± 0.006 | |
| FL encoder_head | 0.105 ± 0.004 | 0.109 / 0.102 / 0.103 |
| FL naive | 0.053 ± 0.006 | 0.048 / 0.052 / 0.059 |

- Hybrid versus zero-filled FedAvg (the key comparison): hybrid has the higher mean test
  PR-AUC, the higher validation PR-AUC on every seed (0.19-0.21 vs 0.15-0.18) and the higher
  mean test PR-AUC at 4 of 5 banks, but zero-filled wins seed 42 on test. With 3 seeds the
  test gap (+0.020) is not statistically significant. Hybrid matches or beats zero-filled
  while keeping each bank's column list private.
- Hybrid beats encoder_head on every seed, and beats the local-only MLP for all 5 banks
  (mean per-bank test PR-AUC). Its spread over seeds is the largest.
- Sharing only the head (`encoder_head`) gives no gain over training alone; sharing the
  core encoder is what helps.
- Naive FedAvg flags 3-4x more legitimate transactions (FPR 0.0047 vs 0.0013), so its
  per-pattern recall is not comparable with the other setups.
- Hybrid is still below local-only XGBoost and well below centralized training.
- Runs are not bit-for-bit reproducible: the order in which the 5 banks' replies arrive changes
  the floating-point sums. Rerunning hybrid seed 42 (to save its models for the dashboard) gave
  0.116 instead of 0.119.

### Ablation: profiles without label signal (`configs/ablation_signal0.yaml`)

The synthetic customer profiles of the main runs carry a weak label signal (`label_signal` 0.1:
some attributes of accounts that laundered in the training period are drawn riskier). Rerun with
pure-noise profiles (`label_signal` 0), seeds 42/43/44:

| Setup | Main runs (signal 0.1) | Signal 0 | Per seed (signal 0) |
|---|---|---|---|
| Centralized XGBoost | 0.340 ± 0.007 | 0.355 ± 0.004 | |
| Centralized MLP | 0.241 ± 0.008 | 0.257 ± 0.010 | |
| **FL hybrid** | **0.139 ± 0.027** | **0.171 ± 0.033** | 0.133 / 0.190 / 0.191 |
| Local-only XGBoost | 0.159 ± 0.006 | 0.154 ± 0.023 | |
| FL zero_filled | 0.119 ± 0.005 | 0.133 ± 0.004 | 0.129 / 0.133 / 0.137 |
| Local-only MLP | 0.105 ± 0.006 | 0.111 ± 0.006 | |

- The results do not depend on the synthetic signal: without it every model does as well or
  better, and the order of the setups is the same.
- Likely reason: the signal marks only accounts that laundered in the training period, so models
  learn a cue that does not carry over to new launderers in the test period.
- Without the signal, hybrid beats zero-filled FedAvg on every seed and passes local-only XGBoost.

## Training

- FedAvg (`strategy.AuditedFedAvg`), all 5 banks every round, weighted by training rows.
  100 rounds by default (`fl.num_rounds`).
- Local training: each round, undersample the bank's training data again
  (`seed + round`), then `fl.local_epochs` epochs with `pos_weight` in the loss. `lr` and
  `batch_size` come from the `mlp:` section so the results are comparable with the baselines.
- `AuditedFedAvg` stores the SHA-256 of every aggregated model and of every bank's update in
  `audit.json`, for the blockchain audit log (Phase 4). Each bank declares its update's hash in
  its reply; the server refuses the round if the update it received hashes differently.

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
