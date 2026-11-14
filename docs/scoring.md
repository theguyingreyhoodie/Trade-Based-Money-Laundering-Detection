# Risk score and explainable alerts

Code: `src/tbml/explain/`. Script: `scripts/generate_alerts.py`. Outputs: `results/alerts/`.
All numbers below are from the primary run (seed 0, 3 institutions, Dirichlet 0.5, global model of
round 29) on the test period.

An alert is a prompt for human review. Nothing in this system decides that a trade is laundering.

## 1. The score

```
risk = 100 * C( w * P_model + (1 - w) * P_rules )        w = 0.7
```

| Part | What it is | Calibration |
|---|---|---|
| `P_model` | output of the global FL model on the trade's indicators | isotonic, on validation |
| `P_rules` | noisy-OR score of the fired red-flag rules (`docs/rules.md`) | isotonic, on validation |
| `C` | final calibration of the blend | isotonic, on validation |

Bands: Low below 25, Medium 25-50, High 50-75, Critical from 75. Alerts are raised from Medium.

Everything is fitted from **validation score histograms** summed over the institutions
(`scoring.py::fit_risk_scorer`). No predictions or labels are pooled; a deployment would exchange
three small histograms. Platt scaling is available with `scoring.calibration: platt`.

## 2. Why this weighting

- The model ranks far better than the rules (validation PR-AUC 0.751 against 0.275), so it
  carries most of the weight.
- The rules are transparent, encode recognised red flags and are partly independent evidence, so
  they keep a material weight. The blend ranks better than either part alone.
- `w = 0.7` was fixed in advance. The validation sweep below shows that the result is not
  sensitive to it between 0.6 and 0.9; the best value on validation is 0.8, which is 0.007 better.

| Model weight `w` | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | **0.7** | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Validation PR-AUC | 0.275 | 0.501 | 0.587 | 0.673 | 0.712 | 0.732 | 0.749 | **0.757** | 0.764 | 0.763 | 0.751 |
| Validation best F1 | 0.372 | 0.574 | 0.635 | 0.693 | 0.705 | 0.718 | 0.723 | **0.720** | 0.719 | 0.714 | 0.709 |

On the test period:

| Score | F1 | Precision | Recall | PR-AUC | False-positive rate |
|---|---|---|---|---|---|
| Risk score (blend) | 0.745 | 0.833 | 0.674 | 0.770 | 0.5% |
| Global model alone | 0.726 | 0.774 | 0.683 | 0.782 | 0.7% |
| Rules alone (on local features) | 0.349 | 0.244 | 0.610 | 0.288 | 7.1% |

The blend gives a better operating point (higher F1 and precision). Its PR-AUC is lower than
the model's because of the final calibration step, not because of the blend: before that step
the blend ranks the test trades exactly as well as the model alone (PR-AUC 0.783 for both),
but isotonic calibration maps its 637 distinct values onto 120, and the resulting ties cost
0.012 PR-AUC. Ties do not matter for the band an alert falls into.

## 3. Calibration

| | Brier score | Expected calibration error |
|---|---|---|
| Model output as trained | 0.0163 | 0.0106 |
| Model output, calibrated | 0.0144 | 0.0023 |
| Risk score (calibrated blend) | 0.0141 | 0.0025 |

A calibrated score can be read as a probability: of the trades scored in a band, about that share
was illicit.

| Band | Score | Trades | Illicit | Precision | Share of all illicit trades |
|---|---|---|---|---|---|
| Low | 0 - 25 | 12,052 | 132 | 1.1% | 29.3% |
| Medium | 25 - 50 | 71 | 22 | 31.0% | 4.9% |
| High | 50 - 75 | 118 | 70 | 59.3% | 15.5% |
| Critical | 75 - 100 | 237 | 227 | 95.8% | 50.3% |

426 alerts on 12,478 trades (3.4% of trades reviewed), 75% of them on truly illicit trades. 29% of
the illicit trades stay in the Low band and are missed.

## 4. Rule floor

Because the blend is recalibrated, a rule hit that the model disagrees with gets a low score: on
validation such trades are mostly legitimate, so that is the right probability. It also means a
blind, drifting or poisoned model could keep a trade with strong red flags out of review.

The scorer therefore applies an explicit policy: a trade whose calibrated rule probability alone
is at least `scoring.rule_floor_probability` (0.6) is scored at least 25 (Medium). The floor is a
review policy, not a probability, and alerts raised by it are marked `raised_by_rule_floor`.

In this run the floor raised 2 alerts, and both were legitimate trades. It is cheap insurance
here, not a source of detections. (An earlier draft of this design claimed that the 0.3 rule
weight alone guaranteed such a floor. A test showed that the final calibration removes it, which
is why it is now explicit.)

## 5. Explanations

**SHAP.** Attributions are computed on the model's log-odds, where they add up exactly: base
value plus contributions equals the trade's log-odds. The background sample is 200 of the
explaining institution's own training rows, so no data from other institutions is needed.
DeepSHAP is tried first on each institution's batch of alerts and rejected if its contributions
for any trade miss the model's output by more than 0.05. In this run it was rejected for all
three institutions. The model clips standardised inputs at +/- 8, and alerted trades often have
an indicator beyond the clip, where DeepSHAP's linear rule fails: on 60 illicit test trades of
one institution, 45% had at least one clipped indicator and the worst trade's contributions
missed its log-odds by 45. All explanations therefore come from the permutation explainer,
whose contributions add up exactly (largest error 1e-6).

The permutation explainer averages over random orderings of the indicators. SHAP's default
budget allows only 3 orderings for 81 indicators, which makes the top-5 list noisy, so each
explanation uses 20 orderings (`scoring.shap_permutations`), seeded per institution so that
explanations are reproducible (tested). Measured on 40 alerts of one institution against a
100-ordering reference:

| Orderings | Top-5 indicators shared with the reference | Error of the contribution values | Time per alert |
|---|---|---|---|
| 3 (SHAP default) | 93% | 11% | 27 ms |
| 10 | 96% | 6% | 74 ms |
| **20 (used)** | **97%** | **4%** | **171 ms** |
| 40 | 98% | 3% | 408 ms |

"Error" is the L1 distance to the reference relative to the reference's L1 size; the reference
itself is an estimate. Moving from 3 to 20 orderings changed the suspected typology of none of
the 426 alerts and kept 94% of their top-5 indicators.

**Plain English.** Every indicator has a phrase (`explain/phrases.py`, tested for all 81):
"unit price 3.4x the HS reference price", "2 invoices quote the same bill of lading", "younger
party registered only 4 months before the invoice". The alert lists the five largest
contributions with their direction.

**Rules fired** are listed by id and name.

**Suspected typology.** Model evidence is the positive SHAP mass of the indicators that target
each typology (per the feature dictionary); rule evidence is the severity of the fired rules of
each typology. Both are normalised and averaged. Among alerts on illicit trades the suspected
typology is right for 83% and within the top two for 96%.

| Typology | Alerts on it | Attributed correctly |
|---|---|---|
| High-risk routing | 21 | 100% |
| Phantom shipment | 38 | 95% |
| Over-/under-shipment | 37 | 92% |
| Shell company | 28 | 86% |
| Third-party payment | 45 | 84% |
| Misdescription | 42 | 83% |
| Over-invoicing | 36 | 83% |
| Multiple invoicing | 37 | 81% |
| Under-invoicing | 13 | 62% |
| Circular trading | 22 | 36% |

Circular trading is attributed poorly: its defining indicators come from the trade graph, which
each institution sees only in part, so the evidence that remains points elsewhere.

## 6. What an alert contains

`results/alerts/alerts.json`; a readable sample is in `results/alerts/report.md`.

- `alert_id`, `trade_id`, `institution`, `created_at`
- `risk_score`, `risk_band`, `raised_by_rule_floor`, `components` (both probabilities, weight)
- `top_features`: indicator, value, SHAP contribution, direction, plain-English text
- `fired_rules`: id, name, typology, severity
- `suspected_typology` and the top three typology scores
- `trade`: parties, goods, value, route (off-chain, for the institution's own investigator)
- `model`: run id, round and SHA-256 of the global model that produced the score
- `ledger`: SHA-256 of the alert, transaction id, block, registering institution

The hash covers everything except the `ledger` block and includes a random 128-bit salt.
Lowering a score, changing the band, removing a fired rule or editing an explanation after the
fact produces a hash that is not on the ledger (tested). Registration took 6.5 ms (p50) and
103,030 gas per alert. No ground-truth field is ever part of an alert.

## 7. Limitations

- Calibration holds for the validation period it was fitted on; it drifts when behaviour does.
- SHAP explains the model, not the world: a contribution says what moved the score, not what the
  parties did. Correlated indicators share credit in ways that are hard to read.
- The suspected typology is a heuristic over the feature dictionary, not a classifier.
- Bands are cut at fixed scores; an institution would set them from its review capacity.
