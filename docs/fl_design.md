# Federated learning design

Code: `src/tbml/fl/`. Experiments: `scripts/run_fl.py` (primary run) and
`scripts/run_fl_experiments.py` (sweep). Framework: Flower 1.43, message API
(`ServerApp` / `ClientApp`), simulation runtime with the Ray backend.

## 1. Horizontal, not vertical

The architecture slide shows Bank A (payments), Bank B (trade records) and Customs. Read
literally that is a **vertical** split: each party holds different *fields* of the same trades.
FedAvg is a **horizontal** method: every party has the same feature space and different *trades*.

The primary system is therefore horizontal:

- A **client** is an institution or consortium that holds the complete linked records (invoice,
  shipment, payments, customs declaration) of its own customers' trades.
- All clients compute the same 81 indicators (`docs/features.md`), each from its own records.
- The server averages model parameters with FedAvg.

A small vertical-FL (split neural network) demo that matches the Bank / Bank / Customs picture
is described in section 9.

## 2. What a client sees

`fl/data.py::prepare_clients` deals every client its own trades; it is the only code that touches
all the data, in the role of "the world". After that a client process reads one file, its own.

A client's local view:

- its own trades with their linked documents and their labels (the outcome of its own
  investigations);
- registry attributes of the entities that appear in those trades;
- the public reference tables (HS reference prices, jurisdiction risk tiers, ports, FX).

**Context indicators are local.** Entity history, the trade graph, PageRank, cycles and
bill-of-lading sharing are computed on the client's own trades only. A pattern that spans
institutions is invisible to each of them: a duplicate invoice financed at another bank, a
trading loop whose legs sit with different banks, an entity's turnover at other banks. This is
the main reason federated results stay below the centralised model (section 7).

## 3. Protocol

| Step | Server sends | Client returns | Purpose |
|---|---|---|---|
| Round 0 (`query`) | request | per-feature count, sum and sum of squares of its training rows | one shared feature scaler without sharing rows |
| Train (`train`) | global model, learning rate, local epochs, FedProx `mu` | updated model, number of training rows, loss, SHA-256 of the update | FedAvg |
| Evaluate (`evaluate`) | global model | score histograms (per typology class, 800 bins) on its validation and test rows, loss | threshold selection and all metrics |

- **Standardisation** is part of the model (non-trainable buffers), so one set of arrays is a
  complete model. The scaler built from the clients' sums equals the scaler of the pooled
  training rows (tested).
- **Aggregation** is Flower's weighted average (`aggregate_arrayrecords`), weights = number of
  local training rows. Replies are sorted by client id first so sums are reproducible.
- **Evaluation is federated.** The server adds up the histograms, picks the decision threshold
  that maximises F1 on the summed *validation* histogram and reads every metric off the summed
  *test* histogram. Nobody pools predictions or labels. Centralised and local-only models are
  scored with the same histogram code, so all numbers are comparable.
- **Model selection.** The global model of the round with the best validation PR-AUC is kept,
  the same rule as early stopping for the centralised MLP.
- **Variant: client thresholds.** Each client can pick its own threshold for the global model on
  its own validation rows. This needs no communication and is reported as a second number.

### What leaves a client

| Message | Content | Size |
|---|---|---|
| statistics reply | 2 x 81 floats + count | 1.4 kB |
| training reply | model arrays (7,361 trainable parameters + scaler), 4 scalars, one hash | 31.7 kB |
| evaluation reply | 2 x 8,800 integer counts, 2 scalars | about 17.7 kB |

Nothing else. `fl/boundary.py` enforces this on the server for every reply before it is used:
exactly the model's array names, shapes and dtypes; only whitelisted metric keys with bounded
length; only the hash in the config record; a byte budget fixed by the model and the histogram
resolution. A reply that fails raises `BoundaryViolation`. The test suite checks twelve ways of
smuggling rows (extra record, larger array, longer list, free text, ...) and checks the server's
own log of a real simulated run (`tests/test_fl.py::test_server_only_ever_receives_parameters`).

The aggregates are not free of information. Feature moments reveal a client's feature means;
score histograms reveal how many trades of each typology a client holds; a model update can
leak about training rows (gradient inversion, membership inference). Secure aggregation and
DP-SGD (section 8) address the last point.

## 4. Model and training settings

MLP: 81 inputs -> 64 -> 32 -> 1, ReLU, dropout 0.1, weighted BCE (positive weight 6), SGD with
momentum 0.9, batch 256. The same network and loss are used centrally, locally and federated.

| Setting | Value | How chosen |
|---|---|---|
| rounds | 30 | fixed |
| local epochs | 1 | validation |
| client learning rate | 0.05, multiplied by 0.95 each round | validation |
| participation | all clients every round | fixed |
| FedProx `mu` | 0.01 | fixed |

The learning rate, its decay and the number of local epochs were compared on *validation* PR-AUC
of seed 0 (IID and Dirichlet 0.5, six combinations). All were within 0.01 of each other; the
decaying rate gave the most stable curve. Test data played no part.

## 5. Data splits between clients

`fl/partition.py`:

- **IID**: random equal shares.
- **Dirichlet(alpha)**, alpha in {0.1, 0.5, 1.0}: for every class (legitimate plus each of the ten
  typologies) the shares across clients are drawn from a Dirichlet distribution (Hsu et al.,
  2019). This skews the positive rate (label skew), the typology mix and the client sizes.
- **Quantity skew**: client sizes drawn from Dirichlet(0.5), same class mix everywhere.

**Sanity bound.** With a 96% majority class an unconstrained draw can produce a client with a few
hundred trades of which 90% are illicit (this happened in the first run). That is not a plausible
institution and it distorts the comparison, because such a client is "right" by flagging
everything. Draws in which a client would get less than a quarter of an equal share of the
legitimate trades are rejected. The typology classes are drawn without constraint. Example,
alpha = 0.1, 3 clients, seed 0: sizes 41.6k / 12.2k / 6.5k, positive rates 1.9% / 9.6% / 6.5%,
and two clients see only 5 and 6 of the 10 typologies.

The primary scenario (used for the ledger, alerts and dashboard) is 3 clients, Dirichlet 0.5.

## 6. What is compared

| Method | Data pooled? | Deployable without sharing rows? |
|---|---|---|
| Centralised MLP | yes, raw records | no (upper bound) |
| Pooled model on local features | yes, but only locally computed features | no (diagnostic) |
| FedAvg / FedProx | no | yes |
| Local-only | no | yes (status quo) |

The diagnostic separates two effects. Centralised -> pooled-local is the cost of **information
fragmentation** (context indicators computed per client). Pooled-local -> FedAvg is the cost of
**federated optimisation**. All four are evaluated on the same test trades.

## 7. Findings (3 seeds, see `results/RESULTS.md` for the full tables)

- FedAvg matches the pooled-local model within about 0.01 F1 for 3 clients (0.702 vs 0.698 IID).
  Parameter averaging loses almost nothing; at 10 clients the gap is 0.03.
- FedAvg beats local-only when data is spread thin, and the gain grows with the number of
  clients: +0.05 F1 at 3 clients, +0.06 at 5, +0.11 at 10 (IID).
- Everything that keeps raw data local is below the centralised MLP (0.753), because of
  fragmentation. The typologies that depend on seeing related trades together lose most:
  multiple invoicing (recall 1.00 -> 0.60) and circular trading (0.81 -> 0.57) on seed 0.
- Under strong skew (alpha = 0.1) local-only (0.801) beats FedAvg (0.737) and even the
  centralised MLP. When a typology concentrates at one institution, its local graph is intact
  and its own threshold fits its own positive rate. Client thresholds on the global model
  recover part of the gap (0.752).
- FedProx (mu = 0.01) does not improve on FedAvg at any alpha (differences between -0.011 and +0.004 F1).

## 8. Threat model and limitations

- **Honest-but-curious server, honest clients** is the assumption of the core protocol.
- **Model poisoning and backdoors are not prevented.** A malicious client can send an arbitrary
  update (boosted, sign-flipped, or trained to ignore one counterparty). FedAvg has no defence:
  with 3 clients one attacker controls a large share of the average. A simple defence, clipping
  the L2 norm of every client update (`fl.norm_clip`), is implemented and measured in section 10;
  it limits boosted updates but not a patient, norm-bounded backdoor. Robust aggregation
  (median, Krum) needs more clients than a three-party consortium has.
- **Aggregates leak something** (section 3).
- **No dropout handling.** Every client takes part in every round; a missing client aborts the
  run. Partial participation is supported by Flower (`fl.fraction_train`) but not evaluated.
- **Simulation only.** Clients are Ray actors on one machine; network latency, TLS and
  authentication are out of scope. Flower marks `run_simulation()` as deprecated in favour of
  the `flwr run` CLI; the version is pinned (`flwr<1.44`).
- **Labels are local and clean** in the simulation; real institutions have few confirmed cases.

## 9. Design note: vertical FL for the Bank / Bank / Customs picture

In the vertical reading each party holds different columns of the *same* trade:

| Party | Holds | Indicators it can compute alone |
|---|---|---|
| Bank (payments) | payments, KYC | payer, timing, currency, counterparty age and registry links |
| Trade registry (invoices, shipments) | invoices, bills of lading | price against reference, route, quantities invoiced and shipped |
| Customs | declarations | declared HS code, value, quantity and weight |

Design (split neural network, implemented as a demo in `fl/vertical.py`):

1. **Alignment.** Parties agree on the set of common trades by private set intersection on the
   invoice number (assumed done; the demo uses the shared key directly).
2. **Bottom models.** Each party runs its own small network on its own columns and sends an
   8-dimensional embedding per trade, not the columns.
3. **Top model.** One coordinator that holds the labels (for example the FIU) concatenates the
   embeddings, computes the loss and returns the gradient of each embedding to its party.
4. Each party back-propagates through its bottom model locally.

What it costs: one embedding up and one gradient down per trade per party per epoch, so
communication grows with the number of trades, unlike FedAvg where it grows with the model size.
What it cannot do: indicators that compare two parties' fields directly (invoice HS code against
customs HS code, amount paid against amount invoiced) can no longer be computed by anyone; the
top model has to learn them from embeddings. Embeddings and their gradients can leak labels and
features, so a deployment would add secure computation on the top model.

Measured (3 seeds, `results/stretch/vertical.json`). Columns per party: bank 23, trade 41,
customs 11.

| Model | F1 | PR-AUC |
|---|---|---|
| Customs alone | 0.266 ± 0.031 | 0.212 |
| Bank alone | 0.346 ± 0.030 | 0.312 |
| Trade registry alone | 0.429 ± 0.006 | 0.440 |
| Split neural network, all three parties | 0.568 ± 0.029 | 0.604 |
| One model on all three parties' columns (bound for the split network) | 0.593 ± 0.014 | 0.642 |
| Centralised MLP with all 81 indicators, including cross-party ones | 0.753 ± 0.008 | 0.820 |

Working together clearly beats any party alone and comes close to pooling the same columns. The
remaining gap to the 81-indicator model is the price of not being able to compare two parties'
fields directly. Traffic: 192 bytes per trade per epoch, 6.8 MB per epoch for 35,383 training
trades (FedAvg: 0.34 MB per round for the whole federation).

## 10. Extensions (stretch goals)

Code: `fl/extensions.py`, `fl/dp.py`, `fl/local.py`, `fl/vertical.py`. Experiments:
`scripts/run_stretch.py`, results in `results/stretch/`. All on the primary scenario
(3 clients, Dirichlet 0.5) unless stated.

### Secure aggregation

Pairwise additive masking after Bonawitz et al. (2017), without dropout recovery. A client
encodes its weighted update `n_i * w_i` (and `n_i` itself) as fixed-point 64-bit integers and adds,
for every other participant, a mask derived from a key the two share: the client with the smaller
id adds it, the other subtracts it. Arithmetic is modulo 2^64, so the masks cancel exactly in the
sum and each masked update on its own is uniform noise. The server adds the masked arrays,
decodes the sum and divides by the summed `n_i`.

- After one round the result equals plain FedAvg to within 1.2e-7 per weight (one float32
  rounding step), with an identical confusion matrix. Over 30 rounds the rounding differences are
  amplified by training: F1 0.730 against 0.726.
- The upload doubles (1.95x) because 64-bit integers replace 32-bit floats.
- The boundary guard switches to "masked" mode: only `uint64` arrays of the model's shapes are
  accepted, an unmasked update is refused.
- Not covered: a client that drops out mid-round (its masks no longer cancel; the full protocol
  recovers them with secret sharing). Pairwise keys are derived from one secret shared by the
  clients, standing in for a Diffie-Hellman exchange. Server-side norm clipping is impossible by
  construction. Feature moments and score histograms are still sent in the clear.

### DP-SGD

Clients train with per-example gradient clipping and Gaussian noise (Opacus), which bounds what
an update reveals about any single training row. Each client computes the noise multiplier that
spends its budget over all its local steps of the run (Renyi DP accounting, Poisson sampling,
delta = 1e-5); a small client needs more noise than a large one.

| Budget per client | Mean noise multiplier | F1 | PR-AUC |
|---|---|---|---|
| no DP | - | 0.722 ± 0.005 | 0.772 |
| epsilon = 8 | 0.94 | 0.577 ± 0.008 | 0.605 |
| epsilon = 3 | 1.68 | 0.548 ± 0.012 | 0.570 |
| epsilon = 1 | 4.10 | 0.415 ± 0.046 | 0.395 |

Privacy is expensive here: even a loose budget costs about 0.15 F1, because the positive class
is rare and its gradients are exactly the ones clipping and noise damage most. The clipping norm
(2.0) was chosen on validation at epsilon = 3 on seed 0 from 12 combinations of clipping norm,
batch size and learning rate. The first, untuned setting (norm 1.0) reached validation PR-AUC
0.41 against 0.54, so an untuned DP baseline would have overstated the cost. DP covers training
only; the evaluation histograms and feature moments are not noised. DP and secure aggregation
are demonstrated separately, not combined.

### Poisoning and norm clipping

One of three clients (the smallest) sends its update sign-flipped and scaled by 5 in every round.

| Condition | F1, model of the last round | F1, round selected on validation |
|---|---|---|
| No attack | 0.720 ± 0.005 | 0.722 ± 0.005 |
| Attack, no defence | 0.070 ± 0.004 | 0.372 ± 0.120 (round 1) |
| Attack, median norm clipping | 0.592 ± 0.028 | 0.584 ± 0.038 |
| No attack, median norm clipping | 0.697 ± 0.014 | 0.700 ± 0.020 |

- Without a defence training diverges. Selecting the round on validation only rolls back to the
  first round; it repairs nothing.
- Clipping every update to the round's median norm keeps training alive (0.59) but does not
  restore the clean result (0.72), and costs 0.02 F1 when nobody attacks.
- The size of an honest update grows with the number of local steps, so the bound mostly shrinks
  the *largest honest* client. A small attacker whose boosted update stays under the median is
  not clipped at all, and a patient attacker within the bound (a backdoor) is not addressed.

### Local fine-tuning of the global model

After federated training each client trains the global model for up to 10 more epochs on its own
rows (learning rate 0.01), keeps whichever version validates best on its own validation rows
(possibly the unchanged global model) and picks its own threshold. Nothing is sent.

| Split (3 clients) | Local-only | FedAvg | FedAvg + fine-tuning |
|---|---|---|---|
| IID | 0.653 | 0.702 | 0.696 |
| Dirichlet 1.0 | 0.708 | 0.726 | 0.735 |
| Dirichlet 0.5 | 0.722 | 0.722 | 0.750 |
| Dirichlet 0.1 | 0.801 | 0.737 | 0.804 |

Fine-tuning is within 0.01 F1 of the better of the two alternatives in every scenario and
clearly ahead under moderate skew. It answers the objection that institutions with skewed data
gain nothing from joining: they keep their local advantage and still get what the others learned.
