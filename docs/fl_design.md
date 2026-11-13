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
- FedProx (mu = 0.01) is not distinguishable from FedAvg in any scenario.

## 8. Threat model and limitations

- **Honest-but-curious server, honest clients** is the assumption of the core protocol.
- **Model poisoning and backdoors are not prevented.** A malicious client can send an arbitrary
  update (boosted, sign-flipped, or trained to ignore one counterparty). FedAvg has no defence:
  with 3 clients one attacker controls a third of the average. A simple defence, clipping the
  L2 norm of every client update (`fl.norm_clip`), is implemented and demonstrated in the
  stretch experiments; it limits boosted updates but not a patient, norm-bounded backdoor.
  Robust aggregation (median, Krum) needs more clients than a three-party consortium has.
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
