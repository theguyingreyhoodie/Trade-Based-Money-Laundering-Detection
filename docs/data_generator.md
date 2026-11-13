# Synthetic trade data generator

No public, labelled TBML dataset exists, so the project runs on a parameterised simulator
(`src/tbml/data/`). This document lists every modelling assumption. All numbers are defaults from
`configs/default.yaml` (section `data`) and can be changed there or from the CLI.

**Everything produced here is synthetic. Results on this data say how the methods compare under
these assumptions; they are not estimates of real-world detection performance.**

```
python scripts/generate_data.py --all-seeds      # data/seed_<s>/ for every seed in project.seeds
python scripts/generate_data.py --seed 7 --set data.n_trades=20000
```

## 1. Output

```
data/seed_<s>/
  raw/bank/      payments.csv  kyc_entities.csv  banks.csv        <- bank view
  raw/trade/     invoices.csv  shipments.csv  shipping_lines.csv  <- exporter / importer / carrier view
  raw/customs/   declarations.csv                                 <- customs view
  reference/     hs_reference.csv  countries.csv  ports.csv  fx_rates.csv   <- public tables
  ground_truth/  labels.csv  entity_truth.csv  joined_trades.csv  <- evaluation and centralised baselines only
  meta.json
```

- The three `raw/` folders are the records as their owners hold them. None of them contains a label
  or any hidden attribute (enforced by a test).
- `ground_truth/joined_trades.csv` is one row per invoice with all linked documents plus the label.
  It is used only for evaluation and for the centralised (pooled) baseline.
- The same config and seed always reproduce the same files (enforced by a test).

One *trade* is one invoice. Its linked records:

| Document | Owner | Key fields |
|---|---|---|
| Invoice | exporter / importer | HS code, goods description, quantity, unit price, currency, value, B/L number quoted, payment terms |
| Shipment (bill of lading) | shipping line | route (origin, transshipment ports, destination), dates, shipped quantity, gross weight, containers |
| Payment(s) | banks | payer, payee, amount, currency, both banks, date; several rows for instalments |
| Customs declaration | customs | declared HS code, value (USD), quantity, weight, origin |

Linkage keys mirror practice: a payment quotes the invoice number, the invoice quotes a B/L number,
the customs declaration quotes both.

## 2. The world

**Jurisdictions.** 35 fictional jurisdictions in six regions, each with a trade weight, a home
currency and a public risk tier. The tiers copy the *structure* of public lists without naming real
countries: `low`, `medium`, `offshore` (free-trade zones and offshore centres, 6) and `high`
(a FATF-style list, 5). Five jurisdictions are transshipment hubs; three of the hubs are also free
zones, so passing through a free zone is common and not suspicious by itself. Risk scores used by
features and rules: low 0.10, medium 0.35, offshore 0.70, high 1.00.

**Ports and distances.** One main seaport per jurisdiction. Distances are great-circle distances
between fictional coordinates; sailing time is distance at 14 knots plus two days per stop.

**Goods.** 49 HS-6 codes in ten sectors, each with a unit, a reference price (USD per unit), a
price dispersion (log-sigma between 0.05 for gold and 0.90 for watches), a typical weight per unit
and a typical order size. Prices are illustrative, not real statistics.

**Reference prices as a detector sees them.** `reference/hs_reference.csv` is the "published"
table: the true median with 3% estimation noise and the dispersion with about 9% noise.

**Entities** (default 3,000 ordinary companies, 25 finance houses, 45 banks, 10 shipping lines):

- Role: exporter 33%, importer 37%, trader (both) 30%. Exporters sell 2-4 HS codes from 1-2
  sectors, weighted by region; importers buy from 1-3 sectors.
- Size is lognormal and drives how much a firm trades.
- Age: gamma-distributed (mean about 10 years); 5% were founded shortly before the period and 6%
  are founded during it. A firm only trades after it exists.
- 15% of firms share a business-centre address (40% of young firms), and 12% share a director with
  other firms (business groups, nominee providers). These are the benign look-alikes of shells.
- Every firm has a home bank in its own jurisdiction.

**Laundering networks** (default 45, ground truth only; on seed 0: 189 shells, 68 payer vehicles,
143 complicit firms). A network has:

- 3-6 **shells**. A shell is short-lived: it is registered 1-8 months before its first use, used
  for 4-12 months and then dropped, so a network works through several of them (median shell age
  at the time of a trade: about 9 months). 20% are aged "shelf" companies. Most sit at a
  formation-agent address with a nominee director; 35% are *hidden* (own address and director).
  Shells sit mostly in offshore and medium-risk jurisdictions.
- 2-4 **complicit firms** recruited from ordinary, established companies that have real trading
  relationships. Their illicit trades are mixed into a genuine trading history.
- 1-2 **payer vehicles** in offshore or listed jurisdictions.
- 2-3 typologies it specialises in, 2-4 preferred goods (biased towards goods whose prices are
  hard to verify) and an active window of 6-24 months.

Identifiers are randomised after construction, so no id prefix or sequence number reveals a shell,
a network or an illicit trade (tested).

## 3. Legitimate trade

- **Standing relationships.** Each importer has 1 + Poisson(1.5) suppliers that share a sector and
  are in another jurisdiction (same region preferred 3:1). About 87% of legitimate trades come from
  these relationships.
- **Spot deals** (12%): one-off trades between firms with no relationship.
- **Returned goods** (0.8%): part of an earlier order shipped back, which creates benign
  reciprocal edges in the trade graph.
- **Trading rings** (0.6%): 12 rings of 3-4 established commodity traders that pass the same goods
  around. They are the benign look-alike of circular trading.
- **Price.** `log price = log(reference) + market drift + pair effect + trade noise`. Market drift
  is a monthly random walk per HS code (sd 2% per month). The pair effect (0.5 x HS sigma) is
  persistent: some suppliers are always dear or cheap. Trade noise is Student-t (5 d.f.,
  0.75 x HS sigma), so legitimate prices have fat tails. 0.4% of trades are clearance or premium
  deals at about half or 1.8 times the usual price.
- **Quantity** is lognormal around the HS typical order, scaled by importer size. 5% of invoices
  are negotiated lump sums (round totals).
- **Route.** Direct in most cases; one transshipment with probability 8% (same region) or 28%
  (cross-region), two with 3%. A hub is only used when it lies roughly on the way (at most 1.8
  times the direct distance; smaller detours are preferred), otherwise the trade goes direct. 3%
  of trades add a feeder call at a nearby ordinary port, which can be in a listed jurisdiction.
  The 99th percentile of the legitimate detour ratio is about 1.8.
- **Payment.** Terms: advance 12%, at sight 38%, open account 50% (30/60/90 days). 18% are paid in
  2-3 instalments. 5% carry bank charges, 2% a credit note. Benign third-party payers: a related
  group company (1.5%), a trade-finance house (1.0%), an unrelated legitimate firm (1.2%). 2% of
  exporters collect through a bank in a hub jurisdiction; 1.5% of payments are in another currency.
- **Customs.** Declared value is the invoice value plus a 0-6% CIF uplift.
- **Split invoicing** (1.5%): two invoices of the same parties share one shipment and together add
  up to it. This is the benign look-alike of multiple invoicing.

### Benign noise that looks suspicious

Measured on seed 0 (58,005 legitimate trades):

| Look-alike | Share of legitimate trades | Mimics |
|---|---|---|
| shipment record cannot be linked (lost record or mistyped B/L number) | 1.08% | phantom shipment |
| customs declaration missing | 0.88% | phantom shipment |
| both missing | 0.11% | phantom shipment |
| third-party payer | 3.70% | third-party payment |
| customs HS code differs from invoice (75% same heading, 20% other heading, 5% other chapter) | 1.94% | misdescription |
| transshipped at least once | 22.6% | high-risk routing |
| two or more invoices on one B/L | 2.74% | multiple invoicing |
| round invoice total | 5.0% | shell company |
| paid before shipment | 9.9% | phantom shipment, shell company |
| shipped quantity differs from invoiced by more than 8% (tolerances, short shipment, split consignment, keying errors) | 3.8% | over-/under-shipment |

Raw tables also contain 0.2% exact duplicate rows and 1% messy codes (stray spaces, lower case)
that the cleaning step has to remove.

## 4. Injected typologies

About 4% of trades are illicit (2,403 of 60,408 on seed 0). Each illicit trade has one primary
typology as its label sub-type. A quarter of the cases are *subtle*: the manipulation is small
enough to overlap with benign behaviour.

**Who trades.** A date inside the network's active window is drawn first; the parties must exist
on that date, and a shell qualifies only while it is in use. The parties are then drawn in one of
three ways: a complicit firm with one of its existing ordinary partners (45%), two complicit firms
of one network (35%), or a shell on one side (20%; always for the shell-company typology). On
seed 0, 34% of illicit trades involve a shell and 35% are between parties that also trade
legitimately with each other. This
matters: if illicit trades only ran between parties that never trade legitimately, a model could
detect "who" instead of "what", and the task would be trivial.

| Typology | Share | Manipulation | Subtle variant |
|---|---|---|---|
| Over-invoicing | 14% | unit price x lognormal(median 2.4), at least 1.4; in 40% customs still sees the true value | x 1.2-1.5 |
| Under-invoicing | 12% | unit price x lognormal(median 0.42), at most 0.72 | x 0.65-0.85 |
| Multiple invoicing | 9% | 2-3 invoices and payments for one shipment (same B/L); 60% of duplicates are financed through another bank | none: only visible to someone who sees all the invoices |
| Phantom shipment | 10% | invoice and payment but no goods: 60% no shipment and no customs record, 25% a near-empty container (2-15% of the expected weight) and no customs record | 15% have a complete set of forged documents |
| Over-/under-shipment | 10% | shipped quantity x 0.2-0.7 (60%) or x 1.4-3.0 | x 0.75-0.90 or x 1.10-1.30 |
| Misdescription | 10% | customs HS code is a different product (70% another sector); weight follows the true goods | sibling HS code with modest weight and price anomalies |
| Third-party payment | 10% | paid by a network vehicle (25%: by a shell), 30% in another currency, 40% split | not applicable |
| Circular trading | 9% | loops of 3-5 network members trade the same goods in sequence, 2-10 days per hop, price +2-8% per hop | not applicable |
| Shell company | 9% | a shell on one side: inflated volume (x 1.5-4), 40% round totals, 50% advance payment, 30% offshore collecting bank, price +/- 30% | hidden and shelf shells |
| High-risk routing | 7% | 1-3 needless stops in free zones (65%) or listed jurisdictions (35%), 60% offshore collecting bank, 50% mispriced by 15-25% | one stop that happens to lie on the way |

## 5. Design choices and known simplifications

- **One label per trade.** Real schemes combine typologies; here each illicit trade has one primary
  sub-type so that per-typology recall is well defined.
- **Clean labels.** Ground truth is exact. Real labels come from investigations and are noisy,
  delayed and incomplete.
- **Settled snapshot.** Every document of the period exists when the data is read. Trades still in
  transit (invoice issued, goods not yet shipped) are not modelled.
- **Static FX rates** and no inflation beyond the per-HS price drift.
- **Seaborne goods trade only.** No services, no air freight, no letters of credit.
- **Stationary behaviour.** Networks do not adapt to detection. Real launderers do.
- **Illicit activity is roughly uniform over time.** Networks are placed so that the density of
  illicit trades is flat; the positive rate still varies between the train, validation and test
  periods by chance (4.0%, 4.2% and 3.6% on seed 0).
- **Difficulty is a modelling choice.** Manipulation strength, the subtle share and the benign
  noise rates determine how separable the classes are. They were set so that a strong centralised
  model is clearly better than rules and linear models but far from perfect (centralised LightGBM:
  test PR-AUC about 0.87, F1 about 0.81; rules engine: F1 about 0.43). Other settings would give
  other absolute numbers.
