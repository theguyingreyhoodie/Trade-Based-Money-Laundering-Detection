# Rules engine: red-flag rule book

Generated from `src/tbml/rules/engine.py` and `configs/default.yaml` (run `python scripts/run_baselines.py --docs-only`).

20 transparent rules over the risk indicators in `docs/features.md`. A rule either fires for a trade or
it does not; the engine reports the fired rules per trade, so every alert can be read without a model.

- **Thresholds** live in `rules.thresholds` in the config. They are expert settings informed by the unlabelled
  distribution of each indicator (for example, a price z-score beyond 3 occurs in well under 1% of trades).
  They are not fitted to labels.
- **Severity** is an expert judgement of how strongly one hit suggests laundering (0 to 1). It can be overridden
  per rule with `rules.severity`.
- **Rule score** = `1 - prod(1 - severity)` over the fired rules (noisy-OR). A single strong rule gives a high
  score and several weak rules add up, but the score never reaches 1.
- As a stand-alone baseline, a trade is flagged when its rule score reaches a threshold chosen on the
  validation period. Measured quality per rule is in `results/RESULTS.md`.

## Rules

| Rule | Red flag | Typology | Severity | Description |
|---|---|---|---|---|
| R01 | Unit price far above reference | over invoicing | 0.60 | Unit price exceeds the HS reference by more than the price-ratio and z-score thresholds. |
| R02 | Unit price far below reference | under invoicing | 0.60 | Unit price is below the HS reference by more than the price-ratio and z-score thresholds. |
| R03 | Customs value differs from invoice value | over invoicing | 0.50 | The value declared to customs and the invoiced value differ by more than the tolerance. |
| R04 | Several invoices exceed one shipment | multiple invoicing | 0.90 | Two or more invoices quote the same bill of lading and together exceed the shipped quantity. |
| R05 | Paid, but no shipment and no customs record | phantom shipment | 0.70 | Payment was made although neither a shipment nor a customs declaration can be found. |
| R06 | No customs record and paid early or near-empty shipment | phantom shipment | 0.50 | The customs declaration is missing and the trade was paid before shipment, or the cargo weighs far less than the goods should. |
| R07 | Shipped or declared quantity differs from invoice | over under shipment | 0.60 | Shipped or customs-declared quantity deviates from the invoiced quantity by more than the tolerance. |
| R08 | Goods declared under a different HS heading | misdescription | 0.60 | The HS code declared to customs is in a different 4-digit heading than the invoiced goods. |
| R09 | Cargo weight inconsistent with the goods | misdescription | 0.40 | Weight per unit (shipped or declared) is far from the typical weight of the invoiced goods. |
| R10 | Unrelated third-party payer in a high-risk jurisdiction | third party payment | 0.80 | The invoice was paid by a party that is neither related to the importer nor a finance house and sits in a high-risk or offshore jurisdiction. |
| R11 | Unrelated third-party payer abroad | third party payment | 0.40 | The invoice was paid by an unrelated party in a different jurisdiction than the importer. |
| R12 | Same goods traded round a loop | circular trading | 0.70 | The trade lies on a cycle of three or more parties trading the same HS code. |
| R13 | Goods resold within days inside a loop | circular trading | 0.40 | The seller bought the same goods only days earlier and the trade lies on a cycle of trades in that HS code. |
| R14 | Young company at a mass-registration address | shell company | 0.60 | A party is younger than the threshold and shares its address with several other entities. |
| R15 | Young company with very high turnover | shell company | 0.50 | A party is younger than the threshold but trades an unusually high volume for its age. |
| R16 | Round amount paid in advance to or by a young company | shell company | 0.40 | A round invoice total, payment before shipment and a party younger than twice the age threshold. |
| R17 | Needless transshipment | high risk routing | 0.50 | The route has many stops or is much longer than the direct route. |
| R18 | Stop in a listed high-risk jurisdiction | high risk routing | 0.50 | The goods were routed through a jurisdiction on the high-risk list. |
| R19 | Proceeds collected in an offshore bank | high risk routing | 0.40 | The exporter is paid into a bank outside its own jurisdiction, in a high-risk or offshore centre. |
| R20 | Party in a listed high-risk jurisdiction | high risk routing | 0.10 | The exporter or the importer is established in a jurisdiction on the high-risk list. |

## Thresholds

| Parameter | Value |
|---|---|
| `price_z_high` | 3.0 |
| `price_ratio_high` | 1.8 |
| `price_ratio_low` | 0.55 |
| `value_mismatch_customs` | 0.2 |
| `qty_mismatch` | 0.15 |
| `weight_per_unit_dev` | 0.6 |
| `invoices_per_bl_min` | 2 |
| `bl_cover_ratio_high` | 1.5 |
| `advance_payment_days` | -5 |
| `third_party_risk_min` | 0.6 |
| `rapid_resale_days` | 14 |
| `young_entity_days` | 365 |
| `shared_address_min` | 4 |
| `volume_per_age_high` | 7.0 |
| `transshipments_high` | 2 |
| `detour_ratio_high` | 1.8 |
