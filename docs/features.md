# Feature dictionary

Generated from `src/tbml/features/dictionary.py`; do not edit by hand (run `python scripts/build_features.py --docs-only`).

81 risk indicators in five groups. **Scope** is `row` when the indicator uses only the trade's own
documents and `context` when it depends on the other trades visible to whoever computes it (entity history, trade graph).
Context indicators are computed per FL client from that client's own trades only. **Owner** names the data owner that
could compute the indicator alone; `cross` needs records linked across owners.

Design notes:

- No identifiers (entity, bank, HS code, country names) are used as features; only behaviour and public risk tiers.
- Ratios are log-ratios clipped to a fixed range; counts are `log1p`-transformed. Missing documents give neutral
  values plus an explicit `missing_*` flag, so that absence is a signal and not an imputation artefact.
- Context indicators for a trade use only trades up to the end of that trade's split period (train, validation or
  test), and never use labels.

## Invoice / price

| Feature | Scope | Owner | What it measures | Typologies targeted |
|---|---|---|---|---|
| `price_log_ratio` | row | trade | log of unit price over the HS reference price (signed) | over invoicing, under invoicing |
| `price_z` | row | trade | price_log_ratio divided by the HS reference dispersion (signed z-score) | over invoicing, under invoicing |
| `price_abs_z` | row | trade | absolute price z-score | over invoicing, under invoicing, misdescription |
| `price_abs_z_declared_hs` | row | cross | absolute price z-score against the reference of the HS code declared to customs | misdescription |
| `price_dev_exporter_hs` | context | trade | log of unit price over the exporter's own median price for this HS code | over invoicing, under invoicing |
| `invoice_value_log` | row | trade | log10 of the invoice value in USD | shell company, over invoicing |
| `round_amount_level` | row | trade | number of trailing zeros of a whole-number invoice total (0 if not whole) | shell company |
| `round_amount_flag` | row | trade | invoice total is a multiple of the round-amount base | shell company |
| `inv_pay_gap` | row | cross | (amount paid - invoice value) / invoice value | multiple invoicing, third party payment |
| `inv_pay_abs_gap` | row | cross | absolute relative gap between amount paid and invoice value | multiple invoicing, third party payment |
| `customs_value_log_ratio` | row | cross | log of customs declared value over invoice value | over invoicing, under invoicing, misdescription |
| `customs_value_abs_dev` | row | cross | absolute size of the gap between customs value and invoice value | over invoicing, under invoicing, misdescription |

## Shipment / customs

| Feature | Scope | Owner | What it measures | Typologies targeted |
|---|---|---|---|---|
| `missing_shipment` | row | cross | no shipment record could be linked to the invoice | phantom shipment |
| `missing_customs` | row | cross | no customs declaration could be linked to the invoice | phantom shipment |
| `missing_both` | row | cross | neither shipment nor customs record exists | phantom shipment |
| `ship_qty_log_ratio` | context | cross | log of shipped quantity over the quantity invoiced against the same bill of lading | over under shipment, multiple invoicing |
| `ship_qty_abs_dev` | context | cross | absolute size of the gap between shipped and invoiced quantity | over under shipment, multiple invoicing |
| `customs_qty_log_ratio` | row | cross | log of customs declared quantity over invoiced quantity | over under shipment |
| `customs_qty_abs_dev` | row | cross | absolute size of the gap between declared and invoiced quantity | over under shipment |
| `weight_per_unit_log_ratio` | row | cross | log of shipped weight per unit over the typical weight of the invoiced goods | misdescription, phantom shipment, over under shipment |
| `weight_per_unit_abs_dev` | row | cross | absolute size of the shipped weight-per-unit anomaly | misdescription, phantom shipment, over under shipment |
| `customs_weight_per_unit_log_ratio` | row | cross | log of declared weight per unit over the typical weight of the invoiced goods | misdescription |
| `hs_mismatch` | row | cross | invoice HS code differs from the customs HS code | misdescription |
| `hs_heading_mismatch` | row | cross | invoice and customs HS codes differ in the 4-digit heading | misdescription |
| `hs_chapter_mismatch` | row | cross | invoice and customs HS codes differ in the 2-digit chapter | misdescription |
| `n_transshipments` | row | trade | number of transshipment stops | high risk routing |
| `route_detour_ratio` | row | trade | route length over the direct distance | high risk routing |
| `route_via_high_risk` | row | trade | number of stops in listed high-risk jurisdictions | high risk routing |
| `route_via_ftz` | row | trade | number of stops in free-trade-zone / offshore jurisdictions | high risk routing |
| `route_max_stop_risk` | row | trade | highest jurisdiction risk score among the stops | high risk routing |
| `route_rarity` | context | trade | -log share of this corridor's trades that used the same route | high risk routing |
| `days_invoice_to_ship` | row | trade | days from invoice to departure | circular trading, phantom shipment |
| `invoices_per_bl` | context | cross | number of invoices quoting the same bill of lading | multiple invoicing |
| `bl_payer_banks` | context | cross | number of distinct paying banks across invoices on the same bill of lading | multiple invoicing |

## Payment

| Feature | Scope | Owner | What it measures | Typologies targeted |
|---|---|---|---|---|
| `third_party_payer` | row | bank | a payer is not the importer | third party payment |
| `n_distinct_payers` | row | bank | number of distinct payers | third party payment |
| `payer_country_differs` | row | bank | third-party payer is in a different jurisdiction than the importer | third party payment |
| `payer_risk` | row | bank | highest jurisdiction risk score among third-party payers | third party payment |
| `payer_related` | row | bank | third-party payer shares an address or director with the importer (benign signal) | third party payment |
| `payer_is_finance_house` | row | bank | third-party payer is a registered trade-finance house (benign signal) | third party payment |
| `payer_age_log` | row | bank | log1p age in days of the youngest third-party payer (0 if none) | third party payment, shell company |
| `days_invoice_to_payment` | row | cross | days from invoice to first payment | phantom shipment, shell company, circular trading |
| `days_payment_minus_ship` | row | cross | days from departure to first payment (negative = paid before shipment) | phantom shipment, shell company |
| `paid_before_shipment` | row | cross | first payment precedes departure | phantom shipment, shell company |
| `n_payments` | row | bank | number of payment instalments | third party payment |
| `payment_span_days` | row | bank | days between first and last instalment | third party payment |
| `currency_mismatch` | row | cross | payment currency differs from invoice currency | third party payment |
| `currency_unusual` | row | trade | invoice currency is neither USD nor a home currency of the parties | third party payment, high risk routing |
| `payee_bank_offshore` | row | bank | exporter is paid into a bank outside its own jurisdiction | high risk routing, shell company |
| `payee_bank_risk` | row | bank | jurisdiction risk score of that bank (0 if domestic) | high risk routing, shell company |
| `payer_bank_offshore` | row | bank | payer pays from a bank outside its own jurisdiction | multiple invoicing, third party payment |

## Counterparty

| Feature | Scope | Owner | What it measures | Typologies targeted |
|---|---|---|---|---|
| `exporter_age_log` | row | bank | log1p age in days of the exporter at invoice date | shell company |
| `importer_age_log` | row | bank | log1p age in days of the importer at invoice date | shell company |
| `min_party_age_log` | row | bank | log1p age in days of the younger party | shell company |
| `exporter_country_risk` | row | bank | jurisdiction risk score of the exporter | high risk routing, shell company |
| `importer_country_risk` | row | bank | jurisdiction risk score of the importer | high risk routing, shell company |
| `max_party_risk` | row | bank | higher jurisdiction risk score of the two parties | high risk routing, shell company |
| `parties_related` | row | bank | exporter and importer share an address or a director | circular trading, shell company |
| `max_addr_shared_log` | context | bank | log1p number of other known entities at a party's address (max of both parties) | shell company |
| `max_dir_shared_log` | context | bank | log1p number of other known entities with a party's director (max of both parties) | shell company, circular trading |
| `exporter_n_trades_log` | context | trade | log1p number of trades of the exporter seen so far | shell company |
| `importer_n_trades_log` | context | trade | log1p number of trades of the importer seen so far | shell company |
| `younger_party_volume_per_age` | context | cross | log10 of USD volume traded per year of age, for the younger of the two parties | shell company |
| `pair_n_trades_log` | context | trade | log1p number of trades between this exporter and importer | circular trading, multiple invoicing |
| `pair_share_exporter` | context | trade | share of the exporter's volume that goes to this importer | circular trading, shell company |
| `exporter_hhi` | context | trade | concentration (HHI) of the exporter's volume across importers | circular trading, shell company |
| `importer_hhi` | context | trade | concentration (HHI) of the importer's volume across exporters | circular trading, shell company |
| `max_hs_diversity` | context | trade | number of distinct HS chapters traded by a party (max of both parties) | shell company |
| `exporter_import_share` | context | trade | share of the exporter's trades in which it is the buyer | circular trading |
| `importer_export_share` | context | trade | share of the importer's trades in which it is the seller | circular trading |

## Trade network (graph)

| Feature | Scope | Owner | What it measures | Typologies targeted |
|---|---|---|---|---|
| `exporter_out_degree_log` | context | trade | log1p number of distinct buyers of the exporter | shell company, circular trading |
| `exporter_in_degree_log` | context | trade | log1p number of distinct suppliers of the exporter | circular trading |
| `importer_in_degree_log` | context | trade | log1p number of distinct suppliers of the importer | shell company, circular trading |
| `importer_out_degree_log` | context | trade | log1p number of distinct buyers of the importer | circular trading |
| `exporter_pagerank_log` | context | trade | log of the exporter's PageRank in the trade graph (scaled by graph size) | circular trading, shell company |
| `importer_pagerank_log` | context | trade | log of the importer's PageRank in the trade graph (scaled by graph size) | circular trading, shell company |
| `edge_reciprocal` | context | trade | the importer also sells to the exporter | circular trading |
| `edge_in_cycle` | context | trade | the trade lies on a directed cycle of the trade graph | circular trading |
| `edge_in_cycle_same_hs` | context | trade | the trade lies on a directed cycle formed by trades in the same HS code | circular trading |
| `cycle_len_same_hs` | context | trade | length of the shortest same-HS cycle through the trade (0 if none) | circular trading |
| `exporter_resale_days_log` | context | trade | log1p days since the exporter last bought the same HS code (365 if never): rapid resale | circular trading, shell company |

## Typology coverage

| Typology | Indicators |
|---|---|
| over invoicing | `price_log_ratio`, `price_z`, `price_abs_z`, `price_dev_exporter_hs`, `invoice_value_log`, `customs_value_log_ratio`, `customs_value_abs_dev` |
| under invoicing | `price_log_ratio`, `price_z`, `price_abs_z`, `price_dev_exporter_hs`, `customs_value_log_ratio`, `customs_value_abs_dev` |
| multiple invoicing | `inv_pay_gap`, `inv_pay_abs_gap`, `ship_qty_log_ratio`, `ship_qty_abs_dev`, `invoices_per_bl`, `bl_payer_banks`, `payer_bank_offshore`, `pair_n_trades_log` |
| phantom shipment | `missing_shipment`, `missing_customs`, `missing_both`, `weight_per_unit_log_ratio`, `weight_per_unit_abs_dev`, `days_invoice_to_ship`, `days_invoice_to_payment`, `days_payment_minus_ship`, `paid_before_shipment` |
| over under shipment | `ship_qty_log_ratio`, `ship_qty_abs_dev`, `customs_qty_log_ratio`, `customs_qty_abs_dev`, `weight_per_unit_log_ratio`, `weight_per_unit_abs_dev` |
| misdescription | `price_abs_z`, `price_abs_z_declared_hs`, `customs_value_log_ratio`, `customs_value_abs_dev`, `weight_per_unit_log_ratio`, `weight_per_unit_abs_dev`, `customs_weight_per_unit_log_ratio`, `hs_mismatch`, `hs_heading_mismatch`, `hs_chapter_mismatch` |
| third party payment | `inv_pay_gap`, `inv_pay_abs_gap`, `third_party_payer`, `n_distinct_payers`, `payer_country_differs`, `payer_risk`, `payer_related`, `payer_is_finance_house`, `payer_age_log`, `n_payments`, `payment_span_days`, `currency_mismatch`, `currency_unusual`, `payer_bank_offshore` |
| circular trading | `days_invoice_to_ship`, `days_invoice_to_payment`, `parties_related`, `max_dir_shared_log`, `pair_n_trades_log`, `pair_share_exporter`, `exporter_hhi`, `importer_hhi`, `exporter_import_share`, `importer_export_share`, `exporter_out_degree_log`, `exporter_in_degree_log`, `importer_in_degree_log`, `importer_out_degree_log`, `exporter_pagerank_log`, `importer_pagerank_log`, `edge_reciprocal`, `edge_in_cycle`, `edge_in_cycle_same_hs`, `cycle_len_same_hs`, `exporter_resale_days_log` |
| shell company | `invoice_value_log`, `round_amount_level`, `round_amount_flag`, `payer_age_log`, `days_invoice_to_payment`, `days_payment_minus_ship`, `paid_before_shipment`, `payee_bank_offshore`, `payee_bank_risk`, `exporter_age_log`, `importer_age_log`, `min_party_age_log`, `exporter_country_risk`, `importer_country_risk`, `max_party_risk`, `parties_related`, `max_addr_shared_log`, `max_dir_shared_log`, `exporter_n_trades_log`, `importer_n_trades_log`, `younger_party_volume_per_age`, `pair_share_exporter`, `exporter_hhi`, `importer_hhi`, `max_hs_diversity`, `exporter_out_degree_log`, `importer_in_degree_log`, `exporter_pagerank_log`, `importer_pagerank_log`, `exporter_resale_days_log` |
| high risk routing | `n_transshipments`, `route_detour_ratio`, `route_via_high_risk`, `route_via_ftz`, `route_max_stop_risk`, `route_rarity`, `currency_unusual`, `payee_bank_offshore`, `payee_bank_risk`, `exporter_country_risk`, `importer_country_risk`, `max_party_risk` |
