"""Risk indicators (Step 02, objective 1).

``build_features`` turns linked trades into the feature matrix described in
``features/dictionary.py``. It is deliberately a pure function of the records it
is given: an FL client calls it on its own trades and gets indicators computed
from its own view only, while the centralised baseline calls it on pooled data.

Context indicators (entity history, trade graph) are computed on cumulative
snapshots: a trade in the validation period sees the trades up to the end of the
validation period, never later ones. Labels are never used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from tbml.config import Config
from tbml.data.reference import haversine_nm
from tbml.data.typologies import TYPOLOGY_CLASSES
from tbml.features.dictionary import CONTEXT_FEATURES, FEATURE_NAMES, ROW_FEATURES
from tbml.models.split import assign_split, split_cutoffs


@dataclass
class Reference:
    """Public reference data in lookup form (HS prices, jurisdiction risk, distances)."""

    hs: pd.DataFrame
    countries: pd.DataFrame
    fx: dict[str, float]
    country_pos: dict[str, int]
    distance_nm: np.ndarray

    @classmethod
    def from_tables(cls, tables: dict[str, pd.DataFrame]) -> "Reference":
        hs = tables["hs_reference"].set_index("hs_code")
        countries = tables["countries"].set_index("country")
        ports = tables["ports"].drop_duplicates("country").set_index("country").reindex(countries.index)
        lat, lon = ports["lat"].to_numpy(), ports["lon"].to_numpy()
        dist = haversine_nm(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
        fx = dict(zip(tables["fx_rates"]["currency"], tables["fx_rates"]["usd_per_unit"].astype(float)))
        return cls(hs=hs, countries=countries, fx=fx, country_pos={c: i for i, c in enumerate(countries.index)}, distance_nm=dist)


@dataclass
class FeatureSet:
    """Feature matrix plus labels and bookkeeping for one set of trades."""

    X: np.ndarray  # float32 [n, d], raw (unstandardised) indicators
    y: np.ndarray  # int8, 1 = illicit
    typology: np.ndarray  # int8 index into TYPOLOGY_CLASSES (0 = none)
    split: np.ndarray  # "train" | "val" | "test"
    trade_id: np.ndarray
    feature_names: list[str]

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, mask: np.ndarray) -> "FeatureSet":
        return FeatureSet(self.X[mask], self.y[mask], self.typology[mask], self.split[mask], self.trade_id[mask], self.feature_names)

    def part(self, name: str) -> "FeatureSet":
        return self.subset(self.split == name)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, X=self.X, y=self.y, typology=self.typology, split=self.split.astype("U5"),
                            trade_id=self.trade_id.astype("U16"), feature_names=np.array(self.feature_names))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FeatureSet":
        with np.load(path, allow_pickle=False) as z:
            return cls(z["X"], z["y"], z["typology"], z["split"].astype(object), z["trade_id"].astype(object), [str(n) for n in z["feature_names"]])


def _safe_log_ratio(num: np.ndarray, den: np.ndarray, clip: float, valid: np.ndarray | None = None) -> np.ndarray:
    """log(num / den) clipped to +/- clip; 0 where the ratio is undefined or not ``valid``."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    ok = (num > 0) & (den > 0) & np.isfinite(num) & np.isfinite(den)
    out = np.zeros(len(num))
    out[ok] = np.clip(np.log(num[ok] / den[ok]), -clip, clip)
    zero_num = (num <= 0) & (den > 0)  # e.g. nothing shipped against a positive invoice
    out[zero_num] = -clip
    if valid is not None:
        out[~valid] = 0.0
    return out


def _days(later: pd.Series, earlier: pd.Series) -> np.ndarray:
    return (later - earlier).dt.days.to_numpy(dtype=float)


def row_features(linked: pd.DataFrame, kyc: pd.DataFrame, banks: pd.DataFrame, ref: Reference, cfg: Config) -> pd.DataFrame:
    """Indicators computed from one trade's own documents and registry data."""
    L = linked.reset_index(drop=True)
    n = len(L)
    clip = float(cfg.features.clip_log_ratio)
    kyc = kyc.set_index("entity_id") if "entity_id" in kyc.columns else kyc
    bank_country = banks.set_index("bank_id")["country"] if "bank_id" in banks.columns else banks["country"]
    has_ship = L["has_shipment"].to_numpy() == 1
    has_cust = L["has_customs"].to_numpy() == 1
    F: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------ invoice / price
    hs = ref.hs.reindex(L["hs_code"])
    ref_price = hs["ref_price_usd"].to_numpy(dtype=float)
    sigma = hs["ref_log_sigma"].to_numpy(dtype=float)
    kg = hs["kg_per_unit"].to_numpy(dtype=float)
    price = L["unit_price_usd"].to_numpy(dtype=float)
    plr = _safe_log_ratio(price, ref_price, clip)
    z = np.clip(plr / sigma, -10, 10)
    F["price_log_ratio"] = plr
    F["price_z"] = z
    F["price_abs_z"] = np.abs(z)
    dhs = ref.hs.reindex(L["declared_hs_code"])
    known = has_cust & dhs["ref_price_usd"].notna().to_numpy()
    z_decl = np.clip(_safe_log_ratio(price, dhs["ref_price_usd"].to_numpy(dtype=float), 12.0) / np.where(known, dhs["ref_log_sigma"].to_numpy(dtype=float), 1.0), -10, 10)
    F["price_abs_z_declared_hs"] = np.where(known, np.abs(z_decl), np.abs(z))

    value_usd = L["invoice_value_usd"].to_numpy(dtype=float)
    F["invoice_value_log"] = np.log10(np.maximum(value_usd, 1.0))
    value = L["invoice_value"].to_numpy(dtype=float)
    whole = np.abs(value - np.round(value)) < 0.005
    ivalue = np.round(value).astype(np.int64)
    level = np.zeros(n)
    for k in range(1, 7):
        level += whole & (ivalue % 10**k == 0)
    F["round_amount_level"] = level
    F["round_amount_flag"] = (whole & (ivalue % int(cfg.features.round_amount_base) == 0)).astype(float)
    gap = np.clip((L["paid_usd"].to_numpy(dtype=float) - value_usd) / value_usd, -2, 2)
    F["inv_pay_gap"] = gap
    F["inv_pay_abs_gap"] = np.abs(gap)
    F["customs_value_log_ratio"] = _safe_log_ratio(L["declared_value_usd"], value_usd, clip, has_cust)
    F["customs_value_abs_dev"] = np.abs(F["customs_value_log_ratio"])

    # ------------------------------------------------------------ shipment / customs
    F["missing_shipment"] = (~has_ship).astype(float)
    F["missing_customs"] = (~has_cust).astype(float)
    F["missing_both"] = (~has_ship & ~has_cust).astype(float)
    qty = L["quantity"].to_numpy(dtype=float)
    F["customs_qty_log_ratio"] = _safe_log_ratio(L["declared_quantity"], qty, 3.0, has_cust)
    F["customs_qty_abs_dev"] = np.abs(F["customs_qty_log_ratio"])
    ship_qty = L["shipped_quantity"].to_numpy(dtype=float)
    weight = L["gross_weight_kg"].to_numpy(dtype=float)
    F["weight_per_unit_log_ratio"] = _safe_log_ratio(weight, ship_qty * kg, 6.0, has_ship & (ship_qty > 0))
    F["weight_per_unit_abs_dev"] = np.abs(F["weight_per_unit_log_ratio"])
    F["customs_weight_per_unit_log_ratio"] = _safe_log_ratio(L["declared_weight_kg"], L["declared_quantity"].to_numpy(dtype=float) * kg, 6.0,
                                                             has_cust & (L["declared_quantity"].to_numpy(dtype=float) > 0))
    inv_hs = L["hs_code"].to_numpy(dtype=object).astype(str)
    dec_hs = L["declared_hs_code"].to_numpy(dtype=object).astype(str)
    F["hs_mismatch"] = (has_cust & (inv_hs != dec_hs)).astype(float)
    F["hs_heading_mismatch"] = (has_cust & np.array([a[:4] != b[:4] for a, b in zip(inv_hs, dec_hs)])).astype(float)
    F["hs_chapter_mismatch"] = (has_cust & np.array([a[:2] != b[:2] for a, b in zip(inv_hs, dec_hs)])).astype(float)

    exp_country = kyc["country"].reindex(L["exporter_id"]).to_numpy(dtype=object)
    imp_country = kyc["country"].reindex(L["importer_id"]).to_numpy(dtype=object)
    risk = ref.countries["risk_score"]
    listed = ref.countries["fatf_listed"]
    ftz = ref.countries["is_ftz"]
    hops = np.zeros(n)
    detour = np.ones(n)
    via_listed = np.zeros(n)
    via_ftz = np.zeros(n)
    stop_risk = np.zeros(n)
    via = L["transshipment_ports"].to_numpy(dtype=object)
    routed = has_ship & (via != "")
    if routed.any():
        keys = pd.DataFrame({"o": L["origin_port"].str[:3], "v": via, "d": L["dest_port"].str[:3]})[routed]
        for (o, v, d), rows in keys.groupby(["o", "v", "d"]).groups.items():
            stops = [p[:3] for p in v.split("|")]
            path = [ref.country_pos[c] for c in [o, *stops, d]]
            length = sum(ref.distance_nm[a, b] for a, b in zip(path[:-1], path[1:]))
            direct = ref.distance_nm[path[0], path[-1]]
            idx = np.asarray(rows)
            hops[idx] = len(stops)
            detour[idx] = min(10.0, length / direct) if direct > 0 else 1.0
            via_listed[idx] = sum(int(listed[c]) for c in stops)
            via_ftz[idx] = sum(int(ftz[c]) for c in stops)
            stop_risk[idx] = max(float(risk[c]) for c in stops)
    F["n_transshipments"] = hops
    F["route_detour_ratio"] = detour
    F["route_via_high_risk"] = via_listed
    F["route_via_ftz"] = via_ftz
    F["route_max_stop_risk"] = stop_risk
    F["days_invoice_to_ship"] = np.where(has_ship, np.clip(np.nan_to_num(_days(L["ship_date"], L["invoice_date"])), -30, 120), 0.0)

    # ------------------------------------------------------------ payment
    importer = L["importer_id"].to_numpy(dtype=object)
    pay = pd.DataFrame({"row": np.arange(n), "payer": L["payers"].str.split("|")}).explode("payer")
    pay = pay[pay["payer"].notna() & (pay["payer"] != "")]
    prow = pay["row"].to_numpy()
    pk = kyc.reindex(pay["payer"])
    third = pay["payer"].to_numpy(dtype=object) != importer[prow]
    imp_k = kyc.reindex(importer[prow])
    related = third & ((pk["address_id"].to_numpy(dtype=object) == imp_k["address_id"].to_numpy(dtype=object)) | (pk["director_id"].to_numpy(dtype=object) == imp_k["director_id"].to_numpy(dtype=object)))
    payer_country = pk["country"].to_numpy(dtype=object)
    first_pay = L["first_payment_date"]
    payer_age = (first_pay.to_numpy()[prow] - pk["registration_date"].to_numpy()).astype("timedelta64[D]").astype(float)
    per = pd.DataFrame(
        {
            "row": prow,
            "third": third.astype(float),
            "differs": (third & (payer_country != imp_country[prow])).astype(float),
            "risk": np.where(third, risk.reindex(payer_country).to_numpy(dtype=float), 0.0),
            "related": related.astype(float),
            "finance": (third & (pk["entity_type"].to_numpy(dtype=object) == "finance_house")).astype(float),
            "age": np.clip(np.nan_to_num(payer_age, nan=3650.0), 0, None),
        }
    ).groupby("row").agg({"third": "max", "differs": "max", "risk": "max", "related": "max", "finance": "max", "age": "min"}).reindex(np.arange(n))
    F["third_party_payer"] = per["third"].fillna(0).to_numpy()
    F["n_distinct_payers"] = L["n_distinct_payers"].to_numpy(dtype=float)
    F["payer_country_differs"] = per["differs"].fillna(0).to_numpy()
    F["payer_risk"] = per["risk"].fillna(0).to_numpy()
    F["payer_related"] = per["related"].fillna(0).to_numpy()
    F["payer_is_finance_house"] = per["finance"].fillna(0).to_numpy()
    F["payer_age_log"] = np.log1p(per["age"].fillna(3650.0).to_numpy())
    F["days_invoice_to_payment"] = np.clip(np.nan_to_num(_days(first_pay, L["invoice_date"])), -30, 400)
    pay_minus_ship = np.nan_to_num(_days(first_pay, L["ship_date"]))
    F["days_payment_minus_ship"] = np.where(has_ship, np.clip(pay_minus_ship, -120, 400), 0.0)
    F["paid_before_shipment"] = (np.where(has_ship, pay_minus_ship < 0, True) & (L["has_payment"].to_numpy() == 1)).astype(float)
    F["n_payments"] = L["n_payments"].to_numpy(dtype=float)
    F["payment_span_days"] = np.clip(np.nan_to_num(_days(L["last_payment_date"], first_pay)), 0, 400)
    cur = L["currency"].to_numpy(dtype=object)
    F["currency_mismatch"] = ((L["pay_currency"].to_numpy(dtype=object) != cur) | (L["n_pay_currencies"].to_numpy() > 1)).astype(float) * (L["has_payment"].to_numpy() == 1)
    home = ref.countries["currency"]
    F["currency_unusual"] = ((cur != "USD") & (cur != home.reindex(exp_country).to_numpy(dtype=object)) & (cur != home.reindex(imp_country).to_numpy(dtype=object))).astype(float)
    payee_bank_country = bank_country.reindex(L["payee_bank_id"]).to_numpy(dtype=object)
    offshore = (payee_bank_country != exp_country) & (L["has_payment"].to_numpy() == 1)
    F["payee_bank_offshore"] = offshore.astype(float)
    F["payee_bank_risk"] = np.where(offshore, risk.reindex(payee_bank_country).fillna(0).to_numpy(dtype=float), 0.0)
    primary_country = kyc["country"].reindex(L["payer_id"]).to_numpy(dtype=object)
    payer_bank_country = bank_country.reindex(L["payer_bank_id"]).to_numpy(dtype=object)
    F["payer_bank_offshore"] = ((payer_bank_country != primary_country) & (L["has_payment"].to_numpy() == 1)).astype(float)

    # ------------------------------------------------------------ counterparty (registry data)
    exp_k = kyc.reindex(L["exporter_id"])
    imp_k2 = kyc.reindex(L["importer_id"])
    inv_date = L["invoice_date"].to_numpy()
    exp_age = np.clip((inv_date - exp_k["registration_date"].to_numpy()).astype("timedelta64[D]").astype(float), 0, None)
    imp_age = np.clip((inv_date - imp_k2["registration_date"].to_numpy()).astype("timedelta64[D]").astype(float), 0, None)
    F["exporter_age_log"] = np.log1p(exp_age)
    F["importer_age_log"] = np.log1p(imp_age)
    F["min_party_age_log"] = np.log1p(np.minimum(exp_age, imp_age))
    exp_risk = risk.reindex(exp_country).to_numpy(dtype=float)
    imp_risk = risk.reindex(imp_country).to_numpy(dtype=float)
    F["exporter_country_risk"] = exp_risk
    F["importer_country_risk"] = imp_risk
    F["max_party_risk"] = np.maximum(exp_risk, imp_risk)
    F["parties_related"] = ((exp_k["address_id"].to_numpy(dtype=object) == imp_k2["address_id"].to_numpy(dtype=object))
                            | (exp_k["director_id"].to_numpy(dtype=object) == imp_k2["director_id"].to_numpy(dtype=object))).astype(float)

    out = pd.DataFrame(F, index=pd.Index(L["trade_id"], name="trade_id"))
    return out[ROW_FEATURES]


def _same_hs_cycles(S: pd.DataFrame, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Per trade: does it lie on a cycle of same-HS trades, and how short is that cycle."""
    n = len(S)
    in_cycle = np.zeros(n)
    cycle_len = np.zeros(n)
    exp = S["exporter_id"].to_numpy(dtype=object)
    imp = S["importer_id"].to_numpy(dtype=object)
    for _, rows in S.groupby("hs_code", sort=False).indices.items():
        edges = set(zip(exp[rows], imp[rows]))
        if len(edges) < 2:
            continue
        g = nx.DiGraph()
        g.add_edges_from(edges)
        comp_of: dict[str, int] = {}
        for cid, comp in enumerate(nx.strongly_connected_components(g)):
            if len(comp) > 1:
                for node in comp:
                    comp_of[node] = cid
        if not comp_of:
            continue
        cache: dict[tuple[str, str], int] = {}
        for r in rows:
            u, v = exp[r], imp[r]
            if u in comp_of and comp_of.get(v, -1) == comp_of[u]:
                in_cycle[r] = 1.0
                if (u, v) not in cache:
                    lengths = nx.single_source_shortest_path_length(g, v, cutoff=max_len - 1)
                    cache[(u, v)] = lengths.get(u, max_len) + 1
                cycle_len[r] = cache[(u, v)]
    return in_cycle, cycle_len


def context_features(S: pd.DataFrame, kyc: pd.DataFrame, ref: Reference, cfg: Config, as_of: pd.Timestamp) -> pd.DataFrame:
    """Indicators that depend on the set of trades ``S`` visible to the computing party."""
    S = S.reset_index(drop=True)
    n = len(S)
    clip = float(cfg.features.clip_log_ratio)
    kyc = kyc.set_index("entity_id") if "entity_id" in kyc.columns else kyc
    exp = S["exporter_id"]
    imp = S["importer_id"]
    value = S["invoice_value_usd"].astype(float)
    F: dict[str, np.ndarray] = {}

    # price against the exporter's own history for the same goods
    grp = S.groupby(["exporter_id", "hs_code"])["unit_price_usd"]
    med = grp.transform("median").to_numpy()
    cnt = grp.transform("size").to_numpy()
    F["price_dev_exporter_hs"] = _safe_log_ratio(S["unit_price_usd"], med, clip, cnt >= 3)

    # bill-of-lading level: how much was invoiced against one shipment, and by how many banks
    bl = S.groupby("bl_id")
    per_bl = bl["quantity"].transform("sum").to_numpy(dtype=float)
    F["invoices_per_bl"] = bl["trade_id"].transform("size").to_numpy(dtype=float)
    F["bl_payer_banks"] = bl["payer_bank_id"].transform("nunique").to_numpy(dtype=float)
    has_ship = S["has_shipment"].to_numpy() == 1
    F["ship_qty_log_ratio"] = _safe_log_ratio(S["shipped_quantity"], per_bl, 3.0, has_ship)
    F["ship_qty_abs_dev"] = np.abs(F["ship_qty_log_ratio"])

    # how usual the route is on its corridor
    exp_country = kyc["country"].reindex(exp).to_numpy(dtype=object)
    imp_country = kyc["country"].reindex(imp).to_numpy(dtype=object)
    corridor = pd.Series(exp_country, dtype=object).str.cat(pd.Series(imp_country, dtype=object), sep=">")
    route = corridor.str.cat(S["transshipment_ports"].astype(str), sep=":")
    shipped = pd.Series(has_ship)
    c_corr = corridor[shipped].map(corridor[shipped].value_counts())
    c_route = route[shipped].map(route[shipped].value_counts())
    rarity = np.zeros(n)
    rarity[has_ship] = -np.log((c_route.to_numpy(dtype=float) + 1.0) / (c_corr.to_numpy(dtype=float) + 2.0))
    F["route_rarity"] = rarity

    # registry neighbourhood among the entities this party knows
    payers = S["payers"].str.split("|").explode()
    known = pd.unique(pd.concat([exp, imp, payers[payers.notna() & (payers != "")]], ignore_index=True))
    reg = kyc.reindex(known)
    addr_n = reg["address_id"].map(reg["address_id"].value_counts()) - 1
    dir_n = reg["director_id"].map(reg["director_id"].value_counts()) - 1
    addr_n.index = known
    dir_n.index = known
    F["max_addr_shared_log"] = np.log1p(np.maximum(addr_n.reindex(exp).to_numpy(dtype=float), addr_n.reindex(imp).to_numpy(dtype=float)))
    F["max_dir_shared_log"] = np.log1p(np.maximum(dir_n.reindex(exp).to_numpy(dtype=float), dir_n.reindex(imp).to_numpy(dtype=float)))

    # activity of each entity
    n_exp = exp.value_counts()
    n_imp = imp.value_counts()
    vol_exp = value.groupby(exp).sum()
    vol_imp = value.groupby(imp).sum()
    entities = n_exp.index.union(n_imp.index)
    n_as_exp = n_exp.reindex(entities, fill_value=0)
    n_as_imp = n_imp.reindex(entities, fill_value=0)
    total_n = n_as_exp + n_as_imp
    total_vol = vol_exp.reindex(entities, fill_value=0.0) + vol_imp.reindex(entities, fill_value=0.0)
    age_years = np.maximum((as_of - kyc["registration_date"].reindex(entities)).dt.days.to_numpy(dtype=float), 30.0) / 365.25
    vol_per_age = pd.Series(np.log10(total_vol.to_numpy() / age_years + 1.0), index=entities)
    F["exporter_n_trades_log"] = np.log1p(total_n.reindex(exp).to_numpy(dtype=float))
    F["importer_n_trades_log"] = np.log1p(total_n.reindex(imp).to_numpy(dtype=float))
    F["max_volume_per_age"] = np.maximum(vol_per_age.reindex(exp).to_numpy(), vol_per_age.reindex(imp).to_numpy())
    F["exporter_import_share"] = (n_as_imp / total_n).reindex(exp).to_numpy(dtype=float)
    F["importer_export_share"] = (n_as_exp / total_n).reindex(imp).to_numpy(dtype=float)

    # pair concentration
    pair = S.groupby(["exporter_id", "importer_id"])
    pair_n = pair["trade_id"].transform("size").to_numpy(dtype=float)
    pair_vol = pair["invoice_value_usd"].transform("sum").to_numpy(dtype=float)
    exp_total = vol_exp.reindex(exp).to_numpy(dtype=float)
    F["pair_n_trades_log"] = np.log1p(pair_n)
    F["pair_share_exporter"] = pair_vol / exp_total
    pairs = pair["invoice_value_usd"].sum().reset_index()
    pairs["s_exp"] = (pairs["invoice_value_usd"] / pairs["exporter_id"].map(vol_exp)) ** 2
    pairs["s_imp"] = (pairs["invoice_value_usd"] / pairs["importer_id"].map(vol_imp)) ** 2
    F["exporter_hhi"] = pairs.groupby("exporter_id")["s_exp"].sum().reindex(exp).to_numpy(dtype=float)
    F["importer_hhi"] = pairs.groupby("importer_id")["s_imp"].sum().reindex(imp).to_numpy(dtype=float)

    chapters = pd.concat(
        [pd.DataFrame({"e": exp.to_numpy(), "c": S["hs_code"].str[:2].to_numpy()}), pd.DataFrame({"e": imp.to_numpy(), "c": S["hs_code"].str[:2].to_numpy()})],
        ignore_index=True,
    ).drop_duplicates()
    diversity = chapters.groupby("e").size()
    F["max_hs_diversity"] = np.maximum(diversity.reindex(exp).to_numpy(dtype=float), diversity.reindex(imp).to_numpy(dtype=float))

    # trade graph: exporter -> importer
    g = nx.DiGraph()
    g.add_edges_from(zip(pairs["exporter_id"], pairs["importer_id"]))
    out_deg = pd.Series(dict(g.out_degree()))
    in_deg = pd.Series(dict(g.in_degree()))
    F["exporter_out_degree_log"] = np.log1p(out_deg.reindex(exp).to_numpy(dtype=float))
    F["exporter_in_degree_log"] = np.log1p(in_deg.reindex(exp).to_numpy(dtype=float))
    F["importer_in_degree_log"] = np.log1p(in_deg.reindex(imp).to_numpy(dtype=float))
    F["importer_out_degree_log"] = np.log1p(out_deg.reindex(imp).to_numpy(dtype=float))
    rank = pd.Series(nx.pagerank(g, alpha=float(cfg.features.pagerank_alpha), max_iter=200, tol=1e-8)) * g.number_of_nodes()
    F["exporter_pagerank_log"] = np.log(rank.reindex(exp).to_numpy(dtype=float))
    F["importer_pagerank_log"] = np.log(rank.reindex(imp).to_numpy(dtype=float))
    edge_set = set(zip(pairs["exporter_id"], pairs["importer_id"]))
    exp_arr = exp.to_numpy(dtype=object)
    imp_arr = imp.to_numpy(dtype=object)
    F["edge_reciprocal"] = np.fromiter(((v, u) in edge_set for u, v in zip(exp_arr, imp_arr)), dtype=float, count=n)
    comp_of: dict[str, int] = {}
    for cid, comp in enumerate(nx.strongly_connected_components(g)):
        if len(comp) > 1:
            for node in comp:
                comp_of[node] = cid
    F["edge_in_cycle"] = np.fromiter((u in comp_of and comp_of.get(v, -1) == comp_of[u] for u, v in zip(exp_arr, imp_arr)), dtype=float, count=n)
    F["edge_in_cycle_same_hs"], F["cycle_len_same_hs"] = _same_hs_cycles(S, int(cfg.features.cycle_max_len))

    # rapid resale: how recently did the seller itself buy the same goods
    sells = pd.DataFrame({"entity": exp_arr, "hs_code": S["hs_code"].to_numpy(dtype=object), "date": S["invoice_date"].to_numpy(), "row": np.arange(n)})
    buys = pd.DataFrame({"entity": imp_arr, "hs_code": S["hs_code"].to_numpy(dtype=object), "buy_date": S["invoice_date"].to_numpy()})
    matched = pd.merge_asof(sells.sort_values("date", kind="stable"), buys.sort_values("buy_date", kind="stable"),
                            left_on="date", right_on="buy_date", by=["entity", "hs_code"], direction="backward")
    gap = (matched["date"] - matched["buy_date"]).dt.days.fillna(365.0).clip(0, 365).to_numpy(dtype=float)
    resale = np.empty(n)
    resale[matched["row"].to_numpy()] = gap
    F["exporter_resale_days_log"] = np.log1p(resale)

    out = pd.DataFrame(F, index=pd.Index(S["trade_id"], name="trade_id"))
    return out[CONTEXT_FEATURES]


def build_features(
    linked: pd.DataFrame,
    kyc: pd.DataFrame,
    banks: pd.DataFrame,
    ref: Reference,
    cfg: Config,
    labels: pd.DataFrame | None = None,
) -> FeatureSet:
    """Feature matrix for ``linked`` trades, computed only from the records passed in.

    ``labels`` (trade_id, is_illicit, typology) are attached to the result for
    training and evaluation; they play no part in computing the indicators.
    """
    linked = linked.sort_values(["invoice_date", "trade_id"], kind="stable").reset_index(drop=True)
    split = assign_split(linked["invoice_date"], cfg)
    rows = row_features(linked, kyc, banks, ref, cfg)
    train_end, val_end = split_cutoffs(cfg)
    end = pd.Timestamp(cfg.data.end_date)
    parts = []
    for name, cutoff in [("train", train_end), ("val", val_end), ("test", end)]:
        mask = split == name
        if not mask.any():
            continue
        snapshot = linked[linked["invoice_date"] <= cutoff]
        ctx = context_features(snapshot, kyc, ref, cfg, as_of=cutoff)
        parts.append(ctx.loc[linked.loc[mask, "trade_id"]])
    context = pd.concat(parts).reindex(rows.index)
    table = pd.concat([rows, context], axis=1)[FEATURE_NAMES]
    X = table.to_numpy(dtype=np.float32)
    if not np.isfinite(X).all():
        bad = [FEATURE_NAMES[j] for j in np.flatnonzero(~np.isfinite(X).all(axis=0))]
        raise ValueError(f"non-finite feature values in {bad}")

    if labels is not None:
        lab = labels.set_index("trade_id").reindex(linked["trade_id"])
        y = lab["is_illicit"].to_numpy(dtype=np.int8)
        typ = lab["typology"].map({name: i for i, name in enumerate(TYPOLOGY_CLASSES)}).to_numpy(dtype=np.int8)
    else:
        y = np.zeros(len(linked), dtype=np.int8)
        typ = np.zeros(len(linked), dtype=np.int8)
    return FeatureSet(X=X, y=y, typology=typ, split=split, trade_id=linked["trade_id"].to_numpy(dtype=object), feature_names=list(FEATURE_NAMES))
