"""Ordinary trade behaviour and benign noise.

The generator works on a table of *primitives*: one row per invoice holding the
independent quantities a trade is made of (price, quantity, ratios between
documents, lags, parties, route). Typologies later mutate primitives, and
``generator.finalise`` derives the actual documents. Keeping manipulation at
the primitive level guarantees that documents stay internally consistent except
where a typology (or a benign error) deliberately breaks them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tbml.config import Config
from tbml.data.entities import Entities
from tbml.data.reference import SECTORS, World

#: spec columns every trade starts from
SPEC_COLUMNS = [
    "exporter_id", "importer_id", "hs_code", "invoice_date", "typology", "network_id",
    "pair_effect", "qty_override", "price_usd_override", "group_key", "dup_of", "payer_override",
]


def empty_spec() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in SPEC_COLUMNS})


def build_price_index(rng: np.random.Generator, world: World, n_months: int, monthly_sd: float) -> np.ndarray:
    """Per-HS log price index: a monthly random walk (legitimate market moves)."""
    steps = rng.normal(0.0, monthly_sd, size=(len(world.hs), n_months))
    steps[:, 0] = 0.0
    return np.cumsum(steps, axis=1)


def build_relationships(rng: np.random.Generator, world: World, ents: Entities, cfg: Config) -> pd.DataFrame:
    """Standing supplier relationships: one row per (exporter, importer, HS code)."""
    lcfg = cfg.data.legit
    start = pd.Timestamp(cfg.data.start_date)
    end = pd.Timestamp(cfg.data.end_date)
    comp = ents.companies
    ordinary = comp[(comp["entity_type"] == "company") & (~comp["is_shell"])]
    exporters = ordinary[ordinary["role"].isin(["exporter", "trader"])]
    importers = ordinary[ordinary["role"].isin(["importer", "trader"])]

    sec_pos = {s: i for i, s in enumerate(SECTORS)}
    exp_sec = np.zeros((len(exporters), len(SECTORS)), dtype=bool)
    for r, secs in enumerate(exporters["sectors"]):
        exp_sec[r, [sec_pos[s] for s in secs]] = True
    exp_ids = exporters.index.to_numpy()
    exp_country = exporters["country"].to_numpy()
    exp_region = exporters["region"].to_numpy()
    exp_size = exporters["size"].to_numpy()
    exp_products = exporters["products"].to_numpy()
    hs_sector = world.hs["sector"].to_dict()

    rows: list[tuple] = []
    for imp_id, imp in importers.iterrows():
        want = np.zeros(len(SECTORS), dtype=bool)
        want[[sec_pos[s] for s in imp["sectors"]]] = True
        mask = exp_sec[:, want].any(axis=1) & (exp_country != imp["country"]) & (exp_ids != imp_id)
        cand = np.flatnonzero(mask)
        if not len(cand):
            continue
        k = min(len(cand), 1 + int(rng.poisson(max(0.0, lcfg.suppliers_per_importer_mean - 1.0))))
        w = exp_size[cand] * np.where(exp_region[cand] == imp["region"], lcfg.same_region_preference, 1.0)
        chosen = rng.choice(cand, size=k, replace=False, p=w / w.sum())
        for e in chosen:
            tradable = [h for h in exp_products[e] if hs_sector[h] in imp["sectors"]]
            if not tradable:
                continue
            n_prod = min(len(tradable), int(rng.integers(1, 3)))
            picks = rng.choice(tradable, size=n_prod, replace=False)
            weight = np.sqrt(exp_size[e] * imp["size"]) * rng.lognormal(0.0, 0.8)
            effect = rng.normal()
            for h in picks:
                rows.append((exp_ids[e], imp_id, h, weight / n_prod, effect))

    rel = pd.DataFrame(rows, columns=["exporter_id", "importer_id", "hs_code", "weight", "pair_effect"])
    reg_e = comp.loc[rel["exporter_id"], "registration_date"].to_numpy()
    reg_i = comp.loc[rel["importer_id"], "registration_date"].to_numpy()
    earliest = np.maximum(np.maximum(reg_e, reg_i) + np.timedelta64(30, "D"), np.datetime64(start))
    rel["earliest"] = earliest
    rel = rel[rel["earliest"] < np.datetime64(end - pd.Timedelta(days=10))].reset_index(drop=True)
    return rel


def legit_specs(rng: np.random.Generator, rel: pd.DataFrame, n: int, end: pd.Timestamp) -> pd.DataFrame:
    """Sample ``n`` ordinary trades from the standing relationships.

    A relationship that starts late (a party was founded during the period) gets
    proportionally fewer trades, which keeps the number of trades per day flat.
    """
    lo_all = rel["earliest"].to_numpy().astype("datetime64[D]")
    span_all = (np.datetime64(end, "D") - lo_all).astype(int)
    p = rel["weight"].to_numpy() * span_all
    idx = rng.choice(len(rel), size=n, p=p / p.sum())
    lo = lo_all[idx]
    span = span_all[idx]
    dates = lo + (rng.random(n) * span).astype(int).astype("timedelta64[D]")
    spec = pd.DataFrame(
        {
            "exporter_id": rel["exporter_id"].to_numpy()[idx],
            "importer_id": rel["importer_id"].to_numpy()[idx],
            "hs_code": rel["hs_code"].to_numpy()[idx],
            "invoice_date": pd.to_datetime(dates),
            "typology": "none",
            "network_id": "",
            "pair_effect": rel["pair_effect"].to_numpy()[idx],
            "qty_override": np.nan,
            "price_usd_override": np.nan,
            "group_key": "",
            "dup_of": -1,
            "payer_override": "",
        }
    )
    return spec


def _spec_frame(rows: list[dict]) -> pd.DataFrame:
    spec = pd.DataFrame(rows, columns=SPEC_COLUMNS)
    spec["invoice_date"] = pd.to_datetime(spec["invoice_date"])
    return spec


def _base_row(exporter: str, importer: str, hs: str, date, pair_effect: float, **extra) -> dict:
    row = {
        "exporter_id": exporter, "importer_id": importer, "hs_code": hs, "invoice_date": date, "typology": "none",
        "network_id": "", "pair_effect": pair_effect, "qty_override": np.nan, "price_usd_override": np.nan,
        "group_key": "", "dup_of": -1, "payer_override": "",
    }
    row.update(extra)
    return row


def spot_specs(rng: np.random.Generator, world: World, ents: Entities, cfg: Config, n: int) -> pd.DataFrame:
    """One-off deals between firms that have no standing relationship."""
    start = pd.Timestamp(cfg.data.start_date)
    end = pd.Timestamp(cfg.data.end_date)
    comp = ents.companies
    ordinary = comp[(comp["entity_type"] == "company") & (~comp["is_shell"])]
    exporters = ordinary[ordinary["role"].isin(["exporter", "trader"])]
    importers = ordinary[ordinary["role"].isin(["importer", "trader"])]
    sec_pos = {sec: i for i, sec in enumerate(SECTORS)}
    imp_sec = np.zeros((len(importers), len(SECTORS)), dtype=bool)
    for r, secs in enumerate(importers["sectors"]):
        imp_sec[r, [sec_pos[x] for x in secs]] = True
    imp_ids = importers.index.to_numpy()
    imp_country = importers["country"].to_numpy()
    imp_w = np.sqrt(importers["size"].to_numpy())
    imp_reg = importers["registration_date"].to_numpy()
    exp_w = np.sqrt(exporters["size"].to_numpy())
    hs_sector = world.hs["sector"].to_dict()
    rows: list[dict] = []
    for e in rng.choice(len(exporters), size=n, p=exp_w / exp_w.sum()):
        ex = exporters.iloc[e]
        hs = str(rng.choice(ex["products"]))
        cand = np.flatnonzero(imp_sec[:, sec_pos[hs_sector[hs]]] & (imp_country != ex["country"]) & (imp_ids != exporters.index[e]))
        if not len(cand):
            continue
        i = int(rng.choice(cand, p=imp_w[cand] / imp_w[cand].sum()))
        lo = max(start, ex["registration_date"] + pd.Timedelta(days=30), pd.Timestamp(imp_reg[i]) + pd.Timedelta(days=30))
        date = start + pd.Timedelta(days=int(rng.integers(0, (end - start).days + 1)))
        if date < lo:  # a party did not exist yet: no deal (keeps trades per day flat over time)
            continue
        rows.append(_base_row(exporters.index[e], imp_ids[i], hs, date.normalize(), float(rng.normal())))
    return _spec_frame(rows)


def return_specs(rng: np.random.Generator, world: World, legit: pd.DataFrame, cfg: Config, n: int) -> pd.DataFrame:
    """Returned goods: part of an earlier order travels back from buyer to seller."""
    end = pd.Timestamp(cfg.data.end_date)
    early = legit[legit["invoice_date"] <= end - pd.Timedelta(days=100)]
    if not len(early) or n <= 0:
        return _spec_frame([])
    picks = early.iloc[rng.choice(len(early), size=min(n, len(early)), replace=False)]
    typical = world.hs.loc[picks["hs_code"], "typical_qty"].to_numpy()
    rows = [
        _base_row(r.importer_id, r.exporter_id, r.hs_code, r.invoice_date + pd.Timedelta(days=int(rng.integers(20, 91))), float(r.pair_effect),
                  qty_override=float(typical[k] * rng.uniform(0.05, 0.30)))
        for k, r in enumerate(picks.itertuples())
    ]
    return _spec_frame(rows)


def ring_specs(rng: np.random.Generator, world: World, ents: Entities, cfg: Config, n: int) -> pd.DataFrame:
    """Benign commodity trading rings: established traders that pass the same goods around.

    These are legitimate same-goods cycles (location swaps, back-to-back trading)
    and are the look-alike of circular trading in the trade graph.
    """
    lcfg = cfg.data.legit
    start = pd.Timestamp(cfg.data.start_date)
    end = pd.Timestamp(cfg.data.end_date)
    comp = ents.companies
    traders = comp[(comp["entity_type"] == "company") & (~comp["is_shell"]) & (comp["role"] == "trader")
                   & (comp["registration_date"] < start - pd.Timedelta(days=365))]
    commodities = world.hs.index[world.hs["sector"].isin(["agri", "energy", "metals", "chemicals"])].to_numpy()
    rings: list[tuple[list[str], str, float]] = []
    attempts = 0
    while len(rings) < int(lcfg.n_trading_rings) and attempts < 400 and len(traders) >= 3:
        attempts += 1
        length = int(rng.integers(3, 5))
        members = list(rng.choice(traders.index.to_numpy(), size=length, replace=False))
        countries = [comp.at[m, "country"] for m in members]
        if any(countries[j] == countries[(j + 1) % length] for j in range(length)):
            continue
        rings.append((members, str(rng.choice(commodities)), float(rng.normal(0.0, 0.5))))
    rows: list[dict] = []
    for _ in range(n if rings else 0):
        members, hs, effect = rings[int(rng.integers(len(rings)))]
        j = int(rng.integers(len(members)))
        date = start + pd.Timedelta(days=int(rng.integers(0, (end - start).days + 1)))
        rows.append(_base_row(members[j], members[(j + 1) % len(members)], hs, date, effect))
    return _spec_frame(rows)


def _hub_probabilities(world: World, scale_nm: float = 900.0) -> np.ndarray:
    """P(hub | origin, destination) for legitimate transshipment: prefer small detours."""
    d = world.distance_nm
    hubs = world.hub_idx
    n = d.shape[0]
    detour = d[:, hubs][:, None, :] + d[hubs, :].T[None, :, :] - d[:, :, None]  # [o, d, h]
    logits = -detour / scale_nm
    for j, h in enumerate(hubs):  # a hub cannot serve its own country as a stopover
        logits[h, :, j] = -np.inf
        logits[:, h, j] = -np.inf
    logits -= logits.max(axis=2, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=2, keepdims=True)


def build_primitives(
    rng: np.random.Generator,
    spec: pd.DataFrame,
    world: World,
    ents: Entities,
    cfg: Config,
    price_index: np.ndarray,
    bl_offset: int = 0,
) -> pd.DataFrame:
    """Turn trade specs into primitives, following ordinary commercial behaviour."""
    lcfg = cfg.data.legit
    start = pd.Timestamp(cfg.data.start_date)
    n = len(spec)
    comp = ents.companies
    P = spec.reset_index(drop=True).copy()
    exp = comp.loc[P["exporter_id"]]
    imp = comp.loc[P["importer_id"]]
    hs_idx = world.hs_index(P["hs_code"])
    hs = world.hs.iloc[hs_idx]
    sigma = hs["ref_log_sigma"].to_numpy()
    dates = pd.DatetimeIndex(P["invoice_date"])

    # ---- invoice: price and quantity
    month = np.clip((dates.year - start.year) * 12 + dates.month - start.month, 0, price_index.shape[1] - 1)
    t_noise = rng.standard_t(5, size=n) / np.sqrt(5.0 / 3.0)
    log_price = (
        np.log(hs["ref_price_usd"].to_numpy())
        + price_index[hs_idx, month]
        + sigma * (lcfg.pair_price_effect * P["pair_effect"].to_numpy(dtype=float) + lcfg.trade_price_noise * t_noise)
    )
    extreme = rng.random(n) < lcfg.extreme_deal_rate
    log_price[extreme] += np.where(rng.random(int(extreme.sum())) < 0.5, np.log(0.5), np.log(1.8)) + rng.normal(0, 0.1, int(extreme.sum()))
    price = np.exp(log_price)
    over = P["price_usd_override"].to_numpy(dtype=float)
    price = np.where(np.isnan(over), price, over)

    qty = hs["typical_qty"].to_numpy() * imp["size"].to_numpy() ** 0.3 * rng.lognormal(0.0, 0.7, size=n)
    qover = P["qty_override"].to_numpy(dtype=float)
    qty = np.where(np.isnan(qover), qty, qover)
    P["qty"] = np.maximum(1, np.round(qty)).astype(np.int64)
    P["unit_price_usd"] = price

    exp_cur = world.countries.loc[exp["country"], "currency"].to_numpy()
    imp_cur = world.countries.loc[imp["country"], "currency"].to_numpy()
    u = rng.random(n)
    P["currency"] = np.where(u < 0.60, exp_cur, np.where(u < 0.95, "USD", imp_cur))
    P["round_base"] = np.where(rng.random(n) < lcfg.lump_sum_rate, 1000, 0)

    # ---- shipment
    # random numbering: the B/L number must not reveal the order trades were generated in
    P["bl_id"] = [f"BL-{bl_offset + i:08d}" for i in rng.permutation(n) + 1]
    P["bl_ref"] = P["bl_id"]
    P["has_shipment"] = True
    P["owns_shipment"] = True
    P["companion_qty"] = 0
    lines = ents.shipping_lines
    share = lines["market_share"].to_numpy()
    P["shipping_line_id"] = rng.choice(lines.index.to_numpy(), size=n, p=share / share.sum())
    o_idx = world.country_index(exp["country"])
    d_idx = world.country_index(imp["country"])
    P["origin_port"] = [f"{c}01" for c in exp["country"]]
    P["dest_port"] = [f"{c}01" for c in imp["country"]]
    same_region = exp["region"].to_numpy() == imp["region"].to_numpy()
    p1 = np.where(same_region, lcfg.transshipment_p1.same_region, lcfg.transshipment_p1.cross_region)
    r = rng.random(n)
    n_hops = np.where(r < lcfg.transshipment_p2, 2, np.where(r < lcfg.transshipment_p2 + p1, 1, 0))
    hub_p = _hub_probabilities(world)
    hubs = world.hub_idx
    codes = world.country_codes
    via = np.full(n, "", dtype=object)
    for i in np.flatnonzero(n_hops > 0):
        p = hub_p[o_idx[i], d_idx[i]]
        k = int(min(n_hops[i], np.count_nonzero(p > 0)))
        chosen = rng.choice(len(hubs), size=k, replace=False, p=p)
        # order stopovers by distance from the origin so the path is sensible
        chosen = sorted(chosen, key=lambda j: world.distance_nm[o_idx[i], hubs[j]])
        via[i] = "|".join(f"{codes[hubs[j]]}01" for j in chosen)
    # occasional extra regional stop at an ordinary (non-hub) port near either end
    regions = world.countries["region"].to_numpy()
    non_hub = world.countries["is_hub"].to_numpy() == 0
    for i in np.flatnonzero(rng.random(n) < lcfg.feeder_stop_rate):
        near_origin = rng.random() < 0.5
        region = regions[o_idx[i]] if near_origin else regions[d_idx[i]]
        pool = np.flatnonzero((regions == region) & non_hub & (np.arange(len(codes)) != o_idx[i]) & (np.arange(len(codes)) != d_idx[i]))
        if not len(pool):
            continue
        stop = f"{codes[int(rng.choice(pool))]}01"
        stops = [x for x in via[i].split("|") if x]
        via[i] = "|".join([stop, *stops] if near_origin else [*stops, stop])
    P["via_ports"] = via
    P["ship_lag_days"] = rng.integers(1, 16, size=n)
    P["transit_noise_days"] = rng.integers(0, 4, size=n)
    P["ship_qty_ratio"] = 1.0
    P["ship_qty_typo"] = 1.0
    P["weight_hs"] = P["hs_code"]
    P["weight_factor"] = 1.05 * rng.lognormal(0.0, 0.08, size=n)

    # ---- customs declaration (import side)
    P["has_customs"] = True
    P["owns_customs"] = True
    P["decl_hs_code"] = P["hs_code"]
    P["decl_value_ratio"] = rng.uniform(1.00, 1.06, size=n)  # CIF uplift over the invoice value
    P["decl_qty_ratio"] = 1.0
    P["decl_weight_ratio"] = rng.normal(1.0, 0.01, size=n)
    P["decl_lag_days"] = rng.integers(0, 6, size=n)

    # ---- payment plan
    terms = list(lcfg.payment_terms.keys())
    tp = np.array(list(lcfg.payment_terms.values()), dtype=float)
    term = rng.choice(terms, size=n, p=tp / tp.sum())
    lag = np.where(
        term == "advance",
        rng.integers(0, 6, size=n),
        np.where(term == "sight", rng.integers(3, 16, size=n), rng.choice([30, 60, 90], size=n) + rng.integers(-5, 6, size=n)),
    )
    P["pay_term"] = term
    P["pay_lag_days"] = lag
    sp = np.array(lcfg.split_payment_probs, dtype=float)
    P["n_payments"] = rng.choice(np.arange(1, len(sp) + 1), size=n, p=sp / sp.sum())
    P["pay_gap_days"] = rng.integers(15, 46, size=n)
    r = rng.random(n)
    P["pay_ratio"] = np.where(r < 0.93, 1.0, np.where(r < 0.98, 1.0 - rng.uniform(0.001, 0.006, size=n), 1.0 - rng.uniform(0.02, 0.10, size=n)))
    P["payer_id"] = P["importer_id"]
    P["payer2_id"] = ""
    P["payer_bank_id"] = imp["home_bank_id"].to_numpy()
    P["payee_bank_id"] = exp["home_bank_id"].to_numpy()
    P["pay_currency"] = P["currency"]

    # benign third-party payers: a related group company, or a trade-finance house
    by_director = comp[comp["entity_type"] == "company"].groupby("director_id").groups
    imp_dir = imp["director_id"].to_numpy()
    p_group = min(1.0, lcfg.group_payer_rate / max(cfg.data.entities.group_director_share, 1e-6))
    for i in np.flatnonzero(imp["in_group"].to_numpy() & (rng.random(n) < p_group)):
        relatives = [e for e in by_director[imp_dir[i]] if e != P.at[i, "importer_id"] and not comp.at[e, "is_shell"]]
        if relatives:
            payer = str(rng.choice(relatives))
            P.at[i, "payer_id"] = payer
            P.at[i, "payer_bank_id"] = comp.at[payer, "home_bank_id"]
    finance = comp.index[comp["entity_type"] == "finance_house"].to_numpy()
    if len(finance):
        fw = comp.loc[finance, "size"].to_numpy()
        fin_rows = np.flatnonzero(rng.random(n) < lcfg.finance_payer_rate)
        payers = rng.choice(finance, size=len(fin_rows), p=fw / fw.sum())
        P.loc[fin_rows, "payer_id"] = payers
        P.loc[fin_rows, "payer_bank_id"] = comp.loc[payers, "home_bank_id"].to_numpy()

    ordinary = comp.index[(comp["entity_type"] == "company") & (~comp["is_shell"])].to_numpy()
    ordinary_country = comp.loc[ordinary, "country"].to_numpy()
    imp_country = imp["country"].to_numpy()
    for i in np.flatnonzero(rng.random(n) < lcfg.unrelated_payer_rate):
        local = ordinary[(ordinary_country == imp_country[i]) & (ordinary != P.at[i, "importer_id"])]
        pool = local if (len(local) and rng.random() < 0.7) else ordinary
        payer = str(rng.choice(pool))
        if payer != P.at[i, "importer_id"]:
            P.at[i, "payer_id"] = payer
            P.at[i, "payer_bank_id"] = comp.at[payer, "home_bank_id"]

    hub_banks = ents.banks.index[ents.banks["country"].isin(codes[hubs])].to_numpy()
    off = np.flatnonzero(rng.random(n) < lcfg.offshore_account_rate)
    P.loc[off, "payee_bank_id"] = rng.choice(hub_banks, size=len(off))
    conv = rng.random(n) < lcfg.currency_convert_rate
    P.loc[conv, "pay_currency"] = np.where(P.loc[conv, "currency"] == "USD", "EUR", "USD")
    return P


def sibling_hs(code: str, rng: np.random.Generator) -> str:
    """A different HS-6 code under the same heading (first four digits)."""
    tail = int(code[4:])
    new = tail
    while new == tail:
        new = int(rng.integers(10, 100))
    return f"{code[:4]}{new:02d}"


def other_heading_hs(code: str, rng: np.random.Generator) -> str:
    """A different HS-6 code in the same chapter but under another heading."""
    heading = int(code[2:4])
    new = heading
    while new == heading:
        new = int(rng.integers(1, 100))
    return f"{code[:2]}{new:02d}{code[4:]}"


def apply_benign_noise(rng: np.random.Generator, P: pd.DataFrame, cfg: Config, world: World) -> pd.DataFrame:
    """Data-quality problems and commercial tolerances that look suspicious but are not."""
    ncfg = cfg.data.noise
    n = len(P)

    r = rng.random(n)
    tol = r < ncfg.qty_tolerance_rate
    short = (r >= ncfg.qty_tolerance_rate) & (r < ncfg.qty_tolerance_rate + ncfg.short_shipment_rate)
    P.loc[tol, "ship_qty_ratio"] = np.clip(rng.normal(1.0, 0.02, int(tol.sum())), 0.94, 1.06)
    P.loc[short, "ship_qty_ratio"] = rng.uniform(0.90, 0.99, int(short.sum()))

    typo = np.flatnonzero(rng.random(n) < ncfg.qty_typo_rate)
    factor = np.where(rng.random(len(typo)) < 0.5, 10.0, 0.1)
    in_customs = rng.random(len(typo)) < 0.5
    P.loc[typo[in_customs], "decl_qty_ratio"] = factor[in_customs]
    P.loc[typo[~in_customs], "ship_qty_typo"] = factor[~in_customs]

    catalogue = world.hs.index.to_numpy()
    for i in np.flatnonzero(rng.random(n) < ncfg.hs_reclass_rate):
        code = P.at[i, "hs_code"]
        u = rng.random()
        if u < ncfg.hs_reclass_chapter_p:  # keying error: another product entirely
            others = catalogue[np.array([c[:2] != code[:2] for c in catalogue])]
            P.at[i, "decl_hs_code"] = str(rng.choice(others))
        elif u < ncfg.hs_reclass_chapter_p + ncfg.hs_reclass_heading_p:
            P.at[i, "decl_hs_code"] = other_heading_hs(code, rng)
        else:
            P.at[i, "decl_hs_code"] = sibling_hs(code, rng)

    # goods sent in two consignments, but only the first bill of lading is quoted on the invoice
    split = rng.random(n) < ncfg.split_consignment_rate
    P.loc[split, "ship_qty_ratio"] = rng.uniform(0.40, 0.60, int(split.sum()))

    adj = rng.random(n) < ncfg.customs_value_adjust_rate
    P.loc[adj, "decl_value_ratio"] *= np.exp(rng.normal(0.0, 0.15, int(adj.sum())))

    # a mistyped B/L number on the invoice: the shipment exists but cannot be linked
    for i in np.flatnonzero(rng.random(n) < ncfg.bl_typo_rate):
        P.at[i, "bl_ref"] = "BL-9" + P.at[i, "bl_id"][4:]

    r = rng.random(n)
    both = r < ncfg.missing_both_rate
    ship = (r >= ncfg.missing_both_rate) & (r < ncfg.missing_both_rate + ncfg.missing_shipment_rate)
    cust = (r >= ncfg.missing_both_rate + ncfg.missing_shipment_rate) & (
        r < ncfg.missing_both_rate + ncfg.missing_shipment_rate + ncfg.missing_customs_rate
    )
    P.loc[both | ship, "has_shipment"] = False
    P.loc[both | cust, "has_customs"] = False
    return P
