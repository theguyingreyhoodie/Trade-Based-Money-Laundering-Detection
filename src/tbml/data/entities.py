"""Entities of the simulated world: companies, banks, shipping lines, laundering networks.

The company table carries hidden ground-truth columns (``is_shell``,
``is_complicit``, ``network_id`` ...). They are never written to the per-owner
views; only the KYC/registry attributes are (see ``generator.kyc_view``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tbml.config import Config
from tbml.data.reference import REGION_EXPORT_SECTORS, SECTORS, World

_SECTOR_WORDS = {
    "agri": "Agro", "energy": "Energy", "chemicals": "Chem", "metals": "Metals", "precious": "Bullion",
    "textiles": "Textiles", "electronics": "Electronics", "vehicles": "Motors", "consumer": "Homeware", "building": "Materials",
}
_SUFFIXES = ["Ltd", "GmbH", "LLC", "S.A.", "Pte", "Co.", "Trading", "Holdings", "Exports", "Imports"]
_SHELL_WORDS = ["Global", "Prime", "Apex", "Summit", "Orbit", "Vertex", "Nova", "Zenith", "Atlas", "Pacific"]
_SHELL_NOUNS = ["Trade Solutions", "General Trading", "Commerce", "Ventures", "Logistics", "International", "Enterprises"]


@dataclass
class Network:
    """A laundering network (ground truth only)."""

    network_id: str
    typologies: list[str]
    hs_codes: list[str]
    start: pd.Timestamp
    end: pd.Timestamp
    shells: list[str] = field(default_factory=list)
    complicit: list[str] = field(default_factory=list)
    payers: list[str] = field(default_factory=list)
    #: period during which each shell is in use (shells are short-lived and get replaced)
    shell_windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=dict)

    def active_shells(self, date: pd.Timestamp) -> list[str]:
        return [s for s in self.shells if self.shell_windows[s][0] <= date <= self.shell_windows[s][1]]

    @property
    def members(self) -> list[str]:
        return [*self.shells, *self.complicit]


@dataclass
class Entities:
    companies: pd.DataFrame  # indexed by entity_id, includes hidden truth columns
    banks: pd.DataFrame  # indexed by bank_id
    shipping_lines: pd.DataFrame  # indexed by shipping_line_id
    networks: list[Network]


def build_banks(rng: np.random.Generator, world: World, n_banks: int) -> pd.DataFrame:
    """At least one bank per jurisdiction; the rest follow trade weight."""
    codes = world.country_codes
    n_extra = max(0, n_banks - len(codes))
    weights = world.countries["trade_weight"].to_numpy()
    extra = rng.choice(codes, size=n_extra, p=weights / weights.sum()) if n_extra else np.array([], dtype=object)
    countries = np.concatenate([codes, extra])
    banks = pd.DataFrame(
        {
            "bank_id": [f"BNK-{i:03d}" for i in range(1, len(countries) + 1)],
            "bank_name": [f"{world.countries.loc[c, 'name']} Bank {i}" for i, c in enumerate(countries, start=1)],
            "country": countries,
        }
    ).set_index("bank_id")
    return banks


def build_shipping_lines(rng: np.random.Generator, n_lines: int) -> pd.DataFrame:
    share = rng.dirichlet(np.full(n_lines, 2.0))
    return pd.DataFrame(
        {
            "shipping_line_id": [f"SHL-{i:02d}" for i in range(1, n_lines + 1)],
            "line_name": [f"Carrier {chr(64 + i)} Lines" for i in range(1, n_lines + 1)],
            "market_share": share.round(4),
        }
    ).set_index("shipping_line_id")


def _bank_for(rng: np.random.Generator, banks: pd.DataFrame, country: str) -> str:
    local = banks.index[banks["country"].to_numpy() == country]
    return str(rng.choice(local))


def _sample_products(rng: np.random.Generator, world: World, sectors: list[str], lo: int, hi: int) -> tuple[str, ...]:
    pool = world.hs.index[world.hs["sector"].isin(sectors)].to_numpy()
    k = int(min(len(pool), rng.integers(lo, hi + 1)))
    return tuple(sorted(rng.choice(pool, size=k, replace=False)))


def build_companies(rng: np.random.Generator, world: World, banks: pd.DataFrame, cfg: Config, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Ordinary companies (exporters, importers, traders) and benign finance houses."""
    ecfg = cfg.data.entities
    n = int(ecfg.n_companies)
    codes = world.country_codes
    weights = world.countries["trade_weight"].to_numpy()
    country = rng.choice(codes, size=n, p=weights / weights.sum())
    region = world.countries.loc[country, "region"].to_numpy()
    roles = list(ecfg.role_shares.keys())
    role = rng.choice(roles, size=n, p=np.array(list(ecfg.role_shares.values())) / sum(ecfg.role_shares.values()))

    sectors: list[tuple[str, ...]] = []
    products: list[tuple[str, ...]] = []
    for i in range(n):
        if role[i] == "importer":
            k = int(rng.integers(1, 4))
            sec = tuple(sorted(rng.choice(SECTORS, size=k, replace=False)))
            prod: tuple[str, ...] = ()
        else:
            w = np.array(REGION_EXPORT_SECTORS[region[i]])
            k = int(rng.integers(1, 3))
            sec = tuple(sorted(rng.choice(SECTORS, size=k, replace=False, p=w / w.sum())))
            prod = _sample_products(rng, world, list(sec), 2, 4)
        sectors.append(sec)
        products.append(prod)

    # registration dates: established firms, recently founded firms, entrants during the period
    period_days = (end - start).days
    age_years = np.clip(rng.gamma(2.0, 5.0, size=n), 1.5, 60.0)
    reg = start - pd.to_timedelta((age_years * 365.25).astype(int), unit="D")
    kind = rng.random(n)
    recent = kind < 0.05
    entrant = (kind >= 0.05) & (kind < 0.05 + ecfg.new_entrant_share)
    reg = pd.Series(reg)
    reg[recent] = start - pd.to_timedelta(rng.integers(60, 540, size=int(recent.sum())), unit="D")
    reg[entrant] = start + pd.to_timedelta(rng.integers(0, max(1, period_days - 150), size=int(entrant.sum())), unit="D")

    # addresses: mostly unique, some firms share a business-centre address
    address = np.array([f"ADR-{i:06d}" for i in range(1, n + 1)], dtype=object)
    n_bc = int(ecfg.n_business_centres)
    bc_country = rng.choice(codes, size=n_bc, p=weights / weights.sum())
    bc_pop = rng.pareto(1.5, size=n_bc) + 1.0
    young = np.asarray(recent | entrant)
    in_bc = rng.random(n) < np.where(young, ecfg.startup_business_centre_p, ecfg.shared_address_share)
    for i in np.flatnonzero(in_bc):
        local = np.flatnonzero(bc_country == country[i])
        if len(local):
            p = bc_pop[local] / bc_pop[local].sum()
            address[i] = f"ADR-BC{int(rng.choice(local, p=p)) + 1:03d}"

    # directors: mostly unique, some firms belong to a business group / use a nominee provider
    director = np.array([f"DIR-{i:06d}" for i in range(1, n + 1)], dtype=object)
    n_gd = int(ecfg.n_group_directors)
    gd_pop = rng.pareto(1.8, size=n_gd) + 1.0
    in_group = rng.random(n) < ecfg.group_director_share
    director[in_group] = [f"DIR-G{int(g) + 1:03d}" for g in rng.choice(n_gd, size=int(in_group.sum()), p=gd_pop / gd_pop.sum())]

    names = [
        f"{world.countries.loc[country[i], 'name'].split()[0]} {_SECTOR_WORDS[sectors[i][0]]} {rng.choice(_SUFFIXES)} {i + 1:04d}"
        for i in range(n)
    ]
    companies = pd.DataFrame(
        {
            "entity_id": [f"ENT-{i:05d}" for i in range(1, n + 1)],
            "name": names,
            "entity_type": "company",
            "role": role,
            "country": country,
            "registration_date": reg.dt.normalize().to_numpy(),
            "address_id": address,
            "director_id": director,
            "home_bank_id": [_bank_for(rng, banks, c) for c in country],
            "sectors": sectors,
            "products": products,
            "size": rng.lognormal(0.0, 1.0, size=n),
            "is_shell": False,
            "is_complicit": False,
            "is_payer": False,
            "network_id": "",
        }
    )

    # benign finance houses that settle invoices on behalf of importers
    n_fh = int(ecfg.n_finance_houses)
    safe = world.countries.index[world.countries["risk_tier"].isin(["low"])].to_numpy()
    fh_country = rng.choice(safe, size=n_fh)
    finance = pd.DataFrame(
        {
            "entity_id": [f"FIN-{i:04d}" for i in range(1, n_fh + 1)],
            "name": [f"{world.countries.loc[c, 'name'].split()[0]} Trade Finance {i}" for i, c in enumerate(fh_country, start=1)],
            "entity_type": "finance_house",
            "role": "payer",
            "country": fh_country,
            "registration_date": (start - pd.to_timedelta(rng.integers(3650, 14600, size=n_fh), unit="D")).normalize(),
            "address_id": [f"ADR-F{i:05d}" for i in range(1, n_fh + 1)],
            "director_id": [f"DIR-F{i:05d}" for i in range(1, n_fh + 1)],
            "home_bank_id": [_bank_for(rng, banks, c) for c in fh_country],
            "sectors": [()] * n_fh,
            "products": [()] * n_fh,
            "size": rng.lognormal(1.0, 0.5, size=n_fh),
            "is_shell": False,
            "is_complicit": False,
            "is_payer": False,
            "network_id": "",
        }
    )
    return pd.concat([companies, finance], ignore_index=True).set_index("entity_id")


def _weighted_typologies(rng: np.random.Generator, mix: dict[str, float], k: int) -> list[str]:
    names = list(mix.keys())
    p = np.array([mix[t] for t in names], dtype=float)
    return list(rng.choice(names, size=min(k, len(names)), replace=False, p=p / p.sum()))


def build_networks(
    rng: np.random.Generator,
    world: World,
    companies: pd.DataFrame,
    banks: pd.DataFrame,
    cfg: Config,
    start: pd.Timestamp,
    end: pd.Timestamp,
    active: set[str],
) -> tuple[pd.DataFrame, list[Network]]:
    """Create laundering networks: new shells and payer vehicles plus recruited firms.

    ``active`` is the set of ordinary firms that have standing trade
    relationships; complicit firms are recruited among them, so that their
    illicit trades are mixed into a genuine trading history.

    Returns the extended company table and the list of networks. Typologies are
    assigned round-robin first, so every typology has specialised networks.
    """
    ncfg = cfg.data.networks
    mix = cfg.data.typology_mix.to_dict()
    n_networks = int(ncfg.n_networks)
    period_days = (end - start).days
    tiers = world.countries["risk_tier"]
    shell_w = tiers.map({"low": 0.7, "medium": 1.5, "offshore": 3.0, "high": 1.0}).to_numpy(dtype=float)
    payer_w = tiers.map({"low": 0.2, "medium": 1.0, "offshore": 3.0, "high": 2.0}).to_numpy(dtype=float)
    codes = world.country_codes
    hs_w = (world.hs["ref_log_sigma"].to_numpy() + 0.15) * np.log10(world.hs["ref_price_usd"].to_numpy() * world.hs["typical_qty"].to_numpy())
    hs_codes = world.hs.index.to_numpy()

    typ_names = list(mix.keys())
    assigned: list[list[str]] = []
    for i in range(n_networks):
        k = int(rng.integers(ncfg.typologies_per_network[0], ncfg.typologies_per_network[1] + 1))
        chosen = _weighted_typologies(rng, mix, k)
        forced = typ_names[i % len(typ_names)]  # round-robin guarantees coverage
        if forced not in chosen:
            chosen[0] = forced
        assigned.append(chosen)

    group_directors = sorted({d for d in companies["director_id"] if str(d).startswith("DIR-G")})
    business_addresses = sorted({a for a in companies["address_id"] if str(a).startswith("ADR-BC")})
    agent_addresses = [f"ADR-AG{i:03d}" for i in range(1, max(2, n_networks // 2) + 1)]  # formation agents

    recruitable = companies.index[
        (companies["entity_type"] == "company")
        & (companies["registration_date"] < start - pd.Timedelta(days=730))
        & companies.index.isin(list(active))
    ].to_numpy()
    role_names = list(ncfg.shell_roles.keys())
    role_p = np.array(list(ncfg.shell_roles.values()), dtype=float)
    recruited: set[str] = set()
    new_rows: list[dict] = []
    networks: list[Network] = []
    shell_seq = 0
    payer_seq = 0

    for i in range(n_networks):
        nid = f"NET-{i + 1:03d}"
        # A network is active for `duration` days starting anywhere from before the data
        # period to just before its end; the window is then clipped to the period. This
        # keeps the density of illicit activity constant over time (no thin edges).
        months = int(rng.integers(ncfg.active_months[0], ncfg.active_months[1] + 1))
        duration = int(months * 30.4)
        founded = start + pd.Timedelta(days=int(rng.integers(-duration + 45, period_days - 45)))
        t0 = max(start, founded)
        t1 = min(end, founded + pd.Timedelta(days=duration))
        k_hs = int(rng.integers(2, 5))
        net = Network(
            network_id=nid,
            typologies=assigned[i],
            hs_codes=list(rng.choice(hs_codes, size=k_hs, replace=False, p=hs_w / hs_w.sum())),
            start=t0,
            end=t1,
        )
        sectors = tuple(sorted(set(world.hs.loc[net.hs_codes, "sector"])))

        # --- shells: young, shared mass-registration address and nominee director
        n_shell = int(rng.integers(ncfg.shells_per_network[0], ncfg.shells_per_network[1] + 1))
        shell_countries = rng.choice(codes, size=n_shell, replace=False, p=shell_w / shell_w.sum())
        addr_pool = [str(rng.choice(agent_addresses))]
        if business_addresses and rng.random() < 0.3:
            addr_pool.append(str(rng.choice(business_addresses)))  # hides among benign tenants
        nominee_pool = [f"DIR-N{i + 1:03d}"]
        if group_directors and rng.random() < 0.25:
            nominee_pool.append(str(rng.choice(group_directors)))  # nominee who also serves benign firms
        for position, c in enumerate(shell_countries):
            shell_seq += 1
            # each shell is used for a limited time; the first one from the day the network starts
            latest = max(0, (t1 - t0).days - int(ncfg.shell_active_days[0]))
            active_from = t0 + pd.Timedelta(days=0 if position == 0 else int(rng.integers(0, latest + 1)))
            active_to = min(t1, active_from + pd.Timedelta(days=int(rng.integers(ncfg.shell_active_days[0], ncfg.shell_active_days[1] + 1))))
            first_use = founded if position == 0 else active_from
            age_months = int(rng.integers(ncfg.shell_age_months_at_start[0], ncfg.shell_age_months_at_start[1] + 1))
            if rng.random() < ncfg.shelf_company_p:
                age_months = int(rng.integers(36, 97))  # aged "shelf" company
            hidden = rng.random() < ncfg.hidden_shell_p
            eid = f"ENT-S{shell_seq:04d}"
            new_rows.append(
                {
                    "entity_id": eid,
                    "name": f"{rng.choice(_SHELL_WORDS)} {rng.choice(_SHELL_NOUNS)} {shell_seq:03d} {rng.choice(_SUFFIXES)}",
                    "entity_type": "company",
                    "role": str(rng.choice(role_names, p=role_p / role_p.sum())),
                    "country": c,
                    "registration_date": (first_use - pd.Timedelta(days=int(age_months * 30.4))).normalize(),
                    "address_id": f"ADR-S{shell_seq:05d}" if hidden else str(rng.choice(addr_pool)),
                    "director_id": f"DIR-S{shell_seq:05d}" if hidden else str(rng.choice(nominee_pool)),
                    "home_bank_id": _bank_for(rng, banks, c),
                    "sectors": sectors,
                    "products": tuple(net.hs_codes),
                    "size": float(rng.lognormal(0.5, 0.6)),
                    "is_shell": True,
                    "is_complicit": False,
                    "is_payer": False,
                    "network_id": nid,
                }
            )
            net.shells.append(eid)
            net.shell_windows[eid] = (active_from, active_to)

        # --- complicit established firms recruited from the ordinary population
        n_comp = int(rng.integers(ncfg.complicit_per_network[0], ncfg.complicit_per_network[1] + 1))
        pool = [e for e in recruitable if e not in recruited]
        picks = list(rng.choice(pool, size=min(n_comp, len(pool)), replace=False))
        for eid in picks:
            recruited.add(eid)
            net.complicit.append(str(eid))

        # --- payer vehicles used for third-party settlements
        n_pay = int(rng.integers(ncfg.payers_per_network[0], ncfg.payers_per_network[1] + 1))
        for c in rng.choice(codes, size=n_pay, replace=False, p=payer_w / payer_w.sum()):
            payer_seq += 1
            eid = f"ENT-P{payer_seq:04d}"
            new_rows.append(
                {
                    "entity_id": eid,
                    "name": f"{rng.choice(_SHELL_WORDS)} Capital {payer_seq:03d} {rng.choice(_SUFFIXES)}",
                    "entity_type": "company",
                    "role": "payer",
                    "country": c,
                    "registration_date": (founded - pd.Timedelta(days=int(rng.integers(60, 1100)))).normalize(),
                    "address_id": str(rng.choice(addr_pool)) if rng.random() < 0.5 else f"ADR-P{payer_seq:05d}",
                    "director_id": str(rng.choice(nominee_pool)) if rng.random() < 0.5 else f"DIR-P{payer_seq:05d}",
                    "home_bank_id": _bank_for(rng, banks, c),
                    "sectors": (),
                    "products": (),
                    "size": float(rng.lognormal(0.5, 0.5)),
                    "is_shell": True,
                    "is_complicit": False,
                    "is_payer": True,
                    "network_id": nid,
                }
            )
            net.payers.append(eid)
        networks.append(net)

    companies = companies.copy()
    for net in networks:
        companies.loc[net.complicit, "is_complicit"] = True
        companies.loc[net.complicit, "network_id"] = net.network_id
    extra = pd.DataFrame(new_rows).set_index("entity_id")
    extra["registration_date"] = pd.to_datetime(extra["registration_date"])
    return pd.concat([companies, extra]), networks


def build_entities(rng: np.random.Generator, world: World, cfg: Config) -> Entities:
    """Ordinary entities only; laundering networks are attached by :func:`attach_networks`."""
    start = pd.Timestamp(cfg.data.start_date)
    end = pd.Timestamp(cfg.data.end_date)
    banks = build_banks(rng, world, int(cfg.data.entities.n_banks))
    lines = build_shipping_lines(rng, int(cfg.data.entities.n_shipping_lines))
    companies = build_companies(rng, world, banks, cfg, start, end)
    companies["registration_date"] = pd.to_datetime(companies["registration_date"])
    companies["region"] = world.countries.loc[companies["country"], "region"].to_numpy()
    companies["in_group"] = companies["director_id"].str.startswith("DIR-G")
    return Entities(companies=companies, banks=banks, shipping_lines=lines, networks=[])


def attach_networks(rng: np.random.Generator, world: World, ents: Entities, relationships: pd.DataFrame, cfg: Config) -> Entities:
    """Add laundering networks (shells, payer vehicles, recruited firms) to the world."""
    start = pd.Timestamp(cfg.data.start_date)
    end = pd.Timestamp(cfg.data.end_date)
    active = set(relationships["exporter_id"]) | set(relationships["importer_id"])
    companies, networks = build_networks(rng, world, ents.companies, ents.banks, cfg, start, end, active)
    companies["registration_date"] = pd.to_datetime(companies["registration_date"])
    companies["region"] = world.countries.loc[companies["country"], "region"].to_numpy()
    companies["in_group"] = companies["in_group"].fillna(False).astype(bool)
    return Entities(companies=companies, banks=ents.banks, shipping_lines=ents.shipping_lines, networks=networks)


def anonymise(rng: np.random.Generator, ents: Entities, relationships: pd.DataFrame) -> tuple[Entities, pd.DataFrame]:
    """Replace the construction-time identifiers with neutral random ones.

    While the world is being built, ids carry meaning (``ENT-S...`` is a shell,
    ``ADR-AG...`` a formation agent). Published identifiers must not, otherwise
    an identifier prefix would give the label away.
    """
    companies = ents.companies.copy()
    old = companies.index.to_numpy()
    new = np.array([f"ENT-{i:05d}" for i in rng.permutation(len(old)) + 1], dtype=object)
    mapping = dict(zip(old, new))
    companies.index = pd.Index(new, name="entity_id")
    for col, prefix in [("address_id", "ADR"), ("director_id", "DIR")]:
        values = pd.unique(companies[col])
        codes = dict(zip(values, [f"{prefix}-{i:06d}" for i in rng.permutation(len(values)) + 1]))
        companies[col] = companies[col].map(codes)
    for net in ents.networks:
        net.shell_windows = {mapping[e]: window for e, window in net.shell_windows.items()}
        net.shells = [mapping[e] for e in net.shells]
        net.complicit = [mapping[e] for e in net.complicit]
        net.payers = [mapping[e] for e in net.payers]
    rel = relationships.copy()
    rel["exporter_id"] = rel["exporter_id"].map(mapping)
    rel["importer_id"] = rel["importer_id"].map(mapping)
    out = Entities(companies=companies.sort_index(), banks=ents.banks, shipping_lines=ents.shipping_lines, networks=ents.networks)
    return out, rel
