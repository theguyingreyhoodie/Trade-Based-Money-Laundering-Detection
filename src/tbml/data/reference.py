"""Public reference tables of the simulated world.

Everything here is *public knowledge* in the simulation: jurisdictions and their
risk tiers, ports and distances, the HS goods catalogue with reference prices,
and FX rates. Every participant (and every FL client) may use these tables.

Jurisdictions are fictional on purpose. The risk tiers mimic the structure of
public lists (a FATF-style high-risk list, free-trade zones / offshore centres,
transshipment hubs) without labelling any real country.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# code, name, region, lat, lon, trade weight, tier, fatf-listed, free-trade zone, hub, currency
_COUNTRIES = [
    # Northern Union: developed, low risk
    ("ALD", "Aldoria", "NOR", 55.0, 10.0, 9.0, "low", 0, 0, 0, "EUR"),
    ("BRV", "Brevia", "NOR", 50.0, 5.0, 7.0, "low", 0, 0, 0, "EUR"),
    ("CRN", "Corvania", "NOR", 48.0, 15.0, 6.0, "low", 0, 0, 0, "EUR"),
    ("DLM", "Dalmora", "NOR", 58.0, 20.0, 3.0, "low", 0, 0, 0, "EUR"),
    ("ELS", "Elsmark", "NOR", 52.0, -2.0, 6.0, "low", 0, 0, 1, "GBP"),
    ("FRH", "Fernholt", "NOR", 45.0, 22.0, 2.0, "medium", 0, 0, 0, "EUR"),
    ("LRN", "Lorne", "NOR", 60.0, -20.0, 1.0, "low", 0, 0, 0, "EUR"),
    # Eastern Rim: manufacturing
    ("GHZ", "Ghanzu", "EAS", 30.0, 115.0, 12.0, "low", 0, 0, 0, "CNY"),
    ("HKR", "Hakaro", "EAS", 35.0, 135.0, 8.0, "low", 0, 0, 0, "JPY"),
    ("JVR", "Javira", "EAS", 10.0, 105.0, 5.0, "medium", 0, 0, 0, "USD"),
    ("KSN", "Kesan Straits", "EAS", 22.0, 110.0, 6.0, "low", 0, 0, 1, "USD"),
    ("LMB", "Lumbai", "EAS", 15.0, 100.0, 4.0, "medium", 0, 0, 0, "USD"),
    ("MYL", "Myrland", "EAS", 5.0, 120.0, 3.0, "medium", 0, 0, 0, "USD"),
    # Southern Shelf: commodities
    ("NVR", "Novara", "SOU", -15.0, -50.0, 5.0, "medium", 0, 0, 0, "USD"),
    ("ORT", "Ortega", "SOU", -30.0, -60.0, 4.0, "medium", 0, 0, 0, "USD"),
    ("PLM", "Palmyra Coast", "SOU", -5.0, -40.0, 3.0, "medium", 0, 0, 0, "USD"),
    ("QRT", "Quarta", "SOU", -25.0, 25.0, 4.0, "medium", 0, 0, 0, "USD"),
    ("RSK", "Roskan", "SOU", -10.0, 35.0, 2.0, "high", 1, 0, 0, "USD"),
    ("SBL", "Sabela", "SOU", 5.0, 10.0, 3.0, "medium", 0, 0, 0, "USD"),
    ("TKR", "Takoro", "SOU", 0.0, 30.0, 2.0, "high", 1, 0, 0, "USD"),
    # Western Federation: large consumer markets
    ("UNL", "Unoland", "WES", 40.0, -95.0, 14.0, "low", 0, 0, 0, "USD"),
    ("VST", "Vestria", "WES", 50.0, -100.0, 5.0, "low", 0, 0, 0, "USD"),
    ("WMR", "Wymar", "WES", 25.0, -100.0, 5.0, "medium", 0, 0, 0, "USD"),
    ("XND", "Xandia Canal Zone", "WES", 10.0, -80.0, 2.0, "offshore", 0, 1, 1, "USD"),
    # Middle Gulf: energy and trading hubs
    ("YZD", "Port Yazdan", "MID", 25.0, 55.0, 4.0, "offshore", 0, 1, 1, "AED"),
    ("ZHR", "Zahira", "MID", 28.0, 48.0, 4.0, "medium", 0, 0, 0, "USD"),
    ("AQR", "Aqara", "MID", 20.0, 45.0, 3.0, "medium", 0, 0, 0, "USD"),
    ("BKT", "Bakhtar", "MID", 33.0, 60.0, 1.5, "high", 1, 0, 0, "USD"),
    ("CSP", "Caspia", "MID", 40.0, 50.0, 2.0, "medium", 0, 0, 0, "USD"),
    ("DRN", "Durran", "MID", 15.0, 48.0, 1.0, "high", 1, 0, 0, "USD"),
    ("KRM", "Karmin", "MID", 35.0, 30.0, 1.5, "high", 1, 0, 0, "USD"),
    # Island States: offshore centres and free zones
    ("EVR", "Everis", "ISL", 18.0, -65.0, 0.8, "offshore", 0, 1, 0, "USD"),
    ("FLR", "Florin Isles", "ISL", 12.0, -60.0, 0.6, "offshore", 0, 1, 0, "USD"),
    ("GRB", "Garuba Atoll", "ISL", 5.0, 75.0, 0.8, "offshore", 0, 1, 1, "USD"),
    ("HVN", "Haven Cay", "ISL", -20.0, 57.0, 0.7, "offshore", 0, 1, 0, "USD"),
]

#: numeric jurisdiction risk used by features and rules (public, list-based)
RISK_SCORE = {"low": 0.10, "medium": 0.35, "offshore": 0.70, "high": 1.00}

#: static FX rates to USD (the simulation has no FX volatility)
FX_TO_USD = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27, "CNY": 0.14, "JPY": 0.0067, "AED": 0.2723}

# code, description, sector, unit, median USD/unit, log-sigma, kg/unit, typical quantity
_HS = [
    ("100630", "Rice, milled", "agri", "kg", 0.55, 0.18, 1.0, 200000),
    ("090111", "Coffee, not roasted", "agri", "kg", 4.20, 0.22, 1.0, 40000),
    ("151190", "Palm oil, refined", "agri", "kg", 0.95, 0.15, 1.0, 300000),
    ("170199", "Cane sugar, refined", "agri", "kg", 0.50, 0.15, 1.0, 250000),
    ("030617", "Shrimps and prawns, frozen", "agri", "kg", 9.00, 0.25, 1.0, 20000),
    ("080390", "Bananas, fresh", "agri", "kg", 0.62, 0.15, 1.0, 180000),
    ("020230", "Beef, boneless, frozen", "agri", "kg", 6.10, 0.18, 1.0, 25000),
    ("220421", "Wine, bottled", "agri", "litre", 5.50, 0.45, 1.3, 15000),
    ("240220", "Cigarettes", "agri", "thousand", 28.0, 0.30, 1.1, 5000),
    ("270900", "Crude petroleum oil", "energy", "barrel", 78.0, 0.12, 136.0, 200000),
    ("271019", "Gas oil / diesel", "energy", "tonne", 820.0, 0.12, 1000.0, 5000),
    ("310210", "Urea fertiliser", "chemicals", "tonne", 380.0, 0.18, 1000.0, 5000),
    ("390110", "Polyethylene, primary form", "chemicals", "tonne", 1150.0, 0.12, 1000.0, 800),
    ("300490", "Medicaments, retail packs", "chemicals", "pack", 14.0, 0.60, 0.08, 50000),
    ("330499", "Cosmetics and skincare", "chemicals", "unit", 6.50, 0.50, 0.15, 40000),
    ("260111", "Iron ore", "metals", "tonne", 105.0, 0.18, 1000.0, 50000),
    ("720839", "Hot-rolled steel coil", "metals", "tonne", 640.0, 0.14, 1000.0, 2000),
    ("721420", "Steel reinforcing bar", "metals", "tonne", 590.0, 0.12, 1000.0, 3000),
    ("740311", "Copper cathodes", "metals", "tonne", 8900.0, 0.10, 1000.0, 300),
    ("760110", "Aluminium, unwrought", "metals", "tonne", 2400.0, 0.10, 1000.0, 500),
    ("732690", "Articles of steel", "metals", "kg", 3.80, 0.45, 1.0, 40000),
    ("710812", "Gold, unwrought", "precious", "gram", 64.0, 0.05, 0.001, 50000),
    ("710239", "Diamonds, worked", "precious", "carat", 1500.0, 0.70, 0.0002, 800),
    ("711319", "Gold jewellery", "precious", "gram", 78.0, 0.25, 0.001, 20000),
    ("910211", "Wrist watches", "precious", "unit", 85.0, 0.90, 0.12, 5000),
    ("520100", "Cotton, raw", "textiles", "tonne", 1900.0, 0.15, 1000.0, 400),
    ("610910", "T-shirts, cotton", "textiles", "unit", 3.20, 0.40, 0.18, 60000),
    ("620342", "Trousers, cotton", "textiles", "unit", 8.50, 0.40, 0.45, 30000),
    ("640399", "Footwear, leather", "textiles", "pair", 24.0, 0.45, 0.90, 12000),
    ("420221", "Handbags, leather", "textiles", "unit", 55.0, 0.80, 0.70, 6000),
    ("847130", "Laptop computers", "electronics", "unit", 620.0, 0.30, 2.4, 1500),
    ("847330", "Computer parts", "electronics", "unit", 45.0, 0.60, 0.40, 20000),
    ("851713", "Smartphones", "electronics", "unit", 290.0, 0.40, 0.35, 4000),
    ("852872", "Television sets", "electronics", "unit", 260.0, 0.35, 12.0, 1500),
    ("854143", "Photovoltaic modules", "electronics", "unit", 95.0, 0.25, 21.0, 5000),
    ("854231", "Integrated circuits", "electronics", "unit", 12.0, 0.80, 0.005, 100000),
    ("850760", "Lithium-ion batteries", "electronics", "unit", 38.0, 0.50, 0.90, 10000),
    ("870323", "Passenger cars, 1.5-3.0 l", "vehicles", "unit", 21000.0, 0.30, 1450.0, 60),
    ("870899", "Motor vehicle parts", "vehicles", "kg", 9.0, 0.50, 1.0, 20000),
    ("401110", "Car tyres", "vehicles", "unit", 62.0, 0.30, 9.0, 5000),
    ("841810", "Refrigerators", "consumer", "unit", 310.0, 0.30, 62.0, 800),
    ("940360", "Wooden furniture", "consumer", "unit", 140.0, 0.45, 32.0, 1500),
    ("950300", "Toys", "consumer", "unit", 4.50, 0.50, 0.35, 50000),
    ("392690", "Articles of plastics", "consumer", "kg", 4.50, 0.45, 1.0, 30000),
    ("901890", "Medical instruments", "consumer", "unit", 180.0, 0.70, 1.5, 2000),
    ("252329", "Portland cement", "building", "tonne", 68.0, 0.15, 1000.0, 10000),
    ("690721", "Ceramic tiles", "building", "m2", 7.50, 0.30, 19.0, 30000),
    ("440710", "Sawn softwood", "building", "m3", 290.0, 0.20, 480.0, 1500),
    ("480256", "Printing paper", "building", "tonne", 900.0, 0.15, 1000.0, 400),
]

SECTORS = ["agri", "energy", "chemicals", "metals", "precious", "textiles", "electronics", "vehicles", "consumer", "building"]

#: relative weight of each sector among exporters of a region (importers buy anything)
REGION_EXPORT_SECTORS = {
    "NOR": [1.0, 0.3, 1.6, 0.8, 0.8, 0.6, 1.0, 1.8, 1.3, 0.8],
    "EAS": [0.8, 0.2, 0.9, 0.9, 0.5, 2.0, 2.4, 1.0, 1.8, 0.8],
    "SOU": [2.4, 0.8, 0.6, 1.8, 1.0, 0.8, 0.2, 0.2, 0.5, 1.2],
    "WES": [1.4, 0.8, 1.4, 0.8, 0.6, 0.5, 1.0, 1.4, 1.2, 1.0],
    "MID": [0.4, 2.6, 1.4, 0.8, 1.6, 0.4, 0.3, 0.3, 0.5, 0.8],
    "ISL": [0.5, 0.3, 0.6, 0.4, 1.8, 0.8, 1.2, 0.4, 1.2, 0.3],
}

EARTH_RADIUS_NM = 3440.065
SHIP_SPEED_KNOTS = 14.0


@dataclass
class World:
    """Public reference data plus fast lookup arrays used by the generator."""

    countries: pd.DataFrame  # indexed by country code
    ports: pd.DataFrame  # indexed by port code
    hs: pd.DataFrame  # indexed by HS code
    fx: dict[str, float]
    distance_nm: np.ndarray  # [n_countries, n_countries]
    hub_idx: np.ndarray  # country indices that are transshipment hubs

    @property
    def country_codes(self) -> np.ndarray:
        return self.countries.index.to_numpy()

    def country_index(self, codes) -> np.ndarray:
        return self.countries.index.get_indexer(pd.Index(codes))

    def hs_index(self, codes) -> np.ndarray:
        return self.hs.index.get_indexer(pd.Index(codes))


def haversine_nm(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in nautical miles (vectorised)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_NM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_countries() -> pd.DataFrame:
    cols = ["country", "name", "region", "lat", "lon", "trade_weight", "risk_tier", "fatf_listed", "is_ftz", "is_hub", "currency"]
    df = pd.DataFrame(_COUNTRIES, columns=cols).set_index("country")
    df["risk_score"] = df["risk_tier"].map(RISK_SCORE)
    for col in ["fatf_listed", "is_ftz", "is_hub"]:
        df[col] = df[col].astype(int)
    return df


def build_ports(countries: pd.DataFrame) -> pd.DataFrame:
    """One main seaport per jurisdiction; hubs are the transshipment ports."""
    ports = pd.DataFrame(
        {
            "port": [f"{code}01" for code in countries.index],
            "port_name": [f"Port of {name}" for name in countries["name"]],
            "country": countries.index,
            "lat": countries["lat"].to_numpy(),
            "lon": countries["lon"].to_numpy(),
            "is_hub": countries["is_hub"].to_numpy(),
        }
    ).set_index("port")
    return ports


def build_hs_catalogue() -> pd.DataFrame:
    cols = ["hs_code", "description", "sector", "unit", "ref_price_usd", "ref_log_sigma", "kg_per_unit", "typical_qty"]
    df = pd.DataFrame(_HS, columns=cols).set_index("hs_code")
    df["chapter"] = df.index.str[:2]
    df["heading"] = df.index.str[:4]
    return df


def build_world() -> World:
    """Assemble the public reference tables (deterministic, seed-independent)."""
    countries = build_countries()
    ports = build_ports(countries)
    hs = build_hs_catalogue()
    lat = countries["lat"].to_numpy()
    lon = countries["lon"].to_numpy()
    dist = haversine_nm(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    hub_idx = np.flatnonzero(countries["is_hub"].to_numpy() == 1)
    return World(countries=countries, ports=ports, hs=hs, fx=dict(FX_TO_USD), distance_nm=dist, hub_idx=hub_idx)


def published_hs_reference(world: World, rng: np.random.Generator, error: float) -> pd.DataFrame:
    """HS reference prices as a detector would know them.

    Real reference prices are estimated from trade statistics, so they carry
    estimation error. ``error`` is the relative noise on the median; the
    dispersion estimate gets about three times that noise.
    """
    hs = world.hs
    ref = pd.DataFrame(
        {
            "hs_code": hs.index,
            "description": hs["description"].to_numpy(),
            "chapter": hs["chapter"].to_numpy(),
            "sector": hs["sector"].to_numpy(),
            "unit": hs["unit"].to_numpy(),
            "ref_price_usd": hs["ref_price_usd"].to_numpy() * np.exp(rng.normal(0, error, len(hs))),
            "ref_log_sigma": hs["ref_log_sigma"].to_numpy() * np.exp(rng.normal(0, 3 * error, len(hs))),
            "kg_per_unit": hs["kg_per_unit"].to_numpy(),
        }
    )
    ref["ref_price_usd"] = ref["ref_price_usd"].round(6)
    ref["ref_log_sigma"] = ref["ref_log_sigma"].round(4)
    return ref


def route_distance_nm(world: World, path_idx: list[int]) -> float:
    """Length of a route given as a list of country indices (origin..destination)."""
    return float(sum(world.distance_nm[a, b] for a, b in zip(path_idx[:-1], path_idx[1:])))


def transit_days(distance_nm: np.ndarray, n_hops: np.ndarray) -> np.ndarray:
    """Expected sailing time in days: distance at service speed plus port time per hop."""
    return distance_nm / (SHIP_SPEED_KNOTS * 24.0) + 2.0 * n_hops + 1.0
