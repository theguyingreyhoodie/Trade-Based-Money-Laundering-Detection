"""Per-client data for federated learning.

``prepare_clients`` deals every institution its own records and lets each one
build its features from that local view only. It is the only place where the
simulation touches all the data at once, in the role of "the world". Everything
after it (client code, server code) reads one client's file through
:func:`load_client` and nothing else.

Local view of client k:

* its own trades: invoices, the shipments and customs declarations they refer
  to, and their payments (complete linked records of its customers' trades);
* registry attributes of the entities that appear in those trades;
* the public reference tables (HS reference prices, jurisdiction risk, ports).

Context indicators (entity history, trade graph, B/L sharing) are therefore
computed on the client's own sub-graph. Patterns that span institutions, such
as a duplicate invoice financed at another bank, are invisible locally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tbml.config import Config
from tbml.data.generator import load_dataset
from tbml.data.typologies import TYPOLOGY_CLASSES
from tbml.features.build import FeatureSet, Reference, build_features
from tbml.features.link import link_dataset
from tbml.fl.partition import Scenario, partition_summary, partition_trades
from tbml.models.split import assign_split
from tbml.utils import derive_seed, get_logger, load_json, save_json

log = get_logger(__name__)


def scenario_dir(cfg: Config, seed: int, scenario: Scenario) -> Path:
    return cfg.seed_dir("artifacts_dir", seed, "fl", scenario.name)


def client_path(root: Path, client: int) -> Path:
    return Path(root) / "clients" / f"client_{client}.npz"


def load_client(root: str | Path, client: int) -> FeatureSet:
    """Feature set of one client. This is the only data a client process reads."""
    return FeatureSet.load(client_path(Path(root), int(client)))


def prepare_clients(cfg: Config, seed: int, scenario: Scenario, force: bool = False) -> dict[str, Any]:
    """Partition the trades of ``seed`` and build each client's local features."""
    root = scenario_dir(cfg, seed, scenario)
    summary_path = root / "partition.json"
    if summary_path.is_file() and not force and all(client_path(root, c).is_file() for c in range(scenario.n_clients)):
        return load_json(summary_path)

    ds = load_dataset(cfg.seed_dir("data_dir", seed))
    linked = link_dataset(ds)
    labels = ds.tables["labels"].set_index("trade_id").reindex(linked["trade_id"])
    typology = labels["typology"].map({name: i for i, name in enumerate(TYPOLOGY_CLASSES)}).to_numpy(dtype=np.int64)
    owner = partition_trades(typology, scenario, derive_seed(seed, "partition", scenario.name), min_share=float(cfg.fl.experiments.min_client_share))
    split = assign_split(linked["invoice_date"], cfg)

    ref = Reference.from_tables(ds.tables)
    kyc = ds.tables["kyc_entities"]
    banks = ds.tables["banks"]
    for client in range(scenario.n_clients):
        own = linked[owner == client]
        payers = own["payers"].str.split("|").explode()
        known = set(own["exporter_id"]) | set(own["importer_id"]) | set(payers[payers.notna() & (payers != "")])
        own_labels = ds.tables["labels"][ds.tables["labels"]["trade_id"].isin(own["trade_id"])]
        fs = build_features(own, kyc[kyc["entity_id"].isin(known)], banks, ref, cfg, own_labels)
        fs.save(client_path(root, client))

    summary = partition_summary(owner, typology, split)
    summary.update({"scenario": scenario.to_dict(), "seed": seed, "n_trades": int(len(linked))})
    save_json(summary, summary_path)
    pd.DataFrame({"trade_id": linked["trade_id"], "client": owner}).to_csv(root / "owners.csv", index=False)
    sizes = [c["n"] for c in summary["clients"]]
    log.info("seed %d %s: client sizes %s, illicit rate %.3f-%.3f", seed, scenario.name, sizes, *summary["illicit_rate_range"])
    return summary
