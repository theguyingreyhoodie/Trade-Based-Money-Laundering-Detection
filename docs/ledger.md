# Blockchain evidence layer

Code: `src/tbml/ledger/`, contract: `chain/contracts/EvidenceRegistry.sol`.
Scripts: `ledger_register.py`, `ledger_tamper_demo.py`, `ledger_benchmark.py`, `chain.py`.

The evidence layer runs alongside the detection pipeline. It does not take part in detection; it
makes the pipeline's inputs and outputs **tamper-evident and attributable**: who registered what,
and when, and has it changed since.

## 1. What is registered

| Evidence | When | Registered by | On the ledger |
|---|---|---|---|
| Trade documents (invoice, shipment, payment, customs declaration) | at ingestion (Step 1) | the owning organisation | Merkle root of a batch of 500 document hashes |
| Client model update | every FL round | that client (`fl_client_<k>`) | SHA-256 of the update as transmitted |
| Global model | every FL round | `fl_aggregator` | SHA-256 of the aggregated model |
| Alert | when raised (Step 6) | the institution whose trade it is | SHA-256 of the alert, salted |

Only a 32-byte hash, a document type, an opaque reference id and two timestamps are stored.
Raw documents, model weights and alert contents stay off-chain with their owners. A test checks
that no trade or entity identifier appears on the ledger.

## 2. Hashing

`ledger/hashing.py`. SHA-256 over **canonical JSON**, so that two parties holding the same
document compute the same hash:

- keys sorted, no insignificant whitespace, UTF-8;
- integers as integers; floats rounded to 6 decimals, and a whole-number float written as an
  integer (`1200.0` and `1200` hash alike); NaN and missing values as `null`;
- dates as `YYYY-MM-DD`; strings stripped of surrounding whitespace.

Model updates are hashed over the raw array bytes together with each array's name, dtype and
shape. The FL server recomputes the hash of every update it receives and refuses an update that
does not match the hash its client reported.

Alerts carry a random 128-bit salt before hashing. An alert is short and partly guessable
(trade id, score); the salt prevents anyone from testing guesses against the public hash.
Individual document hashes are never published (only batch roots are), so they need no salt.

## 3. Merkle batching

`ledger/merkle.py`, `ledger/evidence.py`. One transaction per document does not scale: 252,546
documents at about 103,000 gas each would be 26 billion gas. Instead each owner commits a batch of
500 document hashes with a single hash, the Merkle root (RFC 6962 construction: distinct prefixes
for leaves and inner nodes).

To verify one document later:

1. recompute its hash from the document as presented now;
2. rebuild the batch root from that hash and the sibling hashes on its path (9 hashes for a batch
   of 500);
3. ask the ledger whether **that root** was registered, by whom and when.

The off-chain index (SQLite) that remembers each document's batch and position is a convenience,
not a source of trust. Verification never reads the document's hash from the index. Changing a
document, or the index, or both, produces a root the ledger has never seen. Corrupting the index
can at worst block a verification; it cannot make a forgery pass (tested).

## 4. The contract

`EvidenceRegistry.sol` (Solidity 0.8.28, 4.2 kB of bytecode, 980,755 gas to deploy):

- `registerOrganisation(address, name)`, `revokeOrganisation(address)`, `transferAdmin(address)`:
  admin only.
- `registerEvidence(bytes32 hash, string docType, string refId, uint256 ts)`: registered
  organisations only; reverts if the hash is already on record.
- `registerEvidenceBatch(bytes32[] hashes, string docType, string[] refIds, uint256 ts)`: same
  for many hashes; a hash already on record is skipped and reported with a `DuplicateEvidence`
  event, so one stale item cannot block a batch.
- `verify(bytes32 hash)` returns `(exists, submitter, docType, refId, clientTs, blockTs)`.
- Events `EvidenceRegistered`, `DuplicateEvidence`, `OrganisationRegistered`,
  `OrganisationRevoked`, `AdminTransferred` form the audit trail.

**Permissioned** means: the contract keeps an admin-managed allowlist of organisation addresses
and only those addresses can write. Anyone can read. Records are immutable: there is no update
or delete function, and a hash can be registered once. `ts` is the event time claimed by the
submitter; the block timestamp is recorded next to it.

## 5. Backends behind one interface

`ledger/base.py::EvidenceLedger` is the only thing the pipeline talks to.

| Backend | What it is | Used when |
|---|---|---|
| `evm` (`ledger/evm.py`) | the contract on a local Hardhat node, called with web3.py; account 0 is the admin, each organisation writes from its own account | Node.js and Hardhat are installed (default) |
| `python` (`ledger/pyledger.py`) | an append-only JSON-lines log; every entry holds the hash of the previous one and a secp256k1 signature; only the admin key can add organisations and only registered organisations can add evidence | no Node.js, and in the test suite |

`ledger.backend: auto` picks `evm` if a node answers at `ledger.rpc_url` or can be started, and
falls back to `python` otherwise. Both backends pass the same tests (allowlist, immutability,
batching, audit trail, tamper detection).

**Restarts.** A Hardhat node keeps its chain in memory. The client keeps an off-chain journal of
what it registered; if the chain is found empty, the journal is registered again on a fresh
contract (`scripts/chain.py restore`, also automatic). Hashes, organisations and event times are
preserved; transaction ids and block times are new. A real permissioned network persists by
itself and has no such step.

## 6. Verification demo

`scripts/ledger_tamper_demo.py` (and `tests/test_ledger.py`):

| Case | Result |
|---|---|
| invoice as registered | verified |
| invoice value raised by 10% | rejected: content differs from what was registered |
| same, and the attacker rewrites the off-chain index to match | rejected: index does not match the ledger |
| global model of the selected round | verified, registered by `fl_aggregator` |
| one weight changed by 0.001 | rejected: hash is not on the ledger |
| alert as raised / score lowered afterwards | verified / rejected |

## 7. Measurements

`results/ledger/benchmark.json`, one laptop (Apple M5), local single-validator node. Latency on
an auto-mining development node is a lower bound; the "1 s blocks" row shows what a block
interval does.

| Measure | EVM (Hardhat) | Pure Python |
|---|---|---|
| single registration, latency p50 / p95 | 6.3 ms / 6.4 ms | 2.3 ms / 2.4 ms |
| single registration, throughput | 146 records/s | 427 records/s |
| cost per record, single | 103,088 gas | 573 bytes |
| batch of 10 / 50 / 100 per write, throughput | 1,266 / 2,627 / 2,793 records/s | 4,047 / 14,297 / 21,135 records/s |
| cost per record in a batch of 100 | 74,209 gas | 105 bytes |
| Merkle root per 500 documents, throughput | 68,381 documents/s | 202,043 documents/s |
| cost per document with Merkle batching | 206 gas | 1.2 bytes |
| ledger storage per record | 96 bytes (3 slots) | 106 bytes |
| ledger storage per document with Merkle batching | 0.19 bytes | 1.2 bytes |
| read (`verify`) latency p50 | 1.6 ms | < 0.01 ms |
| with 1 s blocks: latency of one registration | 1.0 s (one block) | n/a |
| with 1 s blocks: 100 registrations submitted together | all final after 1.1 s, in one block | n/a |

Gas and storage are deterministic. Timings vary from run to run on a laptop: an earlier run of
the same benchmark measured 8.9 ms (p50) and 100 records/s for single EVM registrations, and
579 / 1,706 / 1,960 records/s for the three batch sizes.

Pipeline totals on seed 0: 252,546 trade documents anchored as 508 roots in 6.8 s (2.7 s of it
hashing); 120 model hashes for a 30-round, 3-client run at 7.8 ms p50 and 142,566 gas each;
426 alerts at 6.5 ms p50 and 103,030 gas each.

Batches are capped at 100 hashes per transaction because recent EVM rules limit a transaction
to about 16.7 million gas.

## 8. Limitations

- **Not a distributed network.** The EVM backend is one development node with one validator; the
  Python backend is one file. Neither has consensus, replication or Byzantine fault tolerance.
  A deployment would run the same contract on a permissioned network (for example Hyperledger
  Besu with IBFT/QBFT) with one node per organisation.
- **Integrity, not truth.** A hash proves a document has not changed since registration. It says
  nothing about whether the document was true when registered ("garbage in, garbage forever").
- **Admin is a single key.** Whoever holds it controls the allowlist. A deployment would use a
  multi-signature or governance contract.
- **Metadata is visible** to every reader: who registers how much and when, and the round
  structure of a federated run.
- **Keys.** Organisation keys are the node's unlocked development accounts (EVM) or derived
  deterministically from a name (Python). Neither is acceptable outside a prototype.
- **Block time on a development node is not wall-clock time.** Every record carries two
  timestamps: the event time reported by the submitting organisation and the block time set by
  the chain. On the Hardhat node used here the two diverged in both directions. When blocks are
  mined faster than one per second, each block is still stamped one second after the previous
  one (observed: consecutive blocks 1 s apart while mined about 6 ms apart), so after the
  benchmarks the block time ran ahead: the alerts' blocks are stamped about 2 h 35 min after
  their real registration time. Later the node's clock fell behind real time (2 h 40 min behind
  after about 22 hours on a laptop that slept repeatedly; the cause of this part was not pinned
  down). The dashboard therefore shows both times, and the event time is the one to read. On a
  real permissioned network, block time comes from the validators' clocks and blocks are paced.
- **Gas has no price here.** Gas is reported as a measure of work and state growth; a
  permissioned chain typically sets the gas price to zero.
