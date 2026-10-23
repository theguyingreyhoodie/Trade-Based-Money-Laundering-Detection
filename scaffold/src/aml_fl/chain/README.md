# Blockchain audit layer (Phase 4)

```
chain/
├── hardhat/                    Hardhat 3 project (Node.js 22/24)
│   ├── contracts/FLAudit.sol   the audit contract
│   └── test/FLAudit.t.sol      Solidity tests (npx hardhat test)
└── chain_client.py             web3.py wrapper used by scripts/audit_to_chain.py (and later the dashboard)
```

Only hashes go on-chain, never raw data or model weights. Everything runs on a local Hardhat
chain (`npx hardhat node`, chain id 31337).

## FLAudit contract

| Job | Who writes | What is stored |
|---|---|---|
| Bank registry | operator (the FL server, the deployer) | `isBank[address]` |
| Round audit | operator | `globalModelHash[runId][round]`: SHA-256 of the aggregated model |
| | each bank | `updateHash[runId][round][bank]`: SHA-256 of the update it sent |
| Alert log | each bank | `alertBank[hash(txn_id + score + top SHAP features + model_version)]` |

- Every hash is written once; a second write to the same slot reverts (`AlreadyRecorded`).
- `runId` is `keccak256` of the run name, e.g. `"fl_hybrid_s42"`.
- `verifyGlobalModel(runId, round, hash)` lets anyone check a model: re-hash its weights with
  `fl.strategy.hash_parameters` and compare.

## Workflow

1. `scripts/run_fl.py` trains and writes `experiments/results/<run>/audit.json`: per round, the
   global model hash and each bank's update hash (`strategy.AuditedFedAvg`). Each bank declares
   the hash of the update it sends; the server re-hashes what it received and stops the run if
   they differ.
2. `scripts/audit_to_chain.py <run>` deploys the contract on first use (address kept in
   `hardhat/deployments/localhost.json`, not committed), registers accounts 1..5 as the banks
   `client_0..client_4`, records every global model hash (as operator) and every update hash
   (from that bank's own account), and checks the file against the chain.
3. Later, `scripts/audit_to_chain.py <run> --verify-only` shows whether `audit.json` was changed
   since it was recorded: it lists the rounds that no longer match and exits with code 1.

The hashes are written after the run instead of during training, so the FL simulation does not
depend on a running chain. The chain still makes the record tamper-evident from that point on.

## Status

- Done: contract, Solidity tests, Python client, recording and verifying global model and
  per-bank update hashes (`tests/test_chain.py` runs the client against a throwaway Hardhat
  node). Runs made before the update hashes were added only have global model hashes.
- Phase 5: alert hashes from the dashboard (`AuditChain.log_alert`).

Hardhat node accounts are public test keys. Never reuse them, or this setup, on a real network.
