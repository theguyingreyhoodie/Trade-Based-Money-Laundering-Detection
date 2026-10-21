# Blockchain audit layer (Phase 4)

Planned layout:

```
chain/
├── hardhat/                 # `npx hardhat init` here (Node.js project)
│   └── contracts/FLAudit.sol
└── chain_client.py          # web3.py wrapper used by the FL code and the dashboard
```

The contract does three jobs (see section 06 of `docs/AML_FL_Implementation_Plan.pdf`):

- **Bank registry.** Only registered bank addresses can write.
- **FL round audit.** Records each bank's update hash and the server's global-model hash for every round.
- **Alert log.** Stores `hash(txn_id + score + top SHAP features + model_version)` with a timestamp.

Only hashes go on-chain, never raw data or model weights.

Use Hardhat or Anvil as the local chain. Ganache and Truffle are discontinued.
