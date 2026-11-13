/**
 * Hardhat config for the local permissioned-chain prototype.
 *
 * The node is a single-validator development chain. "Permissioned" is enforced
 * at the contract level: EvidenceRegistry keeps an admin-managed allowlist of
 * organisation addresses, and only those addresses can write evidence.
 *
 * No plugins are needed: Hardhat compiles the contract and runs the node, and
 * the Python side (web3.py) deploys and calls it from the compiled artifact.
 */
module.exports = {
  solidity: {
    version: "0.8.28",
    settings: {
      optimizer: { enabled: true, runs: 200 },
      evmVersion: "cancun",
    },
  },
  networks: {
    hardhat: {
      chainId: 31337,
      // 20 deterministic, pre-funded accounts. Account 0 is the registry admin;
      // the following accounts are assigned to participating organisations.
      accounts: { count: 20 },
      // Batched registrations of a few hundred hashes need more than the
      // default 30M block gas limit would comfortably allow in one block.
      blockGasLimit: 60_000_000,
    },
  },
  paths: {
    sources: "./contracts",
    artifacts: "./artifacts",
    cache: "./cache",
  },
};
