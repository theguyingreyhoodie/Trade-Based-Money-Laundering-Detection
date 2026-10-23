// Hardhat 3 config. `npx hardhat build` compiles contracts/, `npx hardhat test` runs the
// Solidity tests in test/, `npx hardhat node` starts a local chain on 127.0.0.1:8545.
export default {
  solidity: {
    version: "0.8.28",
    settings: { optimizer: { enabled: true, runs: 200 } },
  },
};
