// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/// @title FLAudit: tamper-evident log of a federated anti-money-laundering consortium.
/// @notice Stores only hashes (SHA-256 of model weights or alerts), never data or weights.
///         Anyone holding a model or an alert can re-hash it and check it against this log.
///   - Bank registry: the operator (the FL server) registers bank addresses.
///   - Round audit:   the operator records the global model hash of every FL round; each bank
///                    may record the hash of the update it sent in that round.
///   - Alert log:     a bank records hash(txn_id + score + top features + model_version).
///   Nothing can be overwritten: every hash is written once.
contract FLAudit {
    address public immutable operator;
    mapping(address => bool) public isBank;

    // runId (e.g. keccak256("fl_hybrid_s42")) => round => hash
    mapping(bytes32 => mapping(uint256 => bytes32)) public globalModelHash;
    // runId => round => bank => hash
    mapping(bytes32 => mapping(uint256 => mapping(address => bytes32))) public updateHash;
    // alert hash => bank that logged it (zero address: not logged)
    mapping(bytes32 => address) public alertBank;

    event BankRegistered(address indexed bank);
    event GlobalModelRecorded(bytes32 indexed runId, uint256 indexed round, bytes32 modelHash);
    event UpdateRecorded(bytes32 indexed runId, uint256 indexed round, address indexed bank, bytes32 updateHash);
    event AlertLogged(bytes32 indexed alertHash, address indexed bank);

    error NotOperator();
    error NotBank();
    error AlreadyRecorded();
    error EmptyHash();

    modifier onlyOperator() {
        if (msg.sender != operator) revert NotOperator();
        _;
    }

    modifier onlyBank() {
        if (!isBank[msg.sender]) revert NotBank();
        _;
    }

    constructor() {
        operator = msg.sender;
    }

    function registerBank(address bank) external onlyOperator {
        if (isBank[bank]) revert AlreadyRecorded();
        isBank[bank] = true;
        emit BankRegistered(bank);
    }

    function recordGlobalModel(bytes32 runId, uint256 round, bytes32 modelHash) external onlyOperator {
        if (modelHash == bytes32(0)) revert EmptyHash();
        if (globalModelHash[runId][round] != bytes32(0)) revert AlreadyRecorded();
        globalModelHash[runId][round] = modelHash;
        emit GlobalModelRecorded(runId, round, modelHash);
    }

    function recordUpdate(bytes32 runId, uint256 round, bytes32 hash) external onlyBank {
        if (hash == bytes32(0)) revert EmptyHash();
        if (updateHash[runId][round][msg.sender] != bytes32(0)) revert AlreadyRecorded();
        updateHash[runId][round][msg.sender] = hash;
        emit UpdateRecorded(runId, round, msg.sender, hash);
    }

    function logAlert(bytes32 alertHash) external onlyBank {
        if (alertHash == bytes32(0)) revert EmptyHash();
        if (alertBank[alertHash] != address(0)) revert AlreadyRecorded();
        alertBank[alertHash] = msg.sender;
        emit AlertLogged(alertHash, msg.sender);
    }

    /// @return true if `modelHash` is the recorded global model of that run and round.
    function verifyGlobalModel(bytes32 runId, uint256 round, bytes32 modelHash) external view returns (bool) {
        return modelHash != bytes32(0) && globalModelHash[runId][round] == modelHash;
    }
}
