// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title EvidenceRegistry
/// @notice Permissioned, append-only registry of evidence fingerprints for the
///         TBML detection pipeline. It stores SHA-256 hashes plus minimal audit
///         metadata (who, what kind of document, an opaque reference id, when).
///         Raw trade documents, model weights and alert contents stay off-chain.
/// @dev    "Permissioned" is enforced here: an admin manages an allowlist of
///         organisation addresses and only allowlisted addresses can write.
///         Records are immutable. A hash can be registered once; there is no
///         update or delete function.
contract EvidenceRegistry {
    struct Organisation {
        string name;
        bool active;
        uint64 registeredAt;
    }

    /// @dev submitter (20 bytes) + two uint48 timestamps pack into one slot.
    struct Evidence {
        address submitter;
        uint48 clientTs; // event time claimed by the submitter (off-chain clock)
        uint48 blockTs; // block time when the record was accepted
        string docType; // e.g. "invoice", "model_update", "alert", "doc_batch"
        string refId; // opaque off-chain reference, never raw document fields
    }

    address public admin;
    uint256 public evidenceCount;
    uint256 public organisationCount;

    mapping(address => Organisation) private organisations;
    mapping(bytes32 => Evidence) private evidence;

    event AdminTransferred(address indexed previousAdmin, address indexed newAdmin);
    event OrganisationRegistered(address indexed org, string name);
    event OrganisationRevoked(address indexed org);
    event EvidenceRegistered(
        bytes32 indexed hash,
        address indexed submitter,
        string docType,
        string refId,
        uint256 clientTs,
        uint256 blockTs
    );
    /// @notice Emitted when a batch contains a hash that is already on record.
    ///         Useful as an audit signal (e.g. one document presented twice).
    event DuplicateEvidence(bytes32 indexed hash, address indexed submitter, string refId);

    error NotAdmin();
    error NotRegisteredOrganisation(address caller);
    error AlreadyRegistered(bytes32 hash);
    error EmptyHash();
    error LengthMismatch();
    error ZeroAddress();

    modifier onlyAdmin() {
        if (msg.sender != admin) revert NotAdmin();
        _;
    }

    modifier onlyOrganisation() {
        if (!organisations[msg.sender].active) revert NotRegisteredOrganisation(msg.sender);
        _;
    }

    constructor() {
        admin = msg.sender;
        emit AdminTransferred(address(0), msg.sender);
    }

    // ----------------------------------------------------------------- admin

    function transferAdmin(address newAdmin) external onlyAdmin {
        if (newAdmin == address(0)) revert ZeroAddress();
        emit AdminTransferred(admin, newAdmin);
        admin = newAdmin;
    }

    function registerOrganisation(address org, string calldata name) external onlyAdmin {
        if (org == address(0)) revert ZeroAddress();
        if (!organisations[org].active) organisationCount += 1;
        organisations[org] = Organisation({name: name, active: true, registeredAt: uint64(block.timestamp)});
        emit OrganisationRegistered(org, name);
    }

    function revokeOrganisation(address org) external onlyAdmin {
        if (organisations[org].active) {
            organisations[org].active = false;
            organisationCount -= 1;
            emit OrganisationRevoked(org);
        }
    }

    function isOrganisation(address org) external view returns (bool) {
        return organisations[org].active;
    }

    function organisationName(address org) external view returns (string memory) {
        return organisations[org].name;
    }

    // ----------------------------------------------------------------- write

    /// @notice Register one evidence hash. Reverts if the hash is already on record.
    function registerEvidence(
        bytes32 hash,
        string calldata docType,
        string calldata refId,
        uint256 ts
    ) external onlyOrganisation {
        if (hash == bytes32(0)) revert EmptyHash();
        if (evidence[hash].submitter != address(0)) revert AlreadyRegistered(hash);
        _store(hash, docType, refId, ts);
    }

    /// @notice Register many hashes of one document type in a single transaction.
    ///         Hashes already on record are skipped and reported through
    ///         DuplicateEvidence, so one stale item cannot block a whole batch.
    /// @return stored number of new records written
    function registerEvidenceBatch(
        bytes32[] calldata hashes,
        string calldata docType,
        string[] calldata refIds,
        uint256 ts
    ) external onlyOrganisation returns (uint256 stored) {
        if (hashes.length != refIds.length) revert LengthMismatch();
        for (uint256 i = 0; i < hashes.length; i++) {
            bytes32 h = hashes[i];
            if (h == bytes32(0)) revert EmptyHash();
            if (evidence[h].submitter != address(0)) {
                emit DuplicateEvidence(h, msg.sender, refIds[i]);
                continue;
            }
            _store(h, docType, refIds[i], ts);
            stored += 1;
        }
    }

    function _store(bytes32 hash, string calldata docType, string calldata refId, uint256 ts) private {
        evidence[hash] = Evidence({
            submitter: msg.sender,
            clientTs: uint48(ts),
            blockTs: uint48(block.timestamp),
            docType: docType,
            refId: refId
        });
        evidenceCount += 1;
        emit EvidenceRegistered(hash, msg.sender, docType, refId, ts, block.timestamp);
    }

    // ------------------------------------------------------------------ read

    /// @notice Look up a hash. `exists` is false if it was never registered.
    function verify(bytes32 hash)
        external
        view
        returns (
            bool exists,
            address submitter,
            string memory docType,
            string memory refId,
            uint256 clientTs,
            uint256 blockTs
        )
    {
        Evidence storage e = evidence[hash];
        exists = e.submitter != address(0);
        return (exists, e.submitter, e.docType, e.refId, e.clientTs, e.blockTs);
    }
}
