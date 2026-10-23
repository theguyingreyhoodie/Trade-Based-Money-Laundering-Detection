// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import {FLAudit} from "../contracts/FLAudit.sol";

/// Stands in for a bank (or an outsider): a separate address that calls the contract.
contract Actor {
    function recordUpdate(FLAudit a, bytes32 runId, uint256 round, bytes32 h) external {
        a.recordUpdate(runId, round, h);
    }

    function logAlert(FLAudit a, bytes32 h) external {
        a.logAlert(h);
    }

    function recordGlobalModel(FLAudit a, bytes32 runId, uint256 round, bytes32 h) external {
        a.recordGlobalModel(runId, round, h);
    }
}

/// Run with `npx hardhat test`. This contract deploys FLAudit, so it is the operator.
contract FLAuditTest {
    FLAudit audit;
    Actor bank;
    Actor outsider;
    bytes32 constant RUN = keccak256("fl_hybrid_s42");
    bytes32 constant H1 = keccak256("model round 1");
    bytes32 constant H2 = keccak256("model round 2");

    function setUp() public {
        audit = new FLAudit();
        bank = new Actor();
        outsider = new Actor();
        audit.registerBank(address(bank));
    }

    function _reverts(bytes memory call, bytes4 expected) internal returns (bool) {
        (bool ok, bytes memory err) = address(this).call(call);
        return !ok && err.length >= 4 && bytes4(err) == expected;
    }

    // --- helpers called through _reverts (so their revert can be inspected) ---
    function asOutsiderRecordUpdate() external { outsider.recordUpdate(audit, RUN, 1, H1); }
    function asOutsiderLogAlert() external { outsider.logAlert(audit, H1); }
    function asBankRecordGlobal() external { bank.recordGlobalModel(audit, RUN, 1, H1); }
    function recordGlobal(uint256 round, bytes32 h) external { audit.recordGlobalModel(RUN, round, h); }
    function bankUpdate(bytes32 h) external { bank.recordUpdate(audit, RUN, 1, h); }
    function bankAlert(bytes32 h) external { bank.logAlert(audit, h); }
    function register(address a) external { audit.registerBank(a); }

    function test_OperatorIsDeployer() public view {
        require(audit.operator() == address(this), "operator");
        require(audit.isBank(address(bank)) && !audit.isBank(address(outsider)), "registry");
    }

    function test_RecordAndVerifyGlobalModel() public {
        audit.recordGlobalModel(RUN, 1, H1);
        audit.recordGlobalModel(RUN, 2, H2);
        require(audit.globalModelHash(RUN, 1) == H1, "stored");
        require(audit.verifyGlobalModel(RUN, 1, H1), "verifies");
        require(!audit.verifyGlobalModel(RUN, 1, H2), "wrong hash");
        require(!audit.verifyGlobalModel(RUN, 3, bytes32(0)), "unrecorded round");
        require(!audit.verifyGlobalModel(keccak256("other run"), 1, H1), "other run");
    }

    function test_GlobalModelCannotBeOverwritten() public {
        audit.recordGlobalModel(RUN, 1, H1);
        require(_reverts(abi.encodeCall(this.recordGlobal, (1, H2)), FLAudit.AlreadyRecorded.selector), "overwrite");
        require(audit.globalModelHash(RUN, 1) == H1, "unchanged");
    }

    function test_EmptyHashRejected() public {
        require(_reverts(abi.encodeCall(this.recordGlobal, (1, bytes32(0))), FLAudit.EmptyHash.selector), "global");
        require(_reverts(abi.encodeCall(this.bankUpdate, (bytes32(0))), FLAudit.EmptyHash.selector), "update");
        require(_reverts(abi.encodeCall(this.bankAlert, (bytes32(0))), FLAudit.EmptyHash.selector), "alert");
    }

    function test_OnlyOperatorWritesGlobalModelAndRegistry() public {
        require(_reverts(abi.encodeCall(this.asBankRecordGlobal, ()), FLAudit.NotOperator.selector), "bank global");
        require(_reverts(abi.encodeCall(this.register, (address(bank))), FLAudit.AlreadyRecorded.selector), "twice");
    }

    function test_BankRecordsUpdateOnce() public {
        bank.recordUpdate(audit, RUN, 1, H1);
        require(audit.updateHash(RUN, 1, address(bank)) == H1, "stored");
        require(_reverts(abi.encodeCall(this.bankUpdate, (H2)), FLAudit.AlreadyRecorded.selector), "overwrite");
    }

    function test_OnlyBanksWriteUpdatesAndAlerts() public {
        require(_reverts(abi.encodeCall(this.asOutsiderRecordUpdate, ()), FLAudit.NotBank.selector), "update");
        require(_reverts(abi.encodeCall(this.asOutsiderLogAlert, ()), FLAudit.NotBank.selector), "alert");
    }

    function test_AlertLoggedOnceWithItsBank() public {
        bank.logAlert(audit, H1);
        require(audit.alertBank(H1) == address(bank), "bank");
        require(audit.alertBank(H2) == address(0), "unlogged");
        require(_reverts(abi.encodeCall(this.bankAlert, (H1)), FLAudit.AlreadyRecorded.selector), "twice");
    }
}
