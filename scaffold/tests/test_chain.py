"""FLAudit through the Python client, against a throwaway local Hardhat node.

Skipped when Node.js / Hardhat or the compiled contract is missing
(cd src/aml_fl/chain/hardhat && npm install && npx hardhat build).
"""
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("web3")
from aml_fl.chain.chain_client import ARTIFACT, HARDHAT_DIR, AuditChain, connect, run_key, to_bytes32  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("npx") is None or not ARTIFACT.exists() or not (HARDHAT_DIR / "node_modules").exists(),
    reason="needs Node.js, npm install and npx hardhat build in src/aml_fl/chain/hardhat")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def rpc():
    port = _free_port()
    proc = subprocess.Popen(["npx", "hardhat", "node", "--port", str(port)], cwd=HARDHAT_DIR,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                connect(url)
                break
            except ConnectionError:
                time.sleep(0.25)
        else:
            pytest.fail("Hardhat node did not start")
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture
def chain(rpc):
    return AuditChain.deploy(connect(rpc))


def _audit(n=3):
    from hashlib import sha256
    return [{"round": r, "hash": sha256(f"model {r}".encode()).hexdigest()} for r in range(1, n + 1)]


def test_record_run_then_verify_detects_tampering(chain):
    audit = _audit()
    assert chain.record_run("fl_hybrid_s42", audit) == 3
    assert chain.record_run("fl_hybrid_s42", audit) == 0              # idempotent
    assert chain.verify_run("fl_hybrid_s42", audit) == []
    tampered = [dict(e) for e in audit]
    tampered[1]["hash"] = _audit(4)[3]["hash"]
    assert chain.verify_run("fl_hybrid_s42", tampered) == [2]         # the changed round
    assert chain.verify_run("fl_naive_s42", audit) == [1, 2, 3]        # runs are separate
    with pytest.raises(ValueError):
        chain.record_run("fl_hybrid_s42", tampered)                   # no overwriting


def test_model_hash_from_fl_roundtrips(chain):
    pytest.importorskip("flwr")
    import numpy as np
    from flwr.app import ArrayRecord

    from aml_fl.fl.strategy import hash_parameters

    digest = hash_parameters(ArrayRecord(numpy_ndarrays=[np.ones((2, 3), dtype=np.float32)]))
    chain.record_global_model("run", 1, digest)
    assert chain.global_model_hash("run", 1) == digest
    assert chain.verify_global_model("run", 1, digest)


def test_only_registered_banks_write_updates_and_alerts(chain):
    accounts = chain.w3.eth.accounts
    bank, outsider = accounts[1], accounts[9]
    chain.register_bank(bank)
    h = _audit(1)[0]["hash"]
    chain.record_update("run", 1, h, bank)
    chain.log_alert(h, bank)
    assert chain.alert_bank(h) == bank
    with pytest.raises(Exception):
        chain.log_alert(_audit(2)[1]["hash"], outsider)
    with pytest.raises(Exception):                                    # banks cannot act as operator
        chain._send(chain.contract.functions.recordGlobalModel(run_key("run"), 2, to_bytes32(h)), bank)


def test_get_chain_reuses_deployment(rpc, tmp_path):
    from audit_to_chain import get_chain

    dep = tmp_path / "localhost.json"
    a = get_chain(rpc, 5, dep)
    b = get_chain(rpc, 5, dep)
    assert a.address == b.address
    assert all(a.is_bank(x) for x in a.w3.eth.accounts[1:6]) and not a.is_bank(a.w3.eth.accounts[6])
