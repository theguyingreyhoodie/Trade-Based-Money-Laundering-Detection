"""The ``EvidenceLedger`` interface (objective 4).

The pipeline talks to the evidence layer only through this interface, so the
backend can be swapped: the Solidity ``EvidenceRegistry`` contract on a local
EVM node (``ledger.evm``) or a pure-Python signed hash chain (``ledger.pyledger``).

What goes on the ledger: a SHA-256 hash, a document type, an opaque reference
id, the submitting organisation and timestamps. Raw documents, model weights
and alert contents never do.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


class LedgerError(RuntimeError):
    """A write was refused (unknown organisation, malformed hash, ...)."""


class PermissionDenied(LedgerError):
    """The caller is not a registered organisation (or not the admin)."""


@dataclass
class Receipt:
    """Outcome of one registration."""

    digest: str  # hex SHA-256
    doc_type: str
    ref_id: str
    org: str
    status: str  # "registered" | "duplicate"
    tx_id: str
    block: int
    client_ts: int
    cost: int  # gas on the EVM backend, bytes appended on the Python backend
    latency_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceRecord:
    """What the ledger knows about a hash."""

    digest: str
    exists: bool
    org: str = ""
    submitter: str = ""  # address / key id of the submitting organisation
    doc_type: str = ""
    ref_id: str = ""
    client_ts: int = 0
    block_ts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalise_digest(digest: str | bytes) -> str:
    """Lower-case hex without ``0x``; raises if it is not a 32-byte hash."""
    if isinstance(digest, (bytes, bytearray)):
        digest = bytes(digest).hex()
    text = str(digest).lower().removeprefix("0x")
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise LedgerError(f"not a SHA-256 hex digest: {str(digest)[:80]}")
    if text == "0" * 64:
        raise LedgerError("the all-zero hash cannot be registered")
    return text


class Journal:
    """Append-only off-chain log of everything this client registered.

    The journal is an index for convenience (listing evidence, re-anchoring a
    restarted development chain). It is not trusted: verification always goes
    to the ledger.
    """

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, receipt: Receipt) -> None:
        if self.path and receipt.status == "registered":
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(receipt.to_dict(), sort_keys=True) + "\n")

    def entries(self) -> list[dict[str, Any]]:
        if not self.path or not self.path.is_file():
            return []
        with open(self.path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def clear(self) -> None:
        if self.path and self.path.is_file():
            self.path.unlink()


class EvidenceLedger(ABC):
    """Permissioned, append-only registry of evidence hashes."""

    backend: str = "abstract"
    journal: Journal

    # ------------------------------------------------------------------ organisations
    @abstractmethod
    def register_organisation(self, name: str) -> str:
        """Allow ``name`` to write (admin operation); returns its address / key id."""

    @abstractmethod
    def revoke_organisation(self, name: str) -> None:
        """Withdraw the write permission of ``name`` (admin operation)."""

    @abstractmethod
    def organisations(self) -> dict[str, str]:
        """Registered organisations: name -> address / key id."""

    # ------------------------------------------------------------------ writes
    @abstractmethod
    def register(self, digest: str, doc_type: str, ref_id: str, org: str, ts: int | None = None) -> Receipt:
        """Register one hash. A hash that is already on record yields ``status='duplicate'``."""

    @abstractmethod
    def register_batch(self, digests: list[str], doc_type: str, ref_ids: list[str], org: str, ts: int | None = None) -> list[Receipt]:
        """Register several hashes of one document type in as few ledger writes as possible."""

    # ------------------------------------------------------------------ reads
    @abstractmethod
    def verify(self, digest: str) -> EvidenceRecord:
        """Look a hash up; ``exists`` is False if it was never registered."""

    @abstractmethod
    def audit_trail(self, doc_type: str | None = None) -> list[dict[str, Any]]:
        """All registrations in ledger order, read from the ledger itself."""

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Record count, storage use and cumulative cost."""

    def close(self) -> None:  # noqa: B027  (optional hook)
        """Release resources. The default does nothing."""

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def now() -> int:
        return int(time.time())

    def is_registered(self, digest: str) -> bool:
        return self.verify(digest).exists

    def replay(self, entries: Iterable[dict[str, Any]]) -> int:
        """Re-register journal entries (used when a development chain was restarted)."""
        count = 0
        for e in entries:
            if e["org"] not in self.organisations():
                self.register_organisation(e["org"])
            receipt = self.register(e["digest"], e["doc_type"], e["ref_id"], e["org"], int(e["client_ts"]))
            count += receipt.status == "registered"
        return count


def storage_slots(doc_type: str, ref_id: str) -> int:
    """Contract storage slots one evidence record occupies (EVM layout of ``Evidence``).

    One slot packs submitter + both timestamps; each string takes one slot when
    shorter than 32 bytes, otherwise one length slot plus ``ceil(len / 32)`` data slots.
    """

    def string_slots(text: str) -> int:
        size = len(text.encode("utf-8"))
        return 1 if size < 32 else 1 + (size + 31) // 32

    return 1 + string_slots(doc_type) + string_slots(ref_id)
