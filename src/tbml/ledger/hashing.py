"""Canonical hashing of evidence: trade documents, model updates, alerts.

Only these fingerprints ever reach the ledger. Two parties that hold the same
document must compute the same hash, so the serialisation is canonical:

* JSON with sorted keys and no insignificant whitespace, UTF-8;
* numbers normalised: integers as integers, floats rounded to 6 decimals with
  integral floats written as integers (``12.0`` and ``12`` hash alike),
  NaN/None as ``null``;
* dates as ISO ``YYYY-MM-DD`` (or full ISO timestamps when a time is present);
* strings stripped of surrounding whitespace.

Model updates are hashed over the raw array bytes together with each array's
name, dtype and shape.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Mapping

import numpy as np

FLOAT_DECIMALS = 6


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalise(value: Any) -> Any:
    """Bring a value into canonical JSON form (see module docstring)."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        number = round(number, FLOAT_DECIMALS)
        return int(number) if number == int(number) and abs(number) < 2**53 else number
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        return {str(k): normalise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [normalise(v) for v in (sorted(value) if isinstance(value, set) else value)]
    if hasattr(value, "isoformat"):  # date, datetime, pandas Timestamp
        if value != value:  # NaT
            return None
        text = value.isoformat()
        return text[:10] if text.endswith("T00:00:00") else text
    return str(value)


def canonical_json(obj: Any) -> bytes:
    """Canonical UTF-8 JSON bytes of ``obj``."""
    return json.dumps(normalise(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def hash_record(record: Mapping[str, Any]) -> str:
    """SHA-256 (hex) of one document given as a mapping of field -> value."""
    return sha256_hex(canonical_json(record))


def hash_arrays(arrays: Iterable[tuple[str, np.ndarray]]) -> str:
    """SHA-256 (hex) of a model given as ``(name, array)`` pairs, in the given order."""
    digest = hashlib.sha256()
    for name, array in arrays:
        arr = np.ascontiguousarray(array)
        if arr.dtype.byteorder == ">":
            arr = arr.byteswap().view(arr.dtype.newbyteorder("<"))
        digest.update(name.encode("utf-8"))
        digest.update(b"|")
        digest.update(str(arr.dtype).encode("ascii"))
        digest.update(b"|")
        digest.update(",".join(str(d) for d in arr.shape).encode("ascii"))
        digest.update(b"|")
        digest.update(arr.tobytes())
    return digest.hexdigest()


def hash_file(path) -> str:
    """SHA-256 (hex) of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
