"""Merkle trees for batching document hashes.

Registering every trade document in its own transaction does not scale. A batch
of documents is therefore committed with one hash, the Merkle root of the
documents' SHA-256 hashes. A single document is later verified with its hash,
a short proof (the sibling hashes on the path to the root) and the root that is
on the ledger.

Construction follows RFC 6962: leaves and inner nodes are hashed with different
prefixes (``0x00`` / ``0x01``), which rules out passing an inner node off as a
leaf, and an unpaired node is carried up unchanged.
"""

from __future__ import annotations

import hashlib

LEAF_PREFIX = b"\x00"
NODE_PREFIX = b"\x01"


def _leaf(digest_hex: str) -> bytes:
    return hashlib.sha256(LEAF_PREFIX + bytes.fromhex(digest_hex)).digest()


def _node(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(NODE_PREFIX + left + right).digest()


def _levels(leaves_hex: list[str]) -> list[list[bytes]]:
    if not leaves_hex:
        raise ValueError("a Merkle tree needs at least one leaf")
    levels = [[_leaf(h) for h in leaves_hex]]
    while len(levels[-1]) > 1:
        prev = levels[-1]
        nxt = [_node(prev[i], prev[i + 1]) for i in range(0, len(prev) - 1, 2)]
        if len(prev) % 2:
            nxt.append(prev[-1])
        levels.append(nxt)
    return levels


def merkle_root(leaves_hex: list[str]) -> str:
    """Root (hex) of the tree over the given document hashes, in the given order."""
    return _levels(leaves_hex)[-1][0].hex()


def merkle_proof(leaves_hex: list[str], index: int) -> list[tuple[str, str]]:
    """Proof for leaf ``index``: ``(sibling_hex, side)`` pairs from leaf to root.

    ``side`` is ``"L"`` if the sibling is the left operand, ``"R"`` if it is the right one.
    """
    if not 0 <= index < len(leaves_hex):
        raise IndexError(index)
    proof: list[tuple[str, str]] = []
    for level in _levels(leaves_hex)[:-1]:
        sibling = index ^ 1
        if sibling < len(level):
            proof.append((level[sibling].hex(), "L" if sibling < index else "R"))
        index //= 2
    return proof


def root_from_proof(digest_hex: str, proof: list[tuple[str, str]]) -> str:
    """Root implied by a document hash and its proof."""
    node = _leaf(digest_hex)
    for sibling_hex, side in proof:
        sibling = bytes.fromhex(sibling_hex)
        node = _node(sibling, node) if side == "L" else _node(node, sibling)
    return node.hex()


def verify_proof(digest_hex: str, proof: list[tuple[str, str]], root_hex: str) -> bool:
    return root_from_proof(digest_hex, proof) == root_hex.lower().removeprefix("0x")
