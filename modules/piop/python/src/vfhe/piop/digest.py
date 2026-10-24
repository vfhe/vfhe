# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Digests of transcript entries and statement fields (piop.md section 5).

Every object that can appear in a transcript entry, a statement field or an
index entry has a `digest()` method: the oracles, statements and proofs of
this package, the tables, vectors, polynomials and field elements of
`vfhe.arith`, the Merkle paths of `vfhe.crypto`. `element_digest` is the one
entry point: it digests the values that cannot carry a method -- the
primitives, the containers, a class, a protobuf message -- and calls
`digest()` on everything else.
"""

from __future__ import annotations

from typing import cast

from vfhe.crypto import (
    DigestChain,
    bytes_digest,
    chain_digest,
    hash_bytes,
    int_digest,
    str_digest,
)


def element_digest(value) -> bytes:
    """A 32-byte digest of a transcript entry or statement field: a class
    (a relation named in an index) by its module and qualified name; raw
    bytes, ints, strings and None as primitives; dicts and sequences as the
    chain of their parts; a protobuf message (a circuit) by its
    serialization; any other value by its own `digest()` method. Each branch
    is domain-separated by a tag."""
    if isinstance(value, type):
        name = f"{value.__module__}.{value.__qualname__}"
        return hash_bytes(b"class" + name.encode())
    if isinstance(value, bytes | bytearray | memoryview):
        return bytes_digest(value)
    if isinstance(value, bool | int):
        return int_digest(value)
    if isinstance(value, str):
        return str_digest(value)
    if value is None:
        return hash_bytes(b"none")
    if isinstance(value, dict):
        parts = [b"dict"]
        for key, val in value.items():
            parts += [element_digest(key), element_digest(val)]
        return chain_digest(parts)
    if isinstance(value, list | tuple):
        chain = DigestChain().absorb(b"seq")
        return chain.absorb_packed(b"".join(element_digest(v) for v in value)).digest()
    if hasattr(value, "SerializeToString"):  # a protobuf message, e.g. a circuit
        return hash_bytes(b"proto" + value.SerializeToString())
    digest = getattr(value, "digest", None)
    if callable(digest):
        return cast("bytes", digest())
    raise TypeError(
        f"cannot digest a {type(value).__name__}; give it a digest() method"
    )
