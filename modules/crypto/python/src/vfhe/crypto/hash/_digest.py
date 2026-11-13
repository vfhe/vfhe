# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""`digest`, the BLAKE3 digest of a byte string."""

from __future__ import annotations

from vfhe.util.bindings import ffi, lib

#: Bytes in a digest: BLAKE3's default output length.
DIGEST_LEN = 32

#: What any function here takes as bytes.
Bytes = bytes | bytearray | memoryview


def digest(data: Bytes) -> bytes:
    """Computes the BLAKE3 digest of a byte string. Thread-safe, no state.

    Args:
        data: The bytes to hash, any length; a memoryview of any shape is
            read as its bytes.

    Returns:
        The `DIGEST_LEN`-byte digest.
    """
    view = memoryview(data).cast("B")
    out = ffi.new("uint8_t[]", DIGEST_LEN)
    lib.hash_blake3(out, ffi.from_buffer(view), len(view))
    return bytes(out)
