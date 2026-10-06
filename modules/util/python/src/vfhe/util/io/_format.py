# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The wire format: header and record layouts, record kinds, and the checksum registry."""

from __future__ import annotations

import hashlib
import struct
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

if sys.byteorder != "little":
    raise ImportError("vfhe.util.io writes little-endian words as they are in memory")

MAGIC = b"VFHEIO"
VERSION = (1, 0)
#: Records end with a 32-byte checksum of their tag, meta and payload.
FLAG_CHECKSUM = 1 << 0
#: The stream was written by ``dump_secret``: it may hold secret keys.
FLAG_SECRET = 1 << 1
#: Flag bits 8-15: which checksum the records carry, when they carry one.
_CHECKSUM_SHIFT = 8

#: Checksum algorithms, by the id the header records. BLAKE2b (hashlib) is
#: always available; the faster BLAKE3 is registered by ``vfhe.crypto``, which
#: a reader imports when a stream names it.
CHECKSUM_BLAKE2B = 1
CHECKSUM_BLAKE3 = 2
_CHECKSUM_PROVIDERS = {CHECKSUM_BLAKE3: "vfhe.crypto"}
_CHECKSUMS: dict[int, Callable[[], Any]] = {
    CHECKSUM_BLAKE2B: lambda: hashlib.blake2b(digest_size=_DIGEST),
}


def register_checksum(algorithm: int, factory: Callable[[], Any]) -> None:
    """Register checksum ``algorithm``: ``factory()`` returns a hasher with
    ``update`` and a 32-byte ``digest``. Writers use the highest id."""
    _CHECKSUMS[algorithm] = factory


KIND_END = 0
KIND_OBJECT = 1
KIND_DEFINITION = 2
KIND_LIST = 3
KIND_DICT = 4
KIND_NONE = 5
KIND_VALUE = 6

#: Payloads start at a multiple of this many bytes from the stream's start.
ALIGN = 64
_HEADER = struct.Struct("<6sBBII")
_RECORD = struct.Struct("<B3xIQQ")
_DIGEST = 32
#: The tag of a reference to a definition dumped as a value.
_REF_TAG = "io.ref"
