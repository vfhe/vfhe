# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""`LWECodec`: an LWE sample's limbs as words."""

from __future__ import annotations

from typing import TYPE_CHECKING

from _vfhe_proto.vfhe.mlwe.io.v1 import io_pb2 as pb
from vfhe.arith.codec import RingCodec
from vfhe.mlwe.lwe import LWE, _native_limbs
from vfhe.util.bindings import ffi
from vfhe.util.io.protobuf import ObjectMessageCodec

if TYPE_CHECKING:
    from vfhe.arith import RNSRing


def _limb_order(ring: RNSRing) -> list[int]:
    """`LWE`'s limb positions (by base index) sorted by prime value."""
    limbs = _native_limbs(ring)
    return sorted(range(len(limbs)), key=lambda j: ring.primes[limbs[j]])


class LWECodec(ObjectMessageCodec[LWE, pb.LWE]):
    """An LWE sample: per limb in ascending prime, ``a`` then ``b`` as 8-byte
    little-endian words; its ring through ``rings``."""

    message = pb.LWE

    def __init__(self, rings: RingCodec | None = None) -> None:
        self.rings = rings or RingCodec()

    def to_message(self, obj: LWE, /) -> pb.LWE:
        n = obj.n
        struct = ffi.cast("LWE", obj.obj)
        words = bytearray()
        for j in _limb_order(obj.ring):
            words += ffi.buffer(struct.a[j], 8 * n)
            words += ffi.buffer(struct.b + j, 8)
        return pb.LWE(ring=self.rings.to_message(obj.ring), n=n, words=bytes(words))

    def from_message(self, message: pb.LWE, /) -> LWE:
        ring, n = self.rings.from_message(message.ring), message.n
        order = _limb_order(ring)
        if len(message.words) != 8 * (n + 1) * len(order):
            raise ValueError(f"{len(message.words)} bytes of LWE words: corrupt stream")
        out = LWE(ring, n=n)
        struct = ffi.cast("LWE", out.obj)
        limbs = _native_limbs(ring)
        words = memoryview(message.words)
        for k, j in enumerate(order):
            at = k * 8 * (n + 1)
            ffi.memmove(struct.a[j], words[at : at + 8 * n], 8 * n)
            ffi.memmove(struct.b + j, words[at + 8 * n : at + 8 * (n + 1)], 8)
            q = ring.primes[limbs[j]]
            if max(ffi.unpack(struct.a[j], n)) >= q or struct.b[j] >= q:
                raise ValueError(
                    f"a residue modulo {q} is not below it: corrupt stream"
                )
        return out
