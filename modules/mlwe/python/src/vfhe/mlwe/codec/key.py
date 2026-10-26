# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The secret keys, `MLWEKeyCodec` and `LWEKeyCodec`: small signed coefficients at the narrowest width."""

from __future__ import annotations

import array
from typing import TYPE_CHECKING

from _vfhe_proto.vfhe.mlwe.io.v1 import io_pb2 as pb
from vfhe.arith.codec import RingCodec
from vfhe.mlwe.lwe import LWE_Key
from vfhe.mlwe.mlwe import (
    MLWE_Key,
)
from vfhe.util.io import Codec
from vfhe.util.io.protobuf import MessageBytesCodec, ObjectMessageCodec

from .scheme import MLWESchemeCodec, SchemeCodec

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import IO


def _signed_code(width: int) -> str:
    """The `array` type code of a signed integer of ``width`` bytes."""
    return next(c for c in "bhilq" if array.array(c).itemsize == width)


def _pack_signed(values: Sequence[int]) -> tuple[int, bytes]:
    """The smallest width in 1, 2, 4, 8 bytes holding every value, and the
    values packed at that width."""
    lo, hi = min(values), max(values)
    for width in (1, 2, 4, 8):
        if -(1 << (8 * width - 1)) <= lo and hi < (1 << (8 * width - 1)):
            return width, array.array(_signed_code(width), values).tobytes()
    raise ValueError("a secret key coefficient does not fit in 64 bits")


def _unpack_signed(width: int, data: bytes, count: int) -> list[int]:
    if len(data) != width * count:
        raise ValueError(f"{len(data)} bytes of key coefficients: corrupt stream")
    return array.array(_signed_code(width), data).tolist()


class MLWEKeyCodec(Codec[MLWE_Key]):
    """A secret key: its scheme through ``schemes``, then its small signed coefficients."""

    body = MessageBytesCodec(pb.Key)

    def __init__(self, schemes: SchemeCodec | None = None) -> None:
        self.schemes = schemes or MLWESchemeCodec()

    def encode(self, obj: MLWE_Key, stream: IO[bytes], /) -> None:
        self.schemes.encode(obj.scheme, stream)
        width, packed = _pack_signed([x for comp in obj.key for x in comp])
        self.body.encode(
            pb.Key(
                ring=self.schemes.rings.to_message(obj.ring),
                sigma=obj.sigma_err,
                width=width,
                lengths=[len(comp) for comp in obj.key],
                coefficients=packed,
            ),
            stream,
        )

    def decode(self, stream: IO[bytes], /) -> MLWE_Key:
        scheme = self.schemes.decode(stream)
        message = self.body.decode(stream)
        flat = _unpack_signed(message.width, message.coefficients, sum(message.lengths))
        key, at = [], 0
        for n in message.lengths:
            key.append(flat[at : at + n])
            at += n
        ring = self.schemes.rings.from_message(message.ring)
        return MLWE_Key(key, message.sigma, scheme, ring=ring)


class LWEKeyCodec(ObjectMessageCodec[LWE_Key, pb.LWEKey]):
    """An LWE secret key, as `MLWEKeyCodec` writes one; its ring through ``rings``."""

    message = pb.LWEKey

    def __init__(self, rings: RingCodec | None = None) -> None:
        self.rings = rings or RingCodec()

    def to_message(self, obj: LWE_Key, /) -> pb.LWEKey:
        width, packed = _pack_signed(obj.get_s())
        message = pb.LWEKey(
            ring=self.rings.to_message(obj.ring),
            n=obj.n,
            width=width,
            coefficients=packed,
        )
        if obj.err_sigma is not None:
            message.sigma = obj.err_sigma
        return message

    def from_message(self, message: pb.LWEKey, /) -> LWE_Key:
        s = _unpack_signed(message.width, message.coefficients, message.n)
        sigma = message.sigma if message.HasField("sigma") else None
        return LWE_Key(
            self.rings.from_message(message.ring), key=s, n=message.n, err_sigma=sigma
        )
