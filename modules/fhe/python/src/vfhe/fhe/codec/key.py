# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The bootstrapping keys, `SABKeyCodec` and `CGGI16KeyCodec`: a header, then their parts through the codecs they were built with."""

from __future__ import annotations

from typing import TYPE_CHECKING

from _vfhe_proto.vfhe.fhe.io.v1 import io_pb2 as pb
from vfhe.fhe.cggi16 import CGGI16_Key
from vfhe.fhe.gp25 import SAB_Key
from vfhe.util.io import Codec, ListCodec, OptionalCodec
from vfhe.util.io.protobuf import MessageBytesCodec

if TYPE_CHECKING:
    from typing import IO

    from vfhe.mlwe.codec import KeySwitchKeyCodec, MGSWCodec


class SABKeyCodec(Codec[SAB_Key]):
    """GP25's bootstrapping key: a header, then its parts through ``mgsw`` and ``keys``."""

    header = MessageBytesCodec(pb.SABKey)

    def __init__(self, mgsw: MGSWCodec, keys: KeySwitchKeyCodec) -> None:
        self.gaps = ListCodec(ListCodec(ListCodec(mgsw)))
        self.signs = OptionalCodec(ListCodec(ListCodec(mgsw)))
        self.key = OptionalCodec(keys)

    def encode(self, obj: SAB_Key, stream: IO[bytes], /) -> None:
        header = pb.SABKey(
            n=obj.n, h=obj.h, gap_bits=obj.gap_bits, bits_per_layer=obj.bits_per_layer
        )
        if obj.hw_reducing_lvl is not None:
            header.hw_reducing_lvl = obj.hw_reducing_lvl
        if obj.output_lvl is not None:
            header.output_lvl = obj.output_lvl
        self.header.encode(header, stream)
        self.gaps.encode(obj.gaps, stream)
        self.signs.encode(obj.signs, stream)
        for name in (
            "automorphism_key",
            "trace_repack_key",
            "hw_reducing_key",
            "output_switch_key",
        ):
            self.key.encode(getattr(obj, name), stream)

    def decode(self, stream: IO[bytes], /) -> SAB_Key:
        header = self.header.decode(stream)
        out = SAB_Key()
        out.n, out.h, out.gap_bits = header.n, header.h, header.gap_bits
        out.bits_per_layer = header.bits_per_layer
        out.hw_reducing_lvl = (
            header.hw_reducing_lvl if header.HasField("hw_reducing_lvl") else None
        )
        out.output_lvl = header.output_lvl if header.HasField("output_lvl") else None
        out.gaps = self.gaps.decode(stream)
        out.signs = self.signs.decode(stream)
        for name in (
            "automorphism_key",
            "trace_repack_key",
            "hw_reducing_key",
            "output_switch_key",
        ):
            setattr(out, name, self.key.decode(stream))
        return out


class CGGI16KeyCodec(Codec[CGGI16_Key]):
    """CGGI16's bootstrapping key: a header, then its MGSW ciphertexts through ``mgsw``."""

    header = MessageBytesCodec(pb.CGGI16Key)

    def __init__(self, mgsw: MGSWCodec) -> None:
        self.mgsw = mgsw

    def encode(self, obj: CGGI16_Key, stream: IO[bytes], /) -> None:
        self.header.encode(
            pb.CGGI16Key(n=obj.n, unfolding=obj.unfolding, keys=len(obj.bk)), stream
        )
        for k in obj.bk:
            self.mgsw.encode(k, stream)

    def decode(self, stream: IO[bytes], /) -> CGGI16_Key:
        header = self.header.decode(stream)
        out = CGGI16_Key()
        out.n, out.unfolding = header.n, header.unfolding
        out.bk = [self.mgsw.decode(stream) for _ in range(header.keys)]
        return out
