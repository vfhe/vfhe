# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""`LinearTransformCodec`: its scheme, a header, then the encoded diagonals."""

from __future__ import annotations

from typing import TYPE_CHECKING

from _vfhe_proto.vfhe.fhe.io.v1 import io_pb2 as pb
from vfhe.arith.codec import PolynomialCodec
from vfhe.fhe.linear_transform import CKKS_LinearTransform
from vfhe.util.io import Codec
from vfhe.util.io.protobuf import MessageBytesCodec

from .scheme import CKKSSchemeCodec

if TYPE_CHECKING:
    from typing import IO


class LinearTransformCodec(Codec[CKKS_LinearTransform]):
    """A CKKS linear transform: its scheme through ``schemes``, a header, then its encoded diagonals through ``polynomials``, so the encoding need not be redone."""

    header = MessageBytesCodec(pb.LinearTransform)

    def __init__(
        self,
        schemes: CKKSSchemeCodec | None = None,
        polynomials: PolynomialCodec | None = None,
    ) -> None:
        self.schemes = schemes or CKKSSchemeCodec()
        self.polynomials = polynomials or PolynomialCodec(self.schemes.rings)

    def encode(self, obj: CKKS_LinearTransform, stream: IO[bytes], /) -> None:
        self.schemes.encode(obj.scheme, stream)
        layout = [(j, i) for j, row in obj.plaintexts.items() for i in row]
        self.header.encode(
            pb.LinearTransform(
                n=obj.n,
                lvl=obj.lvl,
                baby_steps=obj.baby_steps,
                scale=obj.scale,
                diagonals=[pb.Diagonal(j=j, i=i) for j, i in layout],
            ),
            stream,
        )
        for j, i in layout:
            self.polynomials.encode(obj.plaintexts[j][i], stream)

    def decode(self, stream: IO[bytes], /) -> CKKS_LinearTransform:
        scheme = self.schemes.decode(stream)
        header = self.header.decode(stream)
        out = CKKS_LinearTransform.__new__(CKKS_LinearTransform)
        out.scheme = scheme
        out.n, out.lvl = header.n, header.lvl
        out.baby_steps, out.scale = header.baby_steps, header.scale
        out.plaintexts = {}
        for d in header.diagonals:
            out.plaintexts.setdefault(d.j, {})[d.i] = self.polynomials.decode(stream)
        out._prepare()  # noqa: SLF001 - the codec belongs to the class
        return out
