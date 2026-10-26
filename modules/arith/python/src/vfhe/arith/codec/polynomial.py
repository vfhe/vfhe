# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""`PolynomialCodec`: a polynomial as its ring, then its rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from _vfhe_proto.vfhe.arith.io.v1 import io_pb2 as pb
from vfhe.arith.impl.rns.polynomial import RNSPolynomial
from vfhe.arith.impl.rns.polynomial import repr as Repr
from vfhe.util.io import Codec
from vfhe.util.io.protobuf import MessageBytesCodec

from .ring import RingCodec
from .rows import (
    DOMAINS,
    RowReader,
    RowWriter,
    held_domain,
    rows_options,
    target_domain,
)

if TYPE_CHECKING:
    from typing import IO


class PolynomialCodec(Codec[RNSPolynomial]):
    """An `RNSPolynomial` as its ring, through ``rings``, then its rows; ``packing`` and ``domain`` as `rows_options` says."""

    body = MessageBytesCodec(pb.Polynomial)

    def __init__(
        self,
        rings: RingCodec | None = None,
        packing: str = "word",
        domain: str = "held",
    ) -> None:
        self.rings = rings or RingCodec()
        self.packing, self.domain = rows_options(packing, domain)

    def encode(self, obj: RNSPolynomial, stream: IO[bytes], /) -> None:
        if obj.repr not in DOMAINS:
            raise ValueError("cannot write a polynomial that holds no value")
        self.rings.encode(obj.ring, stream)
        domain = target_domain(obj.repr, self.domain)
        view = obj._viewed_as(domain)  # noqa: SLF001 - a copy only if it converts
        self.body.encode(
            pb.Polynomial(
                domain=DOMAINS[domain],
                packing=self.packing,
                rows=RowWriter(obj.ring, self.packing).rows(view.obj),
            ),
            stream,
        )

    def decode(self, stream: IO[bytes], /) -> RNSPolynomial:
        ring = self.rings.decode(stream)
        message = self.body.decode(stream)
        domain = held_domain(message.domain)
        if domain == Repr.ntt:
            self.rings.require_same_transform(ring)
        out = RNSPolynomial(ring)
        RowReader(ring, message.packing).read(message.rows, out.obj)
        out.repr = domain
        return out
