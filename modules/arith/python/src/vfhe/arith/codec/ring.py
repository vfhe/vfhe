# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""`RingCodec`: a ring by its parameters and NTT roots."""

from __future__ import annotations

from _vfhe_proto.vfhe.arith.io.v1 import io_pb2 as pb
from vfhe.arith.impl.rns.polynomial import RingParameters, RNSRing
from vfhe.util.io.protobuf import ObjectMessageCodec

from .rows import ntt_roots


class RingCodec(ObjectMessageCodec[RNSRing, pb.Ring]):
    """A ring by its parameters.

    A ring read back is one of ``known`` where the parameters match, else it
    is built once and handed back again for every later record of it. Other
    codecs embed its message through `to_message` and `from_message`. Thread-unsafe:
    the codec remembers what it reads, so each thread decodes through its own.
    """

    message = pb.Ring

    def __init__(self, *known: RNSRing) -> None:
        self.rings: dict[RingParameters, RNSRing] = {r.parameters: r for r in known}
        self._roots: dict[
            RNSRing, list[int]
        ] = {}  # the roots a stream wrote for a ring

    def to_message(self, ring: RNSRing) -> pb.Ring:
        return pb.Ring(
            n=ring.N,
            split_degree=ring.split_degree,
            # In the ring's order, which quotient_ring(ell=...) relies on.
            primes=ring.primes,
            roots=ntt_roots(ring),
        )

    def from_message(self, message: pb.Ring) -> RNSRing:
        key = RingParameters(message.n, message.split_degree, tuple(message.primes))
        ring = self.rings.get(key)
        if ring is None:
            ring = self.rings[key] = RNSRing(
                message.n,
                primes=list(message.primes),
                prime_size=[p.bit_length() for p in message.primes],
                split_degree=message.split_degree,
            )
        written = dict(zip(message.primes, message.roots, strict=True))
        self._roots[ring] = [written[p] for p in ring.primes]
        return ring

    def require_same_transform(self, ring: RNSRing) -> None:
        """Refuses NTT-domain rows written under another NTT convention."""
        roots = self._roots.get(ring)
        if roots is not None and roots != ntt_roots(ring):
            raise ValueError(
                "the stream holds NTT-domain rows written with other roots of unity; "
                "write it with domain='canonical' (and seeded=False) to move it"
            )
