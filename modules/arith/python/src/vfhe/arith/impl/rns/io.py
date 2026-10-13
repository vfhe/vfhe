# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""`vfhe.util.io` codecs for RNS rings and polynomials, and their row encoding.

A ring is written by its primes, not by base index, since base indices depend
on the order a process built its rings in; rows go in ascending prime value.
A ring also records each prime's NTT root, because mul-domain rows (seeded
ones included) only mean the same to a reader with the same roots:
`require_same_transform` checks that before reading them.
"""

from __future__ import annotations

import math
from typing import Any

from vfhe.util.bindings import ffi, lib
from vfhe.util.io import (
    Codec,
    Encoded,
    Payload,
    ReadContext,
    Sink,
    WriteContext,
    register,
)

from .polynomial import RNSPolynomial, RNSRing
from .polynomial import repr as Repr

#: The domains as the files name them.
DOMAIN_NAMES = {Repr.coeff: "coeff", Repr.ntt: "ntt"}
DOMAINS_BY_NAME = {v: k for k, v in DOMAIN_NAMES.items()}


def ring_key(ring: RNSRing) -> tuple:
    """What identifies a ring across processes: its parameters and primes."""
    return (ring.N, ring.split_degree, frozenset(ring.primes))


def ntt_roots(ring: RNSRing) -> list[int]:
    """Each prime's NTT root, in `RNSRing.primes` order."""
    plans = ffi.cast("RNS_Base", ring.base).plans
    return [int(lib.ntt_plan_root(plans[idx])) for idx in ring.prime_indices]


def row_order(ring: RNSRing) -> Any:
    """The ring's base indices in ascending prime value, as a native array."""
    return ffi.new(
        "uint64_t[]",
        [idx for _, idx in sorted(zip(ring.primes, ring.prime_indices, strict=True))],
    )


def rows_size(ring: RNSRing, tight: bool) -> int:
    """Bytes of one polynomial's rows over ``ring``."""
    return sum(int(lib.rns_row_bytes(q, ring.N, tight)) for q in ring.primes)


def target_domain(held: Repr, option: str) -> Repr:
    """The domain rows are written in, for an object holding ``held``."""
    if option == "mul":
        return Repr.ntt
    if option == "canonical":
        return Repr.coeff
    return held


class RowWriter:
    """Writes polynomials' rows over one ring through one reusable buffer."""

    def __init__(self, ring: RNSRing, tight: bool) -> None:
        self.ring = ring
        self.tight = tight
        self.order = row_order(ring)
        self.size = rows_size(ring, tight)
        self._buf = ffi.new("uint8_t[]", self.size)

    def write(self, sink: Sink, handle: Any) -> None:
        lib.polynomial_RNS_write_rows(
            self._buf, handle, self.order, self.ring.ell, self.tight
        )
        sink.write(ffi.buffer(self._buf, self.size))


class RowReader:
    """Reads polynomials' rows over one ring through one reusable buffer."""

    def __init__(self, ring: RNSRing, tight: bool, validate: bool) -> None:
        self.ring = ring
        self.tight = tight
        self.validate = validate
        self.order = row_order(ring)
        self.size = rows_size(ring, tight)
        self._buf = ffi.new("uint8_t[]", self.size)

    def read(self, payload: Payload, handle: Any) -> None:
        payload.readinto(ffi.buffer(self._buf, self.size))
        bad = lib.polynomial_RNS_read_rows(
            handle, self._buf, self.order, self.ring.ell, self.tight, self.validate
        )
        if bad >= 0:
            prime = sorted(self.ring.primes)[bad]
            raise ValueError(
                f"a residue modulo {prime} is not below it: corrupt stream"
            )


def require_same_transform(ring: RNSRing, ctx: ReadContext) -> None:
    """Refuse mul-domain rows written under another NTT convention."""
    roots = ctx.cache.get(("arith.roots", id(ring)))
    if roots is not None and roots != ntt_roots(ring):
        raise ValueError(
            "the stream holds NTT-domain rows written with other roots of unity; "
            "write it with domain='canonical' (and seeded=False) to move it"
        )


class RNSRingCodec(Codec):
    tag = "arith.rns_ring"
    types = (RNSRing,)
    definition = True

    def identity(self, obj: RNSRing, /) -> tuple:
        return ring_key(obj)

    def encode(self, obj: RNSRing, _ctx: WriteContext, /) -> Encoded:
        return Encoded(
            {
                "N": obj.N,
                "split_degree": obj.split_degree,
                # In the ring's order, which quotient_ring(ell=...) relies on.
                "primes": list(obj.primes),
                "roots": ntt_roots(obj),
            }
        )

    def decode(
        self, meta: dict, _payload: Payload, _children: list, ctx: ReadContext, /
    ) -> RNSRing:
        N, split, primes = meta["N"], meta["split_degree"], meta["primes"]
        key = (N, split, frozenset(primes))
        ring = ctx.bound(self.tag, key)
        if ring is None:
            ring = ctx.cache.get(("arith.ring", key))
        if ring is None:
            ring = RNSRing(
                N,
                primes=list(primes),
                prime_size=[math.ceil(math.log2(p)) for p in primes],
                split_degree=split,
            )
        ctx.cache[("arith.ring", key)] = ring
        written = dict(zip(primes, meta["roots"], strict=True))
        ctx.cache[("arith.roots", id(ring))] = [written[p] for p in ring.primes]
        return ring


class RNSPolynomialCodec(Codec):
    tag = "arith.rns_polynomial"
    types = (RNSPolynomial,)

    def encode(self, obj: RNSPolynomial, ctx: WriteContext, /) -> Encoded:
        if obj.repr not in DOMAIN_NAMES:
            raise ValueError("cannot write a polynomial that holds no value")
        domain = target_domain(obj.repr, ctx.options.domain)
        view = obj._viewed_as(domain)  # a copy only if it converts
        tight = ctx.options.packing == "tight"
        rows = RowWriter(obj.ring, tight)
        meta = {
            "ring": ctx.ref(obj.ring),
            "domain": DOMAIN_NAMES[domain],
            "tight": tight,
        }
        return Encoded(meta, rows.size, lambda sink: rows.write(sink, view.obj))

    def decode(
        self, meta: dict, payload: Payload, _children: list, ctx: ReadContext, /
    ) -> RNSPolynomial:
        ring = ctx.deref(meta["ring"])
        domain = DOMAINS_BY_NAME[meta["domain"]]
        if domain == Repr.ntt:
            require_same_transform(ring, ctx)
        out = RNSPolynomial(ring)
        RowReader(ring, meta["tight"], ctx.options.validate).read(payload, out.obj)
        out.repr = domain
        return out


register(RNSRingCodec())
register(RNSPolynomialCodec())
