# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The row encoding of RNS polynomials: its options, sizes and order, and the writer and reader over C."""

from __future__ import annotations

import sys
from typing import Any

from _vfhe_proto.vfhe.arith.io.v1 import io_pb2 as pb
from vfhe.arith.impl.rns.polynomial import RNSRing
from vfhe.arith.impl.rns.polynomial import repr as Repr
from vfhe.util.bindings import ffi, lib

if sys.byteorder != "little":
    raise ImportError("vfhe codecs write little-endian words as they are in memory")

#: The domains by the representation held, and the stream's value.
DOMAINS = {Repr.coeff: pb.DOMAIN_COEFF, Repr.ntt: pb.DOMAIN_NTT}


def held_domain(domain: pb.Domain) -> Repr:
    """The representation the stream's ``domain`` names. Raises `ValueError` for one this library does not know."""
    for held, value in DOMAINS.items():
        if value == domain:
            return held
    raise ValueError(f"unknown row domain {domain}: written by a newer vfhe?")


#: The packings by the name a codec takes, and the stream's value.
PACKINGS: dict[str, pb.Packing] = {
    "word": pb.PACKING_WORD,
    "tight": pb.PACKING_TIGHT,
}


TARGETS = ("held", "mul", "canonical")


def rows_options(packing: str, domain: str) -> tuple[pb.Packing, str]:
    """Checks a codec's row options and returns ``(packing, domain)``, the packing as the stream's value.

    ``packing``: ``"word"``, each residue as a 4- or 8-byte word, copied as is;
    ``"tight"``, each residue in exactly the bits its prime needs. ``domain``:
    ``"held"`` writes rows as held; ``"mul"`` and ``"canonical"`` convert to
    the NTT and the coefficient domain. Canonical rows read the same under
    any NTT convention and cost a transform on load. Raises `ValueError`.
    """
    if packing not in PACKINGS:
        raise ValueError(f"packing must be one of {tuple(PACKINGS)}, got {packing!r}")
    if domain not in TARGETS:
        raise ValueError(f"domain must be one of {TARGETS}, got {domain!r}")
    return PACKINGS[packing], domain


def _tight(packing: pb.Packing) -> bool:
    """Whether the stream's ``packing`` is tight. Raises `ValueError` for one this library does not know."""
    if packing not in PACKINGS.values():
        raise ValueError(f"unknown row packing {packing}: written by a newer vfhe?")
    return packing == pb.PACKING_TIGHT


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
    """Encodes polynomials' rows over one ring, at the stream's ``packing``, through one reusable buffer."""

    def __init__(self, ring: RNSRing, packing: pb.Packing) -> None:
        self.ring = ring
        self.tight = _tight(packing)
        self.order = row_order(ring)
        self.size = rows_size(ring, self.tight)
        self._buf = ffi.new("uint8_t[]", self.size)

    def rows(self, handle: Any) -> bytes:
        """Returns the rows of the polynomial ``handle``."""
        lib.polynomial_RNS_write_rows(
            self._buf, handle, self.order, self.ring.ell, self.tight
        )
        return bytes(ffi.buffer(self._buf, self.size))


class RowReader:
    """Decodes polynomials' rows over one ring, at the stream's ``packing``, through one reusable buffer.

    Every residue is checked to be below its prime: C trusts what it is handed.
    Raises `ValueError` for a packing this library does not know.
    """

    def __init__(self, ring: RNSRing, packing: pb.Packing) -> None:
        self.ring = ring
        self.tight = _tight(packing)
        self.order = row_order(ring)
        self.size = rows_size(ring, self.tight)
        self._buf = ffi.new("uint8_t[]", self.size)

    def read(self, rows: bytes, handle: Any) -> None:
        """Fills the polynomial ``handle`` from ``rows``. Raises `ValueError` for rows of another size or with a residue out of range."""
        if len(rows) != self.size:
            raise ValueError(
                f"{len(rows)} bytes of rows, {self.size} expected: corrupt stream"
            )
        ffi.memmove(self._buf, rows, self.size)
        bad = lib.polynomial_RNS_read_rows(
            handle, self._buf, self.order, self.ring.ell, self.tight, True
        )
        if bad >= 0:
            prime = sorted(self.ring.primes)[bad]
            raise ValueError(
                f"a residue modulo {prime} is not below it: corrupt stream"
            )
