# SPDX-License-Identifier: Apache-2.0
"""Quotient rings ``Z_q[X]/(X^N + 1)`` with ``q`` in RNS form.

A :class:`Ring` is a Python-side view over a shared native ring handle (see
:class:`~vfhe.arith.registry.RingRegistry`): it selects a subset of the
handle's prime pool via a limb mask and carries the derived bookkeeping
(primes, bit sizes, modulus product). Ring *semantics* -- prime generation,
quotient relationships, CRT quantities -- live here; all data lives in C.
"""

from __future__ import annotations

import math

from _vfhe_native import ffi, lib

from .number_theory import is_prime
from .registry import RingRegistry


def next_power_of_2(x) -> int:
    """Smallest power of two >= ``x``."""
    return 1 << int(math.ceil(math.log2(x)))


class Ring:
    """A quotient ring ``Z_q[X]/(X^N + 1)`` with ``q`` split into RNS primes.

    The modulus can be specified several ways (mutually sufficient): a target
    ``mod_size`` in bits, an explicit ``primes`` list, a per-prime
    ``prime_size`` schedule, or a ``mask`` selecting a subset of an existing
    prime pool. Rings sharing an ``(N, split_degree)`` reuse the same native
    handle through the :class:`RingRegistry` singleton.

    Attributes:
        N: ring degree.
        split_degree: incomplete-NTT splitting factor (1 = full NTT).
        ell: number of RNS primes ("levels") in this ring.
        primes: the primes, in level order.
        q_l: product of the primes (the composite modulus).
        mask: limb mask selecting this ring's primes inside the pool.
        prime_indices: pool slot of each prime, aligned with ``primes``.
        handle: the shared native ``ring_t``.
    """

    def __init__(
        self,
        N,
        mod_size: int | None = None,
        split_degree=None,
        primes=None,
        mask=None,
        prime_size: int | list[int] = 49,
        exceptional_set_size=128,
    ) -> None:
        assert (
            (mod_size is not None)
            or (type(prime_size) is list)
            or (primes is not None)
            or (mask is not None)
        ), "must provide mod_size, prime_size, primes, or mask"

        registry = RingRegistry()

        if mask is not None:
            assert primes is not None, "must provide primes when mask is given"
            temp_split_degree = split_degree
            if not temp_split_degree:
                smallest_in_pool = min(math.ceil(math.log2(p)) for p in primes)
                temp_split_degree = next_power_of_2(
                    exceptional_set_size / smallest_in_pool
                )

            key = (N, temp_split_degree)
            prime_map = registry.prime_to_index[key]
            active_primes = [p for p in primes if ((mask >> prime_map[p]) & 1)]
            prime_size = [math.ceil(math.log2(p)) for p in active_primes]
            primes = active_primes

        if primes is not None and prime_size is None:
            prime_size = [math.ceil(math.log2(i)) for i in primes]

        if isinstance(prime_size, list):
            self.ell = len(prime_size)
            self.prime_size = prime_size
            self.smallest_prime = min(prime_size)
        else:
            assert mod_size is not None, (
                "must provide mod_size when prime_size is a scalar"
            )
            self.ell = math.ceil(mod_size / prime_size)
            self.prime_size = [prime_size] * self.ell
            self.smallest_prime = prime_size

        if not split_degree:
            # The exceptional set must fit in one slot; grow the split factor
            # until slot values (bounded by the smallest prime) are large enough.
            split_degree = next_power_of_2(exceptional_set_size / self.smallest_prime)
        self.split_degree = split_degree
        self.N = N
        self.byte_size = self.N * self.ell * 8
        self.lib = lib
        self.registry = registry

        if primes:
            assert len(primes) >= self.ell, (
                "not enough primes for quotient ring for size 2^%d" % mod_size
            )
            self.primes = primes[: self.ell]
        else:
            self.primes = self.gen_primes()

        self.q_l = math.prod(self.primes)
        self.bit_size = math.ceil(math.log2(self.q_l))

        self.prime_indices = registry.register_ring_primes(
            self.primes, self.N, self.split_degree
        )
        self.handle = registry.handle(self.N, self.split_degree)
        self.mask = sum(1 << idx for idx in self.prime_indices)

    # --- pool / native views --------------------------------------------------

    @property
    def pool_size(self) -> int:
        """Total primes in the shared pool (>= ``ell``)."""
        return self.registry.pool_size(self.N, self.split_degree)

    def alloc_polynomial(self):
        """Allocate a native polynomial handle over this ring's limb mask."""
        return lib.poly_new(self.handle, self.mask)

    def get_rou_matrix(self):
        """Per-prime twist tables (bit-reversed root powers), one row per level.

        Returns:
            list[list[int]]: ``ell`` rows of ``N // split_degree`` twiddles.
        """
        n_slots = self.N // self.split_degree
        rou_matrix = []
        for idx in self.prime_indices:
            row = lib.ring_twist(self.handle, idx)
            rou_matrix.append([row[k] for k in range(n_slots)])
        return rou_matrix

    # --- ring relationships ----------------------------------------------------

    def quotient_ring(self, mod_size=None, ell=None, mask=None):
        """A smaller ring over a subset of this ring's primes.

        Exactly one of ``mod_size``, ``ell`` or ``mask`` must be given.
        """
        assert (mod_size is not None) ^ (ell is not None) ^ (mask is not None), (
            "must provide mod_size, ell, or mask"
        )
        if mod_size is not None:
            res = Ring(self.N, mod_size, self.split_degree, primes=self.primes)
        elif ell is not None:
            res = Ring(
                self.N,
                prime_size=self.prime_size[:ell],
                split_degree=self.split_degree,
                primes=self.primes,
            )
        else:
            res = Ring(
                self.N, mask=mask, split_degree=self.split_degree, primes=self.primes
            )
        return res

    def is_quotient_ring(self, parent: Ring) -> bool:
        """True if this ring's primes are a subset of ``parent``'s."""
        return parent.mask & self.mask == self.mask

    def intersec(self, other: Ring) -> Ring:
        """The smaller of two nested rings (asserts they are nested)."""
        if self == other:
            return self
        if self.ell > other.ell:
            assert other.is_quotient_ring(self)
            return other
        assert self.is_quotient_ring(other)
        return self

    def modulus_ratio(self, other_ring: Ring, return_pointer: bool = False):
        """``q_self / q_other`` for a quotient ring, as int or RNS pointer.

        With ``return_pointer`` the ratio is returned reduced per pool slot as
        a ``uint64_t[]`` cdata suitable for ``poly_scale_vec`` /
        ``poly_scaled_lift``.
        """
        assert other_ring.is_quotient_ring(self), (
            "other_ring must be a quotient ring of this ring"
        )

        scaling_mask = self.mask & ~other_ring.mask
        delta_big_int = math.prod(
            [
                self.primes[i]
                for i, idx in enumerate(self.prime_indices)
                if (scaling_mask >> idx) & 1
            ]
        )
        if return_pointer:
            delta_arr = [0] * self.pool_size
            for k, idx in enumerate(self.prime_indices):
                delta_arr[idx] = delta_big_int % self.primes[k]
            return ffi.new("uint64_t[]", delta_arr)
        return delta_big_int

    def scalar_array(self, value):
        """RNS-decompose an int (or per-level list) into a pool-indexed array.

        Returns a ``uint64_t[]`` cdata of length ``pool_size`` with this
        ring's slots filled -- the layout ``poly_scale_vec`` expects.
        """
        scale_arr = [0] * self.pool_size
        if isinstance(value, int):
            for k, idx in enumerate(self.prime_indices):
                scale_arr[idx] = value % self.primes[k]
        else:
            vals = list(value)
            assert len(vals) == self.ell, f"expected {self.ell} values, got {len(vals)}"
            for k, idx in enumerate(self.prime_indices):
                scale_arr[idx] = vals[k]
        return ffi.new("uint64_t[]", scale_arr)

    # --- prime generation ---------------------------------------------------------

    @staticmethod
    def gen_prime(rou_order, prime_size, exclude_list=[]):
        """Largest ``prime_size``-bit prime p with ``p % rou_order == 1``.

        Scans downward so repeated calls with ``exclude_list`` yield distinct
        NTT-friendly primes of the requested size.
        """
        a = ((2**prime_size - 1) // rou_order) | 1
        a -= 2
        while True:
            candidate = a * rou_order + 1
            if candidate not in exclude_list and is_prime(candidate):
                return candidate
            a -= 2

    def gen_primes(self):
        """Generate ``ell`` distinct NTT-friendly primes per ``prime_size``."""
        primes = []
        for p_size in self.prime_size:
            primes.append(
                Ring.gen_prime(
                    2 * self.N // self.split_degree, p_size, exclude_list=primes
                )
            )
        return primes

    # --- sampling shorthands ---------------------------------------------------------

    def random_element(self, ntt=True):
        """Uniform ring element (in EVAL form by default)."""
        from .polynomial import Polynomial

        return Polynomial(self).sample_uniform(ntt)

    def random_gaussian_element(self, sigma, ntt=True):
        """Gaussian ring element with parameter ``sigma``."""
        from .polynomial import Polynomial

        return Polynomial(self).sample_gaussian(sigma, ntt)

    def random_exceptional(self, size="minimal", ntt=True):
        """Uniform element of the exceptional set (constant across slots)."""
        from .polynomial import Polynomial

        return Polynomial(self).sample_exceptional(size, ntt)
