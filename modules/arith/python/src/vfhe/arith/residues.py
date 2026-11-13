# SPDX-FileCopyrightText: 2026 The vfhe authors
# SPDX-License-Identifier: Apache-2.0
"""Integers given by their residues modulo several primes, as word arrays."""

from __future__ import annotations

from array import array
from typing import TYPE_CHECKING

from vfhe.util.bindings import ffi, lib

from .base import Field, FieldVector

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The most primes one call combines (the C kernel's bound).
MAX_PRIMES = 64


def crt_lift_mod_2_128(
    residues: Sequence, primes: Sequence[int]
) -> tuple[array, array]:
    """The integers given by their residues, modulo ``2^128``.

    ``residues[r]`` is a buffer of 64-bit unsigned integers (an
    ``array('Q')``, say) holding, for every one of `n` integers, its residue
    modulo ``primes[r]``, reduced or not; every buffer has the same length
    `n`. The primes are distinct and below ``2^63``, at most `MAX_PRIMES` of
    them. Returns ``(low, high)``, two ``array('Q')``s of `n` words:
    ``low[i] + 2^64 high[i]`` is the integer in ``[0, q)`` with the residues
    at position `i`, `q` the product of the primes, modulo ``2^128``. The
    same contract as `RNSRing.lift_mod_2_128`, for residues held as arrays
    instead of ring elements.

    Garner's mixed-radix conversion [Gar59] over whole vectors: the
    mixed-radix digit at each prime is computed as vector operations over
    `F_p` (a buffer is reduced modulo `p` as it is loaded), and the digits
    are recombined with the weights ``prod_(k < r) primes[k]`` in one native
    pass modulo ``2^128``.
    """
    primes = [int(p) for p in primes]
    if not 0 < len(primes) <= MAX_PRIMES:
        raise ValueError(f"between 1 and {MAX_PRIMES} primes, got {len(primes)}")
    if len(set(primes)) != len(primes) or any(p < 2 or p >> 63 for p in primes):
        raise ValueError("the primes must be distinct and below 2^63")
    if len(residues) != len(primes):
        raise ValueError(f"{len(residues)} residue buffers for {len(primes)} primes")
    views = [memoryview(buffer) for buffer in residues]
    for view in views:
        if view.ndim != 1 or view.itemsize != 8 or view.format not in ("Q", "L"):
            raise TypeError(
                "residues must be one-dimensional buffers of 64-bit unsigned integers"
            )
    n = len(views[0])
    if any(len(view) != n for view in views):
        raise ValueError("the residue buffers differ in length")
    low = array("Q", bytes(8 * n))
    high = array("Q", bytes(8 * n))
    if n == 0:
        return low, high

    # The mixed-radix digits y_r: x = y_0 + p_0 (y_1 + p_1 (y_2 + ...)),
    # y_r = (a_r - (y_0 + p_0 y_1 + ... )) / (p_0 ... p_(r-1)) mod p_r.
    digits: list[array] = []
    for r, prime in enumerate(primes):
        field = Field(prime)
        loaded = FieldVector(field, n)
        loaded.set_uint64(views[r])
        if r:
            partial = FieldVector(field, n)
            partial.set_uint64(digits[0])
            weight = 1
            for k in range(1, r):
                weight = weight * primes[k - 1] % prime
                term = FieldVector(field, n)
                term.set_uint64(digits[k])
                partial = partial.fma(term, field(weight))
            weight = weight * primes[r - 1] % prime
            loaded = (loaded - partial).scale(field(pow(weight, -1, prime)))
        words = loaded.to_uint64()
        if words is None:  # elements of F_p are integers below p
            raise RuntimeError("a residue left the prime field")
        digits.append(words)

    # x = sum_r y_r Q_r with Q_r = p_0 ... p_(r-1), modulo 2^128.
    mask = (1 << 64) - 1
    weights_low, weights_high = array("Q"), array("Q")
    weight = 1
    for prime in primes:
        weights_low.append(weight & mask)
        weights_high.append((weight >> 64) & mask)
        weight *= prime
    buffers = [ffi.from_buffer("uint64_t[]", words) for words in digits]
    terms = ffi.new("const uint64_t *[]", buffers)
    lib.vec_combination_mod_2_128(
        ffi.from_buffer("uint64_t[]", low),
        ffi.from_buffer("uint64_t[]", high),
        terms,
        ffi.from_buffer("uint64_t[]", weights_low),
        ffi.from_buffer("uint64_t[]", weights_high),
        len(primes),
        n,
    )
    return low, high
