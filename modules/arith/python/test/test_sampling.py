# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The entropy-backed uniform sampler: exactly uniform below each prime."""

from collections import Counter

from vfhe.arith import Polynomial, Ring
from vfhe.mlwe import LWE, LWE_Key

N = 4096


def _residues(ring, draws):
    values = []
    for _ in range(draws):
        values.extend(Polynomial(ring).sample_uniform(ntt=False).get_coeff_matrix()[0])
    return values


def _balanced(counts, total, classes, tolerance=0.02):
    expected = total / classes
    return len(counts) == classes and all(
        abs(c - expected) < tolerance * expected for c in counts.values()
    )


def test_low_bits_of_a_wide_prime_are_uniform():
    ring = Ring(N, prime_size=[60], split_degree=1)
    values = _residues(ring, 16)
    assert all(v < ring.primes[0] for v in values)
    # 2^16 values over 16 classes: a rounding pattern in the low bits fails this.
    assert _balanced(Counter(v & 15 for v in values), len(values), 16, 0.05)


def test_a_prime_far_below_its_power_of_two_is_uniform():
    # 3 * 2^30 + 1 sits at 3/4 of 2^32: mapping 32 random bits onto it gives
    # some values two preimages and others one.
    q = 3 * 2**30 + 1
    ring = Ring(N, primes=[q], prime_size=[32], split_degree=1)
    values = _residues(ring, 32)
    assert all(v < q for v in values)
    assert _balanced(Counter(v % 3 for v in values), len(values), 3)
    assert _balanced(Counter(v * 4 // q for v in values), len(values), 4)


def test_lwe_mask_is_uniform():
    q = 3 * 2**30 + 1
    ring = Ring(N, primes=[q], prime_size=[32], split_degree=1)
    key = LWE_Key(ring, sec_sigma=3.2, err_sigma=3.2)
    values = [v for _ in range(32) for v in LWE(ring=ring, m=[0], key=key).get_a()[0]]
    assert all(v < q for v in values)
    assert _balanced(Counter(v * 4 // q for v in values), len(values), 4)
