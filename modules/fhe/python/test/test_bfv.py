# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Tests for the vfhe.fhe BFV scheme.

Batched encode/decode, encrypt/decrypt, the homomorphic operations (add, sub,
plaintext multiplication, ciphertext multiplication with relinearization), slot
rotation and conjugation, and modulus switching.
"""

import pytest
from vfhe.arith import Ring
from vfhe.crypto import entropy
from vfhe.fhe import BFV_Scheme

N = 64

#: Module ranks above 1, with a ring dimension that keeps N*r (and the runtime)
#: in line with the rank-1 tests.
RANK_DIMS = [(2, 32), (4, 16)]


def make_scheme(n=N, rank=1, special_primes=0, prime_size=None):
    """A BFV scheme whose plaintext modulus is the ring's lowest (17-bit) prime."""
    if prime_size is None:
        prime_size = [17, 45, 45, 45] + [50] * special_primes
    ring = Ring(n, prime_size=prime_size, split_degree=1)
    return BFV_Scheme(ring, module_rank=rank, special_primes=special_primes)


def rand_slots(scheme):
    return [entropy.below(scheme.t) - scheme.t // 2 for _ in range(scheme.n_slots)]


def centered(scheme, values):
    return [(v + scheme.t // 2) % scheme.t - scheme.t // 2 for v in values]


def test_encode_decode():
    scheme = make_scheme()
    values = rand_slots(scheme)
    assert scheme.decode(scheme.encode(values)) == centered(scheme, values)


def test_encode_rejects_wrong_length():
    scheme = make_scheme()
    with pytest.raises(ValueError, match="Expected"):
        scheme.encode([1, 2, 3])


def test_encrypt_decrypt():
    scheme = make_scheme()
    key = scheme.key_gen_sparse(N // 8, 3.2)
    values = rand_slots(scheme)
    ct = scheme.encrypt(scheme.encode(values), key)
    assert scheme.decode(scheme.decrypt(ct, key)) == centered(scheme, values)


def test_add_sub_and_plaintext_mul():
    scheme = make_scheme()
    key = scheme.key_gen_sparse(N // 8, 3.2)
    a, b = rand_slots(scheme), rand_slots(scheme)
    ca = scheme.encrypt(scheme.encode(a), key)
    cb = scheme.encrypt(scheme.encode(b), key)

    got = scheme.decode(scheme.decrypt(ca + cb, key))
    assert got == centered(scheme, [x + y for x, y in zip(a, b, strict=True)])

    got = scheme.decode(scheme.decrypt(ca - cb, key))
    assert got == centered(scheme, [x - y for x, y in zip(a, b, strict=True)])

    got = scheme.decode(scheme.decrypt(ca * scheme.encode(b), key))
    assert got == centered(scheme, [x * y for x, y in zip(a, b, strict=True)])


def test_multiply_relinearized():
    scheme = make_scheme()
    key = scheme.key_gen_sparse(N // 8, 3.2)
    rlk = scheme.gen_rlk(key, key)
    a, b = rand_slots(scheme), rand_slots(scheme)
    ca = scheme.encrypt(scheme.encode(a), key)
    cb = scheme.encrypt(scheme.encode(b), key)

    prod = scheme.multiply(ca, cb, rlk)
    got = scheme.decode(scheme.decrypt(prod, key))
    assert got == centered(scheme, [x * y for x, y in zip(a, b, strict=True)])


def test_multiply_extended_then_relinearize():
    """``ksk=None`` returns the extended product; relinearizing it agrees."""
    scheme = make_scheme()
    key = scheme.key_gen_sparse(N // 8, 3.2)
    rlk = scheme.gen_rlk(key, key)
    a, b = rand_slots(scheme), rand_slots(scheme)
    ca = scheme.encrypt(scheme.encode(a), key)
    cb = scheme.encrypt(scheme.encode(b), key)

    ext = scheme.multiply(ca, cb)
    assert ext.is_extended
    assert ext.r == scheme.extended_rank
    got = scheme.decode(scheme.decrypt(scheme.relinearize(ext, rlk), key))
    assert got == centered(scheme, [x * y for x, y in zip(a, b, strict=True)])


def test_multiply_operator_uses_scheme_rlk():
    scheme = make_scheme()
    key = scheme.key_gen_sparse(N // 8, 3.2)
    scheme.rlk = scheme.gen_rlk(key, key)
    a, b = rand_slots(scheme), rand_slots(scheme)
    ca = scheme.encrypt(scheme.encode(a), key)
    cb = scheme.encrypt(scheme.encode(b), key)

    got = scheme.decode(scheme.decrypt(ca * cb, key))
    assert got == centered(scheme, [x * y for x, y in zip(a, b, strict=True)])


def test_multiplication_depth_two():
    scheme = make_scheme(prime_size=[17, 45, 45, 45, 45])
    key = scheme.key_gen_sparse(N // 8, 3.2)
    rlk = scheme.gen_rlk(key, key)
    a, b, c = (rand_slots(scheme) for _ in range(3))
    ct = scheme.multiply(
        scheme.encrypt(scheme.encode(a), key),
        scheme.encrypt(scheme.encode(b), key),
        rlk,
    )
    ct = scheme.multiply(ct, scheme.encrypt(scheme.encode(c), key), rlk)
    got = scheme.decode(scheme.decrypt(ct, key))
    assert got == centered(scheme, [x * y * z for x, y, z in zip(a, b, c, strict=True)])


@pytest.mark.parametrize(("rank", "n"), RANK_DIMS)
def test_multiply_at_higher_rank(rank, n):
    scheme = make_scheme(n=n, rank=rank)
    key = scheme.key_gen_sparse(n // 8, 3.2)
    rlk = scheme.gen_rlk(key, key)
    a, b = rand_slots(scheme), rand_slots(scheme)
    prod = scheme.multiply(
        scheme.encrypt(scheme.encode(a), key),
        scheme.encrypt(scheme.encode(b), key),
        rlk,
    )
    got = scheme.decode(scheme.decrypt(prod, key))
    assert got == centered(scheme, [x * y for x, y in zip(a, b, strict=True)])


@pytest.mark.parametrize("k", [1, 3, 7])
def test_rotation(k):
    scheme = make_scheme(special_primes=1)
    key = scheme.key_gen_sparse(N // 8, 3.2)
    ksk = scheme.gen_rotation_key(key, k)
    values = rand_slots(scheme)
    ct = scheme.rotate(scheme.encrypt(scheme.encode(values), key), k, ksk)

    half = scheme.n_slots // 2
    rows = [values[:half], values[half:]]
    expected = [row[(i + k) % half] for row in rows for i in range(half)]
    assert scheme.decode(scheme.decrypt(ct, key)) == centered(scheme, expected)


def test_conjugation_swaps_the_two_rows():
    scheme = make_scheme(special_primes=1)
    key = scheme.key_gen_sparse(N // 8, 3.2)
    ksk = scheme.gen_conjugation_key(key)
    values = rand_slots(scheme)
    ct = scheme.conjugate(scheme.encrypt(scheme.encode(values), key), ksk)

    half = scheme.n_slots // 2
    expected = values[half:] + values[:half]
    assert scheme.decode(scheme.decrypt(ct, key)) == centered(scheme, expected)


def test_mod_switch_preserves_the_message():
    scheme = make_scheme()
    key = scheme.key_gen_sparse(N // 8, 3.2)
    values = rand_slots(scheme)
    ct = scheme.encrypt(scheme.encode(values), key)
    ct = scheme.mod_switch(ct)
    assert ct.lvl == 1
    assert ct.ring == scheme.rings[1]
    assert scheme.decode(scheme.decrypt(ct, key)) == centered(scheme, values)


def test_multiply_after_mod_switch():
    scheme = make_scheme()
    key = scheme.key_gen_sparse(N // 8, 3.2)
    rlk = scheme.gen_rlk(key, key)
    a, b = rand_slots(scheme), rand_slots(scheme)
    ca = scheme.mod_switch(scheme.encrypt(scheme.encode(a), key))
    cb = scheme.mod_switch(scheme.encrypt(scheme.encode(b), key))
    prod = scheme.multiply(ca, cb, rlk)
    got = scheme.decode(scheme.decrypt(prod, key))
    assert got == centered(scheme, [x * y for x, y in zip(a, b, strict=True)])


def test_multi_prime_plaintext_modulus():
    scheme = BFV_Scheme(
        Ring(N, prime_size=[17, 17, 45, 45, 45], split_degree=1), plaintext_primes=2
    )
    assert scheme.plaintext_ring.ell == 2
    assert scheme.t == scheme.plaintext_ring.q_l
    key = scheme.key_gen_sparse(N // 8, 3.2)
    rlk = scheme.gen_rlk(key, key)
    a, b = rand_slots(scheme), rand_slots(scheme)
    prod = scheme.multiply(
        scheme.encrypt(scheme.encode(a), key),
        scheme.encrypt(scheme.encode(b), key),
        rlk,
    )
    got = scheme.decode(scheme.decrypt(prod, key))
    assert got == centered(scheme, [x * y for x, y in zip(a, b, strict=True)])


def test_primes_wider_than_a_multiprecision_digit():
    """Nothing in the product goes through 52-bit multiprecision digits."""
    scheme = make_scheme(prime_size=[17, 60, 60, 60])
    assert max(scheme.rings[0].primes).bit_length() > 52
    key = scheme.key_gen_sparse(N // 8, 3.2)
    rlk = scheme.gen_rlk(key, key)
    a, b = rand_slots(scheme), rand_slots(scheme)
    prod = scheme.multiply(
        scheme.encrypt(scheme.encode(a), key),
        scheme.encrypt(scheme.encode(b), key),
        rlk,
    )
    got = scheme.decode(scheme.decrypt(prod, key))
    assert got == centered(scheme, [x * y for x, y in zip(a, b, strict=True)])
