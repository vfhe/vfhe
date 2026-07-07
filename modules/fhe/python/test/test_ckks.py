# SPDX-License-Identifier: Apache-2.0
"""Characterization tests for the (reverted) vfhe.fhe CKKS scheme over cffi.

Encode/decode, encrypt/decrypt, slot rotation, and ciphertext multiplication
(ciphertext*ciphertext with relinearization+rescale, and ciphertext*plaintext).
"""

import secrets

import pytest
from vfhe.arith import Ring
from vfhe.fhe import CKKS_Scheme

N = 256
rng = secrets.SystemRandom()


def rand_values(n):
    return [
        complex(rng.choice([-2, -1, 0, 1, 2]), rng.choice([-2, -1, 0, 1, 2]))
        for _ in range(n)
    ]


def test_encode_decode():
    scheme = CKKS_Scheme(
        Ring(N, 300, split_degree=1), scaling_factor=2**25, special_primes=1
    )
    values = rand_values(N // 2)
    dec = scheme.decode(scheme.encode(values))
    assert all(abs(v - dv) < 1e-3 for v, dv in zip(values, dec))


def test_encrypt_decrypt():
    scheme = CKKS_Scheme(
        Ring(N, 300, split_degree=1), scaling_factor=2**25, special_primes=1
    )
    key = scheme.key_gen_sparse(N // 8, 3.2)
    values = rand_values(N // 2)
    poly = scheme.encode(values)
    ct = scheme.encrypt(poly, key)
    dec = scheme.decode(scheme.decrypt(ct, key))
    assert all(abs(v - dv) < 0.05 for v, dv in zip(values, dec))


def test_rotation():
    scheme = CKKS_Scheme(
        Ring(N, 300, split_degree=1), scaling_factor=2**25, special_primes=1
    )
    key = scheme.key_gen_sparse(N // 8, 3.2)
    values = rand_values(N // 2)
    ct = scheme.encrypt(scheme.encode(values), key)
    k = 1
    ksk = scheme.gen_rotation_key(key, k)
    dec_rot = scheme.decode(scheme.decrypt(scheme.rotate(ct, k, ksk), key))
    M = N // 2
    assert all(abs(dec_rot[i] - values[(i + k) % M]) < 0.05 for i in range(M))


def test_ciphertext_multiplication():
    scheme = CKKS_Scheme(
        Ring(N, 300, split_degree=1), scaling_factor=2**49, special_primes=0
    )
    key = scheme.key_gen_sparse(N // 8, 3.2)
    s_0 = key.poly[0]
    scheme.rlk = scheme.gen_ksk(key, [-(s_0 * s_0)])

    v1 = [complex(0.5, 0.5) if i == 0 else 0 for i in range(N // 2)]
    v2 = [complex(0.4, -0.4) if i == 0 else 0 for i in range(N // 2)]
    c1 = scheme.encrypt(scheme.encode(v1), key)
    c2 = scheme.encrypt(scheme.encode(v2), key)

    c_mul = c1 * c2
    dec = scheme.decode(scheme.decrypt(c_mul, key), scaling_factor=c_mul.delta)
    expected = [a * b for a, b in zip(v1, v2)]
    assert all(abs(e - d) < 0.05 for e, d in zip(expected, dec))


def test_ciphertext_plaintext_multiplication():
    scheme = CKKS_Scheme(
        Ring(N, 300, split_degree=1), scaling_factor=2**49, special_primes=0
    )
    key = scheme.key_gen_sparse(N // 8, 3.2)

    v1 = [complex(0.5, 0.5) if i == 0 else 0 for i in range(N // 2)]
    v2 = [complex(0.4, -0.4) if i == 0 else 0 for i in range(N // 2)]
    poly2 = scheme.encode(v2)
    c1 = scheme.encrypt(scheme.encode(v1), key)

    c_mul = c1 * poly2
    dec = scheme.decode(scheme.decrypt(c_mul, key), scaling_factor=c_mul.delta)
    expected = [a * b for a, b in zip(v1, v2)]
    assert all(abs(e - d) < 0.05 for e, d in zip(expected, dec))
