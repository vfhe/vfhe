# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Module-LWE operations: key switching, the MGSW external product and the gates and
packings built on it, and key generation.

Every scheme here is rank 1 with special primes, so key switches divide by them
[GHS12]. Ciphertexts are at the top level, and keys are generated for that level only.
"""

from __future__ import annotations

from harness import Case, Shape, benchmark
from vfhe.arith import Polynomial, Ring, RNSPolynomial, RNSRing
from vfhe.mlwe import MLWE, CMUX_batch, MGSW_Scheme, MLWE_Key, MLWE_Scheme

SHAPES = ["ckks-n12-3p", "ckks-n13-7p", "ckks-n14-7p"]

#: Rings of 50-bit primes and many samples, for the operations that run over a
#: batch: ``count`` samples, a ring of degree 2^log_n.
BATCH_SHAPES = [
    Shape("n10-2p-x16", log_n=10, prime_bits=[50, 50], special_bits=[50], count=16),
    Shape("n13-4p-x256", log_n=13, prime_bits=[50] * 4, special_bits=[50], count=256),
]

# A generator of the cyclic part of the Galois group of X^N + 1.
ROTATION = 5


def scheme_of(shape: Shape) -> MLWE_Scheme:
    """Returns the rank-1 scheme over the shape's chain and special primes."""
    ring = Ring(
        1 << shape["log_n"],
        prime_size=shape["prime_bits"] + shape["special_bits"],
        split_degree=1,
    )
    return MLWE_Scheme(ring, special_primes=len(shape["special_bits"]))


def plaintext_ring(scheme: MLWE_Scheme) -> RNSRing:
    """Returns the ring of the lowest prime, which messages are drawn from."""
    return scheme.rings[0].quotient_ring(ell=1)


def encrypt(scheme: MLWE_Scheme, message: RNSPolynomial, key: MLWE_Key) -> MLWE:
    """Encrypts ``message``, scaled by the top level's modulus over the plaintext's."""
    top = scheme.rings[0]
    delta = top.modulus_ratio(message.ring, return_pointer=True)
    return scheme.sample(message.scaled_lift(top, delta=delta), key)


def decrypt(
    scheme: MLWE_Scheme, c: MLWE, key: MLWE_Key, ring: RNSRing
) -> RNSPolynomial:
    """Returns the message of ``c``, divided back down into ``ring``."""
    return scheme.linear_decrypt(c, key).round_division(ring)


@benchmark(shapes=SHAPES, tier="headline")
def key_switch(shape: Shape) -> Case:
    """One ciphertext switched to another key.

    The input is in the coefficient domain, where it stays after the first call.
    """
    scheme = scheme_of(shape)
    rp = plaintext_ring(scheme)
    key, other = scheme.key_gen_sparse(64, 3.2), scheme.key_gen_sparse(64, 3.2)
    ksk = scheme.gen_ksk(other, key, lvl=0)
    message = rp.random_element()
    c = encrypt(scheme, message, key)

    def check(result: MLWE) -> None:
        assert decrypt(scheme, result, other, rp) == message

    return Case(lambda: scheme.keyswitch(c, ksk), check=check)


@benchmark(shapes=SHAPES, tier="headline")
def external_product(shape: Shape) -> Case:
    """The MGSW encryption of 1 times a ciphertext, into a new ciphertext.

    The ciphertext is in the coefficient domain, where it stays after the first call.
    """
    scheme = scheme_of(shape)
    rp = plaintext_ring(scheme)
    key = scheme.key_gen_sparse(64, 3.2)
    (one,) = MGSW_Scheme(scheme).encrypt_constants([1], key)
    message = rp.random_element()
    c = encrypt(scheme, message, key)

    def check(result: MLWE) -> None:
        assert decrypt(scheme, result, key, rp) == message

    return Case(lambda: one.external_product(c), check=check)


@benchmark(shapes=SHAPES)
def automorphism_batch(shape: Shape) -> Case:
    """``X -> X^5`` on 8 ciphertexts in one call, each switched back to the key."""
    scheme = scheme_of(shape)
    rp = plaintext_ring(scheme)
    key = scheme.key_gen_sparse(64, 3.2)
    ksk = scheme.gen_ksk_automorphism(key, key, ROTATION, lvl=0)
    messages = [rp.random_element() for _ in range(8)]
    cs = [encrypt(scheme, m, key) for m in messages]

    def check(results: list[MLWE]) -> None:
        assert decrypt(scheme, results[0], key, rp) == messages[0].automorphism(
            ROTATION
        )

    return Case(
        lambda: scheme.automorphism_batch(cs, [ROTATION] * 8, [ksk] * 8),
        check=check,
        items=8,
    )


@benchmark(shapes=BATCH_SHAPES)
def cmux_batch(shape: Shape) -> Case:
    """``CMUX`` of ``count`` pairs under one MGSW selector, the selector encrypting 1."""
    scheme = scheme_of(shape)
    rp = plaintext_ring(scheme)
    key = scheme.key_gen_sparse(64, 3.2)
    (selector,) = MGSW_Scheme(scheme).encrypt_constants([1], key)
    count = shape["count"]
    first = [encrypt(scheme, rp.random_element(), key) for _ in range(count)]
    chosen = [rp.random_element() for _ in range(count)]
    second = [encrypt(scheme, m, key) for m in chosen]

    def check(results: list[MLWE]) -> None:
        assert decrypt(scheme, results[0], key, rp) == chosen[0]

    return Case(lambda: CMUX_batch(first, second, selector), check=check, items=count)


@benchmark(shapes=BATCH_SHAPES)
def trace_pack(shape: Shape) -> Case:
    """The constant coefficients of ``count`` samples packed into one [CDKS21, LY26].

    The packing overwrites its inputs, so each call first copies the samples into
    buffers allocated once; the copy is part of the time.
    """
    scheme = scheme_of(shape)
    key = scheme.key_gen_sparse(64, 3.2)
    n, count = scheme.N, shape["count"]
    ksk = scheme.gen_ksk_trace(
        key, key, gens=[(1 << j) + 1 for j in range(1, n.bit_length())], lvl=0
    )
    delta = 1 << 60
    top = scheme.rings[0]
    samples = [
        scheme.sample(Polynomial(top).from_bigint_array([(k + 1) * delta]), key)
        for k in range(count)
    ]
    work = [s.copy() for s in samples]

    def op() -> MLWE:
        for w, s in zip(work, samples, strict=True):
            w.copy_from(s)
        return scheme.trace_pack(work, ksk)

    def check(result: MLWE) -> None:
        d = scheme.linear_decrypt(result, key).get_polynomial(signed=True)
        stride = n // count
        for k in (0, 1, count - 1):
            assert round(d[k * stride] / delta) == k + 1

    return Case(op, check=check)


@benchmark(shapes=SHAPES)
def relinearization_key_generation(shape: Shape) -> Case:
    """The relinearization key of the top level, for the key's square."""
    scheme = scheme_of(shape)
    key = scheme.key_gen_sparse(64, 3.2)
    return Case(lambda: scheme.gen_rlk(key, key, lvl=0))
