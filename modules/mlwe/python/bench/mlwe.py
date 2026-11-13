# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Module-LWE operations: key switching, the MGSW external product and the gates and
packings built on it, and key generation.

Every scheme here is rank 1 with special primes, so key switches divide by them
[GHS12]. Ciphertexts are at the top level, and keys are generated for that level only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from vfhe.arith import (
    IntegerMatrix,
    Polynomial,
    Ring,
    RNSPolynomial,
    RNSRing,
    set_generic_only,
)
from vfhe.mlwe import MLWE, CMUX_batch, MGSW_Scheme, MLWE_Key, MLWE_Scheme

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

#: The CKKS chains arith's polynomial benchmarks run at: ring degree 2^log_n,
#: the chain's prime sizes from level 0 up, the primes key switching adds, and
#: the encoding scale.
CKKS = {
    "n12-3p": {
        "log_n": 12,
        "prime_bits": [60, 40, 40],
        "special_bits": [60],
        "scale_bits": 40,
    },
    "n13-7p": {
        "log_n": 13,
        "prime_bits": [60] + [40] * 6,
        "special_bits": [60],
        "scale_bits": 40,
    },
    "n14-7p": {
        "log_n": 14,
        "prime_bits": [60] + [40] * 6,
        "special_bits": [60],
        "scale_bits": 40,
    },
}
chains = pytest.mark.parametrize("p", list(CKKS.values()), ids=list(CKKS))
#: Rings of 50-bit primes and many samples, for the operations that run over a
#: batch: ``count`` samples over a ring of degree 2^log_n.
BATCHES = {
    "n10-2p-x16": {
        "log_n": 10,
        "prime_bits": [50, 50],
        "special_bits": [50],
        "count": 16,
    },
    "n13-4p-x256": {
        "log_n": 13,
        "prime_bits": [50] * 4,
        "special_bits": [50],
        "count": 256,
    },
}
batches = pytest.mark.parametrize("p", list(BATCHES.values()), ids=list(BATCHES))

# A generator of the cyclic part of the Galois group of X^N + 1.
ROTATION = 5


def _scheme(p: dict) -> MLWE_Scheme:
    """The rank-1 scheme over the chain and its special primes."""
    ring = Ring(
        1 << p["log_n"], prime_size=p["prime_bits"] + p["special_bits"], split_degree=1
    )
    return MLWE_Scheme(ring, special_primes=len(p["special_bits"]))


def _plaintext_ring(scheme: MLWE_Scheme) -> RNSRing:
    """The ring of the lowest prime, which messages are drawn from."""
    return scheme.rings[0].quotient_ring(ell=1)


def _encrypt(scheme: MLWE_Scheme, message: RNSPolynomial, key: MLWE_Key) -> MLWE:
    """Encrypts ``message``, scaled by the top level's modulus over the plaintext's."""
    top = scheme.rings[0]
    delta = top.modulus_ratio(message.ring, return_pointer=True)
    return scheme.sample(message.scaled_lift(top, delta=delta), key)


def _decrypt(
    scheme: MLWE_Scheme, c: MLWE, key: MLWE_Key, ring: RNSRing
) -> RNSPolynomial:
    """The message of ``c``, divided back down into ``ring``."""
    return scheme.linear_decrypt(c, key).round_division(ring)


@chains
def key_switch(benchmark: BenchmarkFixture, p: dict) -> None:
    """One ciphertext switched to another key.

    The input is in the coefficient domain, where it stays after the first call.
    """
    scheme = _scheme(p)
    rp = _plaintext_ring(scheme)
    key, other = scheme.key_gen_sparse(64, 3.2), scheme.key_gen_sparse(64, 3.2)
    ksk = scheme.gen_ksk(other, key, lvl=0)
    message = rp.random_element()
    c = _encrypt(scheme, message, key)
    assert _decrypt(scheme, benchmark(scheme.keyswitch, c, ksk), other, rp) == message


@chains
def external_product(benchmark: BenchmarkFixture, p: dict) -> None:
    """The MGSW encryption of 1 times a ciphertext, into a new ciphertext.

    The ciphertext is in the coefficient domain, where it stays after the first call.
    """
    scheme = _scheme(p)
    rp = _plaintext_ring(scheme)
    key = scheme.key_gen_sparse(64, 3.2)
    (one,) = MGSW_Scheme(scheme).encrypt_constants([1], key)
    message = rp.random_element()
    c = _encrypt(scheme, message, key)
    assert _decrypt(scheme, benchmark(one.external_product, c), key, rp) == message


@chains
def automorphism_batch(benchmark: BenchmarkFixture, p: dict) -> None:
    """``X -> X^5`` on 8 ciphertexts in one call, each switched back to the key."""
    scheme = _scheme(p)
    rp = _plaintext_ring(scheme)
    key = scheme.key_gen_sparse(64, 3.2)
    ksk = scheme.gen_ksk_automorphism(key, key, ROTATION, lvl=0)
    messages = [rp.random_element() for _ in range(8)]
    cs = [_encrypt(scheme, m, key) for m in messages]
    rotated = benchmark(scheme.automorphism_batch, cs, [ROTATION] * 8, [ksk] * 8)
    assert _decrypt(scheme, rotated[0], key, rp) == messages[0].automorphism(ROTATION)


@batches
def cmux_batch(benchmark: BenchmarkFixture, p: dict) -> None:
    """``CMUX`` of ``count`` pairs under one MGSW selector, the selector encrypting 1."""
    scheme = _scheme(p)
    rp = _plaintext_ring(scheme)
    key = scheme.key_gen_sparse(64, 3.2)
    (selector,) = MGSW_Scheme(scheme).encrypt_constants([1], key)
    first = [_encrypt(scheme, rp.random_element(), key) for _ in range(p["count"])]
    chosen = [rp.random_element() for _ in range(p["count"])]
    second = [_encrypt(scheme, m, key) for m in chosen]
    assert (
        _decrypt(scheme, benchmark(CMUX_batch, first, second, selector)[0], key, rp)
        == chosen[0]
    )


@batches
def trace_pack(benchmark: BenchmarkFixture, p: dict) -> None:
    """The constant coefficients of ``count`` samples packed into one [CDKS21, LY26].

    The packing overwrites its inputs, so each call first copies the samples into
    buffers allocated once; the copy is part of the time.
    """
    scheme = _scheme(p)
    key = scheme.key_gen_sparse(64, 3.2)
    n, count = scheme.N, p["count"]
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

    def op():
        for w, s in zip(work, samples, strict=True):
            w.copy_from(s)
        return scheme.trace_pack(work, ksk)

    d = scheme.linear_decrypt(benchmark(op), key).get_polynomial(signed=True)
    for k in (0, 1, count - 1):
        assert round(d[k * (n // count)] / delta) == k + 1


@chains
def relinearization_key_generation(benchmark: BenchmarkFixture, p: dict) -> None:
    """The relinearization key of the top level, for the key's square."""
    scheme = _scheme(p)
    key = scheme.key_gen_sparse(64, 3.2)
    benchmark(scheme.gen_rlk, key, key, lvl=0)


#: Integer matrices times vectors of samples: the scheme's chain and special
#: primes, the matrix's outputs x inputs, and its pattern (dense by default).
MATRICES = {
    "n10-2p-8x64": {
        "log_n": 10,
        "prime_bits": [50, 50],
        "special_bits": [50],
        "outputs": 8,
        "inputs": 64,
    },
    "n12-2p-138x1024": {
        "log_n": 12,
        "prime_bits": [50, 50],
        "special_bits": [50],
        "outputs": 138,
        "inputs": 1024,
    },
    "n12-59b-138x1024": {
        "log_n": 12,
        "prime_bits": [59, 59],
        "special_bits": [60],
        "outputs": 138,
        "inputs": 1024,
    },
    "n12-2p-138x1024-sparse": {
        "log_n": 12,
        "prime_bits": [50, 50],
        "special_bits": [50],
        "outputs": 138,
        "inputs": 1024,
        "pattern": "sparse",
    },
    "n12-2p-512x512-select": {
        "log_n": 12,
        "prime_bits": [50, 50],
        "special_bits": [50],
        "outputs": 512,
        "inputs": 512,
        "pattern": "select",
    },
    "n12-2p-138x2": {
        "log_n": 12,
        "prime_bits": [50, 50],
        "special_bits": [50],
        "outputs": 138,
        "inputs": 2,
    },
}
matrices = pytest.mark.parametrize("p", list(MATRICES.values()), ids=list(MATRICES))


def _matrix_rows(p: dict) -> list[list[int]]:
    n_out, n_in = p["outputs"], p["inputs"]
    pattern = p.get("pattern", "dense")
    if pattern == "select":
        return [[int(j == (7 * r) % n_in) for j in range(n_in)] for r in range(n_out)]
    dense = [
        [(31 * r + 17 * j) % 2049 - 1024 for j in range(n_in)] for r in range(n_out)
    ]
    if pattern == "sparse":
        return [
            [v | 1 if (7 * r + 13 * j) % 33 == 0 else 0 for j, v in enumerate(row)]
            for r, row in enumerate(dense)
        ]
    return dense


def _integer_linear_combinations(
    benchmark: BenchmarkFixture, p: dict, generic: bool
) -> None:
    ring = Ring(
        1 << p["log_n"], prime_size=p["prime_bits"] + p["special_bits"], split_degree=1
    )
    scheme = MLWE_Scheme(ring, special_primes=len(p["special_bits"]))
    key = scheme.key_gen_sparse(64, 3.2)
    top = scheme.rings[0]
    cts = [scheme.sample(top.random_element(), key) for _ in range(p["inputs"])]
    rows = _matrix_rows(p)
    matrix = IntegerMatrix(rows)
    decryptions = [scheme.linear_decrypt(c.copy(), key) for c in cts]

    def op() -> list[MLWE]:
        set_generic_only(generic)
        try:
            return scheme.linear_combinations(cts, matrix)
        finally:
            set_generic_only(False)

    result = benchmark(op)
    expected = decryptions[0] * rows[0][0]
    for w, d in zip(rows[0][1:], decryptions[1:], strict=True):
        expected = expected + d * w
    assert scheme.linear_decrypt(result[0], key) == expected


@matrices
def integer_linear_combinations(benchmark: BenchmarkFixture, p: dict) -> None:
    """An integer matrix times a vector of samples, one matrix kernel per component."""
    _integer_linear_combinations(benchmark, p, generic=False)


@matrices
def integer_linear_combinations_generic_path(
    benchmark: BenchmarkFixture, p: dict
) -> None:
    """The same product on the generic path: one pass over a sample per weight."""
    _integer_linear_combinations(benchmark, p, generic=True)
