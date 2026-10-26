# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Bootstrapping: the CGGI16 functional bootstrap, the GP25 rotation of many
accumulators, and the SHIP bootstrap of CKKS slots.

Each shape list starts with a small parameter set that only exercises the code; the
larger ones are those the schemes are run at. Key generation is part of the setup,
not of the time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from harness import Case, Shape, benchmark
from vfhe.arith import Polynomial, Ring
from vfhe.crypto import entropy
from vfhe.fhe import CGGI16, GP25, SHIP, CKKS_Ciphertext, CKKS_Scheme
from vfhe.mlwe import LWE, LWE_Key, MLWE_Scheme

if TYPE_CHECKING:
    from vfhe.mlwe import MLWE, MLWE_Set

#: Ring degree 2^log_n with the given primes, the last of them special; output
#: keys of ``rank`` polynomials of Hamming weight ``weight`` and noise ``sigma``;
#: binary input LWE keys of dimension ``n``. The second set is the module-rank-7
#: ring of [ZYL+17]-style parameters with a short input key, so the key fits in
#: memory.
CGGI16_SHAPES = [
    Shape(
        "n256-r1-in128",
        log_n=8,
        primes=[30, 45, 45],
        rank=1,
        sigma=3.2,
        n=128,
        weight=64,
    ),
    Shape(
        "n1024-r7-in64",
        log_n=10,
        primes=[30, 45, 45, 45, 45],
        rank=7,
        sigma=2.0**15,
        n=64,
        weight=3584,
    ),
]

#: ``count`` accumulators over a ring of degree 2^log_n, rotated by an encrypted
#: amount of ``bits`` bits.
GP25_SHAPES = [
    Shape(
        "n64-x16-b4", log_n=6, prime_bits=[50, 50], special_bits=[50], count=16, bits=4
    ),
    Shape(
        "n2048-x1024-b1",
        log_n=11,
        prime_bits=[50] * 4,
        special_bits=[55],
        count=1024,
        bits=1,
    ),
    Shape(
        "n2048-x1024-b9",
        log_n=11,
        prime_bits=[50] * 4,
        special_bits=[55],
        count=1024,
        bits=9,
    ),
]

#: [CHKS25, Table 2]'s chains, bottom to top: q0, SlotToCoeff, the levels left to the
#: application and the product tree; then the special primes. ``h`` is the input
#: key's Hamming weight, ``theta`` and ``base`` the column and mux step sizes; the
#: small set also fixes ``gamma`` and the window.
SHIP_SHAPES = [
    Shape(
        "toy-n256",
        log_n=8,
        prime_bits=[25, 20, 20] + [25] * 3,
        special_bits=[28],
        h=4,
        theta=6,
        base=4,
        gamma=32.0,
        window=16,
    ),
    Shape(
        "LL13",
        log_n=13,
        prime_bits=[25, 20, 20] + [25] * 5,
        special_bits=[28],
        h=31,
        theta=6,
        base=4,
        gamma=None,
        window=None,
    ),
]

#: The precision of the functional bootstrap's messages, and the size of its table.
MESSAGE_BITS = 5
TABLE_SIZE = 1 << (MESSAGE_BITS - 1)


def mod_switch(v: int, q: int, p: int) -> int:
    """Returns ``v`` scaled from ``Z_q`` to ``Z_p``, rounded half up."""
    return ((2 * v * p + q) // (2 * q)) % p


@benchmark(shapes=CGGI16_SHAPES, reps=3)
def cggi16_functional_bootstrap(shape: Shape) -> Case:
    """One LWE sample through a lookup table by the CGGI16 blind rotation, extracted
    back to an LWE sample. The rotation runs on the library's threads."""
    ring = Ring(1 << shape["log_n"], prime_size=shape["primes"], split_degree=1)
    scheme = MLWE_Scheme(ring, special_primes=1, module_rank=shape["rank"], max_lvl=1)
    output_key = scheme.key_gen_sparse(shape["weight"], shape["sigma"], ternary=False)
    lwe_ring = scheme.rings[0]
    n = shape["n"]
    input_key = LWE_Key(
        ring=lwe_ring, key=[entropy.below(2) for _ in range(n)], n=n, err_sigma=3.2
    )
    cggi16 = CGGI16(scheme)
    bk = cggi16.generate_bootstrap_key(input_key, output_key)
    table = [(3 * i + 1) % TABLE_SIZE for i in range(TABLE_SIZE)]
    test_vector = cggi16.LUT_packing(table, TABLE_SIZE, MESSAGE_BITS)
    message, q = 5, lwe_ring.q_l
    scaled = mod_switch(message, 1 << MESSAGE_BITS, q)
    sample = LWE(ring=lwe_ring, m=[scaled % p for p in lwe_ring.primes], key=input_key)
    out = LWE(ring=lwe_ring)
    extracted = output_key.extract_lwe_key()
    decryption_key = LWE_Key(ring=lwe_ring, key=extracted.get_s(), n=extracted.n)

    def op() -> LWE:
        cggi16.functional_bootstrap(out, test_vector, sample, bk, torus_base=TABLE_SIZE)
        return out

    def check(result: LWE) -> None:
        value = cast("int", result.linear_decrypt(decryption_key, recompose=True))
        assert mod_switch(value, q, 1 << MESSAGE_BITS) == table[message]

    return Case(op, check=check)


def rotated(d: int, n: int) -> list[int]:
    """What accumulator ``k`` reads after a rotation by ``d``: the value of
    accumulator ``k - d``, negated once per wrap past the start."""
    out = []
    for k in range(n):
        j, wraps = k - d, 0
        while j < 0:
            j, wraps = j + n, wraps + 1
        out.append(j + 1 if wraps % 2 == 0 else -(j + 1))
    return out


@benchmark(shapes=GP25_SHAPES, reps=3)
def gp25_rotate(shape: Shape) -> Case:
    """``count`` accumulators rotated by an encrypted amount, one MGSW key per bit [GP25].

    Rotation works in place, so each call first copies the accumulators into
    buffers allocated once; the copy is part of the time.
    """
    n_ring = 1 << shape["log_n"]
    ring = Ring(
        n_ring, prime_size=shape["prime_bits"] + shape["special_bits"], split_degree=1
    )
    scheme = MLWE_Scheme(ring, special_primes=len(shape["special_bits"]))
    key = scheme.key_gen_sparse(n_ring // 8, 3.2)
    gp25 = GP25(scheme)
    automorphism_key = cast(
        "MLWE_Set", scheme.gen_ksk_automorphism(key, key, 2 * n_ring - 1, lvl=0)
    )
    count, bits = shape["count"], shape["bits"]
    d = (1 << bits) - 1
    (keys,) = gp25.encrypt_rotations([d], bits, count, key)
    delta = 1 << 60
    top = scheme.rings[0]
    accumulators = [
        scheme.sample(Polynomial(top).from_bigint_array([0, (k + 1) * delta]), key)
        for k in range(count)
    ]
    for c in accumulators:
        c.to_coeff()
    work = [c.copy() for c in accumulators]
    expected = rotated(d, count)

    def op() -> list[MLWE]:
        for w, c in zip(work, accumulators, strict=True):
            w.copy_from(c)
        gp25.rotate(work, keys, automorphism_key)
        return work

    def read(c: MLWE) -> int:
        """``k`` for a sample of ``k delta X``, ``-k`` for ``-k delta X^(N-1)``."""
        v = scheme.linear_decrypt(c, key).get_polynomial(signed=True)
        return round(v[1] / delta) if abs(v[1]) > delta // 2 else round(v[-1] / delta)

    def check(result: list[MLWE]) -> None:
        for k in (0, 1, count - 1):
            assert read(result[k]) == expected[k]

    return Case(op, check=check, items=count)


@benchmark(shapes=SHIP_SHAPES, reps=3)
def ship_bootstrap(shape: Shape) -> Case:
    """The SHIP bootstrap [CHKS25] of ``N/2`` real slots, from the SlotToCoeff level.

    The check bounds every slot's error by 1/8: it rejects a wrong result, and says
    nothing about the bootstrap's precision.
    """
    n_ring = 1 << shape["log_n"]
    ring = Ring(
        n_ring, prime_size=shape["prime_bits"] + shape["special_bits"], split_degree=1
    )
    scheme = CKKS_Scheme(
        ring, scaling_factor=2.0**25, special_primes=len(shape["special_bits"])
    )
    options = {k: shape[k] for k in ("gamma", "window") if shape[k] is not None}
    ship = SHIP(
        scheme, h=shape["h"], theta=shape["theta"], base=shape["base"], **options
    )
    dense = scheme.key_gen_sparse(n_ring // 2, 3.2)
    scheme.rlk = scheme.gen_rlk(dense, dense)
    sparse = ship.sample_input_key(3.2)
    key = ship.generate_bootstrap_key(dense, sparse)
    slots = n_ring // 2
    values = [((i * 7919) % 2000) / 1000 - 1 for i in range(slots)]
    lvl = ship.slot_to_coeff_lvl
    c = CKKS_Ciphertext(scheme, lvl=lvl)
    scheme.sample(
        scheme.encode(values, ring=scheme.rings[lvl], scale=ship.input_scale),
        dense,
        out=c,
    )
    c.delta = ship.input_scale

    def check(result: CKKS_Ciphertext) -> None:
        got = scheme.decode(scheme.decrypt(result, dense), result.delta)
        assert max(abs(g.real - v) for g, v in zip(got, values, strict=True)) < 0.125

    return Case(lambda: ship.bootstrap(c, key), check=check)
