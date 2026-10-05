# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Characterization test for the (reverted) vfhe.fhe CGGI16 functional
bootstrap over the cffi boundary: bootstrap-key generation, LUT packing, blind
rotation (CMUX), and LWE extraction, verified against a random lookup table.
"""

import random

import pytest
from vfhe import engine
from vfhe.arith import Ring
from vfhe.fhe import CGGI16, mod_switch
from vfhe.fhe.cggi16 import KEY_COMBINATIONS, PARALLELISMS, unfolded_key_count
from vfhe.mlwe import LWE, LWE_Key, MLWE_Scheme


@pytest.mark.complete
def test_functional_bootstrap(deterministic_prng):
    # Bootstrapping is probabilistic; pin the C PRNG + Python RNG so this
    # exact-equality check is reproducible rather than flaky (seed chosen to
    # decrypt cleanly).
    deterministic_prng(0xB007C0DE)
    random.seed(0xB007C0DE)
    out_N = 256
    msg_prec = 5
    Rq = Ring(out_N, prime_size=[50, 50, 50], split_degree=1)
    _Rp = Rq.quotient_ring(ell=1)
    in_ring = Rq.quotient_ring(ell=2)
    in_scheme = MLWE_Scheme(in_ring, special_primes=0, module_rank=1)
    out_scheme = MLWE_Scheme(Rq, special_primes=1, module_rank=1, max_lvl=1)

    input_key = in_scheme.key_gen_sparse(17, 3.2, ternary=False)
    input_lwe_key = input_key.extract_lwe_key()
    output_key = out_scheme.key_gen_sparse(64, 3.2, ternary=False)

    cggi16 = CGGI16(out_scheme)
    bk_key = cggi16.generate_bootstrap_key(input_lwe_key, output_key)

    lut_size = 1 << (msg_prec - 1)
    lut = [random.randint(0, (1 << (msg_prec - 1)) - 1) for _ in range(lut_size)]  # noqa: S311 - test data, not a key
    rlwe_tv = cggi16.LUT_packing(lut, lut_size, msg_prec)

    msg_val = random.randint(0, lut_size - 1)  # noqa: S311 - test data, not a key
    m_scaled = mod_switch(msg_val, 1 << msg_prec, in_ring.q_l)
    lwe_in = LWE(
        ring=in_ring, m=[m_scaled % q for q in in_ring.primes], key=input_lwe_key
    )

    out_lwe = LWE(ring=in_ring)
    cggi16.functional_bootstrap(
        out_lwe, rlwe_tv, lwe_in, bk_key, torus_base=1 << (msg_prec - 1)
    )

    output_lwe_key = output_key.extract_lwe_key()
    output_lwe_key_ell2 = LWE_Key(
        ring=in_ring, key=output_lwe_key.get_s(), n=output_lwe_key.n
    )
    decryption = out_lwe.linear_decrypt(output_lwe_key_ell2, recompose=True)
    assert mod_switch(decryption, in_ring.q_l, 1 << msg_prec) == lut[msg_val]


class _Setup:
    """Schemes and keys for bootstrapping LWE samples of dimension ``in_N``.

    The inputs are encrypted over ``input_ring``; the bootstrapped samples
    live over ``output_ring``, the bootstrapping ring's first two primes.
    """

    def __init__(self, in_N: int = 256, msg_prec: int = 5):
        Rq = Ring(256, prime_size=[50, 50, 50], split_degree=1)
        self.output_ring = Rq.quotient_ring(ell=2)
        self.input_ring = (
            self.output_ring
            if in_N == Rq.N
            else Ring(in_N, prime_size=[50, 50], split_degree=1)
        )
        in_scheme = MLWE_Scheme(self.input_ring, special_primes=0, module_rank=1)
        self.out_scheme = MLWE_Scheme(Rq, special_primes=1, module_rank=1, max_lvl=1)
        self.input_key = in_scheme.key_gen_sparse(
            17, 3.2, ternary=False
        ).extract_lwe_key()
        self.output_key = self.out_scheme.key_gen_sparse(64, 3.2, ternary=False)
        out_lwe_key = self.output_key.extract_lwe_key()
        self.decryption_key = LWE_Key(
            ring=self.output_ring, key=out_lwe_key.get_s(), n=out_lwe_key.n
        )
        self.msg_prec = msg_prec
        self.lut_size = 1 << (msg_prec - 1)

    def lut_and_inputs(self, cggi16: CGGI16, msgs: list[int]):
        """A random LUT, its test vector, and an encryption of each message."""
        lut = [random.randint(0, self.lut_size - 1) for _ in range(self.lut_size)]  # noqa: S311 - test data, not a key
        tv = cggi16.LUT_packing(lut, self.lut_size, self.msg_prec)
        q = self.input_ring.q_l
        inputs = [
            LWE(
                ring=self.input_ring,
                m=[
                    mod_switch(m, 1 << self.msg_prec, q) % p
                    for p in self.input_ring.primes
                ],
                key=self.input_key,
            )
            for m in msgs
        ]
        return lut, tv, inputs

    def outputs(self, count: int) -> list[LWE]:
        return [LWE(ring=self.output_ring) for _ in range(count)]

    def decrypt(self, outs: list[LWE]) -> list[int]:
        q = self.output_ring.q_l
        return [
            mod_switch(
                o.linear_decrypt(self.decryption_key, recompose=True),
                q,
                1 << self.msg_prec,
            )
            for o in outs
        ]


@pytest.mark.complete
@pytest.mark.parametrize(
    ("unfolding", "variant"),
    [
        (1, "zyl17"),
        (2, "bmmp18"),
        (2, "zyl17"),
        (3, "bmmp18"),
        (3, "zyl17"),
        (4, "bmmp18"),
    ],
)
def test_unfolded_functional_bootstrap(deterministic_prng, unfolding, variant):
    deterministic_prng(0xB007C0DE + unfolding)
    random.seed(0xB007C0DE + unfolding)
    setup = _Setup()
    cggi16 = CGGI16(setup.out_scheme, unfolding=unfolding, variant=variant)
    bk = cggi16.generate_bootstrap_key(setup.input_key, setup.output_key)
    assert len(bk.bk) == unfolded_key_count(setup.input_key.n, unfolding, variant)

    msgs = list(range(setup.lut_size))
    lut, tv, inputs = setup.lut_and_inputs(cggi16, msgs)
    engine.set_num_threads(4)
    try:
        for combination in KEY_COMBINATIONS:
            for parallelism in PARALLELISMS:
                bk.combination, cggi16.parallelism = combination, parallelism
                outs = setup.outputs(len(msgs))
                for out, c in zip(outs, inputs, strict=True):
                    cggi16.functional_bootstrap(
                        out, tv, c, bk, torus_base=setup.lut_size
                    )
                assert setup.decrypt(outs) == [lut[m] for m in msgs], (
                    combination,
                    parallelism,
                )
    finally:
        engine.set_num_threads()


@pytest.mark.parametrize(
    ("unfolding", "variant", "in_N"),
    [(3, "bmmp18", 256), (3, "bmmp18", 128), (2, "zyl17", 128)],
)
def test_blind_rotation_is_deterministic(deterministic_prng, unfolding, variant, in_N):
    """Every combination, parallelism, thread count and the batch give one result.

    With unfolding 3 the last group is one coefficient at n = 256 and two at
    n = 128; n = 128 also bootstraps into a ring of another dimension. The
    combination is switched on the one key, which moves its keys between
    domains.
    """
    deterministic_prng(0xC0FFEE)
    random.seed(0xC0FFEE)
    setup = _Setup(in_N)
    cggi16 = CGGI16(setup.out_scheme, unfolding=unfolding, variant=variant)
    bk = cggi16.generate_bootstrap_key(setup.input_key, setup.output_key)
    msgs = [3, 7, 11]
    lut, tv, inputs = setup.lut_and_inputs(cggi16, msgs)

    def single(threads: int) -> list[LWE]:
        engine.set_num_threads(threads)
        outs = setup.outputs(len(msgs))
        for out, c in zip(outs, inputs, strict=True):
            cggi16.functional_bootstrap(out, tv, c, bk, torus_base=setup.lut_size)
        return outs

    def batch() -> list[LWE]:
        engine.set_num_threads(4)
        outs = setup.outputs(len(msgs))
        cggi16.functional_bootstrap_batch(
            outs, tv, inputs, bk, torus_base=setup.lut_size
        )
        return outs

    results = []
    try:
        for combination in (*KEY_COMBINATIONS, "evaluation"):
            bk.combination = combination
            for parallelism in PARALLELISMS:
                cggi16.parallelism = parallelism
                results += [single(1), single(4)]
            results.append(batch())
    finally:
        engine.set_num_threads()
    reference = results[0]
    for outs in results[1:]:
        for a, b in zip(reference, outs, strict=True):
            assert a.get_b() == b.get_b()
            assert a.get_a() == b.get_a()
    assert setup.decrypt(reference) == [lut[m] for m in msgs]


def test_unfolded_key_layout():
    assert unfolded_key_count(256, 1, "bmmp18") == 256
    assert unfolded_key_count(256, 1, "zyl17") == 512
    assert unfolded_key_count(256, 2, "bmmp18") == 128 * 3
    assert unfolded_key_count(256, 3, "bmmp18") == 85 * 7 + 1
    assert unfolded_key_count(256, 3, "zyl17") == 85 * 8 + 2
    Rq = Ring(64, prime_size=[50, 50], split_degree=1)
    scheme = MLWE_Scheme(Rq, special_primes=1, module_rank=1)
    with pytest.raises(ValueError, match="unfolding"):
        CGGI16(scheme, unfolding=0)
    with pytest.raises(ValueError, match="variant"):
        CGGI16(scheme, variant="cggi")
    with pytest.raises(ValueError, match="combination"):
        CGGI16(scheme, combination="fft")
    with pytest.raises(ValueError, match="parallelism"):
        CGGI16(scheme, parallelism="openmp")
    ternary = LWE_Key(ring=Rq.quotient_ring(ell=1), key=[1, -1] + [0] * 62, n=64)
    with pytest.raises(ValueError, match="binary"):
        CGGI16(scheme).generate_bootstrap_key(
            ternary, scheme.key_gen_sparse(8, 3.2, ternary=False)
        )
