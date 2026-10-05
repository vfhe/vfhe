# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

from vfhe.arith import Polynomial, repr
from vfhe.arith.number_theory import crt
from vfhe.engine import ffi, lib
from vfhe.mlwe.lwe import LWE, LWE_Key, lib_lwe
from vfhe.mlwe.mgsw import MGSW, MGSW_Scheme
from vfhe.mlwe.mlwe import MLWE, MLWE_Key, MLWE_Scheme, lib_rlwe

UNFOLDING_VARIANTS = ("bmmp18", "zyl17")
# How a group's keys are combined: the C enum ZYL17_Combination, in order.
KEY_COMBINATIONS = ("evaluation", "precomputed", "coefficient")
# How one rotation uses several threads: the C enum CGGI16_Parallelism, in order.
PARALLELISMS = ("pipeline", "data_parallel")


def unfolded_group_sizes(n: int, unfolding: int) -> list[int]:
    """The sizes of the groups ``n`` input coefficients split into."""
    full, last = divmod(n, unfolding)
    return [unfolding] * full + ([last] if last else [])


def unfolded_key_count(n: int, unfolding: int, variant: str) -> int:
    """MGSW keys in a bootstrapping key over ``n`` input coefficients.

    One per bit pattern of each group of ``unfolding`` coefficients (the last
    group is shorter when ``unfolding`` does not divide ``n``), less the
    all-zero pattern for ``"bmmp18"``.
    """
    drop = 1 if variant == "bmmp18" else 0
    return sum((1 << size) - drop for size in unfolded_group_sizes(n, unfolding))


class CGGI16_Key:
    """A bootstrapping key: its MGSW keys, group after group (see `CGGI16`)."""

    def __init__(self):
        self.bk: list[MGSW] = []
        self.b_prec = 0
        self.n = 0
        self.unfolding = 1
        self.variant = "bmmp18"
        self.combination = "evaluation"
        self._handles = None  # (bk, combination, outer array, row arrays) for C
        self._monomials = None

    def _key_domains(self, combined_canonical: bool) -> list[bool]:
        """Per key, whether the kernel reads it canonical (else in NTT form).

        Only the pipelined coefficient combination (``combined_canonical``)
        reads keys canonical, and only those of combined groups: a
        ``"bmmp18"`` group of one coefficient is used as is, by the external
        product.
        """
        if not combined_canonical:
            return [False] * len(self.bk)
        drop = 1 if self.variant == "bmmp18" else 0
        domains = []
        for size in unfolded_group_sizes(self.n, self.unfolding):
            combined = self.variant == "zyl17" or size > 1
            domains += [combined] * ((1 << size) - drop)
        return domains

    def native_handles(self, parallelism: str = "pipeline"):
        """The key as the C kernel reads it, kept while `bk` and its domains are.

        Moves every key to the domain the rotation reads it in, on the
        calling thread: canonical for the groups the pipelined
        ``"coefficient"`` combination combines, the NTT domain otherwise.
        """
        canonical = self.combination == "coefficient" and parallelism == "pipeline"
        cached = self._handles
        if cached is None or cached[0] is not self.bk or cached[1] != canonical:
            rows = []
            for mgsw, canonical_key in zip(
                self.bk, self._key_domains(canonical), strict=True
            ):
                if canonical_key:
                    mgsw.to_coeff()
                else:
                    mgsw.to_NTT()
                rows.append(ffi.new("void*[]", [c.obj for c in mgsw.obj]))
            cached = (self.bk, canonical, ffi.new("void*[]", rows), rows)
            self._handles = cached
        return cached[2]

    def monomial_table(self):
        """The precomputed ``X^m`` table over the key's ring, built on first use.

        ``N`` ring elements: the memory the ``"precomputed"`` combination
        spends to avoid a transform per pattern. ``ffi.NULL`` for the other
        combinations.
        """
        if self.combination != "precomputed" or not self.bk:
            return ffi.NULL
        if self._monomials is None:
            ring = self.bk[0].scheme.ring
            table = lib.zyl17_monomial_table_new(ring.arith_ring, 0)
            self._monomials = ffi.gc(table, lib.zyl17_monomial_table_free)
        return self._monomials


class CGGI16:
    """The [CGGI16] functional bootstrap, with an optionally unfolded loop.

    ``unfolding`` groups the input key's coefficients ``u`` at a time and
    spends one external product per group instead of one per coefficient
    [ZYL+17]. Each group's keys are MGSW encryptions of the indicators of
    the bit patterns its coefficients can take, so the input key must be
    binary. ``variant`` picks the key:

    - ``"bmmp18"`` (default): the indicators sum to 1, so the all-zero one
      is left out and the accumulator is added back instead [BMMP18,
      Alg. 1]: ``2^u - 1`` keys per group. ``unfolding=1`` is the plain
      [CGGI16] key, one encryption of ``s_i`` per coefficient.
    - ``"zyl17"``: all ``2^u`` indicators [ZYL+17].

    ``combination`` picks how a group's keys are combined; the result is the
    same, the cost is not:

    - ``"evaluation"`` (default): one forward transform per pattern, then a
      pointwise multiply-accumulate into every row.
    - ``"precomputed"``: the transforms of ``X^m`` come from a table of
      ``N`` elements of the key's ring, built once per key -- no transform,
      at that memory (``N^2`` words per prime).
    - ``"coefficient"``: the keys are rotated and summed in the coefficient
      domain and the sum is transformed once per row, so the transforms per
      group do not grow with the number of patterns; it gains as ``u`` grows.

    The key grows by ``(2^u - 1) / u`` (``"bmmp18"``) and each group builds
    a combined key out of its ``2^u - 1`` (or ``2^u``) keys before the
    external product. Single-threaded that is more work than it saves; the
    gain is in the threads (``vfhe.engine.set_num_threads``), and
    ``parallelism`` picks how one rotation uses them (same result either way):

    - ``"pipeline"`` (default): the other threads combine the keys of the
      next groups while one runs the chain of external products, which
      stays sequential.
    - ``"data_parallel"``: every thread works on every group -- digits,
      products with each pattern's key, rescale -- with a barrier between
      the steps; no combined key is formed, so ``combination`` only decides
      how the factors are computed (``"precomputed"`` reads the table, the
      others transform them).

    `functional_bootstrap_batch` runs one rotation per thread instead.
    """

    def __init__(
        self,
        scheme: MLWE_Scheme,
        gsw_ell: int | None = None,
        unfolding: int = 1,
        variant: str = "bmmp18",
        combination: str = "evaluation",
        parallelism: str = "pipeline",
    ):
        if unfolding < 1:
            raise ValueError("unfolding must be at least 1")
        if variant not in UNFOLDING_VARIANTS:
            raise ValueError(f"variant must be one of {UNFOLDING_VARIANTS}")
        if combination not in KEY_COMBINATIONS:
            raise ValueError(f"combination must be one of {KEY_COMBINATIONS}")
        if parallelism not in PARALLELISMS:
            raise ValueError(f"parallelism must be one of {PARALLELISMS}")
        self.scheme = scheme
        self.mgsw_scheme = MGSW_Scheme(scheme, ell=gsw_ell)
        self.ring = scheme.ring
        self.unfolding = unfolding
        self.variant = variant
        self.combination = combination
        self.parallelism = parallelism

    def _encrypt_constant(self, value: int, output_key: MLWE_Key) -> MGSW:
        poly = Polynomial(self.ring).from_array([value] + [0] * (self.ring.N - 1))
        mgsw = self.mgsw_scheme.encrypt(poly, output_key)
        mgsw.to_NTT()
        return mgsw

    def generate_bootstrap_key(
        self, input_key: MLWE_Key | LWE_Key, output_key: MLWE_Key
    ) -> CGGI16_Key:
        bk = CGGI16_Key()
        bk.unfolding = self.unfolding
        bk.variant = self.variant
        bk.combination = self.combination
        if isinstance(input_key, MLWE_Key):
            lwe_key = input_key.extract_lwe_key()
        else:
            lwe_key = input_key

        s = lwe_key.get_s()
        if any(v not in (0, 1) for v in s):
            raise ValueError("the blind rotation needs a binary input key")
        bk.n = lwe_key.n

        first_pattern = 1 if self.variant == "bmmp18" else 0
        for first in range(0, lwe_key.n, self.unfolding):
            bits = s[first : first + self.unfolding]
            for j in range(first_pattern, 1 << len(bits)):
                indicator = math.prod(
                    b if (j >> t) & 1 else 1 - b for t, b in enumerate(bits)
                )
                bk.bk.append(self._encrypt_constant(indicator, output_key))

        return bk

    def _rotation_exponents(self, lwe: LWE, torus_base: int) -> tuple[int, list[int]]:
        """``b`` (with the half-slot offset) and ``a``, rounded into Z_2N.

        N is the dimension of the bootstrapping ring, not of the input.
        """
        primes = lwe.ring.primes
        q = math.prod(primes)
        two_n = 2 * self.ring.N

        def to_2n(v: int) -> int:
            return ((2 * v * two_n + q) // (2 * q)) % two_n

        b = crt(lwe.get_b(), primes) + q // (4 * torus_base)
        limbs = lwe.get_a()
        a = [crt([row[i] for row in limbs], primes) for i in range(lwe.n)]
        return to_2n(b), [to_2n(v) for v in a]

    def _check_key(self, bk: CGGI16_Key, n: int) -> None:
        if bk.combination not in KEY_COMBINATIONS:
            raise ValueError(f"combination must be one of {KEY_COMBINATIONS}")
        if bk.n != n or len(bk.bk) != unfolded_key_count(n, bk.unfolding, bk.variant):
            raise ValueError("the bootstrapping key does not match the input dimension")

    def _rotated_test_vector(self, tv: MLWE, b: int, out: MLWE | None = None) -> MLWE:
        """``tv * X^-b``, canonical, into ``out`` (a new sample by default)."""
        out = MLWE(self.scheme) if out is None else out
        out.to_coeff()
        tv.to_coeff()
        two_n = 2 * self.ring.N
        self.ring.lib.mlwe_RNSc_mul_by_xai(out.obj, tv.obj, (two_n - b) % two_n)
        out.repr = repr.coeff
        return out

    def _kernel_arguments(self, bk: CGGI16_Key, parallelism: str = "pipeline") -> tuple:
        """The key's arguments to the kernel; the batch runs every rotation
        sequentially, which reads the key as the pipeline does."""
        first = bk.bk[0]
        return (
            bk.native_handles(parallelism),
            bk.unfolding,
            int(bk.variant == "zyl17"),
            KEY_COMBINATIONS.index(bk.combination),
            bk.monomial_table(),
            first.gadget_size,
            first.scheme.radix_log_base or 0,
        )

    def functional_bootstrap_wo_extract(
        self, out: MLWE, tv: MLWE, rlwe_in: LWE, bk: CGGI16_Key, torus_base: int
    ):
        self._check_key(bk, rlwe_in.n)
        b, a = self._rotation_exponents(rlwe_in, torus_base)
        self._rotated_test_vector(tv, b, out=out)
        a_arr = ffi.new("uint64_t[]", a)
        lib.cggi16_blind_rotate(
            out.obj,
            a_arr,
            len(a),
            *self._kernel_arguments(bk, self.parallelism),
            PARALLELISMS.index(self.parallelism),
            0,
        )

    def LUT_packing(self, lut: list[int], size: int, LUT_prec: int):
        rlwe_tv = MLWE(self.scheme)
        N = self.ring.N
        ell = self.ring.ell
        quotient_ring = self.ring.quotient_ring(ell=ell)
        q = quotient_ring.q_l
        normal_ring = self.scheme.rings[0]

        scaled_lut = [(val * q) // (1 << LUT_prec) for val in lut]

        coeffs = [0] * N
        for i in range(N):
            idx = i // (N // size)
            if idx < len(scaled_lut):
                coeffs[i] = scaled_lut[idx]

        poly = Polynomial(normal_ring).from_bigint_array(coeffs)
        lib_rlwe.lib.mlwe_RNS_trivial_sample_of_zero(rlwe_tv.obj)
        rlwe_tv.repr = repr.ntt

        rlwe_tv += poly
        return rlwe_tv

    def _extract_into(self, out: LWE, acc: MLWE) -> None:
        acc.to_coeff()
        lwe_obj = self.ring.lib.mlwe_extract_LWE(acc.obj, 0)
        if out.obj is not None:
            lib_lwe.lib.free_lwe_sample(out.obj)
        out.obj = lwe_obj
        out.n = self.scheme.r * self.ring.N

    def functional_bootstrap(
        self, out: LWE, tv: MLWE, rlwe_in: LWE, bk: CGGI16_Key, torus_base: int
    ):
        rotated_tv = MLWE(self.scheme)
        self.functional_bootstrap_wo_extract(rotated_tv, tv, rlwe_in, bk, torus_base)
        self._extract_into(out, rotated_tv)

    def functional_bootstrap_batch(
        self,
        outs: list[LWE],
        tv: MLWE,
        inputs: list[LWE],
        bk: CGGI16_Key,
        torus_base: int,
    ):
        """`functional_bootstrap` of every input, in parallel over the inputs."""
        if len(outs) != len(inputs):
            raise ValueError("one output per input")
        if not inputs:
            return
        n = inputs[0].n
        if any(c.n != n for c in inputs):
            raise ValueError("the inputs must share one dimension")
        self._check_key(bk, n)
        accs, a = [], []
        for c in inputs:
            b, a_c = self._rotation_exponents(c, torus_base)
            accs.append(self._rotated_test_vector(tv, b))
            a.extend(a_c)
        acc_arr = ffi.new("void*[]", [acc.obj for acc in accs])
        a_arr = ffi.new("uint64_t[]", a)
        lib.cggi16_blind_rotate_batch(
            acc_arr, a_arr, len(inputs), n, *self._kernel_arguments(bk), 0
        )
        for out, acc in zip(outs, accs, strict=True):
            self._extract_into(out, acc)
