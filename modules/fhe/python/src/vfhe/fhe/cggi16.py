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


def unfolded_key_count(n: int, unfolding: int) -> int:
    """MGSW keys in a bootstrapping key over ``n`` input coefficients.

    ``2^u - 1`` for each step of the rotation, which consumes ``u = unfolding``
    coefficients (the last step fewer when ``unfolding`` does not divide ``n``).
    """
    full, last = divmod(n, unfolding)
    return full * ((1 << unfolding) - 1) + ((1 << last) - 1 if last else 0)


class CGGI16_Key:
    """A bootstrapping key: its MGSW keys, step after step (see `CGGI16`)."""

    def __init__(self):
        self.bk: list[MGSW] = []
        self.b_prec = 0
        self.n = 0
        self.unfolding = 1
        self._handles = None  # (bk, outer array, row arrays) passed to C

    def native_handles(self):
        """The key as the C kernel reads it, built once and kept while `bk` is.

        Moves every key to the NTT domain on the calling thread first.
        """
        if self._handles is None or self._handles[0] is not self.bk:
            rows = []
            for mgsw in self.bk:
                mgsw.to_NTT()
                rows.append(ffi.new("void*[]", [c.obj for c in mgsw.obj]))
            self._handles = (self.bk, ffi.new("void*[]", rows), rows)
        return self._handles[1]


class CGGI16:
    """The [CGGI16] functional bootstrap, with an optionally unfolded loop.

    Each step of the blind rotation consumes ``unfolding`` (``u``) input
    coefficients at the price of one gadget decomposition of the accumulator
    [ZYL+17]. A step's keys are MGSW encryptions of the indicators of the
    non-zero bit patterns its coefficients can take, the all-zero one left
    out [BMMP18, Alg. 1], so the input key must be binary and the key grows
    by ``(2^u - 1) / u``. ``unfolding=1`` is the plain [CGGI16] key, one
    encryption of ``s_i`` per coefficient.

    `functional_bootstrap` runs one rotation on the library's threads
    (``vfhe.engine.set_num_threads``), all of them on every step;
    `functional_bootstrap_batch` runs one rotation per thread, for
    throughput. Unfolding multiplies the work and the key traffic by
    ``(2^u - 1) / u`` and divides the number of steps by ``u``, so it pays
    only where the synchronization between steps dominates: many threads,
    across sockets.
    """

    def __init__(
        self,
        scheme: MLWE_Scheme,
        gsw_ell: int | None = None,
        unfolding: int = 1,
    ):
        if unfolding < 1:
            raise ValueError("unfolding must be at least 1")
        self.scheme = scheme
        self.mgsw_scheme = MGSW_Scheme(scheme, ell=gsw_ell)
        self.ring = scheme.ring
        self.unfolding = unfolding

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
        if isinstance(input_key, MLWE_Key):
            lwe_key = input_key.extract_lwe_key()
        else:
            lwe_key = input_key

        s = lwe_key.get_s()
        if any(v not in (0, 1) for v in s):
            raise ValueError("the blind rotation needs a binary input key")
        bk.n = lwe_key.n

        for first in range(0, lwe_key.n, self.unfolding):
            bits = s[first : first + self.unfolding]
            for j in range(1, 1 << len(bits)):
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
        if bk.n != n or len(bk.bk) != unfolded_key_count(n, bk.unfolding):
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

    def _kernel_arguments(self, bk: CGGI16_Key) -> tuple:
        first = bk.bk[0]
        return (
            bk.native_handles(),
            bk.unfolding,
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
        lib.cggi16_blind_rotate(out.obj, a_arr, len(a), *self._kernel_arguments(bk), 0)

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
