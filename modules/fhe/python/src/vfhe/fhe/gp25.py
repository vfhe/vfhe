# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The sparse-amortized bootstrap of [GP25].

Every coefficient of an RLWE sample over ``R_n`` is bootstrapped at once, for
the price of a key that grows with the input key's Hamming weight ``h``
rather than with ``n``: one accumulator per coefficient, and a sparse input
key consumed through the gaps between its non-zero coefficients.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from vfhe.arith import Polynomial, repr
from vfhe.crypto import entropy
from vfhe.engine import ffi, lib
from vfhe.mlwe.lwe import LWE, LWE_Key
from vfhe.mlwe.mgsw import MGSW, MGSW_Scheme
from vfhe.mlwe.mlwe import MLWE, MLWE_Key, MLWE_Scheme, MLWE_Set, lib_rlwe

if TYPE_CHECKING:
    from collections.abc import Sequence


def mod_switch(v, q, p):
    return round((v * p) / q) % p


def _bits(value: int, count: int) -> list[int]:
    return [(value >> b) & 1 for b in range(count)]


class SAB_Key:
    """A GP25 bootstrapping key, made by `GP25.generate_bootstrap_key`.

    For an input key of rank ``r`` with ``h`` non-zero coefficients per
    component: ``gaps[i][t]`` are the MGSW encryptions of the bits of gap
    ``t`` of component ``i``, least significant first, ``gap_bits`` of
    them; ``signs[i][t]`` encrypts whether its ``t``-th non-zero coefficient is
    ``-1`` (``None`` for a binary key). ``automorphism_key`` switches
    ``X -> X^-1``, and the packing or trace key brings the accumulators back
    into one sample.
    """

    def __init__(self):
        self.n = 0
        self.h = 0
        self.gap_bits = 0
        self.gaps: list[list[list[MGSW]]] = []
        self.signs: list[list[MGSW]] | None = None
        self.automorphism_key: MLWE_Set | None = None
        self.packing_key: MLWE_Set | None = None
        self.trace_repack_key: MLWE_Set | None = None
        self._handles = None

    def native_handles(self) -> tuple:
        """``(gap_keys, sign_keys)`` as `gp25_blind_rotate` reads them, built
        once; every key moves to the NTT domain on the calling thread first.
        """
        if self._handles is None or self._handles[0] is not self.gaps:
            rows = []

            def handle(mgsw: MGSW):
                mgsw.to_NTT()
                rows.append(ffi.new("void*[]", [c.obj for c in mgsw.obj]))
                return rows[-1]

            gaps = ffi.new(
                "void*[]",
                [
                    handle(m)
                    for component in self.gaps
                    for gap in component
                    for m in gap
                ],
            )
            signs = (
                ffi.NULL
                if self.signs is None
                else ffi.new(
                    "void*[]",
                    [handle(m) for component in self.signs for m in component],
                )
            )
            self._handles = (self.gaps, gaps, signs, rows)
        return self._handles[1], self._handles[2]


class GP25:
    """The sparse-amortized bootstrap [GP25] into ``scheme``'s level 0.

    An RLWE sample over ``R_n`` under a sparse key becomes ``n`` accumulators
    over ``R_N`` (``N`` the scheme's dimension), one per coefficient, each
    holding the test vector rotated by that coefficient's phase; they are then
    packed back into one sample over ``R_N``, coefficient ``k`` of the input
    landing at ``k * N / n``. Packing goes through LWE extraction and a
    packing key switch, or, with ``trace_repack``, through the trace [CDKS21].

    ``gsw_ell`` and ``radix_log_base`` choose the MGSW gadget as in
    :class:`MGSW_Scheme`; the automorphism and packing keys use the same
    radix. Everything runs on the library's threads
    (``vfhe.engine.set_num_threads``).
    """

    def __init__(
        self,
        scheme: MLWE_Scheme,
        gsw_ell: int | None = None,
        radix_log_base: int | None = None,
        trace_repack: bool = False,
    ):
        self.scheme = scheme
        self.ring = scheme.rings[0]
        self.radix_log_base = radix_log_base
        self.mgsw_scheme = MGSW_Scheme(
            scheme, ell=gsw_ell, radix_log_base=radix_log_base
        )
        self.trace_repack = trace_repack

    # -- input keys ---------------------------------------------------------

    @staticmethod
    def gaps(key: MLWE_Key) -> list[list[int]]:
        """The gaps of each component of ``key``: for non-zero coefficients
        at ``j_1 > ... > j_h``, ``[n - j_1, j_1 - j_2, ..., j_h]``.
        """
        n = key.scheme.N
        out = []
        for poly in key.key:
            previous, gaps = n, []
            for j in range(n - 1, -1, -1):
                if poly[j] != 0:
                    gaps.append(previous - j)
                    previous = j
            gaps.append(previous)
            out.append(gaps)
        return out

    @staticmethod
    def _check_gap_bits(gap_bits: int, n: int) -> None:
        if gap_bits < 1 or 1 << (gap_bits - 1) > n:
            raise ValueError(f"gap_bits must be in [1, log2(n) + 1] (n = {n})")

    @staticmethod
    def sample_input_key(
        scheme: MLWE_Scheme,
        h: int,
        gap_bits: int,
        sigma_err: float,
        ternary: bool = True,
        attempts: int = 1 << 15,
    ) -> MLWE_Key:
        """A key for ``scheme`` with ``h`` non-zero coefficients per component,
        uniform in ``{-1, 1}`` (``{1}`` unless ``ternary``), drawn until every
        gap fits in ``gap_bits`` bits.

        ``gap_bits`` is public: the bootstrapping key holds that many MGSW
        keys per gap. It must be a parameter fixed before the key is drawn,
        never derived from a key -- the fewest bits that fit a given key would
        publish a bound on its gaps. Below ``log2(n) + 1`` it restricts the key
        distribution, which the security estimate has to account for.

        Raises ``RuntimeError`` after ``attempts`` draws.
        """
        n = scheme.N
        GP25._check_gap_bits(gap_bits, n)
        if h > n:
            raise ValueError("h exceeds the ring dimension")
        for _ in range(attempts):
            key = []
            for _ in range(scheme.r):
                poly = [0] * n
                placed = 0
                while placed < h:
                    j = entropy.below(n)
                    if poly[j] == 0:
                        poly[j] = 1 - 2 * entropy.below(2) if ternary else 1
                        placed += 1
                key.append(poly)
            candidate = MLWE_Key(key, sigma_err, scheme)
            if all(g < 1 << gap_bits for gaps in GP25.gaps(candidate) for g in gaps):
                return candidate
        raise RuntimeError(f"no key with these gaps in {attempts} draws")

    # -- keys ---------------------------------------------------------------

    def generate_bootstrap_key(
        self,
        input_key: MLWE_Key,
        output_key: MLWE_Key,
        h: int,
        gap_bits: int,
        ternary: bool = True,
        n_threads: int = 0,
    ) -> SAB_Key:
        """The bootstrapping key from ``input_key`` to ``output_key``.

        ``input_key`` must have exactly ``h`` non-zero coefficients per
        component, in ``{-1, 1}`` (``{1}`` unless ``ternary``), and every gap
        must fit in ``gap_bits`` bits, with ``2^(gap_bits - 1) <= n``:
        `sample_input_key` draws such keys, and
        says why ``gap_bits`` must not come from the key. Errors about the
        input key do not say which coefficient or gap failed. Every MGSW key is drawn in one batch, on ``n_threads`` (0: the library
        limit), and so are the automorphism and packing keys.
        """
        n = input_key.scheme.N
        self._check_gap_bits(gap_bits, n)

        gap_values = self.gaps(input_key)
        signs: list[list[int]] = []
        for poly, gaps in zip(input_key.key, gap_values, strict=True):
            nonzero = [c for c in reversed(poly) if c != 0]
            if len(nonzero) != h:
                raise ValueError(
                    f"the input key must have h = {h} non-zero coefficients "
                    "in every component"
                )
            if any(c not in (-1, 1) or (c == -1 and not ternary) for c in nonzero):
                raise ValueError(
                    "the input key's coefficients must be in {-1, 0, 1}"
                    if ternary
                    else "a binary input key's coefficients must be in {0, 1}"
                )
            if any(g >= 1 << gap_bits for g in gaps):
                raise ValueError(
                    "a gap of the input key does not fit its gap_bits; draw "
                    "the key with GP25.sample_input_key"
                )
            signs.append([int(c == -1) for c in nonzero])

        values = [x for gaps in gap_values for g in gaps for x in _bits(g, gap_bits)]
        if ternary:
            values += [x for row in signs for x in row]
        keys = iter(
            self.mgsw_scheme.encrypt_constants(values, output_key, n_threads=n_threads)
        )

        sab = SAB_Key()
        sab.n, sab.h, sab.gap_bits = n, h, gap_bits
        sab.gaps = [
            [[next(keys) for _ in range(gap_bits)] for _ in gaps] for gaps in gap_values
        ]
        sab.signs = [[next(keys) for _ in row] for row in signs] if ternary else None

        two_n = 2 * self.ring.N
        # At one level these return one set.
        sab.automorphism_key = cast(
            "MLWE_Set",
            self.scheme.gen_ksk_automorphism(
                output_key,
                output_key,
                two_n - 1,
                lvl=0,
                radix_log_base=self.radix_log_base,
                n_threads=n_threads,
            ),
        )
        if self.trace_repack:
            log_N = self.ring.N.bit_length() - 1
            sab.trace_repack_key = cast(
                "MLWE_Set",
                self.scheme.gen_ksk_trace(
                    output_key,
                    output_key,
                    gens=[(1 << j) + 1 for j in range(1, log_N + 1)],
                    lvl=0,
                    radix_log_base=self.radix_log_base,
                    n_threads=n_threads,
                ),
            )
        else:
            sab.packing_key = self.gen_packing_ksk(
                output_key, output_key.extract_lwe_key(), n_threads
            )
        return sab

    def gen_packing_ksk(
        self, key_out: MLWE_Key, lwe_key: LWE_Key, n_threads: int = 0
    ) -> MLWE_Set:
        """The key `packing_keyswitch` takes: for every coefficient ``s_i`` of
        ``lwe_key``, encryptions of ``s_i`` times the gadget under
        ``key_out``, at level 0.
        """
        special_ring = self.scheme.special_rings[0]
        gadget = self.scheme.gadget_scalars(0, self.radix_log_base)
        key_out_special = MLWE_Key(
            key_out.key, key_out.sigma_err, self.scheme, ring=special_ring
        )
        # s_i * g is 1 times the scalar s_i * g, so one message serves the key.
        one = Polynomial(special_ring).from_array([1])
        scalars = [[s_i * g_p for g_p in g] for s_i in lwe_key.get_s() for g in gadget]
        samples = self.scheme.sample_scaled(
            [one], scalars, key_out_special, 0, n_threads
        )
        width = len(gadget)
        return MLWE_Set(
            [samples[i : i + width] for i in range(0, len(samples), width)],
            radix_log_base=self.radix_log_base,
            balanced=self.scheme.balanced,
        )

    # -- the bootstrap ------------------------------------------------------

    def test_vector(self, table: Sequence[int]) -> MLWE:
        """The test vector that makes the bootstrap of a coefficient of phase
        ``p`` (in ``Z_2N``, after the input's switch to ``2N``) read
        ``table[i]`` for ``p`` within half a step of ``i * N / len(table)``,
        and its negative for ``p`` within half a step of ``N + i * N /
        len(table)``.

        Values are integers mod the level-0 modulus ``q``, the message as it
        is to be decrypted (scaled already); ``len(table)`` divides ``N``. The
        rounding offset of half a step is part of the test vector, and so is
        the trace's factor ``N`` with ``trace_repack``.
        """
        N = self.ring.N
        size = len(table)
        if size == 0 or N % size:
            raise ValueError("len(table) must divide N")
        q = self.ring.q_l
        step = N // size
        scale = pow(N, -1, q) if self.trace_repack else 1

        def at(p: int) -> int:
            i = (p + step // 2) // step
            return table[i] if i < size else -table[i - size]

        # Every accumulator ends as tv(X^-1) * X^-p or tv * X^p, whose
        # constant coefficient is tv_0 at p = 0 and -tv_(N - p) otherwise.
        coeffs = [0] * N
        coeffs[0] = at(0) * scale % q
        for p in range(1, N):
            coeffs[N - p] = -at(p) * scale % q
        tv = MLWE(self.scheme)
        lib_rlwe.lib.mlwe_RNS_trivial_sample_of_zero(tv.obj)
        tv.repr = repr.ntt
        tv += Polynomial(self.ring).from_bigint_array(coeffs)
        tv.to_coeff()
        return tv

    def _exponents(self, values: list[int], q: int) -> list[int]:
        two_n = 2 * self.ring.N
        return [((2 * v * two_n + q) // (2 * q)) % two_n for v in values]

    def blind_rotate(
        self, rlwe_in: MLWE, tv: MLWE, key: SAB_Key, in_modulus: int | None = None
    ) -> list[MLWE]:
        """The accumulators: ``n`` samples at level 0, canonical, accumulator
        ``k`` holding ``tv(X^-1) * X^-p_k`` for an input of odd rank and
        ``tv * X^p_k`` for even, ``p_k`` the phase of coefficient ``k`` switched
        to ``Z_2N``. Their constant coefficients are what `test_vector`
        describes.

        ``in_modulus`` is the modulus the input's phase is taken mod, by
        default its ring's.
        """
        n = rlwe_in.ring.N
        rank = rlwe_in.scheme.r
        if key.n != n or len(key.gaps) != rank:
            raise ValueError(
                "the bootstrapping key does not match the input's dimension and rank"
            )
        q = rlwe_in.ring.q_l if in_modulus is None else in_modulus
        rlwe_in.to_coeff()
        tv.to_coeff()
        b = self._exponents(rlwe_in.get_b_poly().get_polynomial(), q)
        a = [
            e
            for j in range(rank)
            for e in self._exponents(rlwe_in.get_a_poly(j).get_polynomial(), q)
        ]
        acc = []
        for b_k in b:
            out = MLWE(self.scheme)
            lib_rlwe.lib.mlwe_RNSc_mul_by_xai(out.obj, tv.obj, b_k)
            out.repr = repr.coeff
            acc.append(out)

        gap_keys, sign_keys = key.native_handles()
        lib.gp25_blind_rotate(
            ffi.new("void*[]", [c.obj for c in acc]),
            n,
            ffi.new("uint64_t[]", a),
            rank,
            key.h,
            key.gap_bits,
            gap_keys,
            sign_keys,
            self._automorphism_key(key).obj,
            self.mgsw_scheme.gadget_size(0),
            self.radix_log_base or 0,
            self.mgsw_scheme.balanced,
            0,
        )
        return acc

    def bootstrap(
        self,
        out: MLWE,
        rlwe_in: MLWE,
        tv: MLWE,
        key: SAB_Key,
        in_modulus: int | None = None,
    ) -> None:
        """Bootstraps every coefficient of ``rlwe_in`` (over ``R_n``) into
        ``out`` (level 0 over ``R_N``): coefficient ``k * N / n`` of ``out``
        decrypts to what `test_vector` maps coefficient ``k``'s phase to, the
        other coefficients to zero.
        """
        acc = self.blind_rotate(rlwe_in, tv, key, in_modulus)
        n, N = len(acc), self.ring.N
        if n > N:
            raise ValueError("the input's dimension exceeds the bootstrapping ring's")
        if self.trace_repack:
            if key.trace_repack_key is None:
                raise ValueError(
                    "the key has no trace key; generate it with trace_repack"
                )
            packed = self.scheme.full_packing_keyswitch_scaled(
                acc, key.trace_repack_key
            )
            log_n = n.bit_length() - 1
            remaining = key.trace_repack_key._children[log_n:]  # type: ignore[attr-defined]  # noqa: SLF001
            if remaining:
                traced = MLWE(self.scheme)
                log_N = N.bit_length() - 1
                lib_rlwe.lib.mlwe_partial_trace(
                    traced.obj,
                    packed.obj,
                    ffi.new(
                        "uint64_t[]",
                        [(1 << j) + 1 for j in range(log_n + 1, log_N + 1)],
                    ),
                    ffi.new("void*[]", [ffi.cast("void *", k.obj) for k in remaining]),
                    len(remaining),
                    0,
                )
                traced.repr = repr.coeff
                packed = traced
        else:
            if key.packing_key is None:
                raise ValueError(
                    "the key has no packing key; generate it without trace_repack"
                )
            packed = self.packing_keyswitch(
                [self.rlwe_extract_lwe(c, 0) for c in acc], key.packing_key, N // n
            )
        out.copy_from(packed)

    # -- building blocks ----------------------------------------------------

    @staticmethod
    def _automorphism_key(key: SAB_Key) -> MLWE_Set:
        if key.automorphism_key is None:
            raise ValueError("the key has no automorphism key")
        return key.automorphism_key

    def _mgsw_arguments(self, mgsw: MGSW) -> tuple:
        return (mgsw.gadget_size, mgsw.scheme.radix_log_base or 0, mgsw.scheme.balanced)

    def rotate(self, acc: list[MLWE], bits: list[MGSW], key: SAB_Key) -> None:
        """Rotates the accumulators in place by ``d``, given as the MGSW
        encryptions of its bits, least significant first (``2^(len(bits) - 1)
        <= len(acc)``): ``acc[k]`` takes ``acc[k - d]``, and for ``k < d``
        ``sigma(acc[k - d + n])``, ``sigma`` the automorphism ``X -> X^-1``.
        """
        if not bits:
            return
        if 1 << (len(bits) - 1) > len(acc):
            raise ValueError("more bits than a rotation by at most len(acc) needs")
        rows = []
        for m in bits:
            m.to_NTT()
            rows.append(ffi.new("void*[]", [c.obj for c in m.obj]))
        for c in acc:
            c.to_coeff()
        lib.gp25_rotate(
            ffi.new("void*[]", [c.obj for c in acc]),
            len(acc),
            ffi.new("void*[]", rows),
            len(bits),
            self._automorphism_key(key).obj,
            *self._mgsw_arguments(bits[0]),
            0,
        )

    def multiply_by_signed_monomials(
        self, acc: list[MLWE], a: list[int], sign: MGSW | None
    ) -> None:
        """``acc[k] *= X^a[k]`` if ``sign`` encrypts 0, ``X^-a[k]`` if it
        encrypts 1; with no ``sign``, ``X^a[k]``. In place; ``a[k] < 2N``.
        """
        if len(a) != len(acc):
            raise ValueError("one exponent per accumulator")
        for c in acc:
            c.to_coeff()
        if sign is None:
            handle, arguments = (
                ffi.NULL,
                (0, self.radix_log_base or 0, self.scheme.balanced),
            )
        else:
            sign.to_NTT()
            handle = ffi.new("void*[]", [c.obj for c in sign.obj])
            arguments = self._mgsw_arguments(sign)
        lib.gp25_multiply_by_signed_monomials(
            ffi.new("void*[]", [c.obj for c in acc]),
            len(acc),
            ffi.new("uint64_t[]", a),
            handle,
            *arguments,
            0,
        )

    def rlwe_extract_lwe(self, rlwe: MLWE, idx: int) -> LWE:
        rlwe.to_coeff()
        lwe_obj = lib_rlwe.lib.mlwe_extract_LWE(rlwe.obj, idx)
        return LWE(ring=rlwe.ring, obj=lwe_obj, n=rlwe.r * rlwe.ring.N)

    def packing_keyswitch(
        self, extracted: list[LWE], packing_key: MLWE_Set, stride: int = 1
    ) -> MLWE:
        """One level-0 sample whose coefficient ``k * stride`` decrypts to
        what ``extracted[k]`` does, the others to zero."""
        if len(extracted) * stride > self.ring.N:
            raise ValueError("the samples do not fit the ring at this stride")
        res = MLWE(self.scheme, lvl=0)
        lib_rlwe.lib.mlwe_full_packing_keyswitch(
            res.obj,
            ffi.new("void*[]", [c.obj for c in extracted]),
            len(extracted),
            stride,
            packing_key.obj,
        )
        res.repr = repr.ntt
        return res
