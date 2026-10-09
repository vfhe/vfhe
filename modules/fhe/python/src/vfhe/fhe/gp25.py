# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The sparse-amortized bootstrap of [GP25].

Every coefficient of an RLWE sample over ``R_n`` is bootstrapped at once, for
the price of a key that grows with the input key's Hamming weight ``h``
rather than with ``n``: one accumulator per coefficient, and a sparse input
key consumed through the gaps between its non-zero coefficients.

Three keys take part. The **output key** is a dense key over ``R_n``: inputs
arrive under it, at the lowest level of its scheme, and outputs leave under
it, so bootstraps compose. The **input key** is a sparse key of the same
scheme, which the input is switched to first (the Hamming-weight-reducing key
switch). The **rotation key** is the key over ``R_N`` the blind rotation runs
under.
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
    ``-1`` (``None`` for a binary key), all under the rotation key, as is
    ``automorphism_key`` (``X -> X^-1``). ``hw_reducing_key`` switches the
    input from the output key to the input key, at level ``hw_reducing_lvl``
    of their scheme; ``packing_key`` repacks into the output key at level
    ``output_lvl``, or ``trace_repack_key`` and ``output_switch_key`` (``None``
    when the rotation and output keys are one) do it through the trace.
    """

    def __init__(self):
        self.n = 0
        self.h = 0
        self.gap_bits = 0
        self.gaps: list[list[list[MGSW]]] = []
        self.signs: list[list[MGSW]] | None = None
        self.automorphism_key: MLWE_Set | None = None
        self.hw_reducing_key: MLWE_Set | None = None
        self.hw_reducing_lvl = 0
        self.output_lvl = 0
        self.packing_key: MLWE_Set | None = None
        self.trace_repack_key: MLWE_Set | None = None
        self.output_switch_key: MLWE_Set | None = None
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
    """The sparse-amortized bootstrap [GP25], rotating over ``scheme``.

    ``scheme`` is the rotation key's, over ``R_N``. An RLWE sample over
    ``R_n`` (the input and output keys' scheme, any ``n``) becomes ``n``
    accumulators at ``scheme``'s level 0, one per coefficient, each holding
    the test vector rotated by that coefficient's phase; they are then
    repacked into one sample over ``R_n`` under the output key, coefficient
    ``k`` at coefficient ``k``. Repacking goes through LWE extraction and a
    packing key switch, or, with ``trace_repack`` and ``n == N``, through the
    trace [CDKS21]. Level 0 of ``scheme`` must have the primes of a level of
    the output key's scheme (by value: the two rings have their own bases),
    which is where outputs land.

    ``gsw_ell`` and ``radix_log_base`` choose the MGSW gadget as in
    :class:`MGSW_Scheme`; every other key uses the same radix. Everything
    runs on the library's threads (``vfhe.engine.set_num_threads``).
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
        rotation_key: MLWE_Key,
        output_key: MLWE_Key,
        h: int,
        gap_bits: int,
        ternary: bool = True,
        hw_reducing_lvl: int | None = None,
        hw_reducing_hybrid: bool = True,
        n_threads: int = 0,
    ) -> SAB_Key:
        """The bootstrapping key: inputs under ``output_key`` (at
        ``hw_reducing_lvl``) to outputs under ``output_key``, through the
        sparse ``input_key`` and a rotation under ``rotation_key``.

        ``input_key`` and ``output_key`` are keys of one scheme, over
        ``R_n``; ``rotation_key`` is a key of this one. ``input_key`` must
        have exactly ``h`` non-zero coefficients per component, in
        ``{-1, 1}`` (``{1}`` unless ``ternary``), and every gap must fit in
        ``gap_bits`` bits, with ``2^(gap_bits - 1) <= n``: `sample_input_key`
        draws such keys, and says why ``gap_bits`` must not come from the key.
        Errors about the input key do not say which coefficient or gap failed.

        The Hamming-weight-reducing key switch, ``output_key -> input_key``,
        is generated for one level only, ``hw_reducing_lvl`` (default: the
        last, the lowest modulus), and inputs must arrive there. It encrypts
        the dense key under the sparse one, so the sparse key only has to be
        secure at that level's modulus: its special ring's, or with
        ``hw_reducing_hybrid=False`` the level's own (a BV key switch; pair it
        with ``radix_log_base``). Never generate it at a higher level.

        Every MGSW key is drawn in one batch, on ``n_threads`` (0: the library
        limit), and so are the other keys.
        """
        io_scheme = input_key.scheme
        if output_key.scheme is not io_scheme:
            raise ValueError("the input and output keys must be keys of one scheme")
        if rotation_key.scheme is not self.scheme:
            raise ValueError("the rotation key must be a key of this GP25's scheme")
        n = io_scheme.N
        self._check_gap_bits(gap_bits, n)
        output_lvl = self._output_level(io_scheme)

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

        trace_switch = False
        if self.trace_repack:
            if n != self.ring.N:
                raise ValueError(
                    "trace repacking needs the input's dimension to be the "
                    "rotation ring's; repack with the packing key instead"
                )
            trace_switch = rotation_key.key != output_key.key
            if trace_switch and self.scheme.r != io_scheme.r:
                raise ValueError(
                    "trace repacking into another key needs the two keys' ranks "
                    "to be equal; repack with the packing key instead"
                )

        values = [x for gaps in gap_values for g in gaps for x in _bits(g, gap_bits)]
        if ternary:
            values += [x for row in signs for x in row]
        keys = iter(
            self.mgsw_scheme.encrypt_constants(
                values, rotation_key, n_threads=n_threads
            )
        )

        sab = SAB_Key()
        sab.n, sab.h, sab.gap_bits, sab.output_lvl = n, h, gap_bits, output_lvl
        sab.gaps = [
            [[next(keys) for _ in range(gap_bits)] for _ in gaps] for gaps in gap_values
        ]
        sab.signs = [[next(keys) for _ in row] for row in signs] if ternary else None

        # At one level these return one set.
        sab.automorphism_key = cast(
            "MLWE_Set",
            self.scheme.gen_ksk_automorphism(
                rotation_key,
                rotation_key,
                2 * self.ring.N - 1,
                lvl=0,
                radix_log_base=self.radix_log_base,
                n_threads=n_threads,
            ),
        )

        lvl = len(io_scheme.rings) - 1 if hw_reducing_lvl is None else hw_reducing_lvl
        if not 0 <= lvl < len(io_scheme.rings):
            raise ValueError("hw_reducing_lvl is not a level of the keys' scheme")
        sab.hw_reducing_lvl = lvl
        sab.hw_reducing_key = cast(
            "MLWE_Set",
            io_scheme.gen_ksk(
                input_key,
                output_key,
                lvl,
                radix_log_base=self.radix_log_base,
                n_threads=n_threads,
                hybrid=hw_reducing_hybrid,
            ),
        )

        if self.trace_repack:
            log_N = self.ring.N.bit_length() - 1
            sab.trace_repack_key = cast(
                "MLWE_Set",
                self.scheme.gen_ksk_trace(
                    rotation_key,
                    rotation_key,
                    gens=[(1 << j) + 1 for j in range(1, log_N + 1)],
                    lvl=0,
                    radix_log_base=self.radix_log_base,
                    n_threads=n_threads,
                ),
            )
            if trace_switch:
                sab.output_switch_key = cast(
                    "MLWE_Set",
                    io_scheme.gen_ksk(
                        output_key,
                        MLWE_Key(rotation_key.key, rotation_key.sigma_err, io_scheme),
                        output_lvl,
                        radix_log_base=self.radix_log_base,
                        n_threads=n_threads,
                    ),
                )
        else:
            sab.packing_key = self.gen_packing_ksk(
                output_key, rotation_key.extract_lwe_key(), output_lvl, n_threads
            )
        return sab

    def _output_level(self, io_scheme: MLWE_Scheme) -> int:
        primes = sorted(self.ring.primes)
        for lvl, ring in enumerate(io_scheme.rings):
            if sorted(ring.primes) == primes:
                return lvl
        raise ValueError(
            "level 0 of the rotation scheme must have the primes of a level of "
            "the input and output keys' scheme"
        )

    def gen_packing_ksk(
        self, key_out: MLWE_Key, lwe_key: LWE_Key, lvl: int = 0, n_threads: int = 0
    ) -> MLWE_Set:
        """The key `packing_keyswitch` takes: for every coefficient ``s_i`` of
        ``lwe_key``, encryptions of ``s_i`` times the gadget under
        ``key_out``, at level ``lvl`` of its scheme (any dimension).
        """
        scheme = key_out.scheme
        special_ring = scheme.special_rings[lvl]
        gadget = scheme.gadget_scalars(lvl, self.radix_log_base)
        key_out_special = MLWE_Key(
            key_out.key, key_out.sigma_err, scheme, ring=special_ring
        )
        # s_i * g is 1 times the scalar s_i * g, so one message serves the key.
        one = Polynomial(special_ring).from_array([1])
        scalars = [[s_i * g_p for g_p in g] for s_i in lwe_key.get_s() for g in gadget]
        samples = scheme.sample_scaled([one], scalars, key_out_special, lvl, n_threads)
        width = len(gadget)
        return MLWE_Set(
            [samples[i : i + width] for i in range(0, len(samples), width)],
            radix_log_base=self.radix_log_base,
            balanced=scheme.balanced,
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

        The input is under the output key, at the level of the key's
        Hamming-weight-reducing switch, and is switched to the input key
        first. ``in_modulus`` is the modulus the input's phase is taken mod
        after that switch, by default its ring's.
        """
        n = rlwe_in.ring.N
        rank = rlwe_in.scheme.r
        if key.n != n or len(key.gaps) != rank:
            raise ValueError(
                "the bootstrapping key does not match the input's dimension and rank"
            )
        if key.hw_reducing_key is None:
            raise ValueError("the key has no Hamming-weight-reducing key switch")
        if rlwe_in.lvl != key.hw_reducing_lvl:
            raise ValueError(
                "the input must be at level "
                f"{key.hw_reducing_lvl}, where the key reduces its weight"
            )
        rlwe_in = rlwe_in.scheme.keyswitch(rlwe_in, key.hw_reducing_key)
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
        rlwe_in: MLWE,
        tv: MLWE,
        key: SAB_Key,
        in_modulus: int | None = None,
    ) -> MLWE:
        """Bootstraps every coefficient of ``rlwe_in`` (over ``R_n``, under the
        output key, at the key's ``hw_reducing_lvl``): the result is a sample
        of the same scheme at the key's ``output_lvl``, under the output key,
        whose coefficient ``k`` decrypts to what `test_vector` maps
        coefficient ``k``'s phase to.
        """
        io_scheme = rlwe_in.scheme
        acc = self.blind_rotate(rlwe_in, tv, key, in_modulus)
        out = MLWE(io_scheme, lvl=key.output_lvl)
        if self.trace_repack:
            if key.trace_repack_key is None:
                raise ValueError(
                    "the key has no trace key; generate it with trace_repack"
                )
            packed = self.scheme.full_packing_keyswitch_scaled(
                acc, key.trace_repack_key
            )
            out.copy_from(packed)
            if key.output_switch_key is not None:
                out = io_scheme.keyswitch(out, key.output_switch_key)
            return out
        if key.packing_key is None:
            raise ValueError(
                "the key has no packing key; generate it without trace_repack"
            )
        return self.packing_keyswitch(
            [self.rlwe_extract_lwe(c, 0) for c in acc], key.packing_key, key.output_lvl
        )

    # -- building blocks ----------------------------------------------------

    @staticmethod
    def _automorphism_key(key: SAB_Key) -> MLWE_Set:
        if key.automorphism_key is None:
            raise ValueError("the key has no automorphism key")
        return key.automorphism_key

    def _mgsw_arguments(self, mgsw: MGSW) -> tuple:
        return (mgsw.gadget_size, mgsw.scheme.radix_log_base or 0, mgsw.scheme.balanced)

    def rotate(
        self, acc: list[MLWE], bits: list[MGSW], automorphism_key: MLWE_Set
    ) -> None:
        """Rotates the accumulators in place by ``d``, given as the MGSW
        encryptions of its bits, least significant first (``2^(len(bits) - 1)
        <= len(acc)``): ``acc[k]`` takes ``acc[k - d]``, and for ``k < d``
        ``sigma(acc[k - d + n])``, ``sigma`` the automorphism ``X -> X^-1``
        that ``automorphism_key`` switches back (an `SAB_Key`'s, or any
        level-0 key for ``2N - 1``).
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
            automorphism_key.obj,
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
        self, extracted: list[LWE], packing_key: MLWE_Set, lvl: int = 0
    ) -> MLWE:
        """One sample at level ``lvl`` of the packing key's scheme whose
        coefficient ``k`` decrypts to what ``extracted[k]`` does. The samples
        may come from a ring of another dimension, but must be over the primes
        of that level.
        """
        first = next(x for x in packing_key.mlwe if x is not None)[0]
        scheme = first.scheme
        ring = scheme.rings[lvl]
        if len(extracted) > ring.N:
            raise ValueError("more samples than the ring has coefficients")
        if sorted(extracted[0].ring.primes) != sorted(ring.primes):
            raise ValueError("the samples are not over the primes of that level")
        res = MLWE(scheme, lvl=lvl)
        lib_rlwe.lib.mlwe_full_packing_keyswitch(
            res.obj,
            ffi.new("void*[]", [c.obj for c in extracted]),
            len(extracted),
            packing_key.obj,
        )
        res.repr = repr.ntt
        return res
