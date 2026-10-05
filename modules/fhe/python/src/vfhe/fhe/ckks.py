# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

from vfhe.arith import (
    ComplexPolynomial,
    ComplexRing,
    RNSPolynomial,
    RNSRing,
    repr,
)
from vfhe.engine import ffi
from vfhe.mlwe.mlwe import MLWE, MLWE_Key, MLWE_Scheme, MLWE_Set

if TYPE_CHECKING:
    from collections.abc import Sequence


class CKKS_Scheme(MLWE_Scheme):
    def __init__(
        self,
        rings: list[RNSRing] | RNSRing,
        scaling_factor: float = 2**30,
        module_rank: int = 1,
        special_primes: int = 0,
        special_rings: list[RNSRing] | None = None,
        balanced: bool = True,
    ):
        """Create a CKKS scheme.

        ``rings`` is either a single :class:`RNSRing` (the level chain is derived
        automatically) or an explicit list of per-level rings paired with
        ``special_rings`` when ``special_primes > 0`` -- e.g. non-nested levels
        for rational rescaling. See :class:`MLWE_Scheme` for both modes, and
        for ``balanced``.
        """
        super().__init__(
            rings,
            special_primes=special_primes,
            special_rings=special_rings,
            module_rank=module_rank,
            balanced=balanced,
        )
        self.scaling_factor = scaling_factor
        self.complex_ring = ComplexRing(self.ring.N // 2, True)

    def encode(
        self,
        values: Sequence[complex | float],
        *,
        ring: RNSRing | None = None,
        scale: float | None = None,
    ) -> RNSPolynomial:
        """Encodes complex values into a plaintext polynomial.

        ``len(values)`` must divide ``N/2``; fewer values are repeated to fill the
        slots (sparse packing), and ``decode(..., slots=len(values))`` reads them
        back. ``ring`` is the plaintext's ring (default: level 0's) and ``scale``
        multiplies the values (default: ``scaling_factor``).
        """
        ring = self.ring if ring is None else ring
        if ring.N != self.ring.N:
            raise ValueError(
                f"Expected a ring of dimension {self.ring.N}, got {ring.N}"
            )
        slots = self.ring.N // 2
        if not values or slots % len(values):
            raise ValueError(
                f"Expected a number of values dividing {slots}, got {len(values)}"
            )
        c_poly = ComplexPolynomial(self.complex_ring)
        c_poly.from_array(list(values) * (slots // len(values)))
        c_poly.IFFT()
        c_poly *= self.scaling_factor if scale is None else scale
        poly = c_poly.round_to_RNS_cpp(ring)
        # Restrict the RNS mask of the encoded polynomial to the primes of `ring`
        ffi.cast("RNS_Polynomial", poly.obj).rns_mask = ring.mask
        return poly

    def decode(
        self,
        poly: RNSPolynomial,
        scaling_factor: float | None = None,
        *,
        slots: int | None = None,
    ) -> list[complex]:
        """Decodes a plaintext polynomial into its slot values, divided by
        ``scaling_factor`` (default: the scheme's).

        Returns all ``N/2`` slots, or the first ``slots`` (a divisor of ``N/2``) of
        a sparsely packed message. Exact over any number of primes, and cheapest
        over few, which is how :meth:`decrypt` returns plaintexts.
        """
        if scaling_factor is None:
            scaling_factor = self.scaling_factor
        N = poly.ring.N
        half = N // 2
        if slots is not None and (slots < 1 or half % slots):
            raise ValueError(f"slots must divide {half}, got {slots}")
        poly.to_coeff()
        c_poly = ComplexPolynomial(self.complex_ring)
        # The scaled coefficients are already the complex polynomial's
        # [real | imaginary] layout.
        self.ring.lib.polynomial_RNSc_to_centered_doubles(
            c_poly.obj, poly.obj, 1.0 / scaling_factor
        )
        c_poly.FFT()
        # As (real, imaginary) pairs, cffi builds the Python complex values.
        pairs = ffi.new("double[]", N)
        self.ring.lib.complex_poly_to_interleaved(pairs, c_poly.obj, half)
        return ffi.unpack(
            ffi.cast("double _Complex *", pairs), half if slots is None else slots
        )

    def encrypt(self, message: RNSPolynomial, key: MLWE_Key) -> CKKS_Ciphertext:
        """Encrypts a plaintext polynomial message under the given MLWE key."""
        out = CKKS_Ciphertext(self)
        self.sample(message, key, out=out)
        return out

    def decrypt(
        self,
        ciphertext: MLWE,
        key: MLWE_Key,
        *,
        message_bound: float | None = None,
        drop: bool = True,
    ) -> RNSPolynomial:
        """Decrypts a ciphertext into its plaintext polynomial.

        To keep decoding cheap, the plaintext is returned over the fewest primes
        that hold it: the lowest level (a quotient of the ciphertext's ring) whose
        modulus exceeds ``2 * delta * message_bound``. ``message_bound`` bounds the
        slots' magnitude; by default it is what the last level holds at the
        scheme's scaling factor, so ordinary ciphertexts drop to the last level and
        those with a larger ``delta`` (e.g. unrescaled products) stay higher.
        ``drop=False`` keeps every prime.
        """
        if not drop:
            return self.linear_decrypt(ciphertext, key)
        delta = getattr(ciphertext, "delta", self.scaling_factor)
        target = self._decryption_ring(ciphertext.ring, delta, message_bound)
        return self.linear_decrypt(ciphertext, key, ring=target)

    def _decryption_ring(
        self, ring: RNSRing, delta: float, message_bound: float | None
    ) -> RNSRing:
        """The quotient of ``ring`` among the scheme's levels with the fewest primes
        whose modulus holds a plaintext of size ``delta * message_bound``.
        """
        if message_bound is None:
            # What the last level holds at the scheme's scale, one bit spare.
            last_bits = math.log2(math.prod(self.rings[-1].primes))
            bound_bits = last_bits - math.log2(self.scaling_factor) - 2
        else:
            bound_bits = math.log2(message_bound)
        needed_bits = 1 + math.log2(delta) + bound_bits
        best, best_ell = ring, ring.ell
        for candidate in self.rings:
            if (
                candidate.ell < best_ell
                and candidate.is_quotient_ring(ring)
                and math.log2(math.prod(candidate.primes)) >= needed_bits
            ):
                best, best_ell = candidate, candidate.ell
        return best

    def rotate(
        self, ciphertext: CKKS_Ciphertext, k: int, ksk: MLWE_Set | list[MLWE_Set]
    ) -> CKKS_Ciphertext:
        """Rotates the slots of the ciphertext by k steps."""
        N = self.ring.N
        # Galois generator for rotation by k slots is 5^k mod 2N
        # The rotation group generated by 5 has order N/2
        k_mod = k % (N // 2)
        gen = pow(5, k_mod, 2 * N)
        return self.automorphism(ciphertext, gen, ksk)

    def gen_rotation_key(
        self, key: MLWE_Key, k: int, n_threads: int = 0
    ) -> MLWE_Set | list[MLWE_Set]:
        """Generates the rotation key (automorphism key-switching key) for rotation
        by k slots, on up to ``n_threads`` threads (0: the library limit)."""
        N = self.ring.N
        k_mod = k % (N // 2)
        gen = pow(5, k_mod, 2 * N)
        return self.gen_ksk_automorphism(key, key, gen, n_threads=n_threads)

    def multiply_plain(
        self,
        ciphertext: CKKS_Ciphertext,
        plaintext: RNSPolynomial,
        scale: float | None = None,
    ) -> CKKS_Ciphertext:
        """Multiplies by a plaintext, without rescaling.

        ``scale`` is the plaintext's encoding scale (default: ``scaling_factor``; 1
        for an unscaled plaintext such as a monomial). The result's ``delta`` is the
        ciphertext's times ``scale``, so several products can be summed before one
        :meth:`rescale`.
        """
        prod = cast("CKKS_Ciphertext", MLWE.__mul__(ciphertext, plaintext))
        prod.delta = ciphertext.delta * (
            self.scaling_factor if scale is None else scale
        )
        return prod

    def linear_combination(
        self,
        cts: Sequence[CKKS_Ciphertext],
        coefficients: Sequence[RNSPolynomial],
        scale: float | None = None,
    ) -> CKKS_Ciphertext:
        """``sum_i coefficients[i] * cts[i]``, without rescaling.

        The ciphertexts must share one ``delta``, and the coefficients must all be
        encoded at ``scale`` (default: ``scaling_factor``).
        """
        return self.linear_combinations(cts, [coefficients], scale, n_threads=1)[0]

    def linear_combinations(
        self,
        cts: Sequence[CKKS_Ciphertext],
        rows: Sequence[Sequence[RNSPolynomial | None]],
        scale: float | None = None,
        n_threads: int = 0,
    ) -> list[CKKS_Ciphertext]:
        """One :meth:`linear_combination` of ``cts`` per row of coefficients, on up to
        ``n_threads`` threads (0: the library limit). ``None`` skips a term.
        """
        if any(c.delta != cts[0].delta for c in cts):
            raise ValueError("the ciphertexts must share one scaling factor")
        outs = super().linear_combinations(cts, rows, n_threads)
        delta = cts[0].delta * (self.scaling_factor if scale is None else scale)
        for out in outs:
            out.delta = delta
        return outs

    def conjugate(
        self, ciphertext: CKKS_Ciphertext, ksk: MLWE_Set | list[MLWE_Set]
    ) -> CKKS_Ciphertext:
        """Conjugates every slot."""
        return self.automorphism(ciphertext, 2 * self.N - 1, ksk)

    def gen_conjugation_key(
        self, key: MLWE_Key, n_threads: int = 0
    ) -> MLWE_Set | list[MLWE_Set]:
        """Generates the key-switching key :meth:`conjugate` needs, on up to
        ``n_threads`` threads (0: the library limit)."""
        return self.gen_ksk_automorphism(key, key, 2 * self.N - 1, n_threads=n_threads)

    def rescale(self, ciphertext: CKKS_Ciphertext) -> CKKS_Ciphertext:
        """Rescale down one level: move the next ring in the chain of rings.

        Levels start at 0 and grow as primes are consumed, so a rescale
        increments ``lvl``. The destination ring is taken from ``self.rings``.
        Only the case where the next level is a quotient of the current ring is
        handled here; non-nested moduli need ``rational_rescale``.
        """
        lvl = ciphertext.lvl
        next_lvl = lvl + 1
        if not (next_lvl < len(self.rings)):
            raise ValueError("no lower level to rescale into")

        current_ring = ciphertext.ring
        next_ring = self.rings[next_lvl]
        if not next_ring.is_quotient_ring(current_ring):
            return self.rational_rescale(ciphertext)

        dropped = math.prod(p for p in current_ring.primes if p not in next_ring.primes)
        ciphertext.round_division(lvl=next_lvl)
        ciphertext.delta = ciphertext.delta / dropped
        return ciphertext

    def rescale_batch(
        self, cts: Sequence[CKKS_Ciphertext], n_threads: int = 0
    ) -> list[CKKS_Ciphertext]:
        """:meth:`rescale` of each ciphertext (distinct, at one level), on up to
        ``n_threads`` threads (0: the library limit).

        In place like :meth:`rescale`, except on non-nested chains, where
        :meth:`rational_rescale` returns new ciphertexts one at a time.
        """
        if not cts:
            return []
        lvl = cts[0].lvl
        if any(c.lvl != lvl or c.ring != cts[0].ring for c in cts):
            raise ValueError("the ciphertexts must share a level")
        if not lvl + 1 < len(self.rings):
            raise ValueError("no lower level to rescale into")
        current_ring, next_ring = cts[0].ring, self.rings[lvl + 1]
        if not next_ring.is_quotient_ring(current_ring):
            return [self.rational_rescale(c) for c in cts]
        dropped = math.prod(p for p in current_ring.primes if p not in next_ring.primes)
        self.round_division_batch(cts, lvl + 1, n_threads)
        for c in cts:
            c.delta = c.delta / dropped
        return list(cts)

    def multiply(
        self,
        in1: CKKS_Ciphertext,
        in2: MLWE,
        ksk: MLWE_Set | list[MLWE_Set] | None = None,
    ) -> CKKS_Ciphertext:
        """:meth:`MLWE_Scheme.multiply` (not rescaled), with ``delta`` set to the
        product of the operands'.
        """
        out = super().multiply(in1, in2, ksk)
        out.delta = in1.delta * cast("CKKS_Ciphertext", in2).delta
        return out

    def multiply_batch(
        self,
        lhs: Sequence[CKKS_Ciphertext],
        rhs: Sequence[MLWE],
        ksk: MLWE_Set | list[MLWE_Set] | None = None,
        n_threads: int = 0,
    ) -> list[CKKS_Ciphertext]:
        """``lhs[i] * rhs[i]`` for every ``i``, relinearized with ``ksk`` but not
        rescaled, on up to ``n_threads`` threads (0: the library limit).
        """
        outs = super().multiply_batch(lhs, rhs, ksk, n_threads)
        for out, a, b in zip(outs, lhs, rhs, strict=True):
            out.delta = a.delta * cast("CKKS_Ciphertext", b).delta
        return outs

    def product(
        self, cts: Sequence[CKKS_Ciphertext], n_threads: int = 0
    ) -> CKKS_Ciphertext:
        """The product of ``cts``, computed as a balanced binary tree.

        The tree is ``ceil(log2(n))`` multiplications deep, one level of the chain
        each, instead of ``n - 1``. Each product is relinearized with :attr:`rlk`
        and rescaled. When a tree level has an odd count, its last factor is carried
        down a level unmultiplied (:meth:`MLWE.mod_reduce`, on a copy), which needs
        nested levels unless ``n`` is a power of two. Each tree level's products run
        on up to ``n_threads`` threads (0: the library limit, see
        `vfhe.engine.set_num_threads`).
        """
        n = len(cts)
        if n == 0:
            raise ValueError("expected at least one factor")
        if self.rlk is None:
            raise ValueError("the product needs the relinearization key (rlk)")
        lvl = cts[0].lvl
        depth = (n - 1).bit_length()
        if lvl + depth >= len(self.rings):
            raise ValueError(f"{n} factors need {depth} levels below level {lvl}")
        if n & (n - 1) and not all(
            self.rings[k + 1].is_quotient_ring(self.rings[k])
            for k in range(lvl, lvl + depth)
        ):
            raise ValueError(
                f"{n} factors is not a power of two, which needs nested levels "
                "to carry the odd factor down"
            )
        layer = list(cts)
        if n == 1:
            return layer[0].copy()
        while len(layer) > 1:
            carry = layer[-1] if len(layer) % 2 else None
            products = self.multiply_batch(
                layer[0::2][: len(layer) // 2], layer[1::2], self.rlk, n_threads
            )
            layer = self.rescale_batch(products, n_threads)
            if carry is not None:
                layer.append(carry.copy().mod_reduce(lvl=layer[0].lvl))
        return layer[0]

    def rational_rescale(self, ct: CKKS_Ciphertext) -> CKKS_Ciphertext:
        """
        Rescale from level (lvl) to (lvl+1) as planned in the residue system.
        """
        level = ct.lvl
        if not (level < len(self.rings) - 1):
            raise ValueError("Cannot rescale a ciphertext at the last level")

        start_ring = ct.ring
        end_ring = self.rings[level + 1]

        # Rescaling to a ring that is not a quotient of the current one needs a
        # ring holding both sets of primes to move through.
        union_ring = start_ring.union(end_ring)

        # Allocate new ciphertext at the next level
        res = CKKS_Ciphertext(self, lvl=level + 1)

        # Scale & lift, then round & divide for each 'a' polynomial
        for i in range(self.r):
            a_poly = ct.get_a_poly(i)
            a_lifted = a_poly.scaled_lift(union_ring)
            a_divided = a_lifted.round_division(end_ring)
            self.ring.lib.polynomial_copy_RNS_polynomial(res.obj_a_i(i), a_divided.obj)

        # Scale & lift, then round & divide for 'b' polynomial
        b_poly = ct.get_b_poly()
        b_lifted = b_poly.scaled_lift(union_ring)
        b_divided = b_lifted.round_division(end_ring)
        self.ring.lib.polynomial_copy_RNS_polynomial(res.obj_b(), b_divided.obj)

        # Calculate scaling factor change
        primes_start = start_ring.primes
        primes_end = end_ring.primes
        p_val = math.prod([p for p in primes_end if p not in primes_start])
        q_val = math.prod([q for q in primes_start if q not in primes_end])

        res.delta = ct.delta * p_val / q_val
        res.repr = repr.coeff

        return res


class CKKS_Ciphertext(MLWE):
    scheme: CKKS_Scheme  # narrows MLWE.scheme for the CKKS-only members

    def __init__(
        self,
        scheme: CKKS_Scheme,
        lvl: int | None = None,
        ring: RNSRing | None = None,
        rank: int | None = None,
    ):
        super().__init__(scheme, lvl=lvl, ring=ring, rank=rank)
        self.delta = scheme.scaling_factor

    def _inherit(self, other: MLWE) -> None:
        # Keep the actual scaling factor of the ciphertext this one was derived
        # from; operations that change it (mul, rescale) set it themselves.
        if isinstance(other, CKKS_Ciphertext):
            self.delta = other.delta

    def rescale(self) -> CKKS_Ciphertext:
        return self.scheme.rescale(self)

    def __mul__(self, other) -> CKKS_Ciphertext:
        if isinstance(other, CKKS_Ciphertext):
            if not (self.scheme == other.scheme):
                raise ValueError("Cannot multiply ciphertexts from different schemes")
            if not (self.ell == other.ell):
                raise ValueError("Cannot multiply ciphertexts at different levels")

            base_prod = super().__mul__(other)
            prod = CKKS_Ciphertext(self.scheme, lvl=base_prod.lvl)
            prod.copy_from(base_prod)
            prod.delta = self.delta * other.delta

            return self.scheme.rescale(prod)
        elif isinstance(other, RNSPolynomial):
            return self.scheme.rescale(self.scheme.multiply_plain(self, other))
        else:
            base_prod = super().__mul__(other)
            prod = CKKS_Ciphertext(self.scheme, lvl=base_prod.lvl)
            prod.copy_from(base_prod)
            prod.delta = self.delta
            return prod


CKKS_Scheme.ciphertext_type = CKKS_Ciphertext
