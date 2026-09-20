# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

from vfhe.arith import (
    Polynomial,
    RNSPolynomial,
    RNSRing,
    crt,
    repr,
)
from vfhe.engine import lib
from vfhe.mlwe.mlwe import MLWE, MLWE_Key, MLWE_Scheme, MLWE_Set


class _CenteredMove:
    """Brings a divided product back into the ciphertext ring.

    The product leaves its round-division over a modulus sharing only the
    plaintext primes with the ciphertext ring, so `RNSPolynomial.base_extend`
    does not apply, and the stray multiple of the source modulus that the
    default conversion leaves behind lands in the answer at full size. The
    move is `RNSPolynomial.convert_base` with ``exact=True``.

    Shifting by half the source modulus before converting, and back after, is
    what that conversion's precondition needs -- and centers the value, which
    a round-division's signed result needs anyway. `BFV_Scheme.mul_rings`
    sizes the source so the value cannot reach a quarter of its modulus, which
    is the margin.
    """

    def __init__(self) -> None:
        self._shift: dict[tuple[int, int], tuple[RNSPolynomial, RNSPolynomial]] = {}

    def _shift_pair(
        self, source: RNSRing, dest: RNSRing
    ) -> tuple[RNSPolynomial, RNSPolynomial]:
        """Half the source modulus, as a constant polynomial of each ring.

        The first is added before the conversion, the second subtracted after.
        """
        key = (source.mask, dest.mask)
        if key not in self._shift:
            half = source.q_l // 2
            pair = []
            for ring, value in ((source, half), (dest, half % dest.q_l)):
                poly = Polynomial(ring).from_bigint_array([value] * ring.N)
                poly.to_coeff()
                pair.append(poly)
            self._shift[key] = (pair[0], pair[1])
        return self._shift[key]

    def move(self, poly: RNSPolynomial, dest: RNSRing) -> RNSPolynomial:
        """``poly``'s centered value, exactly, as an element of ``dest``.

        ``poly`` is left in the coefficient domain; the result is returned in
        it.
        """
        poly.to_coeff()
        shift_source, shift_dest = self._shift_pair(poly.ring, dest)
        shifted = poly + shift_source
        return shifted.convert_base(dest, exact=True) - shift_dest


class BFV_Scheme(MLWE_Scheme):
    """BFV, with the plaintext modulus taken from the ciphertext ring's primes.

    The plaintext ring is built from the lowest ``plaintext_primes`` primes of
    the ciphertext ring, so ``t`` divides ``q`` and ``Delta = q/t`` is exact
    rather than a rounded ``floor(q/t)``: encryption is a scaled lift,
    decryption a round-division, and neither approximates. The level chain
    drops primes from the top, so the plaintext primes survive every level.

    ``t`` is a ring prime, so it is NTT-friendly and :meth:`encode` batches for
    free -- but it cannot be a power of two. Noise grows with ``t`` at every
    multiplication, so prefer a small one: ``prime_size=[17, ...]``.
    """

    def __init__(
        self,
        rings: list[RNSRing] | RNSRing,
        plaintext_primes: int = 1,
        module_rank: int = 1,
        special_primes: int = 0,
        special_rings: list[RNSRing] | None = None,
    ):
        """Create a BFV scheme.

        ``rings`` is either a single :class:`RNSRing` (the level chain is
        derived automatically) or an explicit list of per-level rings paired
        with ``special_rings`` when ``special_primes > 0``. See
        :class:`MLWE_Scheme` for both modes.

        :param plaintext_primes: How many of the ring's lowest primes make up
            the plaintext modulus ``t``.
        """
        super().__init__(
            rings,
            special_primes=special_primes,
            special_rings=special_rings,
            module_rank=module_rank,
        )
        self.plaintext_ring = self.rings[0].quotient_ring(ell=plaintext_primes)
        #: The plaintext modulus.
        self.t = self.plaintext_ring.q_l
        if self.plaintext_ring.split_degree != 1:
            raise ValueError(
                "batched encoding needs a plaintext ring that splits completely; "
                "build the ring with split_degree=1"
            )
        self._mover = _CenteredMove()
        self._delta: dict[int, object] = {}
        self._mul_rings_cache: dict[int, tuple[RNSRing, RNSRing]] = {}
        self._slot_perm = self._build_slot_permutation()

    # --- encoding ---

    @property
    def n_slots(self) -> int:
        """Number of independent plaintext slots: two rows of ``N/2``."""
        return self.N

    def _build_slot_permutation(self) -> list[int]:
        """Slot index -> where that slot's point sits in the transform's output.

        Batching is the plaintext ring's NTT, evaluating at the odd powers of a
        2N-th root of unity, one per slot -- but in an order of its own.
        Indexing them by the hypercube ``(-1)^i * 5^j`` instead is what makes
        ``X -> X^5`` rotate each row of ``N/2`` slots and ``X -> X^(2N-1)``
        swap the two rows.
        """
        N = self.N
        p = self.plaintext_ring.primes[0]
        x = Polynomial(self.plaintext_ring.quotient_ring(ell=1)).from_coeff_matrix(
            [[0, 1] + [0] * (N - 2)], repr=repr.coeff
        )
        x.to_NTT()
        points = x.get_coeff_matrix(repr=repr.ntt)[0]

        zeta = self._primitive_root(p, 2 * N)
        position = {}
        power = 1
        for exponent in range(2 * N):
            position[power] = exponent
            power = power * zeta % p
        at_exponent = {position[v]: k for k, v in enumerate(points)}

        half = N // 2
        return [at_exponent[pow(5, j, 2 * N)] for j in range(half)] + [
            at_exponent[-pow(5, j, 2 * N) % (2 * N)] for j in range(half)
        ]

    @staticmethod
    def _primitive_root(p: int, order: int) -> int:
        """An element of order exactly ``order`` modulo the prime ``p``."""
        if (p - 1) % order:
            raise ValueError(f"{p} admits no root of unity of order {order}")
        factors = set()
        rest = p - 1
        divisor = 2
        while divisor * divisor <= rest:
            while rest % divisor == 0:
                factors.add(divisor)
                rest //= divisor
            divisor += 1
        if rest > 1:
            factors.add(rest)
        for candidate in range(2, p):
            if all(pow(candidate, (p - 1) // f, p) != 1 for f in factors):
                return pow(candidate, (p - 1) // order, p)
        raise ValueError(f"no generator found modulo {p}")

    def encode(self, values: list[int]) -> RNSPolynomial:
        """Encodes one integer per slot into a plaintext polynomial.

        The result is an element of the plaintext ring, in the coefficient
        domain. Pass it to :meth:`encrypt`, or multiply a ciphertext by it
        directly for the plaintext multiplication.
        """
        if len(values) != self.n_slots:
            raise ValueError(f"Expected {self.n_slots} values, got {len(values)}")
        matrix = []
        for p in self.plaintext_ring.primes:
            row = [0] * self.N
            for slot, value in enumerate(values):
                row[self._slot_perm[slot]] = value % p
            matrix.append(row)
        poly = Polynomial(self.plaintext_ring).from_coeff_matrix(matrix, repr=repr.ntt)
        poly.to_coeff()
        return poly

    def decode(self, poly: RNSPolynomial, signed: bool = True) -> list[int]:
        """Decodes a plaintext polynomial back into one integer per slot.

        ``signed`` returns each slot as a representative in ``(-t/2, t/2]``
        rather than in ``[0, t)``.
        """
        rows = poly.get_coeff_matrix(repr=repr.ntt)
        primes = poly.ring.primes
        modulus = poly.ring.q_l
        values = []
        for slot in range(self.n_slots):
            k = self._slot_perm[slot]
            value = (
                rows[0][k]
                if len(primes) == 1
                else crt([row[k] for row in rows], list(primes))
            )
            if signed and value > modulus // 2:
                value -= modulus
            values.append(value)
        return values

    # --- encryption ---

    def delta(self, lvl: int = 0):
        """The scaling factor ``q/t`` at ``lvl``, as `scaled_lift` takes it."""
        if lvl not in self._delta:
            self._delta[lvl] = self.rings[lvl].modulus_ratio(
                self.plaintext_ring, return_pointer=True
            )
        return self._delta[lvl]

    def encrypt(self, message: RNSPolynomial, key: MLWE_Key, lvl: int = 0) -> MLWE:
        """Encrypts a plaintext polynomial, scaled by ``Delta``, under ``key``."""
        scaled = message.scaled_lift(self.rings[lvl], delta=self.delta(lvl))
        return self.sample(scaled, key, lvl=lvl)

    def decrypt(self, ciphertext: MLWE, key: MLWE_Key) -> RNSPolynomial:
        """Decrypts a ciphertext back into a plaintext polynomial."""
        return self.phase(ciphertext, key).round_division(self.plaintext_ring)

    def mod_switch(self, ciphertext: MLWE, lvl: int | None = None) -> MLWE:
        """Switches the ciphertext down to level ``lvl`` (the next one by default).

        The round-division divides the phase and ``Delta`` alike, so the
        message is unchanged and the noise shrinks with the modulus.
        """
        lvl = ciphertext.lvl + 1 if lvl is None else lvl
        if not (0 <= lvl < len(self.rings)):
            raise ValueError("no such level to switch into")
        return ciphertext.round_division(lvl=lvl)

    # --- slots ---

    def rotate(
        self, ciphertext: MLWE, k: int, ksk: MLWE_Set | list[MLWE_Set]
    ) -> MLWE:
        """Rotates each of the two rows of ``N/2`` slots by ``k`` positions."""
        return self.automorphism(ciphertext, self._rotation_gen(k), ksk)

    def gen_rotation_key(self, key: MLWE_Key, k: int) -> MLWE_Set | list[MLWE_Set]:
        """Generates the key-switching key :meth:`rotate` needs for ``k``."""
        return self.gen_ksk_automorphism(key, key, self._rotation_gen(k))

    def _rotation_gen(self, k: int) -> int:
        return pow(5, k % (self.N // 2), 2 * self.N)

    def conjugate(self, ciphertext: MLWE, ksk: MLWE_Set | list[MLWE_Set]) -> MLWE:
        """Swaps the two rows of slots."""
        return self.automorphism(ciphertext, 2 * self.N - 1, ksk)

    def gen_conjugation_key(self, key: MLWE_Key) -> MLWE_Set | list[MLWE_Set]:
        """Generates the key-switching key :meth:`conjugate` needs."""
        return self.gen_ksk_automorphism(key, key, 2 * self.N - 1)

    # --- multiplication ---

    def multiply(
        self,
        in1: MLWE,
        in2: MLWE,
        ksk: MLWE_Set | list[MLWE_Set] | None = None,
    ) -> MLWE:
        """Multiplies two ciphertexts and (optionally) relinearizes.

        Two ``Delta``-scaled phases multiply to a ``Delta^2``-scaled one, so
        the product is divided by ``Delta`` to come back. That division is over
        the integers, not modulo ``q``: the operands move up into a ring wide
        enough to hold the product, are divided there, and come back. See
        :meth:`mul_rings`.

        Args:
            in1: The first ciphertext.
            in2: The second ciphertext.
            ksk: The relinearization key, from ``gen_rlk``. If ``None``,
                relinearization is skipped and the extended product (rank
                :attr:`extended_rank`) is returned; feed it to
                :meth:`relinearize`.
        """
        if in1.ring != in2.ring:
            raise ValueError("Ciphertexts must be in the same ring")
        if in1.scheme != in2.scheme:
            raise ValueError("Ciphertexts must be from the same scheme")
        if in1.lvl != in2.lvl:
            raise ValueError("Ciphertexts must have the same level")

        lvl = in1.lvl
        wide_ring, divided_ring = self.mul_rings(lvl)
        polys = self.tensor_product(
            self._lift_ciphertext(in1, wide_ring),
            self._lift_ciphertext(in2, wide_ring),
        )

        out = in1.new_like(lvl=lvl, ring=self.rings[lvl], rank=self.extended_rank)
        for slot, poly in enumerate(polys):
            poly.to_coeff()
            poly.round_division(divided_ring)
            component = self._mover.move(poly, self.rings[lvl])
            handle = out.obj_b() if slot == self.extended_rank else out.obj_a_i(slot)
            lib.polynomial_copy_RNS_polynomial(handle, component.obj)
        out.repr = repr.coeff
        out.is_extended = True

        return out if ksk is None else self.relinearize(out, ksk)

    def mul_rings(self, lvl: int) -> tuple[RNSRing, RNSRing]:
        """The two rings :meth:`multiply` works in at ``lvl``.

        The first is level ``lvl``'s ring plus auxiliary primes: the tensor
        product is computed there, wide enough not to wrap. The second is the
        first without ``Delta``'s primes, where the round-division by ``Delta``
        leaves the result.

        Both follow from one bound. A product coefficient stays under
        ``N * (ell * q)^2`` -- ``ell * q`` rather than ``q`` because
        `_lift_ciphertext` lifts through the fast base extension. Auxiliary
        primes are added until the first modulus is four times that, which
        covers both requirements: the product stays inside half of it, so a
        coefficient's centered representative is the integer itself; and the
        divided value stays inside a quarter of the second modulus, the margin
        `RNSPolynomial.convert_base` needs.
        """
        if lvl not in self._mul_rings_cache:
            ring = self.rings[lvl]
            auxiliary: list[int] = []
            taken = list(self.special_rings[0].primes)
            overflow = ring.ell**2
            while math.prod(auxiliary) <= 4 * overflow * self.N * ring.q_l:
                auxiliary.append(
                    RNSRing.gen_prime(
                        2 * self.N // ring.split_degree,
                        max(ring.prime_size),
                        exclude_list=taken + auxiliary,
                    )
                )
            wide = ring.union(
                RNSRing(
                    self.N,
                    primes=auxiliary,
                    prime_size=[p.bit_length() for p in auxiliary],
                    split_degree=ring.split_degree,
                )
            )
            if wide.ell != ring.ell + len(auxiliary):
                raise ValueError("auxiliary primes collided with the ciphertext ring's")
            divided = wide.quotient_ring(
                mask=wide.mask & ~(ring.mask & ~self.plaintext_ring.mask)
            )
            self._mul_rings_cache[lvl] = (wide, divided)
        return self._mul_rings_cache[lvl]

    def _lift_ciphertext(self, ciphertext: MLWE, ring: RNSRing) -> MLWE:
        """``ciphertext``'s components as elements of ``ring``, for the product.

        The fast base extension is enough, though it writes ``x + u*q``: over
        the integers a phase is already ``Delta*m + e + q*r``, ``u*q`` only
        adds to ``r``, and every such term divides out of the product modulo
        ``q``. The price is the operand size `mul_rings` accounts for.
        """
        ciphertext.to_coeff()
        out = MLWE(self, ring=ring, rank=ciphertext.r)
        for i in range(ciphertext.r):
            component = ciphertext.get_a_poly(i).base_extend(ring)
            lib.polynomial_copy_RNS_polynomial(out.obj_a_i(i), component.obj)
        component = ciphertext.get_b_poly().base_extend(ring)
        lib.polynomial_copy_RNS_polynomial(out.obj_b(), component.obj)
        out.repr = repr.coeff
        return out
