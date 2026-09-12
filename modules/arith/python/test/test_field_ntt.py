# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The negacyclic transform over an `ExtensionField`, root in F_(p^d).

The reference is Horner evaluation at ``psi ** (2 * brv(j) + 1)``, worked out
one element at a time: the definition the transform is an algorithm for, and
code the transform shares nothing with. It is quadratic, so the lengths here
are small -- what the transform has to get right is the index arithmetic and
the twiddle order, and those are wrong or right at n = 8 just as at n = 2048.

**Two fields, because the butterfly has two implementations.** Below 2^50 the
fused kernel takes it; above, the generic whole-vector path does. Both primes
are 1 mod 4 with a verified non-residue `w`, so `x**4 - w` is irreducible and
these are fields rather than quotient rings with zero divisors -- which the
transform needs and plain ring arithmetic does not.
"""

from __future__ import annotations

import pytest
from vfhe.arith import ExtensionField, ExtensionFieldElement, FieldVector

#: 49 bits: takes the fused butterfly. v2(p - 1) = 10, so the root's subfield
#: degree climbs 1 -> 2 -> 4 as 2n passes 2^10, 2^11, 2^12 -- which is what
#: `test_the_subfield_degree_is_forced_by_the_prime` pins.
FUSED_PRIME, FUSED_W = 562949953408001, 3
#: 51 bits: above the fused kernel's family, so the generic path runs.
GENERIC_PRIME, GENERIC_W = 2251799813667841, 7

FIELDS = [
    pytest.param(FUSED_PRIME, FUSED_W, id="fused"),
    pytest.param(GENERIC_PRIME, GENERIC_W, id="generic"),
]


def brv(i: int, bits: int) -> int:
    return int(format(i, f"0{bits}b")[::-1], 2) if bits else 0


def horner(coefficients, x, field):
    """P(x) for P given by `coefficients`, low degree first."""
    acc = field.zero
    for c in reversed(coefficients):
        acc = acc * x + c
    return acc


def values(field, n, seed):
    """`n` elements, deterministic in `seed`, as a plain list."""
    out = []
    state = seed
    for _ in range(n):
        coefficients = []
        for _ in range(field.d):
            state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
            coefficients.append(state % field.prime)
        out.append(ExtensionFieldElement(field, coefficients))
    return out


class TestRootOfUnity:
    @pytest.mark.parametrize(("prime", "w"), FIELDS)
    @pytest.mark.parametrize("n", [1, 2, 8, 64])
    def test_is_primitive_of_order_2n(self, prime, w, n):
        field = ExtensionField(prime, 4, w)
        psi = field.root_of_unity(n)
        assert psi**n == -field.one
        assert psi ** (2 * n) == field.one

    def test_is_deterministic(self):
        field = ExtensionField(FUSED_PRIME, 4, FUSED_W)
        again = ExtensionField(FUSED_PRIME, 4, FUSED_W)
        assert field.root_of_unity(64) == again.root_of_unity(64)

    def test_the_subfield_degree_is_forced_by_the_prime(self):
        """Which subfield the root lands in is arithmetic, not a choice: the
        least k with p^k = 1 mod 2n. The root must actually lie in F_p exactly
        when that k is 1, which is the thing a caller needs to be able to
        predict before building a domain out of it."""
        field = ExtensionField(FUSED_PRIME, 4, FUSED_W)
        for n in (2, 8, 512, 1024, 2048):
            order, k, x = 2 * n, 1, field.prime % (2 * n)
            while x != 1:
                x = x * (field.prime % order) % order
                k += 1
            assert field.root_subfield_degree(n) == k

            psi = field.root_of_unity(n)
            in_prime_field = all(psi.value[j] == 0 for j in range(1, field.d))
            assert in_prime_field == (k == 1)

    def test_refuses_a_length_the_field_cannot_serve(self):
        field = ExtensionField(FUSED_PRIME, 4, FUSED_W)
        # v2(p^4 - 1) = v2(p - 1) + 2 = 12, so 2n = 2^13 does not divide it.
        with pytest.raises(ValueError, match="no primitive"):
            field.root_of_unity(1 << 13)
        for bad in (0, 3, -4):
            with pytest.raises(ValueError, match="power of two"):
                field.root_of_unity(bad)


class TestTransform:
    @pytest.mark.parametrize(("prime", "w"), FIELDS)
    @pytest.mark.parametrize(("n", "blocks"), [(8, 1), (8, 4), (16, 3), (32, 8)])
    def test_matches_evaluation_at_the_domain(self, prime, w, n, blocks):
        """Position j of a block must hold P(psi^(2 brv(j) + 1)) for that
        block's polynomial -- the definition, by Horner."""
        field = ExtensionField(prime, 4, w)
        plan = field.ntt_plan(n)
        psi = plan.root_of_unity
        logn = n.bit_length() - 1

        coefficients = values(field, n * blocks, seed=7)
        vector = FieldVector(field, coefficients)
        plan.forward(vector, blocks)
        got = vector.to_list()

        for b in range(blocks):
            # Block-fastest: element i of block b is at i * blocks + b.
            poly = [coefficients[i * blocks + b] for i in range(n)]
            for j in range(n):
                point = psi ** (2 * brv(j, logn) + 1)
                assert got[j * blocks + b] == horner(poly, point, field)

    @pytest.mark.parametrize(("prime", "w"), FIELDS)
    @pytest.mark.parametrize(("n", "blocks"), [(8, 1), (16, 5), (64, 4)])
    def test_round_trips(self, prime, w, n, blocks):
        field = ExtensionField(prime, 4, w)
        plan = field.ntt_plan(n)
        coefficients = values(field, n * blocks, seed=11)
        vector = FieldVector(field, coefficients)

        plan.forward(vector, blocks)
        assert vector.to_list() != coefficients  # it did something
        plan.inverse(vector, blocks)
        assert vector.to_list() == coefficients

    @pytest.mark.parametrize(("prime", "w"), FIELDS)
    def test_a_batch_is_its_blocks(self, prime, w):
        """The batched layout exists for speed and must change nothing: block
        b of a batch has to equal that block transformed on its own."""
        field = ExtensionField(prime, 4, w)
        n, blocks = 16, 4
        plan = field.ntt_plan(n)
        coefficients = values(field, n * blocks, seed=13)

        batched = FieldVector(field, coefficients)
        plan.forward(batched, blocks)
        got = batched.to_list()

        for b in range(blocks):
            alone = FieldVector(field, [coefficients[i * blocks + b] for i in range(n)])
            plan.forward(alone, 1)
            assert [got[i * blocks + b] for i in range(n)] == alone.to_list()

    def test_plans_are_memoized_per_length(self):
        field = ExtensionField(FUSED_PRIME, 4, FUSED_W)
        assert field.ntt_plan(16) is field.ntt_plan(16)
        assert field.ntt_plan(16) is not field.ntt_plan(32)

    def test_the_plan_reports_its_domain(self):
        field = ExtensionField(FUSED_PRIME, 4, FUSED_W)
        assert field.ntt_plan(2048).subfield_degree == 4  # avoids every subfield
        assert field.ntt_plan(8).subfield_degree == 1  # lies in F_p

    def test_rejects_a_batch_that_is_not_the_plan_s_shape(self):
        field = ExtensionField(FUSED_PRIME, 4, FUSED_W)
        plan = field.ntt_plan(16)
        with pytest.raises(ValueError, match="elements"):
            plan.forward(FieldVector(field, 17), 1)
        with pytest.raises(ValueError, match="elements"):
            plan.forward(FieldVector(field, 32), 1)
        with pytest.raises(ValueError, match="blocks"):
            plan.forward(FieldVector(field, 16), 0)
        other = ExtensionField(GENERIC_PRIME, 4, GENERIC_W)
        with pytest.raises(ValueError, match="different field"):
            plan.forward(FieldVector(other, 16), 1)
