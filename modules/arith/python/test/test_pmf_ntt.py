# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The pseudo-Mersenne transform and its roots, against big-int oracles.

The transform is pinned three ways: direct evaluation at the roots (which
fixes the basis and the output order, independent of any butterfly), a
big-int Cooley-Tukey (which scales to the lengths the in-register stages
need), and the negacyclic convolution theorem against a schoolbook product.
The root convention is pinned to the least quadratic non-residue, the same
generator the RNS plans derive from.

The vector kernels are also compared with the scalar element kernels in C,
per engine, by ``test_pmf_ntt.c``.
"""

from __future__ import annotations

import random

import pytest
from vfhe.arith import FieldVector, PseudoMersenneField, PseudoMersenneNTT

rng = random.Random(0x17E7)  # noqa: S311 - test data, not a key

#: Around the in-register stages of the tuned engine: lengths below one
#: group of 8, one group, and several.
LENGTHS = [1, 2, 4, 8, 16, 32, 128]


@pytest.fixture(
    params=[260, 312, 256], ids=["5-limb-aligned", "6-limb-aligned", "5-limb-shifted"]
)
def field(request):
    return PseudoMersenneField.generate(request.param, two_adicity=8)


def bit_reverse(i: int, bits: int) -> int:
    return int(f"{i:0{bits}b}"[::-1], 2) if bits else 0


def evaluate(coeffs: list[int], psi: int, p: int) -> list[int]:
    """Position j holds P(psi^(2 brv(j) + 1)): the basis, spelled out."""
    n = len(coeffs)
    bits = n.bit_length() - 1
    out = []
    for j in range(n):
        x = pow(psi, 2 * bit_reverse(j, bits) + 1, p)
        acc = 0
        for coefficient in reversed(coeffs):
            acc = (acc * x + coefficient) % p
        out.append(acc)
    return out


def cooley_tukey(coeffs: list[int], psi: int, p: int) -> list[int]:
    """Natural-in, bit-reversed-out negacyclic NTT over Python ints."""
    a = list(coeffs)
    n = len(a)
    bits = n.bit_length() - 1
    twiddles = [pow(psi, bit_reverse(i, bits), p) for i in range(n)]
    t, m = n, 1
    while m < n:
        t //= 2
        for i in range(m):
            w = twiddles[m + i]
            for j in range(2 * i * t, 2 * i * t + t):
                v = a[j + t] * w % p
                a[j], a[j + t] = (a[j] + v) % p, (a[j] - v) % p
        m *= 2
    return a


def random_values(f, n, seed):
    local = random.Random(seed)  # noqa: S311 - test data, not a key
    return [local.randrange(f.prime) for _ in range(n)]


class TestRoots:
    """`two_adicity` and `root_of_unity`: orders, generator, and bounds."""

    def test_two_adicity_divides_p_minus_one_exactly(self, field):
        k = field.two_adicity
        assert k >= 8  # what the fixture asked for
        assert (field.prime - 1) % (1 << k) == 0
        assert (field.prime - 1) % (1 << (k + 1)) != 0

    def test_roots_have_exactly_the_requested_order(self, field):
        p = field.prime
        for log_order in range(field.two_adicity + 1):
            root = int(field.root_of_unity(log_order))
            assert pow(root, 1 << log_order, p) == 1
            if log_order:
                assert pow(root, 1 << (log_order - 1), p) == p - 1

    def test_trivial_orders(self, field):
        assert field.root_of_unity(0) == field.one
        assert field.root_of_unity(1) == -field.one

    def test_derived_from_the_least_quadratic_non_residue(self, field):
        """The same generator the RNS plans use, so the two agree by construction."""
        p = field.prime
        g = 2
        while pow(g, (p - 1) // 2, p) != p - 1:
            g += 1
        for log_order in (3, 8):
            expected = pow(g, (p - 1) >> log_order, p)
            assert int(field.root_of_unity(log_order)) == expected

    def test_successive_orders_are_related_by_squaring(self, field):
        for log_order in range(1, field.two_adicity + 1):
            root = field.root_of_unity(log_order)
            assert root * root == field.root_of_unity(log_order - 1)

    def test_roots_are_memoized(self, field):
        assert field.root_of_unity(5) is field.root_of_unity(5)

    def test_out_of_range_orders_are_rejected(self, field):
        with pytest.raises(ValueError, match="2-adicity"):
            field.root_of_unity(field.two_adicity + 1)
        with pytest.raises(ValueError, match="2-adicity"):
            field.root_of_unity(-1)
        with pytest.raises(TypeError):
            field.root_of_unity(True)


class TestPlan:
    """Construction, memoization, and the root a plan exposes."""

    def test_plan_exposes_the_root_of_order_2n(self, field):
        plan = PseudoMersenneNTT(field, 16)
        assert plan.n == 16
        assert plan.root_of_unity == field.root_of_unity(5)
        assert pow(int(plan.root_of_unity), 16, field.prime) == field.prime - 1

    def test_ntt_plan_memoizes_per_length(self, field):
        assert field.ntt_plan(8) is field.ntt_plan(8)
        assert field.ntt_plan(8) is not field.ntt_plan(16)
        assert isinstance(field.ntt_plan(8), PseudoMersenneNTT)

    def test_roots_of_successive_lengths_square_into_each_other(self, field):
        """The property a fold across lengths relies on."""
        for n in (2, 8, 64):
            psi = field.ntt_plan(n).root_of_unity
            assert psi * psi == field.ntt_plan(n // 2).root_of_unity

    def test_length_must_be_a_power_of_two(self, field):
        for n in (0, 3, 12, -8):
            with pytest.raises(ValueError, match="power of two"):
                PseudoMersenneNTT(field, n)

    def test_length_is_bounded_by_the_two_adicity(self, field):
        longest = 1 << (field.two_adicity - 1)
        assert PseudoMersenneNTT(field, longest).n == longest
        with pytest.raises(ValueError, match="longest transform"):
            PseudoMersenneNTT(field, 2 * longest)


class TestTransform:
    """The vector transform against the big-int oracles."""

    @pytest.mark.parametrize("n", LENGTHS)
    def test_forward_evaluates_at_the_roots_in_bit_reversed_order(self, field, n):
        plan = field.ntt_plan(n)
        coeffs = random_values(field, n, n)
        out = plan.forward(FieldVector(field, coeffs))
        expected = evaluate(coeffs, int(plan.root_of_unity), field.prime)
        assert [int(x) for x in out] == expected
        assert [int(x) for x in out] == cooley_tukey(
            coeffs, int(plan.root_of_unity), field.prime
        )

    @pytest.mark.parametrize("n", LENGTHS)
    def test_inverse_undoes_forward(self, field, n):
        plan = field.ntt_plan(n)
        coeffs = random_values(field, n, 100 + n)
        vector = FieldVector(field, coeffs)
        assert plan.inverse(plan.forward(vector)) == vector
        assert plan.forward(plan.inverse(vector)) == vector

    def test_a_long_transform(self, field):
        """Past the in-register stages, against the big-int Cooley-Tukey."""
        n = 1 << (field.two_adicity - 1)
        plan = field.ntt_plan(n)
        coeffs = random_values(field, n, 7)
        out = plan.forward(FieldVector(field, coeffs))
        assert [int(x) for x in out] == cooley_tukey(
            coeffs, int(plan.root_of_unity), field.prime
        )
        assert plan.inverse(out) == FieldVector(field, coeffs)

    def test_adjacent_positions_are_plus_and_minus_pairs(self, field):
        """word[2i] = P(x), word[2i + 1] = P(-x): what a fold reads."""
        n = 32
        plan = field.ntt_plan(n)
        p = field.prime
        coeffs = random_values(field, n, 8)
        word = [int(x) for x in plan.forward(FieldVector(field, coeffs))]
        psi = int(plan.root_of_unity)
        bits = n.bit_length() - 2
        for i in range(n // 2):
            x = pow(psi, 2 * bit_reverse(i, bits) + 1, p)
            plus = sum(c * pow(x, k, p) for k, c in enumerate(coeffs)) % p
            minus = sum(c * pow(p - x, k, p) for k, c in enumerate(coeffs)) % p
            assert (word[2 * i], word[2 * i + 1]) == (plus, minus)

    @pytest.mark.parametrize("n", [4, 16, 64])
    def test_negacyclic_convolution(self, field, n):
        plan = field.ntt_plan(n)
        p = field.prime
        a, b = random_values(field, n, 9), random_values(field, n, 10)
        expected = [0] * n
        for i in range(n):
            for j in range(n):
                sign = 1 if i + j < n else -1
                expected[(i + j) % n] = (expected[(i + j) % n] + sign * a[i] * b[j]) % p
        product = plan.inverse(
            plan.forward(FieldVector(field, a)) * plan.forward(FieldVector(field, b))
        )
        assert [int(x) for x in product] == expected

    def test_results_are_canonical(self, field):
        n = 16
        out = field.ntt_plan(n).forward(
            FieldVector(field, [field.prime - 1] * (n // 2) + [0] * (n // 2))
        )
        for element in out:
            limbs = element.to_limbs()
            assert 0 <= int(element) < field.prime
            assert all(limb < 1 << 52 for limb in limbs[: field.limbs])
            assert limbs[field.limbs :] == [0] * (8 - field.limbs)

    def test_a_short_vector_keeps_its_padding_canonical(self, field):
        """Lengths below one group pair padding lanes with each other."""
        for n in (1, 2, 4):
            plan = field.ntt_plan(n)
            vector = plan.forward(FieldVector(field, random_values(field, n, n)))
            for index in range(n, vector._allocated_n):
                value = sum(
                    int(vector._planes[k][index]) << (52 * k)
                    for k in range(field.limbs)
                )
                assert value < field.prime

    def test_matches_the_scalar_element_kernels(self, field):
        """The same butterflies over single elements, i.e. the scalar C kernels."""
        n = 32
        plan = field.ntt_plan(n)
        psi = plan.root_of_unity
        values = random_values(field, n, 11)
        elements = [field(x) for x in values]
        bits = n.bit_length() - 1
        twiddles = [psi ** bit_reverse(i, bits) for i in range(n)]
        t, m = n, 1
        while m < n:
            t //= 2
            for i in range(m):
                for j in range(2 * i * t, 2 * i * t + t):
                    v = elements[j + t] * twiddles[m + i]
                    elements[j], elements[j + t] = elements[j] + v, elements[j] - v
            m *= 2
        assert plan.forward(FieldVector(field, values)).to_list() == elements


class TestOperands:
    """Copy versus in-place, and the checks on the operand."""

    def test_forward_returns_a_fresh_vector_by_default(self, field):
        plan = field.ntt_plan(8)
        vector = FieldVector(field, random_values(field, 8, 12))
        before = vector.to_list()
        out = plan.forward(vector)
        assert out is not vector
        assert vector.to_list() == before

    def test_in_place_transforms_the_operand(self, field):
        plan = field.ntt_plan(8)
        vector = FieldVector(field, random_values(field, 8, 13))
        expected = plan.forward(vector)
        assert plan.forward(vector, in_place=True) is vector
        assert vector == expected
        assert plan.inverse(vector, in_place=True) is vector
        assert vector.to_list() == plan.inverse(expected).to_list()

    def test_wrong_length_field_or_type_is_rejected(self, field):
        plan = field.ntt_plan(8)
        with pytest.raises(ValueError, match="length mismatch"):
            plan.forward(FieldVector(field, 16))
        other = PseudoMersenneField.generate(312 if field.bits != 312 else 260)
        with pytest.raises(ValueError, match="different field"):
            plan.forward(FieldVector(other, 8))
        with pytest.raises(TypeError):
            plan.forward([field.one] * 8)
