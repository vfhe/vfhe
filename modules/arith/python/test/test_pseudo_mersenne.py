# SPDX-FileCopyrightText: 2026 Daniele Cozzo <daniele.cozzo@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for pseudo-Mersenne field addition and multiplication.

Python big ints are the exact oracle: ``(x + y) % p`` and ``(x * y) % p``. There
is no schoolbook layer to write as there is for the ring product, because a
prime field has no polynomial structure -- the oracle is the arithmetic itself.

Every assertion goes through ``int(element)``, so the limb pack/unpack round trip
is exercised continuously rather than in a test of its own.

Three fields are covered: both limb buckets, and both the aligned (shift == 0)
and unaligned (shift != 0) reduction paths, since a nonzero shift is what makes
the fold constant differ from c and turns on the extra reduction step.
"""

import random

import pytest
from vfhe.arith import PseudoMersenneField

rng = random.Random(0xC0FFEE)

LIMB_BITS = 52
LIMB_MASK = (1 << LIMB_BITS) - 1
LANES = 8
RANDOM_PAIRS = 200


@pytest.fixture(
    params=[260, 312, 256], ids=["5-limb-aligned", "6-limb-aligned", "5-limb-shifted"]
)
def field(request):
    return PseudoMersenneField.generate(request.param, two_adicity=8)


def edge_values(f):
    """Values that sit on a boundary of the representation or the reduction."""
    p = f.prime
    return [
        0,
        1,
        2,
        f.c,  # the reduction constant itself
        f.fold,  # and the scaled fold constant
        p - 1,  # largest canonical value
        p - 2,
        p,  # reduces to 0
        p + 1,  # reduces to 1
        -1,  # negative input must reduce to p - 1
        -p - 7,
        LIMB_MASK,  # a full single limb
        1 << LIMB_BITS,  # first value needing a second limb
        (1 << (LIMB_BITS * f.limbs)) - 1,  # every representable limb bit set
        (1 << f.bits) - 1,  # just below 2^n, above p
    ]


def assert_canonical(el, f):
    """The representation contract every operation must leave intact."""
    limbs = el.to_limbs()
    assert len(limbs) == LANES
    assert 0 <= int(el) < f.prime, "value outside [0, p)"
    assert all(limb <= LIMB_MASK for limb in limbs[: f.limbs]), "limb exceeds 52 bits"
    assert limbs[f.limbs :] == [0] * (LANES - f.limbs), "padding lanes not zero"


def test_field_layout(field):
    """The prime really has the shape the reduction assumes."""
    assert field.prime == (1 << field.bits) - field.c
    assert field.limbs == -(-field.bits // LIMB_BITS)
    assert field.shift == LIMB_BITS * field.limbs - field.bits
    assert field.fold == field.c << field.shift
    assert field.fold < (1 << LIMB_BITS), "fold constant must fit one limb"


def test_addition_matches_bigint(field):
    """a + b against (x + y) % p, over edge values and random pairs."""
    p = field.prime
    pairs = [(x, y) for x in edge_values(field) for y in edge_values(field)]
    pairs += [(rng.randrange(p), rng.randrange(p)) for _ in range(RANDOM_PAIRS)]

    for x, y in pairs:
        result = field(x) + field(y)
        assert int(result) == (x + y) % p, f"({x} + {y}) % p"
        assert_canonical(result, field)


def test_multiplication_matches_bigint(field):
    """a * b against (x * y) % p, over edge values and random pairs."""
    p = field.prime
    pairs = [(x, y) for x in edge_values(field) for y in edge_values(field)]
    pairs += [(rng.randrange(p), rng.randrange(p)) for _ in range(RANDOM_PAIRS)]

    for x, y in pairs:
        result = field(x) * field(y)
        assert int(result) == (x * y) % p, f"({x} * {y}) % p"
        assert_canonical(result, field)


def test_carry_and_wraparound(field):
    """The cases where the reduction has to do real work.

    Products and sums of near-maximal values are what exercise the carry chain
    and the fold; small operands would never leave the low limbs.
    """
    p, top = field.prime, field.prime - 1
    assert int(field(top) + field(1)) == 0, "p-1 plus 1 must wrap to zero"
    assert int(field(top) + field(top)) == (2 * top) % p
    assert int(field(top) * field(top)) == top * top % p

    # One operand pinned to a single high limb, so the fold moves the product
    # across the wrap point rather than leaving it in place.
    high = (1 << (LIMB_BITS * (field.limbs - 1))) % p
    for other in (2, field.c, top, (1 << LIMB_BITS) - 1):
        assert int(field(high) * field(other)) == high * other % p
        assert int(field(high) + field(other)) == (high + other) % p


def test_addition_is_commutative_and_associative(field):
    p = field.prime
    for _ in range(20):
        x, y, z = (rng.randrange(p) for _ in range(3))
        a, b, c = field(x), field(y), field(z)
        assert a + b == b + a
        assert (a + b) + c == a + (b + c)
        assert a + field.zero == a


def test_multiplication_is_commutative_associative_distributive(field):
    p = field.prime
    for _ in range(20):
        x, y, z = (rng.randrange(p) for _ in range(3))
        a, b, c = field(x), field(y), field(z)
        assert a * b == b * a
        assert (a * b) * c == a * (b * c)
        assert a * (b + c) == a * b + a * c
        assert a * field.one == a
        assert a * field.zero == field.zero


def test_int_operands_coerce_on_either_side(field):
    """`a + 1` and `1 + a` must agree with the oracle, via __add__/__radd__."""
    p = field.prime
    x = rng.randrange(p)
    a = field(x)

    for k in (0, 1, 2, -1, field.c, p - 1, p + 3):
        assert int(a + k) == (x + k) % p
        assert int(k + a) == (k + x) % p
        assert int(a * k) == (x * k) % p
        assert int(k * a) == (k * x) % p


def test_repeated_addition_matches_multiplication(field):
    """An independent cross-check: the two kernels must agree with each other.

    Both are compared to the big-int oracle elsewhere, but this catches a shared
    misreading of the modulus that a common oracle would not.
    """
    a = field(rng.randrange(field.prime))
    total = field.zero
    for k in range(1, 12):
        total = total + a
        assert total == a * k
        assert int(total) == (k * int(a)) % field.prime


def test_non_numeric_operands_are_rejected(field):
    a = field(1)
    for bad in (1.5, "2", None, [1], object()):
        with pytest.raises(TypeError):
            _ = a + bad
        with pytest.raises(TypeError):
            _ = a * bad
    # bool is deliberately excluded, so True does not silently mean 1.
    with pytest.raises(TypeError):
        _ = a + True


def test_operands_from_a_different_field_are_rejected():
    small = PseudoMersenneField.generate(260, two_adicity=8)
    large = PseudoMersenneField.generate(312, two_adicity=8)
    with pytest.raises(TypeError):
        _ = small(1) + large(1)
    with pytest.raises(TypeError):
        _ = small(1) * large(1)


def test_output_buffer_may_alias_the_inputs(field):
    """`a * a` and repeated accumulation must not corrupt their own operands."""
    x = rng.randrange(field.prime)
    a = field(x)
    assert int(a * a) == x * x % field.prime
    assert int(a) == x, "operand was mutated"

    acc = field(x)
    for _ in range(5):
        acc = acc * acc
        assert_canonical(acc, field)
    assert int(acc) == pow(x, 2**5, field.prime)


@pytest.mark.complete
def test_addition_and_multiplication_bulk(field):
    """A wider random sweep than the fast suite can afford."""
    p = field.prime
    for _ in range(20000):
        x, y = rng.randrange(p), rng.randrange(p)
        a, b = field(x), field(y)
        assert int(a + b) == (x + y) % p
        assert int(a * b) == x * y % p


# --- subtraction -----------------------------------------------------------
#
# Negation is exercised only as `field(0) - a`, never as `-a`: these tests
# assume subtraction is implemented and should not also depend on __neg__.


def test_subtraction_matches_bigint(field):
    """a - b against (x - y) % p, over edge values and random pairs.

    Python's % returns a non-negative representative, so `(x - y) % p` is the
    oracle for the whole range including x < y.
    """
    p = field.prime
    pairs = [(x, y) for x in edge_values(field) for y in edge_values(field)]
    pairs += [(rng.randrange(p), rng.randrange(p)) for _ in range(RANDOM_PAIRS)]

    for x, y in pairs:
        result = field(x) - field(y)
        assert int(result) == (x - y) % p, f"({x} - {y}) % p"
        assert_canonical(result, field)


def test_subtraction_is_not_symmetric(field):
    """a - b and b - a must differ, and each must match its own oracle.

    This is the test that catches an __rsub__ or a borrow path written with the
    operands the wrong way round -- the single most likely subtraction bug, and
    one that every "a - a == 0" style check would sail straight past.
    """
    p = field.prime
    for _ in range(20):
        x, y = rng.randrange(p), rng.randrange(p)
        if x == y:
            continue
        a, b = field(x), field(y)
        assert int(a - b) == (x - y) % p
        assert int(b - a) == (y - x) % p
        assert a - b != b - a, "a - b must not equal b - a for x != y"
        # The two differences are additive inverses of each other.
        assert (a - b) + (b - a) == field.zero


def test_subtraction_borrows_across_every_limb(field):
    """The cases where the borrow has to propagate the whole way up.

    Subtracting from zero, or from a value with empty low limbs, forces a borrow
    through every limb and the corrective +p at the end.
    """
    p, top = field.prime, field.prime - 1

    assert int(field(0) - field(1)) == p - 1, "0 - 1 must wrap to p - 1"
    assert int(field(0) - field(top)) == 1
    assert int(field(1) - field(top)) == 2 % p
    assert int(field(top) - field(0)) == top
    assert int(field(top) - field(top)) == 0

    # A single high limb minus a small value: the borrow travels down through
    # every intervening zero limb.
    high = (1 << (LIMB_BITS * (field.limbs - 1))) % p
    for other in (1, 2, field.c, (1 << LIMB_BITS) - 1, top):
        result = field(high) - field(other)
        assert int(result) == (high - other) % p
        assert_canonical(result, field)


def test_subtraction_inverts_addition(field):
    """(a + b) - b == a, and a - b == a + (0 - b).

    Cross-checks subtraction against the already-trusted addition kernel rather
    than only against the big-int oracle.
    """
    p = field.prime
    for _ in range(20):
        x, y = rng.randrange(p), rng.randrange(p)
        a, b = field(x), field(y)
        assert (a + b) - b == a
        assert (a - b) + b == a
        assert a - b == a + (field.zero - b)
        assert a - a == field.zero


def test_subtraction_by_zero_and_of_zero(field):
    p = field.prime
    x = rng.randrange(p)
    a = field(x)
    assert a - field.zero == a
    assert int(field.zero - a) == (-x) % p
    assert field.zero - field.zero == field.zero
    # Subtracting p is a no-op, since p reduces to zero.
    assert a - field(p) == a


def test_int_operands_coerce_on_either_side_of_subtraction(field):
    """`a - 1` and `1 - a` must each match their own oracle, not each other's.

    `1 - a` goes through __rsub__, which cannot delegate to __sub__ the way
    __radd__ delegates to __add__.
    """
    p = field.prime
    x = rng.randrange(p)
    a = field(x)

    for k in (0, 1, 2, -1, field.c, p - 1, p + 3):
        assert int(a - k) == (x - k) % p, f"a - {k}"
        assert int(k - a) == (k - x) % p, f"{k} - a"


def test_repeated_subtraction_counts_down(field):
    """Subtracting a k times must equal multiplying by -k."""
    a = field(rng.randrange(field.prime))
    total = field.zero
    for k in range(1, 12):
        total = total - a
        assert int(total) == (-k * int(a)) % field.prime
        assert total + a * k == field.zero


def test_subtraction_output_may_alias_the_inputs(field):
    """`a - a` and repeated accumulation must not corrupt their own operands."""
    x = rng.randrange(field.prime)
    a = field(x)
    assert a - a == field.zero
    assert int(a) == x, "operand was mutated"

    acc = field(x)
    for _ in range(5):
        acc = acc - acc
        assert_canonical(acc, field)
    assert int(acc) == 0


def test_non_numeric_operands_are_rejected_by_subtraction(field):
    a = field(1)
    for bad in (1.5, "2", None, [1], object()):
        with pytest.raises(TypeError):
            _ = a - bad
        with pytest.raises(TypeError):
            _ = bad - a
    # bool is deliberately excluded, so True does not silently mean 1.
    with pytest.raises(TypeError):
        _ = a - True


def test_subtraction_across_different_fields_is_rejected():
    small = PseudoMersenneField.generate(260, two_adicity=8)
    large = PseudoMersenneField.generate(312, two_adicity=8)
    with pytest.raises(TypeError):
        _ = small(1) - large(1)
    with pytest.raises(TypeError):
        _ = large(1) - small(1)


@pytest.mark.complete
def test_subtraction_bulk(field):
    """A wider random sweep than the fast suite can afford."""
    p = field.prime
    for _ in range(20000):
        x, y = rng.randrange(p), rng.randrange(p)
        a, b = field(x), field(y)
        assert int(a - b) == (x - y) % p
        assert int(b - a) == (y - x) % p


# --- negation --------------------------------------------------------------
#
# Now that __neg__ exists, these use it directly rather than spelling it
# `field(0) - a`, and cross-check the two against each other.


def test_negation_matches_bigint(field):
    """-a against (-x) % p, over edge values and random values."""
    p = field.prime
    values = edge_values(field) + [rng.randrange(p) for _ in range(RANDOM_PAIRS)]

    for x in values:
        result = -field(x)
        assert int(result) == (-x) % p, f"(-{x}) % p"
        assert_canonical(result, field)


def test_negation_of_zero_is_zero(field):
    """-0 must be 0, not p.

    The natural implementation is `p - a`, which returns p for a == 0 and would
    be non-canonical. This is the one input that separates the two.
    """
    result = -field.zero
    assert int(result) == 0
    assert result == field.zero
    assert result.to_limbs() == [0] * LANES
    assert_canonical(result, field)
    # And the same via the value that reduces to zero.
    assert int(-field(field.prime)) == 0


def test_negation_agrees_with_subtraction_from_zero(field):
    """-a and (0 - a) must be the same element.

    Two independent code paths -- pmf_neg and pmf_sub -- so this catches a bug
    present in only one of them.
    """
    p = field.prime
    for x in edge_values(field) + [rng.randrange(p) for _ in range(30)]:
        a = field(x)
        assert -a == field.zero - a
        assert int(-a) == int(field.zero - a)


def test_negation_is_an_additive_inverse(field):
    """a + (-a) == 0, and negating twice is the identity."""
    p = field.prime
    for x in edge_values(field) + [rng.randrange(p) for _ in range(30)]:
        a = field(x)
        negated = -a
        assert a + negated == field.zero
        assert -negated == a
        assert int(-negated) == x % p


def test_negation_relates_subtraction_to_addition(field):
    """a - b == a + (-b), the identity that ties the three kernels together."""
    p = field.prime
    for _ in range(30):
        x, y = rng.randrange(p), rng.randrange(p)
        a, b = field(x), field(y)
        assert a - b == a + (-b)
        assert -(a - b) == b - a
        assert (-a) - b == -(a + b)


def test_negation_distributes_over_multiplication(field):
    """(-a) * b == -(a * b) == a * (-b), and (-a) * (-b) == a * b."""
    p = field.prime
    for _ in range(30):
        x, y = rng.randrange(p), rng.randrange(p)
        a, b = field(x), field(y)
        assert (-a) * b == -(a * b)
        assert a * (-b) == -(a * b)
        assert (-a) * (-b) == a * b
        assert int((-a) * b) == (-x * y) % p


def test_negation_of_one_is_p_minus_one(field):
    """Known answers, pinning the modulus rather than only self-consistency."""
    p = field.prime
    assert int(-field.one) == p - 1
    assert int(-field(p - 1)) == 1
    assert int(-field(2)) == p - 2
    assert -field(field.c) == field(p - field.c)


def test_negation_does_not_mutate_its_operand(field):
    x = rng.randrange(field.prime)
    a = field(x)
    _ = -a
    assert int(a) == x, "operand was mutated"


@pytest.mark.complete
def test_negation_bulk(field):
    """A wider random sweep than the fast suite can afford."""
    p = field.prime
    for _ in range(20000):
        x = rng.randrange(p)
        a = field(x)
        assert int(-a) == (-x) % p
        assert a + (-a) == field.zero
