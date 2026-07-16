import random

import pytest
from vfhe.arith import Field, FieldElement

PRIME = (1 << 61) - 1
SEED = b"test_seed_123"


def schoolbook_mul(a, b, prime, w, d):
    res = [0] * (2 * d - 1)
    for i in range(d):
        for j in range(d):
            res[i + j] = (res[i + j] + a[i] * b[j]) % prime
    for i in range(2 * d - 2, d - 1, -1):
        res[i - d] = (res[i - d] + res[i] * w) % prime
    return res[:d]


@pytest.mark.parametrize("d", [1, 2, 4, 8])
def test_field_arithmetic(d):
    w = 3
    field = Field(PRIME, w, d)
    rng = random.Random(42)

    # Random lists of coefficients
    a_coeffs = [rng.randint(0, PRIME - 1) for _ in range(d)]
    b_coeffs = [rng.randint(0, PRIME - 1) for _ in range(d)]

    a = FieldElement(field, a_coeffs)
    b = FieldElement(field, b_coeffs)

    # Test basic properties of zero and one
    assert a + field.zero == a
    assert a * field.one == a
    assert a * field.zero == field.zero

    # Test addition
    c_add = a + b
    expected_add = [(x + y) % PRIME for x, y in zip(a_coeffs, b_coeffs)]
    assert [c_add.value[i] for i in range(d)] == expected_add

    # Test subtraction
    c_sub = a - b
    expected_sub = [(x - y) % PRIME for x, y in zip(a_coeffs, b_coeffs)]
    assert [c_sub.value[i] for i in range(d)] == expected_sub

    # Test negation
    c_neg = -a
    expected_neg = [(-x) % PRIME for x in a_coeffs]
    assert [c_neg.value[i] for i in range(d)] == expected_neg
    assert a + c_neg == field.zero

    # Test multiplication against schoolbook oracle
    c_mul = a * b
    expected_mul = schoolbook_mul(a_coeffs, b_coeffs, PRIME, w, d)
    assert [c_mul.value[i] for i in range(d)] == expected_mul

    # Test exponentiation
    if d > 1:
        c_pow3 = a**3
        c_mul3 = a * a * a
        assert c_pow3 == c_mul3


@pytest.mark.parametrize("d", [2, 4, 8])
def test_field_inversion(d):
    w = 3
    field = Field(PRIME, w, d)
    rng = random.Random(1337)

    # Generate a random non-zero element
    while True:
        coeffs = [rng.randint(0, PRIME - 1) for _ in range(d)]
        if any(c != 0 for c in coeffs):
            break

    a = FieldElement(field, coeffs)
    a_inv = a.inverse()

    prod = a * a_inv
    assert prod == field.one


def test_field_sampling_and_hashing():
    d = 4
    w = 3
    field = Field(PRIME, w, d)

    a = FieldElement(field)
    a.sample_random(SEED)

    # Verify coefficients are within bounds
    for i in range(d):
        assert 0 <= a.value[i] < PRIME

    # Hash should be 32 bytes and consistent
    h1 = a.hash()
    h2 = a.hash()
    assert len(h1) == 32
    assert h1 == h2

    # Different seeds or elements should give different hashes
    b = FieldElement(field)
    b.sample_random(SEED + b"extra")
    assert a != b
    assert a.hash() != b.hash()
