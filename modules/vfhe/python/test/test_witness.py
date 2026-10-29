# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The witness of the use case under [CCCFGS26]: the entries by name and
role, every relation holding on the recorded values and failing on a
tampered one, a quotient-ring view, and the range classes."""

import math

import pytest
from vfhe.arith import Ring
from vfhe.vfhe import cccfgs26, witness

N = 256


@pytest.mark.parametrize("special_primes", [0, 1])
def test_the_entries_are_named_as_in_the_paper(use_case, special_primes):
    scheme, a, b, c, d, operations = use_case(special_primes)
    w = witness.of(operations, cccfgs26)
    assert len(w.steps) == 2
    d2 = w.query("d2")
    assert [e.value for e in d2] == [
        a.get_a_poly(0) * b.get_a_poly(0),
        c.get_a_poly(0) * c.get_a_poly(0),
    ]
    assert [e.role for e in d2] == ["intermediate"] * 2
    digits = w.query("w_ks")
    assert [e.index for e in digits] == [0, 1, 2, 0, 1]  # one level down
    assert [e.bound for e in digits[:3]] == [p / 2 for p in scheme.ring.primes]
    assert all(e.role == "witness" and e.step is w.steps[0] for e in digits[:3])
    assert [e.index for e in w.query("evk")] == [0, 1] * 2
    if special_primes:
        assert [e.index for e in w.query("w_quot_P")] == [0, 1] * 2
    else:
        with pytest.raises(ValueError, match="no entry"):
            w.query("w_quot_P")
    assert [e.value for e in w.query("ct''")][1] == [d.get_a_poly(0), d.get_b_poly()]
    remainders = w.query("w_rmd")
    assert [e.bound for e in remainders] == [
        scheme.ring.primes[-1] / 2,
        scheme.ring.primes[-1] / 2,
        scheme.ring.primes[-2] / 2,
        scheme.ring.primes[-2] / 2,
    ]
    assert w.query("w_quot")[0].bound == math.prod(scheme.ring.primes[:-1])
    with pytest.raises(ValueError, match="no entry"):
        w.query("w_nope")
    names = [r.name for r in w.relations]
    assert names.count("PreMult") == 2 and names.count("KeySwitch") == 2
    assert names.count("Rescale") == 4
    assert names.count("the division by the special primes") == 4 * special_primes
    w.check()


def test_a_tampered_witness_fails_its_relation(use_case):
    _, _, _, _, _, operations = use_case(0)
    w = witness.of(operations, cccfgs26)
    first = w.steps[0]
    digit = first.key_switch.digits[0]
    first.key_switch.digits[0] = digit + 1
    with pytest.raises(ValueError, match=r"w_ks\[0\] recomposes to d2"):
        w.check()
    first.key_switch.digits[0] = digit
    w.check()
    rescale = first.rescales[0]
    rescale.remainder = rescale.remainder + 1
    with pytest.raises(ValueError, match="Rescale does not hold"):
        w.check()
    entry = w.query("w_rmd")[0]
    entry.value = entry.value * (rescale.primes[0] // 2)
    with pytest.raises(ValueError, match="w_rmd exceeds its bound"):
        w.check()


def test_a_quotient_ring_view_reduces_the_values(use_case):
    scheme, a, b, _, _, operations = use_case(0)
    small = scheme.ring.quotient_ring(ell=2)
    w = witness.of(operations, cccfgs26, view=small)
    (first, _) = w.query("d2")
    assert first.value.ring == small
    assert first.value == (a.get_a_poly(0) * b.get_a_poly(0)).mod_reduce(small)
    foreign = Ring(N, prime_size=[40, 40], split_degree=1)
    with pytest.raises(ValueError, match="quotient"):
        witness.of(operations, cccfgs26, view=foreign).query("d2")


def test_the_range_classes_group_the_bounded_entries(use_case):
    scheme, *_, operations = use_case(0)
    w = witness.of(operations, cccfgs26)
    primes = scheme.ring.primes
    assert witness.range_classes(w) == [
        witness.RangeClass("w_ks", 3, 3, int(max(primes) / 2)),
        witness.RangeClass("w_quot", 3, 2, primes[0] * primes[1]),
        witness.RangeClass("w_rmd", 3, 2, int(primes[2] / 2)),
        witness.RangeClass("w_ks", 2, 2, int(max(primes[:2]) / 2)),
        witness.RangeClass("w_quot", 2, 2, primes[0]),
        witness.RangeClass("w_rmd", 2, 2, int(primes[1] / 2)),
    ]
    digits = witness.coefficients(w, "w_ks", ell=3)
    assert len(digits) == 3 and all(len(v) == N for v in digits)
    assert digits[1] == w.query("w_ks")[1].value.get_polynomial(signed=True)
    assert len(witness.coefficients(w, "w_ks")) == 5
    with pytest.raises(ValueError, match="no entry with a bound"):
        witness.coefficients(w, "d2")
