# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The witness of the use case under [CCCFGS26]: the entries by name and
role, every relation holding on the recorded values and failing on a
tampered one, and a quotient-ring view."""

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
    assert w.query("w_quot")[0].bound is None
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
