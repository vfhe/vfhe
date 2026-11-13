# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The ``cuda`` backend of ``rns``: registered with a device, and agreeing with the host."""

import pytest
from vfhe.arith import CUDARing, Polynomial, Ring
from vfhe.arith.registry import backends
from vfhe.util import gpu


@pytest.fixture
def ring():
    return Ring(256, 300, split_degree=1)


def test_the_backend_exists_only_with_a_device(ring):
    assert ("cuda" in backends("rns")) == gpu.available()
    if gpu.available():
        return
    with pytest.raises(LookupError, match="cuda"):
        Ring(256, 300, split_degree=1, backend="cuda")
    with pytest.raises(TypeError, match="no conversion"):
        ring.random_element().to_backend("cuda")


def test_the_device_ring_is_another_parent_over_the_same_primes(ring):
    device = CUDARing.of(ring)
    assert type(device) is CUDARing
    assert device.primes == ring.primes and device.mask == ring.mask
    assert device != ring and ring != device
    assert device.host == ring
    assert CUDARing.of(ring) is device


def test_an_element_is_itself_on_its_own_backend(ring):
    a = ring.random_element()
    assert a.to_backend("ntt") is a


def requires_device():
    if not gpu.available():
        pytest.skip("no CUDA device")


def test_add_sub_negate_agree_with_the_host(ring):
    requires_device()
    a, b = ring.random_element(), ring.random_element()
    da, db = a.to_backend("cuda"), b.to_backend("cuda")
    assert type(da.ring) is CUDARing and da.repr == a.repr
    assert (da + db).to_backend("ntt") == a + b
    assert (da - db).to_backend("ntt") == a - b
    assert (-da).to_backend("ntt") == -a
    da += db
    assert da.to_backend("ntt") == a + b
    assert da.copy() == da and da == a + b


def test_coefficient_domain_rows_round_trip(ring):
    requires_device()
    a = ring.random_element(ntt=False)
    assert a.to_backend("cuda").to_backend("ntt") == a
    assert Polynomial(CUDARing.of(ring)).from_array([1, 2, 3]).get_polynomial()[:3] == [
        1,
        2,
        3,
    ]


def test_mixing_host_and_device_is_refused(ring):
    requires_device()
    a = ring.random_element()
    da = a.to_backend("cuda")
    for left, right in ((a, da), (da, a)):
        with pytest.raises(TypeError, match="no implicit conversion"):
            _ = left + right
        with pytest.raises(TypeError, match="no implicit conversion"):
            _ = bool(left == right)
