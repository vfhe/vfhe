# SPDX-License-Identifier: Apache-2.0
from vfhe.arith import mul64, ntt_forward_32


def test_ntt_forward_32_placeholder_reduces_mod_q():
    assert list(ntt_forward_32([10, 20, 100], 7)) == [3, 6, 2]


def test_ntt_forward_32_empty():
    assert list(ntt_forward_32([], 7)) == []


def test_mul64_assembly_kernel():
    assert mul64(6, 7) == 42
    assert mul64(0, 123) == 0
    # low 64 bits of the product (wraps mod 2**64)
    assert mul64(2**63, 4) == 0
