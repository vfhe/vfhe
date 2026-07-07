# SPDX-License-Identifier: Apache-2.0
"""The runtime "you could be faster" hint fires exactly when a portable build
runs on a CPU that supports AVX-512 IFMA -- and never otherwise."""

import warnings

import pytest
from vfhe.misc.libvfhe import _warn_if_leaving_performance_on_the_table as hint


class _FakeLib:
    def __init__(self, portable, ifma):
        self._portable, self._ifma = portable, ifma

    def vfhe_build_is_portable(self):
        return self._portable

    def vfhe_cpu_has_avx512ifma(self):
        return self._ifma


def test_hint_fires_on_portable_build_with_capable_cpu():
    with pytest.warns(RuntimeWarning, match="AVX-512 IFMA"):
        hint(_FakeLib(portable=1, ifma=1))


@pytest.mark.parametrize("portable,ifma", [(0, 0), (1, 0), (0, 1)])
def test_hint_silent_otherwise(portable, ifma):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would raise
        hint(_FakeLib(portable, ifma))
