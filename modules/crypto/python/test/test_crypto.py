# SPDX-License-Identifier: Apache-2.0
from vfhe.crypto import sample


def test_sample_matches_kernel():
    assert sample(0) == 1442695040888963407
    assert sample(1) == 7806831264735756412
