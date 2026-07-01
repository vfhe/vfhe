# SPDX-License-Identifier: Apache-2.0
from _vfhe_native import lib


def sample(seed: int) -> int:
    """One PRNG step from ``seed`` (delegates to the C kernel)."""
    return lib.crypto_sample(seed)
