# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""LWE, Module-LWE and MGSW over the `vfhe.arith` ring: `MLWE_Scheme` with `MLWE`, `MLWE_Key` and `MLWE_Set`, `LWE_Key` and `LWE`, `MGSW_Scheme` and `MGSW`, and the gates `CMUX` and `NCMUX`.

A scheme is built over `vfhe.arith.RNSRing` levels and draws every key, mask
and noise from its ``random`` source, so ``MLWE_Scheme(ring, random=Random(seed))``
is reproducible. Keys for key switching, MGSW encryptions and the bootstrapping
keys `vfhe.fhe` makes of them are sampled through one kernel on up to
``n_threads`` threads, each sample from seeds of its own taken in order, so a
key never depends on the thread count.

>>> from vfhe.arith import Ring
>>> from vfhe.mlwe import MLWE_Scheme
>>> scheme = MLWE_Scheme(Ring(256, 120, split_degree=1))
>>> key = scheme.key_gen_sparse(32, 3.2)
>>> len(key.key) == scheme.r
True
"""

from .lwe import LWE, LWE_Key
from .mgsw import CMUX, MGSW, NCMUX, CMUX_batch, MGSW_Scheme
from .mlwe import MLWE, GadgetParams, MLWE_Key, MLWE_Scheme, MLWE_Set, PlaintextMatrix

__all__ = [
    "CMUX",
    "LWE",
    "MGSW",
    "MLWE",
    "NCMUX",
    "CMUX_batch",
    "GadgetParams",
    "LWE_Key",
    "MGSW_Scheme",
    "MLWE_Key",
    "MLWE_Scheme",
    "MLWE_Set",
    "PlaintextMatrix",
]
