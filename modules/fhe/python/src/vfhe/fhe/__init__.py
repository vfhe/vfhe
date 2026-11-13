# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The FHE schemes built on `vfhe.mlwe`: `CKKS_Scheme` with `CKKS_Ciphertext` and `CKKS_LinearTransform`, `BFV_Scheme`, and the bootstraps `CGGI16`, `GP25` and `SHIP` with their keys.

A scheme is constructed over `vfhe.arith.Ring` levels and holds the keys its
operations need, such as ``scheme.rlk`` for multiplication; a ciphertext is a
`vfhe.mlwe.MLWE` sample, so ``+``, ``-`` and ``*`` work on it. vFHE is
pre-release and unaudited: take parameters from the literature and keep it out
of production.

>>> from vfhe.arith import Ring
>>> from vfhe.fhe import CKKS_Scheme
>>> scheme = CKKS_Scheme(Ring(256, 300, split_degree=1), scaling_factor=2**49)
>>> key = scheme.key_gen_sparse(32, 3.2)
>>> ct = scheme.encrypt(scheme.encode([1.5 + 0.5j] + [0j] * 127), key)
>>> round(scheme.decode(scheme.decrypt(ct, key))[0].real, 3)
1.5
"""

from .bfv import BFV_Scheme
from .cggi16 import CGGI16, CGGI16_Key
from .ckks import CKKS_Ciphertext, CKKS_Scheme
from .gp25 import GP25, SAB_Key
from .linear_transform import CKKS_LinearTransform
from .ship import SHIP, SHIP_Key

__all__ = [
    "CGGI16",
    "GP25",
    "SHIP",
    "BFV_Scheme",
    "CGGI16_Key",
    "CKKS_Ciphertext",
    "CKKS_LinearTransform",
    "CKKS_Scheme",
    "SAB_Key",
    "SHIP_Key",
]
