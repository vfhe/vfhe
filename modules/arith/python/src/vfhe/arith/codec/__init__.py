# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""`vfhe.util.io` codecs for RNS rings and polynomials: `RingCodec`, `PolynomialCodec`, and their row encoding.

A ring is written by its primes, since base indices depend on the order a
process built its rings in; rows go in ascending prime value. A ring also
records each prime's NTT root, because NTT-domain rows (seeded ones included)
only mean the same to a reader with the same roots: `RingCodec.require_same_transform`
checks that before rows are read.
"""

from .polynomial import PolynomialCodec
from .ring import RingCodec

__all__ = ["PolynomialCodec", "RingCodec"]
