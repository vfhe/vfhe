# SPDX-License-Identifier: Apache-2.0
"""vfhe.arith -- RNS/NTT polynomial arithmetic over the native engine.

Public surface:

* :class:`Ring` / :class:`Polynomial` -- quotient rings and their elements
  (:class:`Domain` tags the representation).
* :class:`ComplexRing` / :class:`ComplexPolynomial` -- CKKS-style encoding.
* :class:`Multiprecision` -- exact big-integer bridge.
* :class:`RingRegistry` -- shared native handle cache (rarely used directly).
* :mod:`vfhe.arith.errors` -- exception hierarchy mirroring native statuses.
"""

from .encoding import ComplexPolynomial, ComplexRing
from .errors import (
    ArgumentError,
    DomainError,
    NotInvertibleError,
    UnsupportedError,
    VfheError,
)
from .multiprecision import Multiprecision, mp_polynomial_to_list, mp_scalar_to_int
from .number_theory import crt, is_prime
from .polynomial import Domain, Polynomial
from .registry import RingRegistry
from .ring import Ring

__all__ = [
    "Ring",
    "Polynomial",
    "Domain",
    "RingRegistry",
    "is_prime",
    "crt",
    "ComplexRing",
    "ComplexPolynomial",
    "Multiprecision",
    "mp_polynomial_to_list",
    "mp_scalar_to_int",
    "VfheError",
    "DomainError",
    "NotInvertibleError",
    "UnsupportedError",
    "ArgumentError",
]
