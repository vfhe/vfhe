# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
# vfhe.arith public API re-exports.
from .base import ArithParent
from .complex import ComplexPolynomial, ComplexRing
from .field import Field, FieldElement
from .multiprecision import Multiprecision
from .number_theory import crt, gen_pseudo_mersenne_prime, is_prime
from .polynomial import Polynomial, Ring, repr
from .pseudo_mersenne import PseudoMersenneElement, PseudoMersenneField
from .registry import (
    backends,
    implementations,
    register_conversion,
    registered,
    resolve,
)
from .spec import Capability, Constraints, Domain, Spec

__all__ = [
    "ArithParent",
    "Capability",
    "ComplexPolynomial",
    "ComplexRing",
    "Constraints",
    "Domain",
    "Field",
    "FieldElement",
    "Multiprecision",
    "Polynomial",
    "PseudoMersenneElement",
    "PseudoMersenneField",
    "Ring",
    "Spec",
    "backends",
    "crt",
    "gen_pseudo_mersenne_prime",
    "implementations",
    "is_prime",
    "register_conversion",
    "registered",
    "repr",
    "resolve",
]
