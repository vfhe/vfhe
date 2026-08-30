# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
# vfhe.arith public API re-exports.
from .base import ArithParent, Field, FieldVector, Polynomial, Ring
from .impl.complex.complex import ComplexPolynomial, ComplexRing
from .impl.field.field import ExtensionField, FieldElement
from .impl.field.vector import ExtensionFieldVector
from .impl.mp.multiprecision import Multiprecision
from .impl.pmf.pseudo_mersenne import PseudoMersenneElement, PseudoMersenneField
from .impl.rns.polynomial import RNSPolynomial, RNSRing, domain_of, repr
from .number_theory import crt, gen_pseudo_mersenne_prime, is_prime
from .registry import (
    backends,
    implementations,
    register_conversion,
    registered,
    resolve,
)
from .spec import Capability, Constraints, Domain, Spec
from .state import (
    rebind as rebind_state,
)
from .state import (
    reset as reset_state,
)

__all__ = [
    "ArithParent",
    "Capability",
    "ComplexPolynomial",
    "ComplexRing",
    "Constraints",
    "Domain",
    "ExtensionField",
    "ExtensionFieldVector",
    "Field",
    "FieldElement",
    "FieldVector",
    "Multiprecision",
    "Polynomial",
    "PseudoMersenneElement",
    "PseudoMersenneField",
    "RNSPolynomial",
    "RNSRing",
    "Ring",
    "Spec",
    "backends",
    "crt",
    "domain_of",
    "gen_pseudo_mersenne_prime",
    "implementations",
    "is_prime",
    "rebind_state",
    "register_conversion",
    "registered",
    "repr",
    "reset_state",
    "resolve",
]
