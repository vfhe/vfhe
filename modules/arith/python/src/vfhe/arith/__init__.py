# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
# vfhe.arith public API re-exports.
from .base import ArithParent, Field, FieldElement, FieldVector, Polynomial, Ring
from .counting import count_occurrences
from .dispatch import generic_only, set_generic_only
from .impl.complex.complex import ComplexPolynomial, ComplexRing
from .impl.field.field import ExtensionField, ExtensionFieldElement
from .impl.field.vector import ExtensionFieldVector
from .impl.mp.multiprecision import Multiprecision
from .impl.pmf.ntt import PseudoMersenneNTT
from .impl.pmf.pseudo_mersenne import PseudoMersenneElement, PseudoMersenneField
from .impl.pmf.vector import PseudoMersenneVector
from .impl.rns.polynomial import (
    Representation,
    RNSPolynomial,
    RNSRing,
    RNSScalar,
    domain_of,
    repr,
)
from .mle import MLE, MLE_Basis, MLE_Variable, SparseMLE
from .number_theory import crt, gen_pseudo_mersenne_prime, is_prime
from .registry import (
    backends,
    implementations,
    register_conversion,
    registered,
    resolve,
)
from .spec import Capability, Constraints, Domain, Spec
from .state import reset as reset_state

__all__ = [
    "MLE",
    "ArithParent",
    "Capability",
    "ComplexPolynomial",
    "ComplexRing",
    "Constraints",
    "Domain",
    "ExtensionField",
    "ExtensionFieldElement",
    "ExtensionFieldVector",
    "Field",
    "FieldElement",
    "FieldVector",
    "MLE_Basis",
    "MLE_Variable",
    "Multiprecision",
    "Polynomial",
    "PseudoMersenneElement",
    "PseudoMersenneField",
    "PseudoMersenneNTT",
    "PseudoMersenneVector",
    "RNSPolynomial",
    "RNSRing",
    "RNSScalar",
    "Representation",
    "Ring",
    "SparseMLE",
    "Spec",
    "backends",
    "count_occurrences",
    "crt",
    "domain_of",
    "gen_pseudo_mersenne_prime",
    "generic_only",
    "implementations",
    "is_prime",
    "register_conversion",
    "registered",
    "repr",
    "reset_state",
    "resolve",
    "set_generic_only",
]
