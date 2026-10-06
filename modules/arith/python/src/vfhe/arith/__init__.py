# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
# vfhe.arith public API re-exports.
from .base import ArithParent, Field, FieldElement, FieldVector, Polynomial, Ring
from .impl.complex.complex import ComplexPolynomial, ComplexRing
from .impl.field.field import ExtensionField, ExtensionFieldElement
from .impl.field.vector import ExtensionFieldVector
from .impl.mp.multiprecision import Multiprecision
from .impl.pmf.ntt import PseudoMersenneNTT
from .impl.pmf.pseudo_mersenne import PseudoMersenneElement, PseudoMersenneField
from .impl.pmf.vector import PseudoMersenneVector

# Registers the vfhe.util.io codecs.
from .impl.rns import io as _io  # noqa: F401  # pyright: ignore[reportUnusedImport]
from .impl.rns.polynomial import (
    Representation,
    RNSPolynomial,
    RNSRing,
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
    "Representation",
    "Ring",
    "SparseMLE",
    "Spec",
    "backends",
    "crt",
    "domain_of",
    "gen_pseudo_mersenne_prime",
    "implementations",
    "is_prime",
    "register_conversion",
    "registered",
    "repr",
    "reset_state",
    "resolve",
]
