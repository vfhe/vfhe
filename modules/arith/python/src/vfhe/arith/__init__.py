# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""RNS polynomial arithmetic over ``Z_q[X]/(X^N+1)`` (`Ring`, `Polynomial`), finite fields and their vectors (`Field`, `FieldElement`, `FieldVector`), and multilinear extensions over any of them (`MLE`, `SparseMLE`).

``Ring(N, bits, split_degree=...)`` resolves to an implementation through the
registry (`resolve`, `implementations`, `backends`): `RNSRing` is the one every
scheme computes in, `ComplexRing` the CKKS encoder's, `ExtensionField` and
`PseudoMersenneField` the SNARK stack's. On a ring or a field, ``random_*``
returns a new element; on an element or a vector, ``sample_*`` overwrites it
in place. Every sampler takes a ``random=`` source (`vfhe.crypto.random.Random`), the
entropy stream by default. Implementations cache native state per process;
`reset_state` empties the caches.

>>> from vfhe.arith import Ring
>>> ring = Ring(256, 120, split_degree=1)
>>> a, b = ring.random_element(), ring.random_element()
>>> (a + b) - b == a
True
"""

from .base import ArithParent, Field, FieldElement, FieldVector, Polynomial, Ring
from .impl.complex.complex import ComplexPolynomial, ComplexRing
from .impl.field.field import ExtensionField, ExtensionFieldElement
from .impl.field.vector import ExtensionFieldVector
from .impl.mp.multiprecision import Multiprecision
from .impl.pmf.ntt import PseudoMersenneNTT
from .impl.pmf.pseudo_mersenne import PseudoMersenneElement, PseudoMersenneField
from .impl.pmf.vector import PseudoMersenneVector
from .impl.rns.cuda import CUDAPolynomial, CUDARing
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
    "CUDAPolynomial",
    "CUDARing",
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
