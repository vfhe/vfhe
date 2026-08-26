# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""What an arithmetic parent is, as data: domains, capabilities, and Spec.

Three axes describe an arithmetic object in this library:

``implementation``
    What an element is, mathematically and as data - RNS residues, a
    multiprecision limb vector, a pseudo-Mersenne field element. Chosen per
    object at runtime; changes the set of representable values and which
    operations exist at all.
``backend``
    How that implementation's operations are carried out - transform choice,
    word width, device. Chosen per object at runtime; the observable values
    are the same, the storage and kernels are not.
``engine``
    Which machine code runs. Chosen once per process by CPU capability
    (``VFHE_ENGINE``, ``vfhe.misc.libvfhe.active_engine``) and not addressed
    here at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntFlag, auto


class Domain(Enum):
    """Which representation an element currently holds.

    Every implementation offers two, and an element is always in one of them
    (or ``EMPTY``, meaning allocated but never written):

    ``CANONICAL``
        A canonical form: one representative per ring element. Equality,
        hashing and serialization are only defined here, because only here
        does equal value imply equal bits.
    ``MUL``
        The image under an injective ring homomorphism chosen so that
        multiplication is cheap - the NTT/CRT decomposition for RNS,
        evaluation form for an FFT backend, and Montgomery-like forms in
        general. Need not be canonical.

    The two may coincide (see ``Capability.DOMAINS_COINCIDE``), in which case
    conversion is the identity and every element reports ``CANONICAL``.
    """

    EMPTY = auto()
    CANONICAL = auto()
    MUL = auto()


class Capability(IntFlag):
    """Which groups of operations an implementation supports.

    Bit positions are the contract shared with the C method table, so a value
    can be handed across the boundary unchanged. A caller that wants to fail
    early tests the flag; one that calls anyway gets the implementation's
    not-implemented answer for that slot.
    """

    #: Allocate, copy, zero, add, sub, negate, scale, multiply, convert
    #: domains. Every implementation has these.
    CORE = auto()
    #: The element is a polynomial in a quotient ring, so Galois automorphisms
    #: and monomial multiplication are meaningful. Absent for a field, whose
    #: elements are scalars.
    QUOTIENT_POLY_RING = auto()
    #: Movement within a modulus tower: round/floor division, modulus
    #: reduction, lifting to a larger modulus.
    TOWER = auto()
    #: Sampling: uniform, gaussian, and exceptional-set elements.
    SAMPLING = auto()
    #: Results are exact. Absent means an approximate representation whose
    #: error the caller must account for.
    EXACT = auto()
    #: The canonical and mul domains are the same representation, so domain
    #: conversion is free and the domain flag never changes.
    DOMAINS_COINCIDE = auto()


@dataclass(frozen=True)
class Constraints:
    """Limits a backend places on the parameters of a parent built on it.

    The resolver applies these *before* parameters are chosen, so a backend
    that can only hold narrow primes gets narrow primes rather than an error
    after the fact. ``None`` means the backend does not constrain that
    parameter.
    """

    #: Largest prime bit width the backend's storage and kernels accept.
    max_prime_bits: int | None = None
    #: Smallest transform length the backend handles; shorter parents must
    #: use another backend.
    min_transform_length: int | None = None

    def accepts_prime_bits(self, bits: int) -> bool:
        """Whether a prime of this bit width fits the backend."""
        return self.max_prime_bits is None or bits <= self.max_prime_bits


@dataclass(frozen=True)
class Spec:
    """One registered (implementation, backend) pair and what it provides.

    ``parent_cls`` is the class the resolver instantiates; ``element_cls`` is
    the type of its elements, or None where an implementation has no separate
    element type.
    """

    implementation: str
    backend: str
    parent_cls: type
    element_cls: type | None = None
    capabilities: Capability = Capability.CORE
    constraints: Constraints = field(default_factory=Constraints)
    #: Preference among backends of one implementation, best first; the
    #: resolver picks the lowest rank whose constraints the parameters meet.
    rank: int = 0

    @property
    def key(self) -> tuple[str, str]:
        return (self.implementation, self.backend)

    def has(self, capability: Capability) -> bool:
        """Whether every flag in `capability` is present."""
        return bool(self.capabilities & capability == capability)

    def __str__(self) -> str:
        return f"{self.implementation}/{self.backend}"
