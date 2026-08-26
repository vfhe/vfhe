# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The contract every arithmetic parent satisfies, and what follows from it.

An implementation supplies a data structure and a handful of primitives, and
states which capability groups it carries; everything expressible from those
lives here once instead of in each implementation. A consumer that needs a
property of the domain asks the parent for it rather than inspecting the
parent's type or attributes.

`ArithParent` is a base class, not a Protocol: the generic methods below are
inherited, and `isinstance` against it is the supported way to ask whether an
object is an arithmetic domain at all.
"""

from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod

from .registry import resolve
from .spec import Capability, Domain, Spec


class ArithParent(ABC):
    """A set of elements with arithmetic: a ring, a field, a quotient ring.

    Subclasses set `spec` and implement `exceptional_set_size`. Everything
    else has a default here that an implementation may override when it can
    do better.
    """

    #: The (implementation, backend) this parent was built for.
    spec: Spec

    @property
    def implementation(self) -> str:
        return self.spec.implementation

    @property
    def backend(self) -> str:
        return self.spec.backend

    @property
    def capabilities(self) -> Capability:
        return self.spec.capabilities

    def supports(self, capability: Capability) -> bool:
        """Whether this parent carries every flag in `capability`.

        The cheap check a caller makes before an operation an implementation
        may not define at all, so the failure is a clear message here rather
        than a not-implemented answer deeper in.
        """
        return self.spec.has(capability)

    def require(self, capability: Capability) -> None:
        """Raise TypeError unless this parent carries `capability`."""
        if not self.supports(capability):
            missing = capability & ~self.capabilities
            raise TypeError(
                f"{type(self).__name__} ({self.spec}) does not support {missing!r}"
            )

    @property
    def domains_coincide(self) -> bool:
        """Whether the canonical and mul domains are one representation."""
        return self.supports(Capability.DOMAINS_COINCIDE)

    def mul_domain(self) -> Domain:
        """The domain multiplication needs operands to be in."""
        return Domain.CANONICAL if self.domains_coincide else Domain.MUL

    @property
    @abstractmethod
    def exceptional_set_size(self) -> int:
        """|A| for an exceptional set of this domain.

        A set whose pairwise differences are all invertible. Protocols with
        soundness in the size of the challenge set read this rather than
        deriving it from the domain's internals, which differ per
        implementation.
        """


class _ImplementationDispatch(ABCMeta):
    """Constructing a generic front class builds its resolved implementation.

    A class that defines ``_concrete`` in its own namespace is a front:
    calling it strips the ``implementation=`` / ``backend=`` keywords, asks
    ``_concrete`` for the class those select, and constructs that class once,
    normally. Subclasses do not inherit front behavior -- a concrete class
    constructs as itself.
    """

    def __call__(cls, *args, **kwargs):
        concrete = cls.__dict__.get("_concrete")
        if concrete is None:
            return super().__call__(*args, **kwargs)
        implementation = kwargs.pop("implementation", None)
        backend = kwargs.pop("backend", None)
        target = concrete(implementation, backend, *args, **kwargs)
        return target(*args, **kwargs)


class Ring(ArithParent, metaclass=_ImplementationDispatch):
    """A quotient polynomial ring, independent of how its elements are stored.

    ``Ring(...)`` builds the default implementation (RNS); pass
    ``implementation=`` / ``backend=`` to name another. Subclasses hold one
    representation each and everything specific to it; what is true for every
    representation belongs here.
    """

    @staticmethod
    def _concrete(
        implementation: str | None, backend: str | None, *args, **kwargs
    ) -> type:
        return resolve(implementation or "rns", backend).parent_cls


class Polynomial(metaclass=_ImplementationDispatch):
    """An element of a `Ring`, in whatever representation the ring uses.

    ``Polynomial(ring, ...)`` builds the ring's own element type: the class
    comes from the ring's spec, so a polynomial is always matched to its
    ring's implementation and a caller never names the concrete class.
    """

    @staticmethod
    def _concrete(
        implementation: str | None, backend: str | None, ring=None, *args, **kwargs
    ) -> type:
        return ring.spec.element_cls


class Field(ArithParent, metaclass=_ImplementationDispatch):
    """A finite field, independent of how its elements are stored.

    ``Field(...)`` builds the default implementation (the extension field over
    a `Modulus`); pass ``implementation=`` / ``backend=`` to name another.
    Subclasses provide `order`; what holds for every field lives here.
    """

    @staticmethod
    def _concrete(
        implementation: str | None, backend: str | None, *args, **kwargs
    ) -> type:
        return resolve(implementation or "field", backend).parent_cls

    @property
    @abstractmethod
    def order(self) -> int:
        """The number of elements of the field."""

    @property
    def exceptional_set_size(self) -> int:
        """|A| for any field is its order: every nonzero difference of two
        field elements is invertible, so the whole field is exceptional."""
        return self.order
