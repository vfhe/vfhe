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

from abc import ABC, abstractmethod

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
