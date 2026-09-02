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

from .registry import registered, resolve
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
        implementation: str | None, backend: str | None, *_args, **_kwargs
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
        _implementation: str | None, _backend: str | None, ring=None, *_args, **_kwargs
    ) -> type:
        return ring.spec.element_cls


class Field(ArithParent, metaclass=_ImplementationDispatch):
    """A finite field F_(p^d).

    ``Field(modulus, degree=1, ...)`` builds a concrete implementation. The
    first two parameters are the ones every implementation takes:

    - ``modulus`` -- the prime p.
    - ``degree`` -- the extension degree d; 1 (the default) is the prime
      field F_p. Some implementations serve only ``degree=1``.

    Anything after them is implementation-specific -- the extension field,
    for instance, takes ``w`` defining ``x**degree - w``.

    Left unnamed, the implementation is chosen from the parameters: a
    ``degree``-1 modulus above 64 bits that is a pseudo-Mersenne prime the
    ``pmf`` kernels cover resolves to ``pmf``; everything else resolves to
    ``field``, the extension field over a `Modulus`. Pass ``implementation=``
    / ``backend=`` to name one explicitly; either way the modulus width is
    checked against the resolved backend's constraints.

    Subclasses provide `order`; what holds for every field lives here.
    """

    @staticmethod
    def _concrete(
        implementation: str | None,
        backend: str | None,
        modulus: int | None = None,
        degree: int = 1,
        *_args,
        **_kwargs,
    ) -> type:
        bits = modulus.bit_length() if isinstance(modulus, int) else None
        if implementation is None:
            implementation = "field"
            if degree == 1 and bits is not None and bits > 64:
                # Imported here: impl modules import this one at load time.
                from .impl.pmf.pseudo_mersenne import is_pseudo_mersenne

                if is_pseudo_mersenne(modulus):
                    implementation = "pmf"
        return resolve(implementation, backend, prime_bits=bits).parent_cls

    @property
    @abstractmethod
    def order(self) -> int:
        """The number of elements of the field."""

    @property
    def exceptional_set_size(self) -> int:
        """|A| for any field is its order: every nonzero difference of two
        field elements is invertible, so the whole field is exceptional."""
        return self.order


class FieldVector(metaclass=_ImplementationDispatch):
    """Many elements of one `Field`, held together in one buffer.

    A vector exists because a length-n operation costs one call into C
    instead of n: the kernels see whole arrays, so they vectorize across
    elements and the per-element boundary crossing disappears. Prefer it to a
    Python list of elements wherever the same operation applies to all of
    them.

    ``FieldVector(field, n)`` builds n zeros and ``FieldVector(field,
    values)`` builds from a sequence; either way the class comes from the
    field's spec, so a vector is always matched to its field's
    implementation and a caller never names the concrete class. A field whose
    implementation has no vector type raises TypeError.

    An implementation must provide the operations whose cost is the point --
    elementwise ``+ - *``, broadcast against a single element, `scale`,
    `sum`, and the movement and encoding it needs. The rest have a default
    here, written once against those: correct for any implementation, and
    replaceable by one that can do better.

    Vectors are mutable through `__setitem__`, so they are not hashable;
    `hash` is the digest of the contents, as on an element. Every arithmetic
    operation returns a fresh vector rather than writing into an operand.
    """

    @staticmethod
    def _concrete(
        _implementation: str | None, _backend: str | None, field=None, *_args, **_kwargs
    ) -> type:
        vector_cls = field.spec.vector_cls
        if vector_cls is None:
            raise TypeError(
                f"{field.spec} has no vector type; "
                f"registered implementations with one: "
                f"{[str(s) for s in registered().values() if s.vector_cls]}"
            )
        return vector_cls

    #: A vector holds elements, so it cannot be a dict key: `hash` below is a
    #: digest of the contents, not an identity.
    __hash__ = None

    def __pow__(self, exponent: int):
        """Elementwise ``self ** exponent`` by square-and-multiply.

        The exponent is a plain integer, not reduced modulo the group order.
        Negative exponents are not accepted: invert first, which is one batch
        inversion rather than n.
        """
        if not isinstance(exponent, int) or isinstance(exponent, bool):
            raise TypeError(f"exponent must be an int, not {type(exponent).__name__}")
        if exponent < 0:
            raise ValueError("negative exponent; call inverse() first")
        result = type(self)(self.field, [self.field.one] * len(self))
        base = self.copy()
        while exponent > 0:
            if exponent & 1:
                result = result * base
            exponent >>= 1
            if exponent:
                base = base * base
        return result

    def inverse(self):
        """The elementwise inverse, by Montgomery's trick.

        One inversion plus three multiplications per element, rather than n
        inversions: the prefix products are formed, the last is inverted, and
        a reverse sweep peels each factor back off. Raises ValueError if any
        element is zero.
        """
        n = len(self)
        if n == 0:
            return type(self)(self.field, 0)
        elements = self.to_list()
        prefix = []
        running = self.field.one
        for element in elements:
            running = running * element
            prefix.append(running)
        running = running.inverse()  # raises for a zero anywhere in the product
        inverses = [None] * n
        for i in range(n - 1, 0, -1):
            inverses[i] = running * prefix[i - 1]
            running = running * elements[i]
        inverses[0] = running
        return type(self)(self.field, inverses)

    @staticmethod
    def concat(vectors: list):
        """One vector holding every element of `vectors`, in order."""
        if not vectors:
            raise ValueError("concat needs at least one vector")
        field = vectors[0].field
        for vector in vectors:
            if vector.field != field:
                raise ValueError("cannot concatenate vectors over different fields")
        return type(vectors[0])(field, [e for v in vectors for e in v])

    def query(self, indices):
        """The elements at `indices`, gathered into a new vector."""
        return type(self)(self.field, [self[i] for i in indices])
