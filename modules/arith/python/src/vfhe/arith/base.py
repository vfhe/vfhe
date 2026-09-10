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

The classes here are also the **declared API**. `Ring`, `Polynomial`, `Field`,
`FieldElement` and `FieldVector` are front classes: calling one builds the
resolved implementation's class (see `_ImplementationDispatch`), so the member
a caller may rely on for *any* implementation is the member declared here. A
front declares only what is representation-independent; a caller that needs an
implementation's own coordinates -- an RNS prime set, a pseudo-Mersenne limb
layout -- names that implementation's class instead.

Every declaration carries the signature and the documentation for the
operation, so an editor can complete a call from the front class and a reader
does not have to find the implementation to learn what a parameter means. A
declaration's body raises NotImplementedError: reaching one means the
implementation resolved for the parent does not provide that operation, which
is the same answer `require` gives from the capability flags, one level later.
"""

from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from vfhe.crypto import entropy

from .registry import registered, resolve
from .spec import Capability, Domain, Spec

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    # Type-checking only, so it adds no runtime dependency: `Self` is what an
    # in-place operator returns, and typing has it only from 3.11.
    from typing_extensions import Self

    from .impl.rns.polynomial import RNSPolynomial, RNSRing

# Seed length for an unseeded field sampler: 256 bits, matching the width the
# C expanders absorb.
_SEED_BYTES = 32


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

    A type checker does not follow this: to it, ``Ring(...)`` is a call to
    `Ring`. Two things make that read correctly. Each front provides a
    concrete body for every member it inherits as abstract, so it is not an
    abstract class to instantiate; and a front whose default implementation is
    known -- `Ring`, `Polynomial` -- declares a ``__new__`` under
    ``TYPE_CHECKING`` giving the constructor's parameters and the class it
    yields. Those declarations exist only for the checker; the dispatch below
    is what runs.
    """

    def __call__(cls, *args, **kwargs):
        concrete = cls.__dict__.get("_concrete")
        if concrete is None:
            return super().__call__(*args, **kwargs)
        implementation = kwargs.pop("implementation", None)
        backend = kwargs.pop("backend", None)
        target = concrete(implementation, backend, *args, **kwargs)
        return target(*args, **kwargs)


def check_spec_keywords(
    cls: type, implementation: str | None, backend: str | None
) -> None:
    """Raise unless ``(implementation, backend)`` resolves to `cls`.

    A front strips both keywords and uses them to pick the class, so an
    implementation reached that way never sees them. Constructed directly it
    does, and this is how it checks them: naming a spec that is not this class
    is an error rather than a silently ignored argument. Both None accepts.
    """
    if implementation is None and backend is None:
        return
    spec = resolve(implementation or cls.spec.implementation, backend)
    if spec.parent_cls is not cls:
        raise LookupError(
            f"{implementation or cls.spec.implementation}/{backend or 'default'} "
            f"resolves to {spec.parent_cls.__name__}, not {cls.__name__}"
        )


class Ring(ArithParent, metaclass=_ImplementationDispatch):
    """A quotient polynomial ring, independent of how its elements are stored.

    ``Ring(...)`` builds the default implementation (RNS, whose class is
    `RNSRing` and whose parameters the constructor below documents); pass
    ``implementation=`` / ``backend=`` to name another. Subclasses hold one
    representation each and everything specific to it; what is true for every
    representation belongs here.

    What that split means for a caller: the members below are the ones any
    implementation of a quotient polynomial ring offers. An RNS ring's own
    coordinates -- its primes, its RNS mask, its shared base, the split degree
    of the incomplete transform -- are `RNSRing`'s, because another
    representation of the same mathematical ring has none of them. Code that
    reads them should say `RNSRing`, which is also what ``Ring(...)`` returns.

    Rings have no value equality: two rings are the same object or they are
    not (`intersec` compares by identity). Match rings across a scheme by the
    implementation's own key -- for RNS, the prime `RNSRing.mask`.
    """

    if TYPE_CHECKING:
        # Declaration only; `_ImplementationDispatch.__call__` is what runs.
        def __new__(
            cls,
            N: int,
            mod_size: int | None = None,
            split_degree: int | None = None,
            primes: list[int] | None = None,
            mask: int | None = None,
            prime_size: int | list[int] = 49,
            exceptional_set_size: int = 128,
            *,
            implementation: str | None = None,
            backend: str | None = None,
        ) -> RNSRing:
            """Build a quotient ring Z_q[X]/(X^N + 1).

            The modulus is given one of four ways, and exactly one is needed:
            a total bit budget (``mod_size``), explicit per-prime bit sizes
            (``prime_size`` as a list), an explicit prime pool (``primes``), or
            a ``mask`` selecting a subset of a pool. The rest of the
            parameters are the ring's shape.

            :param N: Degree of the quotient: elements are polynomials modulo
                ``X**N + 1``. A power of two.
            :param mod_size: Total modulus width in bits. The prime count is
                ``ceil(mod_size / prime_size)`` with ``prime_size`` a scalar.
            :param split_degree: Split degree of the incomplete transform: the
                number of coefficients each residue slot holds. Left None it
                is chosen as the smallest power of two making the residue
                fields at least ``exceptional_set_size`` large.
            :param primes: Pool of RNS primes to draw from, in the base's
                order. Left None the primes are generated to fit
                ``prime_size``.
            :param mask: Bitmask over ``primes`` selecting this ring's primes,
                by their index in the shared RNS base. Needs ``primes``. Use
                it to build a non-nested level chain.
            :param prime_size: Per-prime width in bits: one int for every
                prime, or a list giving each. A list also fixes the prime
                count.
            :param exceptional_set_size: Target size for the ring's
                exceptional set, which is what ``split_degree`` is chosen from
                when it is not given. Ignored once ``split_degree`` is
                explicit.
            :param implementation: Name the representation instead of taking
                the default (``'rns'``).
            :param backend: Name how that representation's operations run
                instead of taking its best available.
            """
            ...

    #: Degree of the quotient: elements are polynomials modulo ``X**N + 1``.
    N: int

    @staticmethod
    def _concrete(
        implementation: str | None, backend: str | None, *args, **kwargs
    ) -> type:
        return resolve(implementation or "rns", backend).parent_cls

    # --- the generic C interface ---

    @property
    def arith_ring(self):
        """This ring's handle in the generic C interface (an ``ArithRing``).

        The handle the representation-independent kernels take. Shared and
        never freed, so it is safe to hold: a structure built over a ring may
        point at it, and nothing can prove the last one is gone.
        """
        raise NotImplementedError

    # --- the tower of quotients ---

    def is_quotient_ring(self, parent: Ring) -> bool:
        """Whether ``self`` is a quotient of ``parent``.

        True when every one of this ring's residues is present in ``parent``,
        so an element of ``parent`` reduces into this ring without a base
        conversion.
        """
        raise NotImplementedError

    def intersec(self, other: Ring) -> Ring:
        """The smaller of two nested rings: where a binary operation lands.

        One of the two must be a quotient of the other, and that one is
        returned; a pair that is not nested is an error rather than a
        silently widened result.
        """
        raise NotImplementedError

    def union(self, other: Ring) -> Ring:
        """The smallest ring both ``self`` and ``other`` are quotients of.

        Neither ring need contain the other, so this is not `intersec`
        reversed.
        """
        raise NotImplementedError

    # --- sampling ---

    def random_element(self, ntt: bool = True) -> Polynomial:
        """A uniform element of the ring.

        :param ntt: Return it in the mul domain (the NTT domain, for RNS)
            rather than the canonical one. The default costs nothing, since
            the sampler writes residues directly.
        """
        raise NotImplementedError

    def random_gaussian_element(self, sigma: float, ntt: bool = True) -> Polynomial:
        """An element with discrete-Gaussian coefficients of width ``sigma``.

        :param sigma: Standard deviation, in coefficients.
        :param ntt: Convert to the mul domain before returning. Sampling is
            always done in the canonical domain, so True adds a transform.
        """
        raise NotImplementedError

    def random_exceptional(self, ntt: bool = True) -> Polynomial:
        """A uniform element of the ring's exceptional set.

        Every difference of two such elements is invertible, which is what a
        protocol challenge needs. `exceptional_set_size` is how large the set
        is.

        :param ntt: Return it in the mul domain rather than the canonical one.
        """
        raise NotImplementedError

    @property
    def exceptional_set_size(self) -> int:
        """|A| for an exceptional set of this ring."""
        raise NotImplementedError


class Polynomial(metaclass=_ImplementationDispatch):
    """An element of a `Ring`, in whatever representation the ring uses.

    ``Polynomial(ring, ...)`` builds the ring's own element type: the class
    comes from the ring's spec, so a polynomial is always matched to its
    ring's implementation and a caller never names the concrete class.

    The members below are the ones every representation of a ring element
    offers. The RNS element's own surface -- its native handle, its
    representation flag and the transforms that move between representations,
    the per-prime coefficient matrix -- is `RNSPolynomial`'s, which is what
    ``Polynomial(ring)`` returns for an `RNSRing`.

    An element is mutable: the in-place operators and the ``from_*`` loaders
    write into the buffer it already owns, while the binary operators return a
    fresh element. Readers never mutate -- reading a value does not change the
    representation of the object read.
    """

    if TYPE_CHECKING:
        # Declaration only; `_ImplementationDispatch.__call__` is what runs.
        def __new__(
            cls,
            ring: Ring,
            repr: Any = ...,
        ) -> RNSPolynomial:
            """Allocate an element of ``ring``, holding no value yet.

            The ``implementation=`` / ``backend=`` keywords the other fronts
            take are accepted and ignored here: the ring has already fixed
            both.

            :param ring: The parent. Its spec decides the class built.
            :param repr: The representation to mark the fresh buffer as
                holding; the default says "allocated, never written", which is
                what a caller wants before a ``from_*`` loader or an output
                parameter fills it in.
            """
            ...

    #: The parent this element belongs to.
    ring: Ring

    @staticmethod
    def _concrete(
        _implementation: str | None,
        _backend: str | None,
        ring: Ring | None = None,
        *_args,
        **_kwargs,
    ) -> type:
        if ring is None:
            raise TypeError("Polynomial(ring) needs the ring it belongs to")
        element_cls = ring.spec.element_cls
        if element_cls is None:
            raise TypeError(f"{ring.spec} has no element type")
        return element_cls

    # --- loading and reading values ---

    def from_array(self, array: Sequence[int]) -> Polynomial:
        """Load integer coefficients, low degree first, and return ``self``.

        Shorter than the ring's degree is padded with zeros. Values are taken
        as signed and reduced into the modulus.
        """
        raise NotImplementedError

    def from_bigint_array(self, array: Sequence[int]) -> Polynomial:
        """Load coefficients too wide for a machine word, and return ``self``.

        The path `from_array` cannot take: it reads each value as an int64,
        so a coefficient at full modulus width has to come through here.
        """
        raise NotImplementedError

    def get_polynomial(self, signed: bool = False) -> list[int]:
        """The coefficients as Python integers, low degree first.

        :param signed: Report each coefficient as the representative of
            smallest absolute value, in ``(-q/2, q/2]``, rather than in
            ``[0, q)``. What to use when the coefficients are noise.
        """
        raise NotImplementedError

    def __iter__(self) -> Iterator:
        """Iterate the value in whatever per-residue rows the representation
        keeps. Present for compatibility with a list; not an efficient
        iterator, and `get_polynomial` is what returns plain integers."""
        raise NotImplementedError

    def copy(self) -> Polynomial:
        """A fresh element with the same value and representation."""
        raise NotImplementedError

    def __copy__(self) -> Polynomial:
        return self.copy()

    # --- identity ---

    def get_hash(self) -> list[int]:
        """A 256-bit digest of the value, as four 64-bit words.

        Taken in the mul domain, so the same value hashes the same however the
        element is currently stored. Host-endian and width-dependent: an
        identifier within one process, not across builds.
        """
        raise NotImplementedError

    def get_hash_pointer(self):
        """`get_hash` left in native memory, for a caller that passes it on."""
        raise NotImplementedError

    def __hash__(self):
        raise TypeError(
            "not a Python hashable object. Call polynomial.get_hash() for cryptographic hash"
        )

    def __eq__(self, value) -> bool:
        """Value equality, against another element, an int, or a list of ints.

        An int compares against the constant polynomial; a list against the
        per-residue constant terms.
        """
        raise NotImplementedError

    # --- the generic C interface ---

    def as_element(self):
        """This element as an ``ArithElement`` for the generic C interface.

        A view, not a copy: the returned struct points at the same storage, so
        it must not outlive this element.
        """
        raise NotImplementedError

    # --- ring structure ---

    def automorphism(self, gen: int):
        """``self(X**gen)``: the Galois automorphism of the quotient ring.

        ``gen`` is odd and below ``2 * ring.N``, so that ``X -> X**gen``
        permutes the quotient. Returns a fresh element.
        """
        raise NotImplementedError

    # --- movement in the tower ---

    def base_extend(
        self, ring: Ring | None = None, out: Polynomial | None = None
    ) -> Polynomial:
        """This value, exactly, as an element of a *larger* ring.

        Pass the destination ``ring`` to allocate the result, or ``out`` to
        write into an element that already exists.
        """
        raise NotImplementedError

    def mod_reduce(
        self, ring: Ring | None = None, out: Polynomial | None = None
    ) -> Polynomial:
        """This value reduced into a smaller ring of the same tower.

        The modulus shrinks and the value is kept: exact where the smaller
        modulus can hold it. `round_division` is the other direction of the
        same move, dividing the value as it drops the residues.

        Pass the destination ``ring`` to allocate the result, or ``out`` to
        write into an element that already exists.
        """
        raise NotImplementedError

    def round_division(self, ring: Ring) -> Polynomial:
        """Divide by the residues ``ring`` drops, rounding, **in place**.

        The rescale primitive: the value is divided by the product of the
        residues that ``ring`` does not have, to the nearest integer, and
        ``self.ring`` becomes ``ring``. Returns ``self``.

        Not to be confused with `MLWE.round_division`, which moves a whole
        ciphertext and takes a level.
        """
        raise NotImplementedError

    def floor_division(self, ring: Ring) -> Polynomial:
        """`round_division` truncating instead of rounding, in place."""
        raise NotImplementedError

    def scaled_lift(self, ring: Ring, delta=None) -> Polynomial:
        """This value lifted to a larger ring and scaled by ``delta``.

        The base-extension behind a rescale between moduli that are not nested
        quotients of one another.

        :param ring: The destination ring, which must contain this one.
        :param delta: The per-residue scaling factor, as the native array the
            ring builds (``RNSRing.modulus_ratio(..., return_pointer=True)``).
            None scales by 1.
        """
        raise NotImplementedError

    # --- sampling into this element ---

    def sample_uniform(self, ntt: bool = True) -> Polynomial:
        """Overwrite with a uniform element of the ring; returns ``self``."""
        raise NotImplementedError

    def sample_gaussian(self, sigma: float, ntt: bool = True) -> Polynomial:
        """Overwrite with discrete-Gaussian coefficients; returns ``self``."""
        raise NotImplementedError

    def sample_exceptional(self, ntt: bool = True) -> Polynomial:
        """Overwrite with an exceptional-set element; returns ``self``."""
        raise NotImplementedError

    # --- arithmetic ---

    def __add__(self, other) -> Polynomial:
        """``self + other``, for another element or an int."""
        raise NotImplementedError

    def __radd__(self, other) -> Polynomial:
        return self.__add__(other)

    def __iadd__(self, other) -> Self:
        """``self += other``, reusing this element's buffer."""
        raise NotImplementedError

    def __sub__(self, other) -> Polynomial:
        """``self - other``, for another element or an int."""
        raise NotImplementedError

    def __rsub__(self, other) -> Polynomial:
        return (-self) + other

    def __isub__(self, other) -> Self:
        """``self -= other``, reusing this element's buffer."""
        raise NotImplementedError

    def __mul__(self, other) -> Polynomial:
        """``self * other``: another element, an int, or one int per residue.

        Both operands are moved to the mul domain first, and the result lands
        in the smaller of the two rings.
        """
        raise NotImplementedError

    def __rmul__(self, other) -> Polynomial:
        return self.__mul__(other)

    def __imul__(self, other) -> Self:
        """``self *= other``, reusing this element's buffer.

        The form to prefer in a loop: the binary operators allocate a result
        per iteration, which dominates the kernels they call.
        """
        raise NotImplementedError

    def __neg__(self) -> Polynomial:
        raise NotImplementedError


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

    Unlike `Ring`, the members below are the whole common surface and every
    implementation carries all of them: the two that exist differ in how an
    element is stored, not in what a field offers. An implementation may add
    to it -- `PseudoMersenneField` also serves transform plans and roots of
    unity -- and code that uses such an addition names that class.
    """

    if TYPE_CHECKING:
        # Declaration only; `_ImplementationDispatch.__call__` is what runs.
        def __init__(
            self,
            modulus: int,
            degree: int = 1,
            w: int | None = None,
            *,
            implementation: str | None = None,
            backend: str | None = None,
        ) -> None:
            """Build F_(modulus^degree).

            :param modulus: The characteristic p, a prime. Capped at 62 bits
                by the native kernels, which reduce from a lazy range four
                times the modulus.
            :param degree: The extension degree d. 1 is the prime field.
            :param w: For ``degree > 1``, the constant in the defining
                polynomial ``x**degree - w``, which the caller is responsible
                for making irreducible. Ignored at ``degree=1``, and not
                accepted by an implementation that serves only prime fields.
            :param implementation: Name the representation instead of letting
                the modulus choose (``'field'`` or ``'pmf'``).
            :param backend: Name how that representation's operations run
                instead of taking its best available.
            """

    @staticmethod
    def _concrete(
        implementation: str | None,
        backend: str | None,
        modulus: int | None = None,
        degree: int = 1,
        *args,
        **kwargs,
    ) -> type:
        bits = modulus.bit_length() if isinstance(modulus, int) else None
        if implementation is None:
            implementation = "field"
            if degree == 1 and isinstance(modulus, int) and modulus.bit_length() > 64:
                # Imported here: impl modules import this one at load time.
                from .impl.pmf.pseudo_mersenne import is_pseudo_mersenne

                if is_pseudo_mersenne(modulus):
                    implementation = "pmf"
        return resolve(implementation, backend, prime_bits=bits).parent_cls

    #: The characteristic: the prime p this field is built over.
    prime: int

    #: The extension degree d. 1 for a prime field. `d` is the short name for
    #: the same value, which is what the coefficient kernels are written in
    #: terms of.
    degree: int
    d: int

    #: 0, 1 and 2 as elements, built once with the field.
    zero: FieldElement
    one: FieldElement
    two: FieldElement

    @property
    def order(self) -> int:
        """The number of elements of the field, ``prime**degree``."""
        raise NotImplementedError

    @property
    def two_adicity(self) -> int:
        """The largest ``k`` with ``2**k`` dividing ``prime - 1``.

        A transform over this field needs a primitive 2n-th root of unity, so
        ``log2(n) + 1`` is at most this. It is 1 for many primes -- among them
        ``2**61 - 1`` -- and such a field admits no transform of length above
        1.
        """
        raise NotImplementedError

    def __call__(self, value: int) -> FieldElement:
        """``value`` reduced into the field: the short way to build an element.

        `FieldElement(field, value)` is the same thing spelled out, and takes
        the forms this does not (a coefficient list, for an extension field).
        """
        raise NotImplementedError

    @property
    def exceptional_set_size(self) -> int:
        """|A| for any field is its order: every nonzero difference of two
        field elements is invertible, so the whole field is exceptional."""
        return self.order

    def _uniform_from_seed(self, seed: bytes) -> FieldElement:
        """One uniform element, a pure function of ``seed``.

        The single sampling primitive an implementation supplies; the public
        samplers below are spelled once here in terms of it.
        """
        raise NotImplementedError

    def random_element(self, seed: bytes | None = None) -> FieldElement:
        """A uniform element of the field.

        Deterministic from ``seed`` when one is given -- the same seed gives
        the same element on every engine -- and expanded from a fresh
        ``vfhe.crypto`` seed otherwise.
        """
        if seed is None:
            seed = entropy.bytes(_SEED_BYTES)
        return self._uniform_from_seed(seed)

    def random_exceptional(self) -> FieldElement:
        """A fresh element of the exceptional set, which for a field is the
        whole field: a uniform element. The name a protocol verifier calls."""
        return self.random_element()

    def exceptional_from_seed(self, seed: bytes) -> FieldElement:
        """The exceptional-set element a Fiat-Shamir transcript derives from
        ``seed``: for a field, a uniform element."""
        return self.random_element(seed)


class FieldElement(metaclass=_ImplementationDispatch):
    """One element of a `Field`, in whatever representation the field uses.

    ``FieldElement(field, value)`` builds the field's own element type: the
    class comes from the field's spec, so an element is always matched to its
    field's implementation and a caller never names the concrete class.

    Elements are immutable in use: every operator returns a fresh element.
    An int on either side of an operator is reduced into the field first, so
    ``element + 3`` and ``3 - element`` both work; anything else returns
    NotImplemented, which is what lets the other operand's reflected method
    take its turn -- ``element * vector`` reaches the vector.

    An implementation must provide the arithmetic primitives and `inverse`,
    `hash` and equality. `square` and division have a default here written
    against those, which an implementation may replace with a cheaper one.

    Prefer a `FieldVector` wherever the same operation applies to many
    elements: one call into C rather than n.
    """

    __slots__ = ()

    if TYPE_CHECKING:
        # Declaration only; `_ImplementationDispatch.__call__` is what runs.
        def __init__(
            self,
            field: Field,
            value: int | Sequence[int] | None = None,
            *,
            implementation: str | None = None,
            backend: str | None = None,
        ) -> None:
            """Build an element of ``field``.

            :param field: The parent. Its spec decides the class built.
            :param value: An int, reduced into the field; a sequence of
                coefficients, low degree first, for an extension field; or
                None for zero. An implementation serving only prime fields
                takes the int form alone.
            :param implementation: Ignored -- the field already fixes the
                implementation. Accepted so that every front takes the same
                keywords.
            :param backend: Ignored, as ``implementation``.
            """

    #: The parent this element belongs to.
    field: Field

    @staticmethod
    def _concrete(
        _implementation: str | None,
        _backend: str | None,
        field: Field | None = None,
        *_args,
        **_kwargs,
    ) -> type:
        if field is None:
            raise TypeError("FieldElement(field) needs the field it belongs to")
        element_cls = field.spec.element_cls
        if element_cls is None:
            raise TypeError(f"{field.spec} has no element type")
        return element_cls

    def __add__(self, other: FieldElement | int) -> FieldElement:
        raise NotImplementedError

    def __radd__(self, other: FieldElement | int) -> FieldElement:
        """``other + self``; addition commutes, so this is `__add__`."""
        return self.__add__(other)

    def __sub__(self, other: FieldElement | int) -> FieldElement:
        raise NotImplementedError

    def __rsub__(self, other: FieldElement | int) -> FieldElement:
        """``other - self``: the operands reversed, not a delegation."""
        raise NotImplementedError

    def __neg__(self) -> FieldElement:
        raise NotImplementedError

    def __mul__(self, other: FieldElement | int) -> FieldElement:
        raise NotImplementedError

    def __rmul__(self, other: FieldElement | int) -> FieldElement:
        """``other * self``; multiplication commutes, so this is `__mul__`."""
        return self.__mul__(other)

    def __pow__(self, exponent: int) -> FieldElement:
        """``self ** exponent``. The exponent is a plain integer, not reduced
        modulo the group order."""
        raise NotImplementedError

    def square(self) -> FieldElement:
        """``self * self``, which an implementation may do cheaper."""
        return self.__mul__(self)

    def inverse(self) -> FieldElement:
        """The multiplicative inverse. Raises ValueError for zero."""
        raise NotImplementedError

    def __truediv__(self, other: FieldElement | int) -> FieldElement:
        """``self / other``: one inversion and one multiplication."""
        rhs = other if isinstance(other, FieldElement) else self.field(other)
        return self.__mul__(rhs.inverse())

    def __rtruediv__(self, other: FieldElement | int) -> FieldElement:
        """``other / self``."""
        lhs = other if isinstance(other, FieldElement) else self.field(other)
        return lhs.__mul__(self.inverse())

    def __int__(self) -> int:
        """The value as a Python integer.

        Defined where the element is a scalar -- a prime field, or an
        extension element whose only nonzero coefficient is the constant one.
        An implementation raises rather than silently dropping coefficients.
        """
        raise NotImplementedError

    def __bool__(self) -> bool:
        """False exactly for zero."""
        return self != self.field.zero

    def hash(self) -> bytes:
        """A 32-byte digest of the value.

        The leaf digest a Merkle tree over field elements takes. Pinned to the
        canonical value, so an element digests the same however it was built;
        host-dependent, so it identifies within one process, not across
        builds.
        """
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


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

    if TYPE_CHECKING:
        # Declaration only; `_ImplementationDispatch.__call__` is what runs.
        def __init__(
            self,
            field: Field,
            values: int | Iterable,
            *,
            implementation: str | None = None,
            backend: str | None = None,
        ) -> None:
            """Build a vector over ``field``.

            :param field: The parent of every element. Its spec decides the
                class built.
            :param values: A length, which builds that many zeros, or a
                sequence of elements and ints to load.
            :param implementation: Ignored -- the field already fixes the
                implementation. Accepted so that every front takes the same
                keywords.
            :param backend: Ignored, as ``implementation``.
            """

    #: The parent every element belongs to.
    field: Field

    # An implementation stores a vector as *planes*: one buffer per
    # coefficient (or limb) of an element, holding that coefficient of every
    # element contiguously, each padded to a length the kernels can run in
    # whole groups. That is what makes an operation a fixed number of calls
    # over length-n runs instead of n calls over one element, and the C side
    # states the contract at the vector declaration in ``arith.h``. The three
    # members below are that storage; they are not private to one
    # implementation, they are the layout every implementation here meets.

    #: The native handle the kernels take (a ``FieldVector`` / ``PMFVector``
    #: struct pointing at this vector's planes). Borrowed: it must not
    #: outlive the vector that owns them.
    _struct: Any
    #: One buffer per coefficient or limb, each `_allocated_n` words long.
    _planes: list[Any]
    #: Padded length of each plane: at least ``len(self)``, rounded up to what
    #: the kernels process in whole groups.
    _allocated_n: int

    @staticmethod
    def _concrete(
        _implementation: str | None,
        _backend: str | None,
        field: Field | None = None,
        *_args,
        **_kwargs,
    ) -> type:
        if field is None:
            raise TypeError("FieldVector(field, values) needs the field")
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
    __hash__: ClassVar[None] = None

    # --- shape and access ---

    def __len__(self) -> int:
        raise NotImplementedError

    def __iter__(self) -> Iterator[FieldElement]:
        raise NotImplementedError

    def __getitem__(self, index: int) -> FieldElement:
        """The element at ``index``, negative counting from the end."""
        raise NotImplementedError

    def __setitem__(self, index: int, value) -> None:
        """Write an element or an int at ``index``, in place."""
        raise NotImplementedError

    def to_list(self) -> list[FieldElement]:
        """The contents as a list of elements, in order.

        One call into C for the whole buffer; the per-element cost that
        remains is building the Python objects.
        """
        raise NotImplementedError

    def copy(self) -> FieldVector:
        """A fresh vector with the same contents."""
        raise NotImplementedError

    # --- arithmetic ---

    def __add__(self, other) -> FieldVector:
        """Elementwise ``self + other``, or broadcast against one element."""
        raise NotImplementedError

    def __radd__(self, other) -> FieldVector:
        raise NotImplementedError

    def __sub__(self, other) -> FieldVector:
        """Elementwise ``self - other``, or broadcast against one element."""
        raise NotImplementedError

    def __rsub__(self, other) -> FieldVector:
        raise NotImplementedError

    def __neg__(self) -> FieldVector:
        raise NotImplementedError

    def __mul__(self, other) -> FieldVector:
        """Elementwise ``self * other``, or broadcast against one element."""
        raise NotImplementedError

    def __rmul__(self, other) -> FieldVector:
        raise NotImplementedError

    def scale(self, value) -> FieldVector:
        """Every element multiplied by one element (or int): the broadcast
        `__mul__` under the name that says there is no second vector."""
        raise NotImplementedError

    def sum(self) -> FieldElement:
        """The sum of every element, as one element."""
        raise NotImplementedError

    def __pow__(self, exponent: int) -> FieldVector:
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

    def inverse(self) -> FieldVector:
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
        inverses: list[Any] = [None] * n
        for i in range(n - 1, 0, -1):
            inverses[i] = running * prefix[i - 1]
            running = running * elements[i]
        inverses[0] = running
        return type(self)(self.field, inverses)

    # --- movement ---

    def split_even_odd(self) -> tuple[FieldVector, FieldVector]:
        """The elements at even and at odd positions, as two half vectors:
        the inverse of `interleave`."""
        raise NotImplementedError

    @staticmethod
    def concat(vectors: list) -> FieldVector:
        """One vector holding every element of `vectors`, in order."""
        if not vectors:
            raise ValueError("concat needs at least one vector")
        field = vectors[0].field
        for vector in vectors:
            if vector.field != field:
                raise ValueError("cannot concatenate vectors over different fields")
        return type(vectors[0])(field, [e for v in vectors for e in v])

    @staticmethod
    def interleave(even, odd) -> FieldVector:
        """The vector with `even` at the even positions and `odd` at the odd
        ones: the inverse of `split_even_odd`. Both must have the same length."""
        if not isinstance(even, FieldVector) or not isinstance(odd, FieldVector):
            raise TypeError("interleave takes two vectors")
        if len(even) != len(odd):
            raise ValueError(f"length mismatch: {len(even)} and {len(odd)}")
        if even.field != odd.field:
            raise ValueError("cannot interleave vectors over different fields")
        merged: list[Any] = [None] * (2 * len(even))
        merged[0::2] = even.to_list()
        merged[1::2] = odd.to_list()
        return type(even)(even.field, merged)

    def query(self, indices) -> FieldVector:
        """The elements at `indices`, gathered into a new vector."""
        return type(self)(self.field, [self[i] for i in indices])

    def fold(self, r) -> FieldVector:
        """``even + r * (odd - even)`` over adjacent pairs, half the length.

        Position ``i`` of the result is ``self[2i] + r * (self[2i+1] -
        self[2i])``: the interpolation that binds the low variable of a
        multilinear table to ``r``, or folds a codeword. ``r`` is one element
        (or an int). Requires an even length.
        """
        even, odd = self.split_even_odd()
        return even + (odd - even).scale(r)

    # --- sampling and digests ---

    def sample_random(self, seed: bytes) -> None:
        """Overwrite every element with a uniform one, in place.

        A pure function of ``seed``: the same seed fills the same vector on
        every engine.
        """
        raise NotImplementedError

    def hash(self) -> bytes:
        """A 32-byte digest of the whole vector's contents."""
        raise NotImplementedError

    def hash_elements(self, group: int = 1, stride: int = 1) -> list[bytes]:
        """One digest per group of elements: the Merkle leaves of a codeword.

        :param group: How many elements go into each digest.
        :param stride: Distance between the elements of one group, so that a
            group can be a column of an interleaved codeword rather than a
            contiguous run.
        """
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)
