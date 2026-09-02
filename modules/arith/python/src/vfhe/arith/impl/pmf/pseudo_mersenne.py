# SPDX-FileCopyrightText: 2026 Daniele Cozzo <daniele.cozzo@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""
Arithmetic over a pseudo-Mersenne prime p = 2^n - c (the Crandall family).

Representation contract, mirrored by the C side and relied upon by the tuned
AVX-512 IFMA kernels:

- An element is ``_LANES`` uint64 words -- one zmm register -- holding
  ``L = ceil(n / 52)`` limbs of 52 bits in lanes ``0..L-1``, little-endian.
  Lanes ``L.._LANES-1`` are ALWAYS zero.
- Buffers are 64-byte aligned, so the tuned engine can load an element as a
  single ``__m512i``.
- Every operation returns a CANONICAL result: value in ``[0, p)``, every limb
  below 2^52, padding lanes zero. Lazy reduction is deliberately not offered --
  IFMA truncates both multiplicands to 52 bits, so carrying back below 2^52
  after a multiply is a hardware requirement, not a policy choice.
- Reduction folds on ``2^(52L) == e (mod p)`` with ``e = c << (52L - n)``. That
  ``e`` must fit one limb is what bounds ``c``, and hence which ``n`` are usable.

None of this is constant-time: exponents and seeds are public values.
"""

from __future__ import annotations

from vfhe.arith._alloc import aligned64
from vfhe.arith.base import Field
from vfhe.arith.number_theory import gen_pseudo_mersenne_prime, is_prime
from vfhe.arith.registry import register
from vfhe.arith.spec import Capability, Constraints, Spec
from vfhe.engine import ffi, lib

_LIMB_BITS = 52
_LIMB_MASK = (1 << _LIMB_BITS) - 1
_LANES = 8
_SUPPORTED_LIMBS = (5, 6)


def _layout(bits: int) -> tuple[int, int]:
    """
    Return ``(limbs, shift)`` for an ``n``-bit pseudo-Mersenne prime.

    ``limbs = ceil(bits / 52)`` and ``shift = 52 * limbs - bits``. Raises
    ValueError when no kernel covers that limb count.
    """
    limbs = -(-bits // _LIMB_BITS)
    if limbs not in _SUPPORTED_LIMBS:
        windows = ", ".join(
            f"{ell} limbs: n in {_LIMB_BITS * (ell - 1) + 1}..{_LIMB_BITS * ell}"
            for ell in _SUPPORTED_LIMBS
        )
        raise ValueError(f"n={bits} needs {limbs} limbs; only {windows} have kernels")
    return limbs, _LIMB_BITS * limbs - bits


def is_pseudo_mersenne(modulus: int) -> bool:
    """Whether ``modulus`` is a pseudo-Mersenne the kernels here cover.

    True when ``modulus = 2^n - c`` needs a supported limb count and the fold
    constant ``c << shift`` fits one 52-bit limb -- the structural test
    `Field` applies before routing a modulus to this implementation.
    Primality is not checked; the constructor does that.
    """
    bits = modulus.bit_length()
    limbs = -(-bits // _LIMB_BITS)
    if limbs not in _SUPPORTED_LIMBS:
        return False
    shift = _LIMB_BITS * limbs - bits
    c = (1 << bits) - modulus
    return (c << shift) >> _LIMB_BITS == 0


def _pack_into(buf, value: int, limbs: int) -> None:
    """Write ``value`` into ``buf`` as ``limbs`` 52-bit limbs, zeroing the rest."""
    for i in range(limbs):
        buf[i] = (value >> (i * _LIMB_BITS)) & _LIMB_MASK
    for i in range(limbs, _LANES):
        buf[i] = 0


def _unpack(buf, limbs: int) -> int:
    """Recombine ``limbs`` 52-bit limbs of ``buf`` into an integer."""
    return sum(int(buf[i]) << (i * _LIMB_BITS) for i in range(limbs))


class PseudoMersenneField(Field):
    """F_p for a pseudo-Mersenne prime, with the kernels in C."""

    def __init__(self, modulus: int, degree: int = 1) -> None:
        """
        Build the field for an explicit prime.

        ``degree`` exists for interface uniformity with `Field`; this
        implementation serves only prime fields, so any value but 1 is
        rejected. Rejects a prime whose bit-length has no kernel, and one
        whose fold constant ``e = c << shift`` does not fit a 52-bit limb.
        Primality is checked with ``number_theory.is_prime``, a fixed-base
        probable-prime test -- adequate for primes from our own generator,
        not a defence against a hostile ``p``.
        """
        if degree != 1:
            raise ValueError(
                f"pmf implements only prime fields; degree={degree} is not supported"
            )
        if not isinstance(modulus, int) or isinstance(modulus, bool):
            raise TypeError(f"modulus must be an int, not {type(modulus).__name__}")
        if modulus <= 2:
            raise ValueError(f"modulus={modulus} must exceed 2")

        self.bits = modulus.bit_length()
        self.limbs, self.shift = _layout(self.bits)
        self.prime = modulus
        self.c = (1 << self.bits) - modulus
        self.lanes = _LANES

        # The bound that actually matters is on the fold constant, not on c.
        c_bits = self.bits - _LIMB_BITS * (self.limbs - 1)
        self.fold = self.c << self.shift
        if self.fold >> _LIMB_BITS:
            raise ValueError(
                f"c={self.c} exceeds {c_bits} bits, so the fold constant "
                f"c << {self.shift} = {self.fold} would not fit a 52-bit limb; "
                f"for n={self.bits} the reduction needs c < 2**{c_bits}"
            )
        if not is_prime(modulus):
            raise ValueError(f"modulus={modulus} is not prime")

        params = lib.pmf_new_params(self.bits, self.c)
        if params == ffi.NULL:
            raise ValueError(
                f"C rejected n={self.bits}, c={self.c}; see stderr for the reason"
            )
        self._params = ffi.gc(params, lib.pmf_free_params)
        if lib.pmf_limbs(self._params) != self.limbs:
            raise RuntimeError("C and Python disagree on the limb count")

        self.zero = self(0)
        self.one = self(1)

    @property
    def order(self) -> int:
        return self.prime

    @classmethod
    def generate(
        cls,
        bits: int,
        two_adicity: int = 0,
        max_two_adicity: int | None = None,
    ) -> PseudoMersenneField:
        """
        Search for a suitable prime of the given bit-length and build the field.

        Delegates to ``number_theory.gen_pseudo_mersenne_prime``, bounding ``c``
        by ``52 - shift`` bits rather than the default 52 -- the real constraint
        is on the fold constant ``c << shift``, not on ``c`` alone.

        The achievable 2-adicity falls off sharply as ``bits`` moves below
        ``52 * limbs``, because the budget for ``c`` is only
        ``bits - 52 * (limbs - 1)`` bits wide. ``bits`` of 260 (5 limbs) and 312
        (6 limbs) are the roomiest choices.
        """
        _, shift = _layout(bits)
        prime = gen_pseudo_mersenne_prime(
            bits,
            two_adicity,
            max_two_adicity=max_two_adicity,
            max_c_bits=_LIMB_BITS - shift,
        )
        return cls(prime)

    def __call__(self, value: int) -> PseudoMersenneElement:
        """Reduce an integer into the field."""
        return PseudoMersenneElement(self, value)

    @property
    def byte_length(self) -> int:
        """Bytes in the canonical encoding of an element."""
        return (self.bits + 7) // 8

    def random(self, seed: bytes) -> PseudoMersenneElement:
        """Sample a uniform element deterministically from ``seed``."""
        buf = self._new_buffer()
        lib.pmf_sample_random(buf, seed, len(seed), self._params)
        return self._wrap(buf)

    def from_bytes(self, data: bytes) -> PseudoMersenneElement:
        """
        Decode a canonical big-endian encoding of exactly ``byte_length`` bytes.

        Rejects anything else: a wrong length, or a value at or above ``p``,
        which would give one field element two encodings.
        """
        if len(data) != self.byte_length:
            raise ValueError(
                f"encoding must be {self.byte_length} bytes, got {len(data)}"
            )
        buf = self._new_buffer()
        if lib.pmf_from_bytes(buf, data, self._params) == 0:
            raise ValueError("encoding is not a canonical element of this field")
        return self._wrap(buf)

    def root_of_unity(self, log_order: int) -> PseudoMersenneElement:
        """
        A primitive ``2^log_order``-th root of unity, memoized.

        Derived from the least quadratic non-residue, so it needs no
        factorization of ``p - 1``. Note that a generator of the full group
        F_p^* is NOT offered: verifying one requires factoring ``p - 1``, which
        is not feasible at these sizes, and an NTT needs only the 2-primary
        subgroup. Raises ValueError for ``log_order`` above ``two_adicity``.
        """
        raise NotImplementedError

    def _wrap(self, buf) -> PseudoMersenneElement:
        """Adopt a C output buffer as an element, without copying."""
        element = object.__new__(PseudoMersenneElement)
        element.field = self
        element._buf = buf
        return element

    def _new_buffer(self):
        """Allocate one zeroed, 64-byte-aligned element buffer."""
        return aligned64("uint64_t[]", _LANES)

    def __eq__(self, other: object) -> bool:
        """
        Two fields are equal when their primes are.

        Compared by prime value rather than object identity, so a field built
        twice in two modules still accepts its own elements on both sides.
        """
        if not isinstance(other, PseudoMersenneField):
            return NotImplemented
        return self.prime == other.prime

    def __hash__(self) -> int:
        """Hash of the prime, so equal fields hash equally."""
        return hash(self.prime)

    def __repr__(self) -> str:
        """Name the shape, e.g. ``PseudoMersenneField(2^260 - 22527, 5 limbs)``."""
        return f"PseudoMersenneField(2^{self.bits} - {self.c}, {self.limbs} limbs)"


class PseudoMersenneElement:
    """
    An element of a ``PseudoMersenneField``.

    Immutable by construction -- there are no in-place operators and no setters
    -- which is what makes ``__hash__`` sound.
    """

    __slots__ = ("_buf", "field")

    def __init__(self, field: PseudoMersenneField, value: int = 0) -> None:
        """Reduce ``value`` into ``field`` and pack it into a fresh buffer."""
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"value must be an int, not {type(value).__name__}")
        self.field = field
        self._buf = field._new_buffer()
        # Python's % already returns a non-negative representative for negatives.
        _pack_into(self._buf, value % field.prime, field.limbs)

    def _coerce(self, other: object) -> PseudoMersenneElement | None:
        """
        Promote ``other`` to an element of the same field, or None if impossible.

        Accepts an element over an equal prime, or an int (excluding bool, so
        that ``True + a`` is a TypeError rather than quietly meaning ``1 + a``).
        Callers turn None into NotImplemented so Python raises, and so the
        reflected operand gets its turn.
        """
        if isinstance(other, PseudoMersenneElement):
            return other if other.field.prime == self.field.prime else None
        if isinstance(other, int) and not isinstance(other, bool):
            return PseudoMersenneElement(self.field, other)
        return None

    def _binary(self, other: object, kernel):
        """Coerce, allocate an output buffer, and make the single C call."""
        rhs = self._coerce(other)
        if rhs is None:
            return NotImplemented
        out = self.field._new_buffer()
        kernel(out, self._buf, rhs._buf, self.field._params)
        return self.field._wrap(out)

    def __add__(self, other: PseudoMersenneElement | int) -> PseudoMersenneElement:
        """``self + other``. Returns NotImplemented if ``other`` will not coerce."""
        return self._binary(other, lib.pmf_add)

    def __radd__(self, other: PseudoMersenneElement | int) -> PseudoMersenneElement:
        """
        ``other + self``, the reflected form, which is what makes ``1 + a`` work.

        Python reaches here only after ``type(other).__add__`` has declined.
        Addition commutes, so this just delegates to ``__add__``.
        """
        return self.__add__(other)

    def __sub__(self, other: PseudoMersenneElement | int) -> PseudoMersenneElement:
        """``self - other``."""
        return self._binary(other, lib.pmf_sub)

    def __rsub__(self, other: PseudoMersenneElement | int) -> PseudoMersenneElement:
        """
        ``other - self``, the reflected form, which is what makes ``1 - a`` work.

        NOT symmetric: subtraction does not commute, so this must compute
        ``other - self`` and cannot delegate to ``__sub__``. Written out rather
        than routed through ``_binary`` precisely so the operand order is visible
        at the call site -- getting it backwards is the classic bug here.
        """
        lhs = self._coerce(other)
        if lhs is None:
            return NotImplemented
        out = self.field._new_buffer()
        lib.pmf_sub(out, lhs._buf, self._buf, self.field._params)
        return self.field._wrap(out)

    def __neg__(self) -> PseudoMersenneElement:
        """``-self``, i.e. ``p - self``, and zero for zero."""
        out = self.field._new_buffer()
        lib.pmf_neg(out, self._buf, self.field._params)
        return self.field._wrap(out)

    def __mul__(self, other: PseudoMersenneElement | int) -> PseudoMersenneElement:
        """``self * other``, the merged multiply-and-reduce."""
        return self._binary(other, lib.pmf_mul)

    def __rmul__(self, other: PseudoMersenneElement | int) -> PseudoMersenneElement:
        """
        ``other * self``, the reflected form, which is what makes ``2 * a`` work.

        Multiplication commutes, so this just delegates to ``__mul__``.
        """
        return self.__mul__(other)

    def square(self) -> PseudoMersenneElement:
        """Equivalent to ``self * self``, via a kernel free to specialize."""
        out = self.field._new_buffer()
        lib.pmf_mul(out, self._buf, self._buf, self.field._params)
        return self.field._wrap(out)

    def __pow__(self, exponent: int) -> PseudoMersenneElement:
        """
        Square-and-multiply. The exponent is not reduced mod ``p - 1``.

        A negative exponent is Python-level sugar: it inverts first, so it costs
        two C calls rather than one.
        """
        if not isinstance(exponent, int) or isinstance(exponent, bool):
            raise TypeError(f"exponent must be an int, not {type(exponent).__name__}")
        base = self.inverse() if exponent < 0 else self
        exponent = abs(exponent)
        result = self.field.one
        while exponent:
            if exponent & 1:
                result = result * base
            exponent >>= 1
            if exponent:
                base = base.square()
        return result

    def inverse(self) -> PseudoMersenneElement:
        """
        The multiplicative inverse, by Fermat ``a^(p-2)``.

        Raises ZeroDivisionError for zero.
        """
        if not self:
            raise ZeroDivisionError("zero has no inverse")
        return self ** (self.field.prime - 2)

    def __truediv__(self, other: PseudoMersenneElement | int) -> PseudoMersenneElement:
        """Sugar for ``self * other.inverse()``; two C calls rather than one."""
        divisor = self._coerce(other)
        if divisor is None:
            return NotImplemented
        return self * divisor.inverse()

    def __rtruediv__(self, other: PseudoMersenneElement | int) -> PseudoMersenneElement:
        """
        ``other / self``, the reflected form, which is what makes ``3 / a`` work.

        NOT symmetric: this is ``other * self.inverse()``, so the inverse is
        taken of ``self``, not of ``other``.
        """
        dividend = self._coerce(other)
        if dividend is None:
            return NotImplemented
        return dividend * self.inverse()

    def __eq__(self, other: object) -> bool:
        """
        Compare by field prime and value.

        An int is never equal to an element: coercion is unambiguous in
        arithmetic, where the modulus comes from the element, but not in
        equality, where an int is not a residue class.
        """
        if not isinstance(other, PseudoMersenneElement):
            return NotImplemented
        if other.field.prime != self.field.prime:
            return False
        return bool(lib.pmf_is_equal(self._buf, other._buf, self.field._params))

    def __hash__(self) -> int:
        """
        Hash of ``(field prime, value)``, consistent with ``__eq__``.

        Sound only because elements are immutable and always canonical: one
        value has exactly one representation, so equal elements hash equally.
        """
        return hash((self.field.prime, int(self)))

    def __bool__(self) -> bool:
        """False for zero, True otherwise, so ``if a:`` tests non-zero."""
        return any(self._buf[i] for i in range(self.field.limbs))

    def __int__(self) -> int:
        """The canonical representative in ``[0, p)``, unpacked from the limbs."""
        return _unpack(self._buf, self.field.limbs)

    def to_bytes(self) -> bytes:
        """Canonical fixed-width big-endian encoding, ``byte_length`` bytes."""
        out = ffi.new("uint8_t[]", self.field.byte_length)
        lib.pmf_to_bytes(out, self._buf, self.field._params)
        return bytes(ffi.buffer(out))

    def to_limbs(self) -> list[int]:
        """All ``_LANES`` words, padding included, for tests that pin the layout."""
        return [int(self._buf[i]) for i in range(_LANES)]

    def digest(self) -> bytes:
        """
        BLAKE3 over the canonical encoding, 32 bytes.

        Hashing the encoding rather than the raw limbs keeps the digest
        independent of the representation and of the padding lanes, so the two
        engines cannot disagree.
        """
        out = ffi.new("uint8_t[32]")
        lib.pmf_hash(out, self._buf, self.field._params)
        return bytes(out)

    def __repr__(self) -> str:
        """Unambiguous form: the hex value plus the modulus it lives in."""
        return (
            f"PseudoMersenneElement({int(self):#x}, "
            f"field=2^{self.field.bits} - {self.field.c})"
        )

    def __str__(self) -> str:
        """The decimal value alone, for printing."""
        return str(int(self))


# Imported here, at the bottom: the vector module imports this one above.
from .vector import PseudoMersenneVector  # noqa: E402

#: F_p for p = 2^n - c, one element per vector register, scalar kernels in C.
#: Its elements are field scalars, so there are no quotient-polynomial-ring or
#: tower operations, and its canonical form is its only representation.
PMF_LIMB = register(
    Spec(
        implementation="pmf",
        backend="limb52",
        parent_cls=PseudoMersenneField,
        element_cls=PseudoMersenneElement,
        vector_cls=PseudoMersenneVector,
        capabilities=(Capability.CORE | Capability.EXACT | Capability.DOMAINS_COINCIDE),
        constraints=Constraints(max_prime_bits=_LIMB_BITS * max(_SUPPORTED_LIMBS)),
    )
)

# `spec` binds to the class because each class here serves exactly one; a
# class serving several must set it per instance instead.
PseudoMersenneField.spec = PMF_LIMB
