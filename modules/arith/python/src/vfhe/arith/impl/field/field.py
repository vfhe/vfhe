# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from vfhe.arith.base import Field, FieldElement
from vfhe.arith.registry import register
from vfhe.arith.spec import Capability, Constraints, Spec
from vfhe.engine import ffi, lib


class ExtensionField(Field):
    """F_(p^d) as F_p[x]/(x^d - w), with scalar 64-bit coefficient kernels."""

    #: 0, 1 and 2 as elements, built once with the field; `inv_two` is the
    #: inverse of `two`, which halving needs on every call.
    zero: ExtensionFieldElement
    one: ExtensionFieldElement
    two: ExtensionFieldElement
    inv_two: ExtensionFieldElement

    def __init__(self, modulus: int, degree: int = 1, w: int | None = None) -> None:
        """
        Build F_(modulus^degree) with defining polynomial ``x^degree - w``.

        ``w`` must make ``x^degree - w`` irreducible over F_modulus, which is
        the caller's responsibility -- it is not verified here. For
        ``degree=1`` no defining polynomial exists and ``w`` is ignored;
        for higher degrees it is required.
        """
        if degree > 1 and w is None:
            raise TypeError(f"degree={degree} needs w, with x^{degree} - w irreducible")
        self.prime = modulus
        self.w = 0 if w is None else w
        self.d = degree
        # A Field is pure modular arithmetic, so the modulus is all it needs.
        self.mod = lib.mod_new(modulus)
        if self.mod == ffi.NULL:
            raise ValueError(
                f"modulus {modulus} is too wide for the native kernels; "
                "see stderr for the bound"
            )

        # Precompute constants
        self.zero = ExtensionFieldElement(self, [0] * degree)
        self.one = ExtensionFieldElement(self, [1] + [0] * (degree - 1))
        self.two = ExtensionFieldElement(self, [2] + [0] * (degree - 1))
        self.inv_two = self.two.inverse()

    def __call__(self, value: int) -> ExtensionFieldElement:
        """``value`` reduced into the field, as its constant coefficient."""
        return ExtensionFieldElement(self, value)

    @property
    def degree(self) -> int:
        """The extension degree. `d` is the same value under the short name
        the coefficient kernels are written in terms of."""
        return self.d

    @property
    def order(self) -> int:
        return self.prime**self.d

    @property
    def two_adicity(self) -> int:
        """
        The largest ``k`` with ``2^k`` dividing ``p - 1``.

        A transform over this field's coefficient planes needs a primitive
        2n-th root of unity in F_p, so ``log2(n) + 1`` at most this. It is 1
        for many primes -- among them ``2^61 - 1`` -- and such a field admits
        no transform of length above 1.
        """
        below = self.prime - 1
        return (below & -below).bit_length() - 1

    def _uniform_from_seed(self, seed: bytes) -> ExtensionFieldElement:
        element = ExtensionFieldElement(self)
        element.sample_random(seed)
        return element

    def element_from_seed(self, seed: bytes, index: int) -> ExtensionFieldElement:
        """Samples the element at position `index` based on `seed` and `index`.
        The result, per element, is the same as calling `FieldVector.sample_random` for the entire vector."""
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError(f"index must be an int, not {type(index).__name__}")
        if index < 0:
            raise ValueError(f"index must not be negative, got {index}")
        value = ffi.new("uint64_t[]", self.d)
        lib.field_vec_sample_random_element(
            value, seed, len(seed), index, self.d, self.prime
        )
        return ExtensionFieldElement(self, value)

    if TYPE_CHECKING:
        # `Field` spells the samplers once against `_uniform_from_seed`;
        # these say what they yield here. Declarations only -- the inherited
        # implementations are what run.
        def random_element(
            self, seed: bytes | None = None
        ) -> ExtensionFieldElement: ...
        def random_exceptional(self) -> ExtensionFieldElement: ...
        def exceptional_from_seed(self, seed: bytes) -> ExtensionFieldElement: ...

    def __del__(self) -> None:
        if getattr(self, "mod", None):
            with contextlib.suppress(Exception):  # lib may be torn down already
                lib.mod_free(self.mod)


class ExtensionFieldElement(FieldElement):
    """One element of an `ExtensionField`: d coefficients over F_p."""

    #: The parent, whose `d`, `w` and native modulus the kernels below read.
    field: ExtensionField
    #: The d coefficients, low degree first, as a native ``uint64_t[d]``.
    value: Any

    def __init__(self, field: ExtensionField, value=None) -> None:
        """
        Build an element of ``field``.

        ``value`` is a list of up to d coefficients, low degree first; an
        int, taken as the constant coefficient; a native ``uint64_t[d]``
        to adopt; or None for zero.
        """
        self.field = field
        if value is None:
            self.value = ffi.new("uint64_t[]", field.d)
        elif isinstance(value, (list, tuple)):
            if not (len(value) <= field.d):
                raise ValueError("len(value) <= field.d")
            self.value = ffi.new("uint64_t[]", field.d)
            for i, val in enumerate(value):
                self.value[i] = val % field.prime
        elif isinstance(value, int):
            self.value = ffi.new("uint64_t[]", field.d)
            self.value[0] = value % field.prime
        else:
            # Assume it's a ffi cdata uint64_t[]
            self.value = value

    def _coerce(self, other: object) -> ExtensionFieldElement | None:
        """
        ``other`` as an element, or None if it cannot be one.

        Accepts an element, or an int (excluding bool) reduced into the field.
        Callers turn None into NotImplemented rather than a TypeError: it is
        what lets the reflected operand take its turn, so `element + vector`
        reaches the vector's __radd__.
        """
        if isinstance(other, ExtensionFieldElement):
            return other
        if isinstance(other, int) and not isinstance(other, bool):
            return ExtensionFieldElement(self.field, other)
        return None

    def __add__(self, other: ExtensionFieldElement | int) -> ExtensionFieldElement:
        rhs = self._coerce(other)
        if rhs is None:
            return NotImplemented
        res_val = ffi.new("uint64_t[]", self.field.d)
        lib.field_ext_add(
            res_val, self.value, rhs.value, self.field.d, self.field.prime
        )
        return ExtensionFieldElement(self.field, res_val)

    def __radd__(self, other: ExtensionFieldElement | int) -> ExtensionFieldElement:
        """``other + self``; addition commutes, so this is `__add__`."""
        return self.__add__(other)

    def __sub__(self, other: ExtensionFieldElement | int) -> ExtensionFieldElement:
        rhs = self._coerce(other)
        if rhs is None:
            return NotImplemented
        res_val = ffi.new("uint64_t[]", self.field.d)
        lib.field_ext_sub(
            res_val, self.value, rhs.value, self.field.d, self.field.prime
        )
        return ExtensionFieldElement(self.field, res_val)

    def __rsub__(self, other: ExtensionFieldElement | int) -> ExtensionFieldElement:
        """``other - self``: the operands reversed, not a delegation to `__sub__`."""
        lhs = self._coerce(other)
        if lhs is None:
            return NotImplemented
        res_val = ffi.new("uint64_t[]", self.field.d)
        lib.field_ext_sub(
            res_val, lhs.value, self.value, self.field.d, self.field.prime
        )
        return ExtensionFieldElement(self.field, res_val)

    def __neg__(self) -> ExtensionFieldElement:
        res_val = ffi.new("uint64_t[]", self.field.d)
        lib.field_ext_neg(res_val, self.value, self.field.d, self.field.prime)
        return ExtensionFieldElement(self.field, res_val)

    def __mul__(self, other: ExtensionFieldElement | int) -> ExtensionFieldElement:
        rhs = self._coerce(other)
        if rhs is None:
            return NotImplemented
        res_val = ffi.new("uint64_t[]", self.field.d)
        lib.field_ext_mul(
            res_val,
            self.value,
            rhs.value,
            self.field.d,
            self.field.w,
            self.field.mod,
        )
        return ExtensionFieldElement(self.field, res_val)

    def __rmul__(self, other: ExtensionFieldElement | int) -> ExtensionFieldElement:
        """``other * self``; multiplication commutes, so this is `__mul__`."""
        return self.__mul__(other)

    def __pow__(self, exponent: int) -> ExtensionFieldElement:
        res_val = ffi.new("uint64_t[]", self.field.d)
        lo = exponent & 0xFFFFFFFFFFFFFFFF
        hi = (exponent >> 64) & 0xFFFFFFFFFFFFFFFF
        lib.field_ext_pow(
            res_val,
            self.value,
            lo,
            hi,
            self.field.d,
            self.field.w,
            self.field.mod,
        )
        return ExtensionFieldElement(self.field, res_val)

    def frobenius(self, k: int = 1) -> ExtensionFieldElement:
        """``self ** (p ** k)``, as the coefficient map it is.

        A permutation of the d coefficient positions with one constant each,
        derived from ``(p, d, w)`` -- not the exponentiation the front
        defines it as. ``k`` counts modulo d, the order of the group.
        """
        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError(f"k must be an int, not {type(k).__name__}")
        if k < 0:
            raise ValueError("k must not be negative; the group is cyclic of order d")
        res_val = ffi.new("uint64_t[]", self.field.d)
        lib.field_ext_frobenius(
            res_val, self.value, k, self.field.d, self.field.w, self.field.mod
        )
        return ExtensionFieldElement(self.field, res_val)

    def inverse(self) -> ExtensionFieldElement:
        res_val = ffi.new("uint64_t[]", self.field.d)
        ret = lib.field_ext_inv(
            res_val, self.value, self.field.d, self.field.w, self.field.mod
        )
        if ret == 0:
            raise ValueError("Element not invertible")
        return ExtensionFieldElement(self.field, res_val)

    def sample_random(self, seed: bytes):
        lib.field_sample_random_element(
            self.value, seed, len(seed), self.field.d, self.field.prime
        )

    def __int__(self) -> int:
        """The constant coefficient, when it is the whole element.

        Raises ValueError for an element of degree above 0, which has no
        integer value: dropping the other coefficients silently would make
        ``int(a) == int(b)`` for elements that differ.
        """
        if any(self.value[i] for i in range(1, self.field.d)):
            raise ValueError(
                f"element of F_(p^{self.field.d}) is not a scalar; no integer value"
            )
        return int(self.value[0])

    def hash(self) -> bytes:
        out = ffi.new("uint8_t[32]")
        lib.field_hash_element(out, self.value, self.field.d)
        return bytes(out)

    def __repr__(self):
        coeffs = [self.value[i] for i in range(self.field.d)]
        return f"ExtensionFieldElement({coeffs})"

    def __eq__(self, other):
        if not isinstance(other, ExtensionFieldElement):
            return False
        return bool(lib.field_ext_is_equal(self.value, other.value, self.field.d))

    def __ne__(self, other):
        return not self.__eq__(other)


# Imported here, at the bottom: the vector module imports ExtensionFieldElement above.
from .vector import ExtensionFieldVector  # noqa: E402

#: A degree-d extension of F_p by scalar modular arithmetic. Its elements are
#: scalars, so it carries no quotient-polynomial-ring or tower operations, and
#: its two domains are the same representation.
FIELD_SCALAR = register(
    Spec(
        implementation="field",
        backend="scalar",
        parent_cls=ExtensionField,
        element_cls=ExtensionFieldElement,
        vector_cls=ExtensionFieldVector,
        capabilities=(
            Capability.CORE
            | Capability.SAMPLING
            | Capability.EXACT
            | Capability.DOMAINS_COINCIDE
        ),
        # 62 bits, not 64: values are lazy in [0, 4q) between butterfly
        # stages, so the widest reduction radix, 2^64, needs 4q <= 2^64.
        constraints=Constraints(max_prime_bits=62),
    )
)

# `spec` binds to the class because each class here serves exactly one; a
# class serving several must set it per instance instead.
ExtensionField.spec = FIELD_SCALAR
