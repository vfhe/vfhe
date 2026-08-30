# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib

from vfhe.misc.libvfhe import ffi, lib

from ...base import Field
from ...registry import register
from ...spec import Capability, Constraints, Spec


class ExtensionField(Field):
    """F_(p^d) as F_p[x]/(x^d - w), with scalar 64-bit coefficient kernels."""

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

        # Precompute constants
        self.zero = FieldElement(self, [0] * degree)
        self.one = FieldElement(self, [1] + [0] * (degree - 1))
        self.two = FieldElement(self, [2] + [0] * (degree - 1))
        self.inv_two = self.two.inverse()

    @property
    def order(self) -> int:
        return self.prime**self.d

    def __del__(self) -> None:
        if getattr(self, "mod", None):
            with contextlib.suppress(Exception):  # lib may be torn down already
                lib.mod_free(self.mod)


class FieldElement:
    def __init__(self, field: Field, value=None):
        self.field = field
        if value is None:
            self.value = ffi.new("uint64_t[]", field.d)
        elif isinstance(value, (list, tuple)):
            assert len(value) <= field.d
            self.value = ffi.new("uint64_t[]", field.d)
            for i, val in enumerate(value):
                self.value[i] = val % field.prime
        elif isinstance(value, int):
            self.value = ffi.new("uint64_t[]", field.d)
            self.value[0] = value % field.prime
        else:
            # Assume it's a ffi cdata uint64_t[]
            self.value = value

    def __add__(self, other: FieldElement) -> FieldElement:
        # NotImplemented, not TypeError: it is what lets the reflected operand
        # take its turn, so `element + vector` reaches the vector's __radd__.
        if not isinstance(other, FieldElement):
            return NotImplemented
        res_val = ffi.new("uint64_t[]", self.field.d)
        lib.field_ext_add(
            res_val, self.value, other.value, self.field.d, self.field.prime
        )
        return FieldElement(self.field, res_val)

    def __sub__(self, other: FieldElement) -> FieldElement:
        if not isinstance(other, FieldElement):
            return NotImplemented
        res_val = ffi.new("uint64_t[]", self.field.d)
        lib.field_ext_sub(
            res_val, self.value, other.value, self.field.d, self.field.prime
        )
        return FieldElement(self.field, res_val)

    def __neg__(self) -> FieldElement:
        res_val = ffi.new("uint64_t[]", self.field.d)
        lib.field_ext_neg(res_val, self.value, self.field.d, self.field.prime)
        return FieldElement(self.field, res_val)

    def __mul__(self, other: FieldElement) -> FieldElement:
        if not isinstance(other, FieldElement):
            return NotImplemented
        res_val = ffi.new("uint64_t[]", self.field.d)
        lib.field_ext_mul(
            res_val,
            self.value,
            other.value,
            self.field.d,
            self.field.w,
            self.field.mod,
        )
        return FieldElement(self.field, res_val)

    def __pow__(self, exponent: int) -> FieldElement:
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
        return FieldElement(self.field, res_val)

    def inverse(self) -> FieldElement:
        res_val = ffi.new("uint64_t[]", self.field.d)
        ret = lib.field_ext_inv(
            res_val, self.value, self.field.d, self.field.w, self.field.mod
        )
        if ret == 0:
            raise ValueError("Element not invertible")
        return FieldElement(self.field, res_val)

    def sample_random(self, seed: bytes):
        lib.field_sample_random_element(
            self.value, seed, len(seed), self.field.d, self.field.prime
        )

    def hash(self) -> bytes:
        out = ffi.new("uint8_t[32]")
        lib.field_hash_element(out, self.value, self.field.d)
        return bytes(out)

    def __repr__(self):
        coeffs = [self.value[i] for i in range(self.field.d)]
        return f"FieldElement({coeffs})"

    def __eq__(self, other):
        if not isinstance(other, FieldElement):
            return False
        return bool(lib.field_ext_is_equal(self.value, other.value, self.field.d))

    def __ne__(self, other):
        return not self.__eq__(other)


# Imported here, at the bottom: the vector module imports FieldElement above.
from .vector import ExtensionFieldVector  # noqa: E402

#: A degree-d extension of F_p by scalar modular arithmetic. Its elements are
#: scalars, so it carries no quotient-polynomial-ring or tower operations, and
#: its two domains are the same representation.
FIELD_SCALAR = register(
    Spec(
        implementation="field",
        backend="scalar",
        parent_cls=ExtensionField,
        element_cls=FieldElement,
        vector_cls=ExtensionFieldVector,
        capabilities=(
            Capability.CORE
            | Capability.SAMPLING
            | Capability.EXACT
            | Capability.DOMAINS_COINCIDE
        ),
        constraints=Constraints(max_prime_bits=64),
    )
)

# `spec` binds to the class because each class here serves exactly one; a
# class serving several must set it per instance instead.
ExtensionField.spec = FIELD_SCALAR
