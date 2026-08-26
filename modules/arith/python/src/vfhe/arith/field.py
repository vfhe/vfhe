# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib

from vfhe.misc.libvfhe import ffi, lib

from .base import ArithParent
from .registry import register
from .spec import Capability, Constraints, Spec


class Field(ArithParent):
    def __init__(self, prime: int, w: int, d: int) -> None:
        self.prime = prime
        self.w = w
        self.d = d
        # A Field is pure modular arithmetic, so the modulus is all it needs.
        self.mod = lib.mod_new(prime)

        # Precompute constants
        self.zero = FieldElement(self, [0] * d)
        self.one = FieldElement(self, [1] + [0] * (d - 1))
        self.two = FieldElement(self, [2] + [0] * (d - 1))
        self.inv_two = self.two.inverse()

    @property
    def exceptional_set_size(self) -> int:
        """|A| for a field: every nonzero difference is invertible, so the
        whole of F_{p^d} is exceptional."""
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
        if not isinstance(other, FieldElement):
            raise TypeError("Can only add FieldElement")
        res_val = ffi.new("uint64_t[]", self.field.d)
        lib.field_ext_add(
            res_val, self.value, other.value, self.field.d, self.field.prime
        )
        return FieldElement(self.field, res_val)

    def __sub__(self, other: FieldElement) -> FieldElement:
        if not isinstance(other, FieldElement):
            raise TypeError("Can only subtract FieldElement")
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
            raise TypeError("Can only multiply FieldElement")
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


#: A degree-d extension of F_p by scalar modular arithmetic. Its elements are
#: scalars, so it carries no quotient-polynomial-ring or tower operations, and
#: its two domains are the same representation.
FIELD_SCALAR = register(
    Spec(
        implementation="field",
        backend="scalar",
        parent_cls=Field,
        element_cls=FieldElement,
        capabilities=(
            Capability.CORE
            | Capability.SAMPLING
            | Capability.EXACT
            | Capability.DOMAINS_COINCIDE
        ),
        constraints=Constraints(max_prime_bits=64),
    )
)

# One spec per class today, so it binds to the class; a class serving
# several backends would set it per instance instead.
Field.spec = FIELD_SCALAR
