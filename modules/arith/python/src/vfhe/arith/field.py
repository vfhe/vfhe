from __future__ import annotations

from vfhe.misc.libvfhe import ffi, lib


class Field:
    def __init__(self, prime: int, w: int, d: int) -> None:
        self.prime = prime
        self.w = w
        self.d = d
        self.proc = lib.field_new_proc(prime)

        # Precompute constants
        self.zero = FieldElement(self, [0] * d)
        self.one = FieldElement(self, [1] + [0] * (d - 1))
        self.two = FieldElement(self, [2] + [0] * (d - 1))
        self.inv_two = self.two.inverse()

    def __del__(self):
        if hasattr(self, "proc") and self.proc:
            # lib.ntt_free_proc is mapped from free_proc, but wait,
            # in arith.h: void ntt_free_proc(NTT_proc proc);
            # in arith.cdef: we don't have ntt_free_proc?
            lib.ntt_free_proc(self.proc)


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
            self.field.proc,
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
            self.field.proc,
        )
        return FieldElement(self.field, res_val)

    def inverse(self) -> FieldElement:
        res_val = ffi.new("uint64_t[]", self.field.d)
        ret = lib.field_ext_inv(
            res_val, self.value, self.field.d, self.field.w, self.field.proc
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
