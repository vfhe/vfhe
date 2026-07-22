from __future__ import annotations

import math
from enum import Enum

from vfhe.misc.libvfhe import ffi, lib

from .ntt import NTT_processor_instance
from .number_theory import crt, is_prime


def next_power_of_2(x):
    return 1 << int(math.ceil(math.log2(x)))


class Ring:
    def __init__(
        self,
        N,
        mod_size=None,
        split_degree=None,
        primes=None,
        mask=None,
        prime_size: "int | list[int]" = 49,
        exceptional_set_size=128,
    ) -> None:
        assert (
            (mod_size is not None)
            or (type(prime_size) is list)
            or (primes is not None)
            or (mask is not None)
        ), "must provide mod_size, prime_size, primes, or mask"

        if mask is not None:
            assert primes is not None, "must provide primes when mask is given"
            temp_split_degree = split_degree
            if not temp_split_degree:
                smallest_in_pool = min(math.ceil(math.log2(p)) for p in primes)
                temp_split_degree = next_power_of_2(
                    exceptional_set_size / smallest_in_pool
                )

            key = (N, temp_split_degree)
            prime_map = NTT_processor_instance.prime_to_index[key]
            active_primes = [p for p in primes if ((mask >> prime_map[p]) & 1)]
            prime_size = [math.ceil(math.log2(p)) for p in active_primes]
            primes = active_primes

        if primes is not None and prime_size is None:
            prime_size = [math.ceil(math.log2(i)) for i in primes]

        if isinstance(prime_size, list):
            self.ell = len(prime_size)
            self.prime_size = prime_size
            self.smallest_prime = min(prime_size)
        else:
            self.ell = math.ceil(mod_size / prime_size)  # type: ignore
            self.prime_size = [prime_size] * self.ell
            self.smallest_prime = prime_size

        if not split_degree:
            split_degree = next_power_of_2(exceptional_set_size / self.smallest_prime)  # type: ignore
        self.split_degree = split_degree
        self.N = N
        self.byte_size = self.N * self.ell * 8
        self.lib = lib

        if primes:
            assert len(primes) >= self.ell, (
                "not enough primes for quotient ring for size 2^%d" % mod_size
            )
            self.primes = primes[: self.ell]
        else:
            self.primes = self.gen_primes()

        self.q_l = math.prod(self.primes)

        self.bit_size = math.ceil(math.log2(self.q_l))

        self.prime_indices = NTT_processor_instance.register_ring_primes(
            self.primes, self.N, self.split_degree
        )
        self.NTT = NTT_processor_instance.incNTTs[(self.N, self.split_degree)]
        self.mask = sum(1 << idx for idx in self.prime_indices)

    def _ntt_l(self):
        return ffi.cast("incNTT", self.NTT).l

    def get_rou_matrix(self):
        w = self.lib.incNTT_get_rou_matrix(self.NTT)
        row_len = self.N // self.split_degree
        rou_matrix = []
        for idx in self.prime_indices:
            rou_matrix.append([w[idx][k] for k in range(row_len)])
        return rou_matrix

    def quotient_ring(self, mod_size=None, ell=None, mask=None):
        assert (mod_size is not None) ^ (ell is not None) ^ (mask is not None), (
            "must provide mod_size, ell, or mask"
        )
        if mod_size is not None:
            res = Ring(self.N, mod_size, self.split_degree, primes=self.primes)
        elif ell is not None:
            res = Ring(
                self.N,
                prime_size=self.prime_size[:ell],
                split_degree=self.split_degree,
                primes=self.primes,
            )
        else:
            res = Ring(
                self.N, mask=mask, split_degree=self.split_degree, primes=self.primes
            )
        return res

    def is_quotient_ring(self, parent: Ring):
        return parent.mask & self.mask == self.mask

    def modulus_ratio(self, other_ring: Ring, return_pointer: bool = False):
        assert other_ring.is_quotient_ring(self), (
            "other_ring must be a quotient ring of this ring"
        )

        scaling_mask = self.mask & ~other_ring.mask
        delta_big_int = math.prod(
            [
                self.primes[i]
                for i, idx in enumerate(self.prime_indices)
                if (scaling_mask >> idx) & 1
            ]
        )
        if return_pointer:
            ntt_len = self._ntt_l()
            delta_arr = [0] * ntt_len
            for k, idx in enumerate(self.prime_indices):
                delta_arr[idx] = delta_big_int % self.primes[k]
            return ffi.new("uint64_t[]", delta_arr)
        return delta_big_int

    def intersec(self, other: Ring):
        if self == other:
            return self
        if self.ell > other.ell:
            assert other.is_quotient_ring(self)
            return other
        assert self.is_quotient_ring(other)
        return self

    # generates the RNS primes of size prime_size
    @staticmethod
    def gen_prime(rou_order, prime_size, exclude_list=[]):
        a = ((2**prime_size - 1) // rou_order) | 1
        a -= 2
        while True:
            candidate = a * rou_order + 1
            if candidate not in exclude_list and is_prime(candidate):
                return candidate
            a -= 2

    def gen_primes(self):
        primes = []
        for p_size in self.prime_size:
            primes.append(
                Ring.gen_prime(
                    2 * self.N // self.split_degree, p_size, exclude_list=primes
                )
            )
        return primes

    # in this file used to test against the definition of Ring, used in other files when efficiency is not required

    def alloc_polynomial(self):
        return self.lib.polynomial_new_RNS_polynomial(self.N, self.mask, self.NTT)

    def scalar_array(self, value):
        ntt_len = self._ntt_l()
        scale_arr = [0] * ntt_len
        if isinstance(value, int):
            for k, idx in enumerate(self.prime_indices):
                scale_arr[idx] = value % self.primes[k]
        else:
            vals = list(value)
            assert len(vals) == self.ell, f"expected {self.ell} values, got {len(vals)}"
            for k, idx in enumerate(self.prime_indices):
                scale_arr[idx] = vals[k]
        return ffi.new("uint64_t[]", scale_arr)

    def random_element(self, ntt=True):
        return Polynomial(self).sample_uniform(ntt)

    def random_gaussian_element(self, sigma, ntt=True):
        return Polynomial(self).sample_gaussian(sigma, ntt)

    def random_exceptional(self, size="minimal", ntt=True):
        return Polynomial(self).sample_exceptional(size, ntt)


repr = Enum("Polynomial Representation", ["empty", "ntt", "coeff"])


class Polynomial:
    def __init__(self, ring: Ring, repr=repr.empty) -> None:
        self.ring = ring
        self.obj = ring.alloc_polynomial()
        self.repr = repr

    @property
    def rns_mask(self):
        return ffi.cast("RNS_Polynomial", self.obj).rns_mask

    def from_array(self, array: list):
        array = list(array) + ([0] * (self.ring.N - len(array)))
        # int_array_to_RNS reads the buffer as int64_t; wrap signed values into
        # their two's-complement uint64 form.
        array = [x & 0xFFFFFFFFFFFFFFFF for x in array]
        self.ring.lib.int_array_to_RNS(self.obj, ffi.new("uint64_t[]", array))
        self.repr = repr.ntt
        return self

    def from_bigint_array(self, array: list):
        ntt_len = self.ring._ntt_l()
        rows = [ffi.NULL] * ntt_len
        keep = []  # keep row buffers alive across the (copying) C call
        for k, idx in enumerate(self.ring.prime_indices):
            p = self.ring.primes[k]
            row = ffi.new("uint64_t[]", [v % p for v in array])
            keep.append(row)
            rows[idx] = row
        matrix = ffi.new("uint64_t*[]", rows)
        self.ring.lib.array_to_RNS(self.obj, matrix)
        self.repr = repr.ntt
        return self

    def fast_inverse(self):
        assert self.repr == repr.ntt, "fast_inverse requires NTT representation"
        out = Polynomial(self.ring)
        rc = self.ring.lib.polynomial_RNS_inverse(out.obj, self.obj)
        if rc == -1:
            raise AssertionError("polynomial_RNS_inverse: invalid ring parameters")
        if rc == -2:
            raise ValueError("polynomial_RNS_inverse: zero slot is not invertible")
        if rc != 0:
            raise RuntimeError(f"polynomial_RNS_inverse failed with code {rc}")
        out.repr = repr.ntt
        return out

    def __del__(self) -> None:
        try:
            self.ring.lib.free_RNS_polynomial(self.obj)
        except Exception:
            pass  # interpreter shutdown

    # def from_int(self, i:int) -> Polynomial:
    def multiply(self, in1, in2):
        assert in1.repr == in2.repr == repr.ntt
        self.ring.lib.polynomial_mul_RNS_polynomial(self.obj, in1.obj, in2.obj)
        self.repr = in1.repr

    def negate(self, in1=None):
        if not in1:
            in1 = self
        self.ring.lib.polynomial_RNSc_negate(self.obj, in1.obj)
        self.repr = in1.repr

    def sub(self, in1, in2):
        assert in1.repr == in2.repr
        if in1.repr == repr.ntt:
            self.ring.lib.polynomial_sub_RNS_polynomial(self.obj, in1.obj, in2.obj)
        else:
            self.ring.lib.polynomial_sub_RNSc_polynomial(self.obj, in1.obj, in2.obj)
        self.repr = in1.repr

    def add(self, in1, in2):
        assert in1.repr == in2.repr
        if in1.repr == repr.ntt:
            self.ring.lib.polynomial_add_RNS_polynomial(self.obj, in1.obj, in2.obj)
        else:
            self.ring.lib.polynomial_add_RNSc_polynomial(self.obj, in1.obj, in2.obj)
        self.repr = in1.repr

    def automorphism(self, gen):
        assert gen < self.ring.N * 2
        res = Polynomial(self.ring)
        self.to_coeff()
        self.ring.lib.polynomial_RNSc_permute(res.obj, self.obj, gen)
        res.repr = repr.coeff
        return res

    def to_NTT(self):
        if self.repr == repr.ntt:
            return
        self.ring.lib.polynomial_RNSc_to_RNS(self.obj, self.obj)
        self.repr = repr.ntt

    def to_coeff(self):
        if self.repr == repr.coeff:
            return
        self.ring.lib.polynomial_RNS_to_RNSc(self.obj, self.obj)
        self.repr = repr.coeff

    def to_repr(self, repr):
        if repr == repr.ntt:
            self.to_NTT()
        if repr == repr.coeff:
            self.to_coeff()

    def sample_uniform(self, ntt=True):
        self.ring.lib.polynomial_gen_random_RNSc_polynomial(self.obj)
        self.repr = repr.ntt if ntt else repr.coeff
        return self

    def sample_gaussian(self, sigma, ntt=True):
        self.ring.lib.polynomial_gen_gaussian_RNSc_polynomial(self.obj, sigma)
        self.repr = repr.coeff
        if ntt:
            self.to_NTT()
        return self

    def sample_exceptional(self, size="minimal", ntt=True):
        self.ring.lib.polynomial_gen_random_RNSc_polynomial(self.obj)
        self.ring.lib.polynomial_RNS_broadcast_slot(self.obj, self.obj, 0)
        self.repr = repr.ntt
        if not ntt:
            self.to_coeff()
        return self

    def base_extend(self, ring: Ring | None = None, out: Polynomial | None = None):
        if out is None and ring is not None:
            out_ = Polynomial(ring)
        elif type(out) is Polynomial:
            out_: Polynomial = out
        assert self.ring.is_quotient_ring(out_.ring), "Not a quotient ring"
        return self.lift_to(out=out_)

    def lift_to(
        self, ring: Ring | None = None, out: Polynomial | None = None, params=None
    ):
        self.to_coeff()
        if out is None and ring is not None:
            out_ = Polynomial(ring)
        elif type(out) is Polynomial:
            out_: Polynomial = out
        if params is None:
            params = NTT_processor_instance.get_conversion_params(
                self.ring.N, self.ring.split_degree, self.rns_mask, out_.rns_mask
            )
        self.ring.lib.polynomial_base_conversion_RNSc(out_.obj, self.obj, params)
        out_.repr = repr.coeff
        return out_

    def mod_reduce(
        self, ring: Ring | None = None, out: Polynomial | None = None
    ) -> Polynomial:
        self.to_coeff()
        assert out is not None or ring is not None, "Must provide ring or out"
        if out is None:
            out_ = Polynomial(ring)  # type: ignore
        else:
            out_: Polynomial = out
        assert out_.ring.is_quotient_ring(self.ring), "Not a quotient ring"
        self.ring.lib.polynomial_RNSc_mod_reduce(out_.obj, self.obj)
        out_.repr = repr.coeff
        return out_

    # floor division, in-place
    def floor_division(self, ring: Ring):
        assert ring.ell < self.ring.ell, "new ring is not smaller than the current one"
        assert ring.is_quotient_ring(self.ring), (
            "Not a quotient ring or contiguous RNS-component subset"
        )
        self.to_coeff()
        divide_mask = self.rns_mask & ~ring.mask
        self.ring.lib.polynomial_floor_division_RNSc_wo_free(self.obj, divide_mask)
        self.ring = ring
        return self

    # round division, in-place
    def round_division(self, ring: Ring):
        assert ring.ell < self.ring.ell, "new ring is not smaller than the current one"
        assert ring.is_quotient_ring(self.ring), (
            "Not a quotient ring or contiguous RNS-component subset"
        )
        self.to_coeff()
        divide_mask = self.rns_mask & ~ring.mask
        self.ring.lib.polynomial_round_division_RNSc_wo_free(self.obj, divide_mask)
        self.ring = ring
        return self

    def scaled_lift(self, ring: Ring, delta=None) -> Polynomial:
        """Lifts the polynomial to a larger ring and scales it by the delta factor."""
        self.to_coeff()
        out = Polynomial(ring)
        self.ring.lib.polynomial_RNSc_scaled_lift(
            out.obj, self.obj, delta if delta is not None else ffi.NULL
        )
        out.repr = repr.coeff
        return out

    def __eq__(self, value) -> bool:
        if type(value) is int:
            self.to_coeff()
            return all([all([value == i[0]] + [j == 0 for j in i[1:]]) for i in self])
        elif type(value) is list:
            self.to_coeff()
            s_list = list(self)
            return all(
                [
                    all([value[i] == s_list[i][0]] + [j == 0 for j in s_list[i][1:]])
                    for i in range(self.ring.ell)
                ]
            )
        else:
            self.to_repr(value.repr)
            return bool(self.ring.lib.polynomial_eq(self.obj, value.obj))

    def __repr__(self) -> str:
        return str(list(self))

    def get_hash_pointer(self):
        self.to_NTT()
        return self.ring.lib.polynomial_RNS_get_hash_p(self.obj)

    def get_hash(self):
        self.to_NTT()
        result = ffi.new("uint64_t[4]")
        self.ring.lib.polynomial_RNS_get_hash(result, self.obj)
        return [result[i] for i in range(4)]

    def __hash__(self):
        raise TypeError(
            "not a Python hashable object. Call polynomial.get_hash() for cryptographic hash"
        )

    # for compatibility with list, not an efficient iterator
    def __iter__(self):
        self.to_coeff()
        out = self.get_coeff_matrix(repr=repr.coeff)
        return iter(out)

    def get_coeff_matrix(self, repr=repr.coeff):
        if self.repr != repr:
            self.to_repr(repr)
        p = ffi.cast("RNS_Polynomial", self.obj)
        values = []
        for idx in self.ring.prime_indices:
            row = p.coeffs[idx]
            values.append([row[k] for k in range(self.ring.N)])
        modMask = self.ring.split_degree - 1
        poly_size = self.ring.N // self.ring.split_degree
        c = values
        out = []
        for i in range(self.ring.ell):
            out_i = [0] * self.ring.N
            for j in range(self.ring.N):
                out_i[j] = c[i][(j & modMask) * poly_size + j // self.ring.split_degree]
            out += [out_i]
        return out

    def from_coeff_matrix(self, matrix, repr=repr.coeff):
        p = ffi.cast("RNS_Polynomial", self.obj)
        modMask = self.ring.split_degree - 1
        poly_size = self.ring.N // self.ring.split_degree
        for k, idx in enumerate(self.ring.prime_indices):
            row = p.coeffs[idx]
            for j in range(self.ring.N):
                row[(j & modMask) * poly_size + j // self.ring.split_degree] = int(
                    matrix[k][j]
                )
        self.repr = repr
        return self

    # returns the list of coefficients of a Polynomial element
    def get_polynomial(self, signed: bool = False) -> list:
        self.to_coeff()
        rns = list(self)
        if self.ring.ell == 1:
            unsigned_res = list(sum(rns, start=[]))
        else:
            unsigned_res = [
                crt([rns[i][j] for i in range(self.ring.ell)], self.ring.primes)
                for j in range(self.ring.N)
            ]
        if not signed:
            return unsigned_res
        else:
            return [
                min(i, -(self.ring.q_l - i), key=lambda x: abs(x)) for i in unsigned_res
            ]

    def copy(self) -> Polynomial:
        res = Polynomial(self.ring)
        self.ring.lib.polynomial_copy_RNS_polynomial(res.obj, self.obj)
        res.repr = self.repr
        return res

    def decompose(self, base, small=False):
        res = []
        lst = self.get_polynomial()
        for j in range(self.ring.bit_size // base):
            dec = []
            for i in range(len(lst)):
                dec.append(lst[i] & ((1 << base) - 1))
                lst[i] >>= base
            res.append(Polynomial(self.ring).from_array(dec))
        return res

    def __mod__(self, value: int):
        self.to_coeff()
        res = Polynomial(self.ring)
        value_idx = self.ring.primes.index(value)
        self.ring.lib.polynomial_RNSc_mod_reduce_lifted(res.obj, self.obj, value_idx)
        return res

    def __copy__(self) -> Polynomial:
        return self.copy()

    def __itruediv__(self, value) -> Polynomial:
        self.to_coeff()
        value_idx = self.ring.primes.index(value)
        self.ring.lib.polynomial_round_division_RNSc_wo_free(self.obj, 1 << value_idx)
        return self

    def __mul__(self, other) -> Polynomial:
        if type(other) is Polynomial:
            self.to_NTT()
            other.to_NTT()
            res = Polynomial(self.ring.intersec(other.ring))
            res.multiply(self, other)
        elif type(other) is int:
            if other == 0:
                return 0  # type: ignore
            if other == 1:
                return self.copy()
            res = Polynomial(self.ring)
            self.ring.lib.polynomial_scale_RNSc_polynomial(res.obj, self.obj, other)
            res.repr = self.repr
        elif type(other) is list:
            res = Polynomial(self.ring)
            assert len(other) == self.ring.ell
            self.ring.lib.polynomial_scale_RNS_polynomial_RNS(
                res.obj, self.obj, self.ring.scalar_array(other)
            )
            res.repr = self.repr
        else:
            print(type(other))
            assert False, "not implemented"
        return res

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return (-self) + other

    def __imul__(self, other):
        if type(other) is Polynomial:
            self.to_NTT()
            other.to_NTT()
            self.ring.lib.polynomial_multo_RNS_polynomial(self.obj, other.obj)
        elif type(other) is int:
            if other == 0:
                return 0
            if other == 1:
                return self
            assert other < 2**self.ring.smallest_prime  # large scaling not implemented
            self.ring.lib.polynomial_scale_RNSc_polynomial(self.obj, self.obj, other)
        else:
            print(type(other))
            assert False  # not implemented
        return self

    def __add__(self, other):
        if type(other) is int:
            if other == 0:
                return self.copy()
            else:
                res = Polynomial(self.ring)
                if self.repr == repr.coeff:
                    self.ring.lib.polynomial_RNSc_add_integer(res.obj, self.obj, other)
                else:
                    self.ring.lib.polynomial_RNS_add_integer(res.obj, self.obj, other)
                res.repr = self.repr
                return res
        if self.repr != other.repr:
            self.to_NTT()
            other.to_NTT()
        res = Polynomial(self.ring.intersec(other.ring))
        res.add(self, other)
        return res

    def __iadd__(self, other):
        if type(other) is int:
            if other == 0:
                return self
            else:
                assert False  # not implemented
        if self.repr != other.repr:
            self.to_NTT()
            other.to_NTT()
        if type(other) is Polynomial:
            self.add(self, other)
            return self
        else:  # assume it's rlwe
            return other + self

    def __isub__(self, other):
        if type(other) is int:
            if other == 0:
                return self
            else:
                assert False  # not implemented
        if self.repr != other.repr:
            self.to_NTT()
            other.to_NTT()
        self.sub(self, other)
        return self

    def __neg__(self):
        res = Polynomial(self.ring)
        res.negate(self)
        return res

    def __sub__(self, other):
        if type(other) is int:
            if other == 0:
                return self.copy()
            else:
                assert False  # not implemented
        if self.repr != other.repr:
            self.to_NTT()
            other.to_NTT()
        res = Polynomial(self.ring.intersec(other.ring))
        res.sub(self, other)
        return res
