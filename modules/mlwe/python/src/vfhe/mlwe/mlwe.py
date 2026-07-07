from __future__ import annotations

import math
import secrets
from typing import TYPE_CHECKING

from vfhe.arith.polynomial import Polynomial, Ring, repr
from vfhe.misc.libvfhe import ffi, lib

if TYPE_CHECKING:
    from .lwe import LWE_Key


class LibMLWE:
    def __init__(self) -> None:
        self.lib = lib


lib_rlwe = LibMLWE()


class MLWE_Scheme:
    def __init__(
        self,
        rings: "list[Ring]|Ring",
        special_primes: int = 0,
        special_rings: "list[Ring]|None" = None,
        max_lvl: "int|None" = None,
        module_rank: int = 1,
    ) -> None:
        if isinstance(rings, Ring):
            max_lvl = max_lvl if max_lvl is not None else len(rings.primes) - 1
            if special_primes == 0:
                self.rings = [
                    rings.quotient_ring(ell=rings.ell - i) for i in range(max_lvl)
                ]
                self.special_rings = self.rings
            else:
                self.rings = [
                    rings.quotient_ring(ell=rings.ell - i)
                    for i in range(special_primes, special_primes + max_lvl)
                ]
                special_mask = ((1 << special_primes) - 1) << self.rings[0].ell
                self.special_rings = [
                    rings.quotient_ring(mask=self.rings[i].mask | special_mask)
                    for i in range(len(self.rings))
                ]
        else:
            max_lvl = max_lvl if max_lvl is not None else len(rings) - 1
            self.rings = [rings[i] for i in range(max_lvl)]
            assert special_rings is not None, (
                "special_rings required when rings is a list"
            )
            self.special_rings = [
                special_rings[i] for i in range(special_primes, max_lvl)
            ]

        self.r = module_rank
        self.N = self.rings[0].N
        self.ell = len(self.rings)
        self.max_lvl = max_lvl
        self.tmp = Polynomial(self.rings[0])
        self.rlk: "MLWE_Set | list | None" = None

    @property
    def ring(self) -> Ring:
        return self.rings[0]

    @property
    def special_primes(self) -> int:
        if self.special_rings and self.rings:
            return self.special_rings[0].ell - self.rings[0].ell
        return 0

    def key_gen_sparse(self, h, sigma_err, ternary=True):
        assert self.N * self.r >= h
        key = [[0] * self.N for _ in range(self.r)]
        crt_h = 0
        sign = 1
        while crt_h < h:
            rnd = secrets.randbelow(self.N * self.r)
            if key[rnd // self.N][rnd % self.N]:
                continue
            key[rnd // self.N][rnd % self.N] = sign
            if ternary:
                sign *= -1
            crt_h += 1
        return MLWE_Key(key, sigma_err, self)

    def gen_ksk_for_level(
        self, key_out: MLWE_Key, key_in: MLWE_Key | list[Polynomial], lvl: int
    ):
        key_poly = key_in if isinstance(key_in, list) else key_in.poly
        assert self == key_out.scheme, "Scheme mismatch"
        assert lvl is not None, "Level must be specified"
        result = []
        special_q = self.special_rings[lvl].modulus_ratio(self.rings[lvl])
        key_out_special = MLWE_Key(
            key_out.key, key_out.sigma_err, self, ring=self.special_rings[lvl]
        )
        for j in range(len(key_poly)):
            poly_j = key_poly[j]
            if poly_j.ring != self.special_rings[lvl]:
                poly_j = Polynomial(self.special_rings[lvl]).from_bigint_array(
                    poly_j.get_polynomial(signed=True)
                )
            result_i = []
            for i in range(self.special_rings[lvl].ell):
                scaling_factor = (
                    [0] * i
                    + [special_q % self.special_rings[lvl].primes[i]]
                    + [0] * (self.special_rings[lvl].ell - 1 - i)
                )
                out = MLWE(self, lvl=lvl, ring=self.special_rings[lvl])
                self.sample(poly_j * scaling_factor, key_out_special, out=out)
                result_i.append(out)
            result.append(result_i)
        return MLWE_Set(result)

    def gen_ksk(
        self,
        key_out: MLWE_Key,
        key_in: "MLWE_Key|list[Polynomial]",
        lvl: "int|None" = None,
    ):
        key_poly = key_in if isinstance(key_in, list) else key_in.poly
        assert self == key_out.scheme, "Scheme mismatch"
        if lvl is not None:
            return self.gen_ksk_for_level(key_out, key_poly, lvl)
        ksk_leveled = []
        for lvl in range(len(self.rings)):
            ksk_leveled.append(self.gen_ksk_for_level(key_out, key_poly, lvl))
        return ksk_leveled

    def gen_ksk_automorphism(
        self, key_out: MLWE_Key, key_in: MLWE_Key, g: int, lvl: "int|None" = None
    ):
        key_perm = [i.automorphism(g) for i in key_in.poly]
        return self.gen_ksk(key_out, key_perm, lvl)

    def gen_ksk_automorphism_set(
        self,
        key_out: MLWE_Key,
        key_in: MLWE_Key,
        generators: list[int],
        lvl: "int|None" = None,
    ):
        result = []
        for g in generators:
            result.append(self.gen_ksk_automorphism(key_out, key_in, g, lvl))
        return result

    def keyswitch(self, c: MLWE, ksk: MLWE_Set | list[MLWE_Set]):
        ksk = ksk if isinstance(ksk, MLWE_Set) else ksk[c.lvl]
        out = MLWE(self, lvl=c.lvl, ring=self.special_rings[c.lvl])
        c.to_coeff()
        lib_rlwe.lib.mlwe_RNSc_GHS_hybrid_keyswitch(out.obj, c.obj, ksk.obj, c.lvl)
        out.repr = repr.coeff
        out.ring = self.rings[c.lvl]
        return out

    def gen_ksk_trace(
        self,
        key_out: MLWE_Key,
        key_in: MLWE_Key,
        gens: "list[int]|None" = None,
        lvl: "int|None" = None,
    ):
        log_N = int(math.log2(self.N))
        gens = (
            gens
            if gens is not None
            else [(1 << (log_N - i + 1)) + 1 for i in range(1, log_N + 1)]
        )
        result = []
        for g in gens:
            result.append(self.gen_ksk_automorphism(key_out, key_in, g, lvl))
        if lvl is not None:
            return MLWE_Set().flatten_array(result)
        result_leveled = []
        for lvl in range(len(result[0])):
            result_lvl_i = [result[i][lvl] for i in range(len(gens))]
            result_leveled.append(MLWE_Set().flatten_array(result_lvl_i))
        return result_leveled

    def automorphism(self, c: MLWE, gen: int, ksk: MLWE_Set | list[MLWE_Set]):
        ksk = ksk if isinstance(ksk, MLWE_Set) else ksk[c.lvl]
        out = MLWE(self, lvl=c.lvl, ring=self.special_rings[c.lvl])
        c.to_coeff()
        lib_rlwe.lib.mlwe_automorphism_RNSc_GHS(out.obj, c.obj, gen, ksk.obj, c.lvl)
        out.repr = repr.coeff
        out.ring = self.rings[c.lvl]
        return out

    def trace(self, c: MLWE, ksk: MLWE_Set | list[MLWE_Set]):
        ksk = ksk if isinstance(ksk, MLWE_Set) else ksk[c.lvl]
        out = MLWE(self, lvl=c.lvl, ring=self.special_rings[c.lvl])
        c.to_coeff()
        lib_rlwe.lib.mlwe_trace(out.obj, c.obj, ksk.obj, c.lvl)
        out.repr = repr.coeff
        out.ring = self.rings[c.lvl]
        return out

    def full_packing_keyswitch_scaled(
        self, vec: list[MLWE], ksk: MLWE_Set | list[MLWE_Set]
    ):
        ell = int(math.log2(len(vec)))
        ksk = ksk if isinstance(ksk, MLWE_Set) else ksk[vec[0].lvl]
        assert (1 << ell) == len(vec)
        for c in vec:
            c.to_coeff()

        # create C array of handles for vec
        vec_ptr_array = ffi.new("void*[]", [c.obj for c in vec])

        lib_rlwe.lib.mlwe_full_packing_keyswitch_scaled(
            vec_ptr_array, ell, ksk.obj, vec[0].lvl
        )

        # The output is in vec[0]
        vec[0].repr = repr.coeff
        return vec[0]

    def sample(
        self,
        msg: Polynomial,
        key: MLWE_Key,
        out: "MLWE|None" = None,
        lvl: "int|None" = None,
    ) -> MLWE:
        """Samples an MLWE ciphertext of the given message polynomial under the given key.

        Args:
            msg: The message polynomial.
            key: The MLWE key.
            out: Optional MLWE object to store the result.

        Returns:
            The sampled MLWE ciphertext.
        """
        if not out:
            out = MLWE(self, lvl=lvl)
        msg.to_coeff()
        lib_rlwe.lib.mlwe_RNSc_sample(out.obj, key.obj, msg.obj)
        out.repr = repr.coeff
        return out

    def phase(self, rlwe: MLWE, key: MLWE_Key, out: "Polynomial|None" = None):
        if not out:
            out = Polynomial(rlwe.ring)
        if key.ring != rlwe.ring:
            key_at_ring = MLWE_Key(key.key, key.sigma_err, self, ring=rlwe.ring)
        else:
            key_at_ring = key
        rlwe.to_NTT()
        lib_rlwe.lib.mlwe_RNS_phase(out.obj, rlwe.obj, key_at_ring.obj)
        out.repr = repr.ntt
        return out

    def discrete_convolution(self, in1: MLWE, in2: MLWE) -> list[Polynomial]:
        """Computes the discrete convolution between the r+1 components of in1 and in2.

        Args:
            in1: The first MLWE ciphertext.
            in2: The second MLWE ciphertext.

        Returns:
            A list of 2*r+1 Polynomial objects.
        """
        assert in1.ring == in2.ring, "Ciphertexts must be in the same ring"
        assert in1.scheme.r == in2.scheme.r, "Ciphertexts must have the same rank"
        r = in1.scheme.r
        in1.to_NTT()
        in2.to_NTT()
        out_polys = [Polynomial(in1.ring) for _ in range(2 * r + 1)]
        for p in out_polys:
            p.repr = repr.ntt
        out_pointers = ffi.new("void*[]", [p.obj for p in out_polys])
        lib_rlwe.lib.mlwe_discrete_convolution(out_pointers, in1.obj, in2.obj)
        return out_polys

    def multiply(self, in1: MLWE, in2: MLWE, ksk: "MLWE_Set | list[MLWE_Set]") -> MLWE:
        """Performs discrete convolution and relinearization of two MLWE ciphertexts.

        Args:
            in1: The first MLWE ciphertext.
            in2: The second MLWE ciphertext.
            ksk: The key-switching key set for relinearization.

        Returns:
            A new MLWE ciphertext of rank r representing the product.
        """
        assert in1.ring == in2.ring, "Ciphertexts must be in the same ring"
        assert in1.scheme == in2.scheme, "Ciphertexts must be from the same scheme"
        assert in1.lvl == in2.lvl, "Ciphertexts must have the same level"
        ksk = ksk if isinstance(ksk, MLWE_Set) else ksk[in1.lvl]
        in1.to_NTT()
        in2.to_NTT()
        out = MLWE(in1.scheme, lvl=in1.lvl)
        lib_rlwe.lib.mlwe_multiply(out.obj, in1.obj, in2.obj, ksk.obj)
        out.repr = repr.ntt
        return out


class MLWE_Key:
    def __init__(
        self,
        key: list[list[int]],
        sigma_err: float,
        scheme: MLWE_Scheme,
        ring: "Ring|None" = None,
    ):
        assert len(key) == scheme.r
        self.key = key
        self.scheme = scheme
        self.sigma_err = sigma_err
        if ring is None:
            ring = scheme.rings[0]
        self.ring = ring
        # the key is copied verbatim into signed int64 coeffs; wrap negatives into
        # two's-complement uint64.
        concat_key = [x & 0xFFFFFFFFFFFFFFFF for x in sum(key, start=[])]
        self.obj = lib_rlwe.lib.mlwe_new_RNS_key_from_array(
            ffi.new("uint64_t[]", concat_key),
            ring.N,
            scheme.r,
            ring.ell,
            ring.NTT,
            sigma_err,
        )
        self.poly = [Polynomial(ring).from_array(s_i) for s_i in key]
        for p in self.poly:
            p.to_NTT()
        struct = ffi.cast("RNS_MLWE_Key", self.obj)
        for i in range(scheme.r):
            ring.lib.polynomial_copy_RNS_polynomial(struct.s_RNS[i], self.poly[i].obj)

    def __del__(self) -> None:
        if hasattr(self, "obj") and self.obj:
            lib_rlwe.lib.free_mlwe_RNS_key(self.obj)

    def extract_lwe_key(self) -> "LWE_Key":
        from .lwe import LWE_Key

        lwe_coeffs = sum(self.key, [])
        n = len(lwe_coeffs)
        return LWE_Key(ring=self.scheme.rings[0], key=lwe_coeffs, n=n)


class MLWE_Set:
    def __init__(self, mlwe: "list[list[MLWE]]|None" = None):
        if mlwe is None:
            return
        self.mlwe = mlwe
        self.dim = 2
        result_obj = ffi.new("void*[]", len(mlwe))
        for j in range(len(mlwe)):
            ell = len(mlwe[j])
            for x in mlwe[j]:
                x.to_NTT()
            tmp = ffi.new("void*[]", [i.obj for i in mlwe[j]])
            result_obj[j] = lib_rlwe.lib.mlwe_create_copy_array(tmp, ell)
        self.obj = result_obj

    # Turn an array of n-D MLWE_Set into a (n+1)-D MLWE_set
    @staticmethod
    def flatten_array(array: list[MLWE_Set]):
        out = MLWE_Set()
        out.mlwe = []
        out.dim = array[0].dim + 1
        result_obj = ffi.new("void*[]", len(array))
        out._children = array  # type: ignore  # keep child MLWE_Set buffers alive
        for j in range(len(array)):
            out.mlwe.append(array[j].mlwe)
            result_obj[j] = ffi.cast("void *", array[j].obj)
        out.obj = result_obj
        return out


class MLWE:
    def __init__(
        self, scheme: MLWE_Scheme, lvl: "int|None" = None, ring: "Ring|None" = None
    ) -> None:
        lvl = lvl if lvl is not None else 0
        if ring is not None:
            self.obj = lib_rlwe.lib.mlwe_alloc_RNS_sample(
                ring.N, scheme.r, ring.mask, ring.NTT
            )
            self.ring = ring
        else:
            ring = scheme.rings[lvl]
            self.obj = lib_rlwe.lib.mlwe_alloc_RNS_sample(
                ring.N, scheme.r, ring.mask, ring.NTT
            )
            self.ring = ring
        self.scheme = scheme
        self.repr = repr.empty
        self.lvl = lvl

    @property
    def ell(self) -> int:
        return self.ring.ell

    def __del__(self) -> None:
        if hasattr(self, "obj") and self.obj:
            lib_rlwe.lib.free_mlwe_RNS_sample(self.obj)

    def to_NTT(self):
        if self.repr == repr.ntt:
            return
        lib_rlwe.lib.mlwe_RNSc_to_RNS(self.obj, self.obj)
        self.repr = repr.ntt

    def to_coeff(self):
        if self.repr == repr.coeff:
            return
        lib_rlwe.lib.mlwe_RNS_to_RNSc(self.obj, self.obj)
        self.repr = repr.coeff

    def multiply_poly(self, in_rlwe, in_poly):
        assert in_rlwe.ring == in_poly.ring, "trying to mul things in different rings"
        assert in_rlwe.repr == in_poly.repr == repr.ntt
        lib_rlwe.lib.mlwe_RNS_mul_by_poly(self.obj, in_rlwe.obj, in_poly.obj)
        self.repr = repr.ntt

    def multiply_scalar(self, in_rlwe, pointer_to_int_list):
        if self is not in_rlwe:
            lib_rlwe.lib.mlwe_copy_RNS_sample(self.obj, in_rlwe.obj)
        lib_rlwe.lib.mlwe_scale_RNS_mlwe_RNS(self.obj, pointer_to_int_list)
        self.repr = in_rlwe.repr

    def add_MLWE(self, in1, in2):
        assert in1.repr == in2.repr
        lib_rlwe.lib.mlwe_add_RNSc_sample(self.obj, in1.obj, in2.obj)
        self.repr = in1.repr

    def sub_MLWE(self, in1, in2):
        assert in1.repr == in2.repr
        lib_rlwe.lib.mlwe_sub_RNSc_sample(self.obj, in1.obj, in2.obj)
        self.repr = in1.repr

    def add_poly(self, in1: MLWE, in2: Polynomial):
        assert in1.ring == in2.ring, "trying to add things in different rings"
        assert in1.repr == in2.repr
        if in1.repr == repr.ntt:
            lib_rlwe.lib.mlwe_RNS_add_polynomial(self.obj, in1.obj, in2.obj)
        else:
            lib_rlwe.lib.mlwe_add_RNSc_polynomial(self.obj, in1.obj, in2.obj)
        self.repr = in1.repr

    def sub_poly(self, in1, in2):
        assert in1.repr == in2.repr
        if in1.repr == repr.ntt:
            lib_rlwe.lib.mlwe_RNS_sub_polynomial(self.obj, in1.obj, in2.obj)
        else:
            lib_rlwe.lib.mlwe_sub_RNSc_polynomial(self.obj, in1.obj, in2.obj)
        self.repr = in1.repr

    def get_a_poly(self, j: int) -> Polynomial:
        res = Polynomial(self.ring)
        self.ring.lib.polynomial_copy_RNS_polynomial(res.obj, self.obj_a_i(j))
        res.repr = self.repr
        return res

    def get_b_poly(self) -> Polynomial:
        res = Polynomial(self.ring)
        self.ring.lib.polynomial_copy_RNS_polynomial(res.obj, self.obj_b())
        res.repr = self.repr
        return res

    def get_a_digit(self, j: int, i: int) -> Polynomial:
        res = Polynomial(self.ring)
        # polynomial_RNSc_mod_reduce(out, in) -- reduces to the base RNS limb.
        lib_rlwe.lib.polynomial_RNSc_mod_reduce(res.obj, self.obj_a_i(j))
        res.repr = repr.coeff
        return res

    def get_b_digit(self, i: int) -> Polynomial:
        res = Polynomial(self.ring)
        lib_rlwe.lib.polynomial_RNSc_mod_reduce(res.obj, self.obj_b())
        res.repr = repr.coeff
        return res

    def obj_a_i(self, j):
        # self.obj points to RNS_MLWE { a, b, r }; a is an array of RNS_Polynomial
        return ffi.cast("RNS_MLWE", self.obj).a[j]

    def obj_b(self):
        # b is the second member of RNS_MLWE
        return ffi.cast("RNS_MLWE", self.obj).b

    def copy(self):
        res = MLWE(self.scheme, lvl=self.lvl)
        lib_rlwe.lib.mlwe_copy_RNS_sample(res.obj, self.obj)
        res.repr = self.repr
        return res

    def copy_from(self, other: MLWE):
        lib_rlwe.lib.mlwe_copy_RNS_sample(self.obj, other.obj)
        self.repr = other.repr

    def round_division(self, ring: Ring):
        assert ring.is_quotient_ring(self.ring), "Not a quotient ring"
        self.to_coeff()
        divide_mask = self.ring.mask ^ ring.mask
        lib_rlwe.lib.mlwe_round_division_RNSc(self.obj, divide_mask)
        self.lvl = ring.ell
        self.ring = ring
        return self

    def __copy__(self):
        return self.copy()

    def _base_extend_cond(self, other: Polynomial):
        if other.ring != self.ring:
            res = other.base_extend(self.ring)
            res.to_NTT()
            return res
        return other

    def __add__(self, other):
        if type(other) == int:
            if other == 0:
                return self.copy()
            else:
                assert False  # not implemented
        if self.repr != other.repr:
            self.to_NTT()
            other.to_NTT()
        res = MLWE(self.scheme, lvl=self.lvl)
        if isinstance(other, MLWE):
            res.add_MLWE(self, other)
        if type(other) == Polynomial:
            res.add_poly(self, other)
        return res

    def __iadd__(self, other):
        if type(other) == int:
            if other == 0:
                return self
            else:
                assert False  # not implemented
        if self.repr != other.repr:
            self.to_NTT()
            other.to_NTT()
        if isinstance(other, MLWE):
            self.add_MLWE(self, other)
        if type(other) == Polynomial:
            self.add_poly(self, other)
        return self

    def __sub__(self, other) -> MLWE:
        if type(other) == int:
            if other == 0:
                return self.copy()
            else:
                assert False  # not implemented
        if self.repr != other.repr:
            self.to_NTT()
            other.to_NTT()
        res = MLWE(self.scheme, lvl=self.lvl)
        if isinstance(other, MLWE):
            res.sub_MLWE(self, other)
        if type(other) == Polynomial:
            res.sub_poly(self, other)
        return res

    def __isub__(self, other):
        if type(other) == int:
            if other == 0:
                return self
            else:
                assert False  # not implemented
        if self.repr != other.repr:
            self.to_NTT()
            other.to_NTT()
        if isinstance(other, MLWE):
            self.sub_MLWE(self, other)
        if type(other) == Polynomial:
            self.sub_poly(self, other)
        return self

    def __mul__(self, other) -> MLWE:
        if type(other) is Polynomial:
            res = MLWE(self.scheme, lvl=self.lvl)
            if other.ring != self.ring:
                assert other.ring.is_quotient_ring(self.ring)
                other_sr = other.base_extend(self.ring)
            else:
                other_sr = other
            self.to_NTT()
            other_sr.to_NTT()
            res.multiply_poly(self, other_sr)
        elif type(other).__name__ == "MGSW":
            return other * self
        elif isinstance(other, MLWE):
            assert self.scheme.rlk is not None, (
                "Relinearization key (rlk) must be set in the scheme"
            )
            return self.scheme.multiply(self, other, self.scheme.rlk)
        else:  # assuming other is a pointer to int
            res = MLWE(self.scheme, lvl=self.lvl)
            res.multiply_scalar(self, other)
        return res

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)
