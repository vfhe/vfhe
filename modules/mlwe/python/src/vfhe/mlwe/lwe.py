from __future__ import annotations

from vfhe.arith.number_theory import crt
from vfhe.arith.polynomial import Ring
from vfhe.misc.libvfhe import ffi, lib


class LibLWE:
    def __init__(self):
        self.lib = lib


lib_lwe = LibLWE()


class LWE_Key:
    def __init__(
        self,
        ring: Ring,
        sec_sigma: "float|None" = None,
        err_sigma: "float|None" = None,
        sparse_h: "int|None" = None,
        key: "list[int]|None" = None,
        n: "int|None" = None,
    ):
        self.ring = ring
        self.n = n if n is not None else ring.N
        self.l = ring.ell
        self.q = ring.primes[0]  # Kept for backward compat

        if key is not None:
            self.obj = lib_lwe.lib.lwe_alloc_key(self.n, self.l, ring.NTT)
            self.set_s(key)
        elif sparse_h is not None and err_sigma is not None:
            self.obj = lib_lwe.lib.lwe_new_sparse_ternary_key(
                self.n, self.l, ring.NTT, sparse_h, err_sigma
            )
        elif sec_sigma is not None and err_sigma is not None:
            self.obj = lib_lwe.lib.lwe_new_key(
                self.n, self.l, ring.NTT, sec_sigma, err_sigma
            )
        else:
            self.obj = lib_lwe.lib.lwe_alloc_key(self.n, self.l, ring.NTT)

    def set_s(self, key: list[int]):
        # Note: key might be flattened RNS polynomials. For LWE extraction,
        # we usually only deal with l=1 or we set identical limbs.
        assert len(key) == self.n
        s = ffi.cast("LWE_Key", self.obj).s  # uint64_t ** s
        for j in range(self.l):
            q_j = self.ring.primes[j]
            for i in range(self.n):
                val = key[i]
                if val < 0:
                    val = (val % q_j + q_j) % q_j
                else:
                    val = val % q_j
                s[j][i] = val

    def get_s(self) -> list[int]:
        s = ffi.cast("LWE_Key", self.obj).s
        q_0 = self.ring.primes[0]
        res = []
        for i in range(self.n):
            val = s[0][i]
            # Recover sign for small integer keys
            if val > q_0 // 2:
                res.append(val - q_0)
            else:
                res.append(val)
        return res


class LWE:
    def __init__(
        self,
        ring: Ring,
        m: "list[int]|None" = None,
        key: "LWE_Key|None" = None,
        is_trivial: bool = False,
        obj=None,
        n: "int|None" = None,
    ):
        self.ring = ring
        self.n = n if n is not None else ring.N
        self.l = ring.ell
        # self.q = ring.primes[0] # kept for backwards compat

        if obj is not None:
            self.obj = obj
        elif key is not None and m is not None:
            self.n = key.n
            m_arr = ffi.new("uint64_t[]", [x & 0xFFFFFFFFFFFFFFFF for x in m])
            self.obj = lib_lwe.lib.lwe_new_sample(m_arr, key.obj)
        elif is_trivial and m is not None:
            m_arr = ffi.new("uint64_t[]", [x & 0xFFFFFFFFFFFFFFFF for x in m])
            self.obj = lib_lwe.lib.lwe_new_trivial_sample(
                m_arr, self.n, self.l, ring.NTT
            )
        else:
            self.obj = lib_lwe.lib.lwe_alloc_sample(self.n, self.l, ring.NTT)

    def __del__(self):
        if hasattr(self, "obj") and self.obj is not None:
            lib_lwe.lib.free_lwe_sample(self.obj)

    def phase(self, key: LWE_Key, recompose: bool = False) -> "list[int]|int":
        out_arr = ffi.new("uint64_t[]", self.l)
        lib_lwe.lib.lwe_phase(out_arr, self.obj, key.obj)
        if recompose:
            return crt(list(out_arr), self.ring.primes)
        else:
            return list(out_arr)

    def subto(self, other: LWE):
        lib_lwe.lib.lwe_subto(self.obj, other.obj)

    def get_a(self) -> list[list[int]]:
        # Returns a list (length l) of list (length n)
        a = ffi.cast("LWE", self.obj).a
        res = []
        for j in range(self.l):
            res.append([int(a[j][i]) for i in range(self.n)])
        return res

    def get_b(self) -> list[int]:
        b = ffi.cast("LWE", self.obj).b
        return [int(b[j]) for j in range(self.l)]
