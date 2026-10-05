# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from vfhe.arith.number_theory import crt
from vfhe.engine import ffi, lib

if TYPE_CHECKING:
    from vfhe.arith import RNSRing


class LibLWE:
    def __init__(self):
        self.lib = lib


lib_lwe = LibLWE()

_T = TypeVar("_T")


def _native_limbs(ring: RNSRing) -> list[int]:
    """The position in ``ring.primes`` of each native limb, in native order.

    Natively a limb is the residue modulo the prime at the i-th set bit of the
    ring's mask, in ascending base index. This API presents limbs in
    ``ring.primes`` order instead, as the rest of the RNS surface does.
    """
    return sorted(range(ring.ell), key=lambda i: ring.prime_indices[i])


class LWE_Key:
    def __init__(
        self,
        ring: RNSRing,
        sec_sigma: float | None = None,
        err_sigma: float | None = None,
        sparse_h: int | None = None,
        key: list[int] | None = None,
        n: int | None = None,
    ):
        """An LWE secret key over ``ring``, of dimension ``n`` (``ring.N``).

        Generated (``sec_sigma`` or ``sparse_h``, with ``err_sigma``), or given
        as its coefficients in ``key``. ``err_sigma`` is the standard deviation
        of the noise the key encrypts with; a key given without one is
        ``err_sigma=None``, decrypts, and refuses to encrypt.
        """
        self.ring = ring
        self.err_sigma: float | None = None
        self.n = n if n is not None else ring.N
        self.l = ring.ell
        self.q = ring.primes[0]  # Kept for backward compat
        self._limbs = _native_limbs(ring)

        if key is not None:
            self.obj = lib_lwe.lib.lwe_alloc_key(self.n, ring.mask, ring.base)
            self.set_s(key)
            if err_sigma is not None:
                ffi.cast("LWE_Key", self.obj).sigma = err_sigma
                self.err_sigma = err_sigma
        elif sparse_h is not None and err_sigma is not None:
            self.obj = lib_lwe.lib.lwe_new_sparse_ternary_key(
                self.n, ring.mask, ring.base, sparse_h, err_sigma
            )
            self.err_sigma = err_sigma
        elif sec_sigma is not None and err_sigma is not None:
            self.obj = lib_lwe.lib.lwe_new_key(
                self.n, ring.mask, ring.base, sec_sigma, err_sigma
            )
            self.err_sigma = err_sigma
        else:
            self.obj = lib_lwe.lib.lwe_alloc_key(self.n, ring.mask, ring.base)

    def set_s(self, key: list[int]):
        # Note: key might be flattened RNS polynomials. For LWE extraction,
        # we usually only deal with l=1 or we set identical limbs.
        if not (len(key) == self.n):
            raise ValueError("len(key) == self.n")
        s = ffi.cast("LWE_Key", self.obj).s  # uint64_t ** s
        for j, pos in enumerate(self._limbs):
            q_j = self.ring.primes[pos]
            for i in range(self.n):
                val = key[i]
                val = (val % q_j + q_j) % q_j if val < 0 else val % q_j
                s[j][i] = val

    def get_s(self) -> list[int]:
        s = ffi.cast("LWE_Key", self.obj).s
        q_0 = self.ring.primes[self._limbs[0]]
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
        ring: RNSRing,
        m: list[int] | None = None,
        key: LWE_Key | None = None,
        is_trivial: bool = False,
        obj=None,
        n: int | None = None,
    ):
        self.ring = ring
        self.n = n if n is not None else ring.N
        self.l = ring.ell
        # self.q = ring.primes[0] # kept for backwards compat
        self._limbs = _native_limbs(ring)

        if obj is not None:
            if ffi.cast("LWE", obj).mask != ring.mask:
                raise ValueError("the sample does not live over the ring's primes")
            self.obj = obj
        elif key is not None and m is not None:
            if key.err_sigma is None:
                raise ValueError(
                    "the key has no noise parameter: build it with err_sigma to "
                    "encrypt under it"
                )
            self.n = key.n
            m_arr = ffi.new("uint64_t[]", self._to_native(m))
            self.obj = lib_lwe.lib.lwe_new_sample(m_arr, key.obj)
        elif is_trivial and m is not None:
            m_arr = ffi.new("uint64_t[]", self._to_native(m))
            self.obj = lib_lwe.lib.lwe_new_trivial_sample(
                m_arr, self.n, ring.mask, ring.base
            )
        else:
            self.obj = lib_lwe.lib.lwe_alloc_sample(self.n, ring.mask, ring.base)

    def _to_native(self, limbs: list[int]) -> list[int]:
        return [limbs[pos] & 0xFFFFFFFFFFFFFFFF for pos in self._limbs]

    def _from_native(self, limbs: list[_T]) -> list[_T]:
        res = list(limbs)
        for j, pos in enumerate(self._limbs):
            res[pos] = limbs[j]
        return res

    def __del__(self):
        if hasattr(self, "obj") and self.obj is not None:
            lib_lwe.lib.free_lwe_sample(self.obj)

    def linear_decrypt(self, key: LWE_Key, recompose: bool = False) -> list[int] | int:
        """The linear part of decrypting this sample under ``key``.

        This is ``b - <a, s>`` per limb (the encoded message plus the noise),
        CRT-recomposed into one integer with ``recompose``.
        """
        out_arr = ffi.new("uint64_t[]", self.l)
        lib_lwe.lib.lwe_linear_decrypt(out_arr, self.obj, key.obj)
        out = self._from_native([int(x) for x in out_arr])
        if recompose:
            return crt(out, self.ring.primes)
        else:
            return out

    def subto(self, other: LWE):
        lib_lwe.lib.lwe_subto(self.obj, other.obj)

    def get_a(self) -> list[list[int]]:
        # Returns a list (length l) of list (length n)
        a = ffi.cast("LWE", self.obj).a
        return self._from_native(
            [[int(a[j][i]) for i in range(self.n)] for j in range(self.l)]
        )

    def get_b(self) -> list[int]:
        b = ffi.cast("LWE", self.obj).b
        return self._from_native([int(b[j]) for j in range(self.l)])
