from __future__ import annotations

from math import log2

from vfhe.misc.libvfhe import ffi, lib

from .polynomial import Polynomial, Ring, repr

# The AVX-512 complex FFT casts these buffers to __m512d and uses aligned loads,
# so they must be 64-byte aligned; more than cffi's default. Back them with the
# engine's posix_memalign allocator (freed via libc free on GC).
_aligned64 = ffi.new_allocator(
    alloc=lambda size: lib.safe_aligned_malloc(size),
    free=lambda ptr: lib.free(ptr),
    should_clear_after_alloc=True,
)


class ComplexRing:
    def __init__(self, N, special_rous=True):
        self.lib = lib
        self.N = N
        self.logN = int(log2(N))
        self.N2 = 2 * N

        if special_rous:
            rous = self.gen_special_rous_hp(self.N2)
            rous_real = [i.real for i in rous]
            rous_imag = [i.imag for i in rous]
            self.CT_rous = self.lib.load_rous_CT(
                ffi.new("double[]", rous_real), ffi.new("double[]", rous_imag), self.N2
            )
            # inv rous
            inv_rous = [i**-1 for i in rous]
            rous_inv_real = [i.real for i in inv_rous]
            rous_inv_imag = [i.imag for i in inv_rous]
            self.GS_rous = self.lib.load_rous_GS(
                ffi.new("double[]", rous_inv_real),
                ffi.new("double[]", rous_inv_imag),
                self.N2,
            )

    def alloc_polynomial(self):
        return _aligned64("double[]", self.N2)

    def exp_complex_polys_ifft_scale_round_to_RNS_batch(
        self,
        e_polys: list[ComplexPolynomial],
        ring: Ring,
        temp_delta: float,
    ) -> list[Polynomial]:
        """Batch: for each exp-domain ComplexPolynomial, IFFT, scale by temp_delta, round to RNS/NTT."""
        n = len(e_polys)
        if n == 0:
            return []
        res = [Polynomial(ring) for _ in range(n)]
        rows = ffi.new(
            "void*[]", [ffi.cast("void *", e_polys[i].obj) for i in range(n)]
        )
        outs = ffi.new("void*[]", [ffi.cast("void *", res[i].obj) for i in range(n)])
        self.lib.complex_polys_ifft_scale_round_to_RNS_batch(
            rows,
            outs,
            n,
            self.N,
            self.logN,
            self.GS_rous,
            temp_delta,
        )
        for p in res:
            p.repr = repr.ntt
        return res

    @staticmethod
    # special RoUs for CKKS
    def gen_special_rous(rou, N):
        def brev(x, size):
            return int(bin(x)[2:].rjust(size, "0")[::-1], 2)

        result = [1] * N
        for k in range(int(log2(N)) - 1, 0, -1):
            for i in range(2 ** (k - 1)):
                result[2 ** (k - 1) + i] = rou ** (
                    (5 ** brev(i, k - 1)) * N // (2 ** (k + 1))
                )
        return result

    @staticmethod
    def gen_special_rous_hp(N):
        N = int(N)
        import mpmath as mp

        mp.mp.prec = 100  # type: ignore
        rou = mp.exp(2 * mp.pi * 1j / (2 * mp.mpf(N)))

        def brev(x, size):
            return int(bin(x)[2:].rjust(size, "0")[::-1], 2)

        result = [1] * N
        for k in range(int(log2(N)) - 1, 0, -1):
            for i in range(2 ** (k - 1)):
                expo = ((5 ** brev(i, k - 1)) * N // (2 ** (k + 1))) % (2 * N)
                result[2 ** (k - 1) + i] = mp.power(rou, expo)
        return [complex(i) for i in result]


class ComplexPolynomial:
    def __init__(self, ring: ComplexRing):
        self.ring = ring
        self.obj = ring.alloc_polynomial()

    def __iter__(self):
        array = [
            self.obj[i] + self.obj[i + self.ring.N] * 1j for i in range(self.ring.N)
        ]
        return iter(array)

    def IFFT(self):
        self.ring.lib.bit_reverse_array(self.obj, self.ring.N, self.ring.logN)
        self.ring.lib.GS_RN(self.obj, self.ring.GS_rous, self.ring.N)
        self.ring.lib.complex_poly_scale_double(
            self.obj, 1.0 / self.ring.N, self.ring.N
        )

    def FFT(self):
        self.ring.lib.CT_NR(self.obj, self.ring.CT_rous, self.ring.N)
        self.ring.lib.bit_reverse_array(self.obj, self.ring.N, self.ring.logN)

    def __imul__(self, other):
        if type(other) is int or type(other) is float:
            self.ring.lib.complex_poly_scale_double(self.obj, float(other), self.ring.N)
            return self
        else:
            print(type(other))
            assert False, "not implemented"

    def __setitem__(self, idx, val: float | complex):
        if type(val) is float or type(val) is int:
            self.obj[idx] = val
            self.obj[idx + self.ring.N] = 0.0
        elif type(val) is complex:
            self.obj[idx] = float(val.real)
            self.obj[idx + self.ring.N] = float(val.imag)
        else:
            print(type(val))
            assert False, "not implemented"

    def from_array(self, v: list[float | complex]) -> ComplexPolynomial:
        for i in range(len(v)):
            self[i] = v[i]
        return self

    def round_to_RNS(self, ring: Ring) -> Polynomial:
        array = [round(self.obj[i]) for i in range(self.ring.N2)]
        return Polynomial(ring).from_array(array)

    # Duplicate of round_to_RNS with no Python per-coefficient overhead (C rounding + int_array_to_RNS).
    def round_to_RNS_cpp(self, ring: Ring) -> Polynomial:
        res = Polynomial(ring)
        self.ring.lib.complex_poly_round_to_RNS(res.obj, self.obj, ring.N)
        res.repr = repr.ntt
        return res
