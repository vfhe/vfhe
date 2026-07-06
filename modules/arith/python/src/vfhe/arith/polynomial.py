# SPDX-License-Identifier: Apache-2.0
"""RNS polynomials over the native engine.

A :class:`Polynomial` wraps an opaque native ``rns_poly_t`` handle. Both
pieces of per-polynomial state -- the active-limb *mask* and the
representation *domain* (:class:`Domain`) -- live in C and are read through
accessors, so Python never shadows engine state. Operations convert
representation lazily (multiplication moves operands to EVAL, tower
operations to COEFF), and native status codes surface as the exceptions in
:mod:`vfhe.arith.errors`.
"""

from __future__ import annotations

from enum import Enum

from _vfhe_native import ffi, lib

from .errors import check
from .number_theory import crt
from .registry import RingRegistry
from .ring import Ring


class Domain(Enum):
    """Representation domain of a polynomial (mirrors ``vfhe_domain``)."""

    COEFF = 0  #: coefficient form
    EVAL = 1  #: evaluation (NTT) form


class Polynomial:
    """An element of a :class:`~vfhe.arith.ring.Ring`.

    Wraps the native handle ``self.obj``. The current representation is the
    :attr:`domain` property (read from C); arithmetic operators accept other
    polynomials, plain ints (scalar ops), or per-level int lists, and
    interoperate by duck typing with higher-level ring elements (e.g. RLWE
    ciphertexts) implementing the reflected operators.
    """

    def __init__(self, ring: Ring) -> None:
        self.ring = ring
        self.obj = ring.alloc_polynomial()

    def __del__(self) -> None:
        try:
            lib.poly_free(self.obj)
        except Exception:
            pass  # interpreter shutdown

    # --- state ------------------------------------------------------------------

    @property
    def rns_mask(self) -> int:
        """Active-limb mask (read from the native handle)."""
        return lib.poly_mask(self.obj)

    @property
    def domain(self) -> Domain:
        """Current representation domain (read from the native handle)."""
        return Domain(lib.poly_domain(self.obj))

    def to_eval(self) -> None:
        """Convert to evaluation (NTT) form in place. No-op if already there."""
        check(lib.poly_to_eval(self.obj, self.obj))

    def to_coeff(self) -> None:
        """Convert to coefficient form in place. No-op if already there."""
        check(lib.poly_to_coeff(self.obj, self.obj))

    def to_domain(self, domain: Domain) -> None:
        """Convert to the given :class:`Domain` in place."""
        if domain == Domain.EVAL:
            self.to_eval()
        else:
            self.to_coeff()

    # --- loading ------------------------------------------------------------------

    def from_array(self, array: list) -> Polynomial:
        """Load signed integer coefficients (zero-padded to N); ends in EVAL."""
        array = list(array) + ([0] * (self.ring.N - len(array)))
        check(lib.poly_from_int_array(self.obj, ffi.new("uint64_t[]", array)))
        return self

    def from_bigint_array(self, array: list) -> Polynomial:
        """Load arbitrary-precision coefficients by reducing per prime; ends in EVAL."""
        matrix = ffi.new("uint64_t*[]", self.ring.pool_size)
        rows = []  # keep the row buffers alive across the (copying) C call
        for k, idx in enumerate(self.ring.prime_indices):
            p = self.ring.primes[k]
            row = ffi.new("uint64_t[]", [v % p for v in array])
            rows.append(row)
            matrix[idx] = row
        check(lib.poly_from_residues(self.obj, matrix))
        return self

    # --- in-place cores (operators build fresh outputs and delegate here) ---------

    def multiply(self, in1: Polynomial, in2: Polynomial) -> None:
        """self = in1 * in2 (both operands must be in EVAL form)."""
        check(lib.poly_mul(self.obj, in1.obj, in2.obj))

    def add(self, in1: Polynomial, in2: Polynomial) -> None:
        """self = in1 + in2 (operands in matching domains)."""
        check(lib.poly_add(self.obj, in1.obj, in2.obj))

    def sub(self, in1: Polynomial, in2: Polynomial) -> None:
        """self = in1 - in2 (operands in matching domains)."""
        check(lib.poly_sub(self.obj, in1.obj, in2.obj))

    def negate(self, in1: Polynomial | None = None) -> None:
        """self = -in1 (defaults to negating in place)."""
        if not in1:
            in1 = self
        lib.poly_negate(self.obj, in1.obj)

    # --- sampling -------------------------------------------------------------------

    def sample_uniform(self, ntt=True) -> Polynomial:
        """Fill with uniform residues, tagged EVAL (default) or COEFF."""
        lib.poly_sample_uniform(
            self.obj, Domain.EVAL.value if ntt else Domain.COEFF.value
        )
        return self

    def sample_gaussian(self, sigma, ntt=True) -> Polynomial:
        """Sample a Gaussian element (COEFF), optionally transforming to EVAL."""
        lib.poly_sample_gaussian(self.obj, sigma)
        if ntt:
            self.to_eval()
        return self

    def sample_exceptional(self, size="minimal", ntt=True) -> Polynomial:
        """Sample from the exceptional set: one uniform value per limb, in
        every evaluation slot."""
        lib.poly_sample_uniform(self.obj, Domain.EVAL.value)
        check(lib.poly_broadcast_slot(self.obj, self.obj, 0))
        if not ntt:
            self.to_coeff()
        return self

    # --- ring maps ---------------------------------------------------------------------

    def automorphism(self, gen: int) -> Polynomial:
        """Apply the Galois automorphism ``X -> X^gen`` (returns COEFF form)."""
        assert gen < self.ring.N * 2
        res = Polynomial(self.ring)
        self.to_coeff()
        check(lib.poly_permute(res.obj, self.obj, gen))
        return res

    def fast_inverse(self) -> Polynomial:
        """Slot-wise inverse via batched inversion (EVAL, split_degree == 1).

        Raises:
            NotInvertibleError: if any evaluation slot is zero.
            UnsupportedError: if the ring has ``split_degree != 1``.
        """
        self.to_eval()
        out = Polynomial(self.ring)
        check(lib.poly_inverse(out.obj, self.obj))
        return out

    # --- evaluation-slot operations ------------------------------------------------

    def rotate_slots(self, rot: int) -> Polynomial:
        """Cyclically rotate the evaluation slots of each block left by ``rot``."""
        self.to_eval()
        out = Polynomial(self.ring)
        check(lib.poly_rotate_slots(out.obj, self.obj, rot))
        return out

    def copy_slot(self, dst: int, src: int) -> Polynomial:
        """Copy slot ``src`` over slot ``dst`` (within each block), in a copy."""
        self.to_eval()
        out = self.copy()
        check(lib.poly_copy_slot(out.obj, dst, self.obj, src))
        return out

    def broadcast_slot(self, slot_idx: int) -> Polynomial:
        """Broadcast slot ``slot_idx`` of each block to the whole block."""
        self.to_eval()
        out = Polynomial(self.ring)
        check(lib.poly_broadcast_slot(out.obj, self.obj, slot_idx))
        return out

    # --- tower operations -----------------------------------------------------------------

    def base_extend(
        self, ring: Ring | None = None, out: Polynomial | None = None
    ) -> Polynomial:
        """Extend to a larger ring's limb set (fast base extension)."""
        if out is None and ring is not None:
            out_ = Polynomial(ring)
        elif type(out) is Polynomial:
            out_: Polynomial = out
        assert self.ring.is_quotient_ring(out_.ring), "Not a quotient ring"
        return self.lift_to(out=out_)

    def lift_to(
        self, ring: Ring | None = None, out: Polynomial | None = None, plan=None
    ) -> Polynomial:
        """Base-convert into ``ring``/``out`` using a (cached) conversion plan."""
        self.to_coeff()
        if out is None and ring is not None:
            out_ = Polynomial(ring)
        elif type(out) is Polynomial:
            out_: Polynomial = out
        if plan is None:
            plan = RingRegistry().get_conversion_plan(
                self.ring.N, self.ring.split_degree, self.rns_mask, out_.rns_mask
            )
        check(lib.poly_base_convert(out_.obj, self.obj, plan))
        return out_

    def mod_reduce(
        self, ring: Ring | None = None, out: Polynomial | None = None
    ) -> Polynomial:
        """Project onto a smaller ring's limb set (natural ring hom)."""
        self.to_coeff()
        assert out is not None or ring is not None, "Must provide ring or out"
        if out is None:
            assert ring is not None
            out_ = Polynomial(ring)
        else:
            out_: Polynomial = out
        assert out_.ring.is_quotient_ring(self.ring), "Not a quotient ring"
        check(lib.poly_mod_reduce(out_.obj, self.obj))
        return out_

    def floor_division(self, ring: Ring) -> Polynomial:
        """In-place floor division by the primes dropped when moving to ``ring``."""
        assert ring.ell < self.ring.ell, "new ring is not smaller than the current one"
        assert ring.is_quotient_ring(self.ring), (
            "Not a quotient ring or contiguous RNS-component subset"
        )
        self.to_coeff()
        divide_mask = self.rns_mask & ~ring.mask
        check(lib.poly_div_floor(self.obj, divide_mask))
        self.ring = ring
        return self

    def round_division(self, ring: Ring) -> Polynomial:
        """In-place rounding division by the primes dropped moving to ``ring``."""
        assert ring.ell < self.ring.ell, "new ring is not smaller than the current one"
        assert ring.is_quotient_ring(self.ring), (
            "Not a quotient ring or contiguous RNS-component subset"
        )
        self.to_coeff()
        divide_mask = self.rns_mask & ~ring.mask
        check(lib.poly_div_round(self.obj, divide_mask))
        self.ring = ring
        return self

    def scaled_lift(self, ring: Ring, delta=None) -> Polynomial:
        """Lift to a larger ring scaled by the modulus ratio (optionally with
        precomputed per-slot factors ``delta``)."""
        self.to_coeff()
        out = Polynomial(ring)
        check(
            lib.poly_scaled_lift(
                out.obj, self.obj, delta if delta is not None else ffi.NULL
            )
        )
        return out

    # --- hashing --------------------------------------------------------------------------

    def get_hash(self) -> list[int]:
        """BLAKE3 digest (4 words) of the canonical (EVAL) representation."""
        self.to_eval()
        result = ffi.new("uint64_t[4]")
        lib.poly_digest(result, self.obj)
        return list(result)

    def get_hash_pointer(self):
        """Digest as a C-owned ``uint64_t[4]`` pointer (caller-side lifetime)."""
        self.to_eval()
        return lib.poly_digest_alloc(self.obj)

    def __hash__(self):
        raise TypeError(
            "not a Python hashable object; call polynomial.get_hash() for a cryptographic hash"
        )

    # --- data access -------------------------------------------------------------------------

    def get_coeff_matrix(self, domain: Domain = Domain.COEFF) -> list[list[int]]:
        """Residue matrix (``ell`` x ``N``) in natural coefficient order.

        Converts to ``domain`` first; undoes the engine's split-block layout.
        """
        self.to_domain(domain)
        mod_mask = self.ring.split_degree - 1
        poly_size = self.ring.N // self.ring.split_degree
        out = []
        for idx in self.ring.prime_indices:
            row = lib.poly_limb_data(self.obj, idx)
            c = [row[k] for k in range(self.ring.N)]
            out_i = [0] * self.ring.N
            for j in range(self.ring.N):
                out_i[j] = c[(j & mod_mask) * poly_size + j // self.ring.split_degree]
            out.append(out_i)
        return out

    def from_coeff_matrix(self, matrix, domain: Domain = Domain.COEFF) -> Polynomial:
        """Load a residue matrix written in natural coefficient order.

        The caller asserts the data's representation via ``domain`` (this is
        the expert path around the engine's domain tracking).
        """
        mod_mask = self.ring.split_degree - 1
        poly_size = self.ring.N // self.ring.split_degree
        for k, idx in enumerate(self.ring.prime_indices):
            row = lib.poly_limb_data(self.obj, idx)
            for j in range(self.ring.N):
                row[(j & mod_mask) * poly_size + j // self.ring.split_degree] = int(
                    matrix[k][j]
                )
        lib.poly_assume_domain(self.obj, domain.value)
        return self

    def get_polynomial(self, signed: bool = False) -> list:
        """Reconstruct the integer coefficients via CRT (optionally signed)."""
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
        return [
            min(i, -(self.ring.q_l - i), key=lambda x: abs(x)) for i in unsigned_res
        ]

    def decompose(self, base, small=False):
        """Digit-decompose the CRT-reconstructed coefficients in ``2^base``."""
        res = []
        lst = self.get_polynomial()
        for _ in range(self.ring.bit_size // base):
            dec = []
            for i in range(len(lst)):
                dec.append(lst[i] & ((1 << base) - 1))
                lst[i] >>= base
            res.append(Polynomial(self.ring).from_array(dec))
        return res

    def copy(self) -> Polynomial:
        """Deep copy (same ring, same mask/domain/data)."""
        res = Polynomial(self.ring)
        lib.poly_copy(res.obj, self.obj)
        return res

    def __copy__(self) -> Polynomial:
        return self.copy()

    def __iter__(self):
        """Iterate residue rows (converts to COEFF; not an efficient path)."""
        self.to_coeff()
        return iter(self.get_coeff_matrix(domain=Domain.COEFF))

    def __repr__(self) -> str:
        return str(list(self))

    # --- operators ---------------------------------------------------------------------------

    def __eq__(self, value: object) -> bool:
        if isinstance(value, int):
            self.to_coeff()
            return all([all([value == i[0]] + [j == 0 for j in i[1:]]) for i in self])
        elif isinstance(value, list):
            self.to_coeff()
            s_list = list(self)
            return all(
                [
                    all([value[i] == s_list[i][0]] + [j == 0 for j in s_list[i][1:]])
                    for i in range(self.ring.ell)
                ]
            )
        assert isinstance(value, Polynomial)
        self.to_domain(value.domain)
        return bool(lib.poly_eq(self.obj, value.obj))

    def __mul__(self, other) -> Polynomial:
        if type(other) is Polynomial:
            self.to_eval()
            other.to_eval()
            res = Polynomial(self.ring.intersec(other.ring))
            res.multiply(self, other)
        elif type(other) is int:
            if other == 0:
                return 0  # type: ignore[return-value]  # scalar 0 annihilates to int 0
            if other == 1:
                return self.copy()
            res = Polynomial(self.ring)
            lib.poly_scale(res.obj, self.obj, other)
        elif type(other) is list:
            res = Polynomial(self.ring)
            assert len(other) == self.ring.ell
            lib.poly_scale_vec(res.obj, self.obj, self.ring.scalar_array(other))
        else:
            raise TypeError(f"cannot multiply Polynomial by {type(other)}")
        return res

    def __rmul__(self, other) -> Polynomial:
        return self.__mul__(other)

    def __imul__(self, other) -> Polynomial:
        if type(other) is Polynomial:
            self.to_eval()
            other.to_eval()
            check(lib.poly_mul_into(self.obj, other.obj))
        elif type(other) is int:
            if other == 0:
                return 0  # type: ignore[return-value]  # scalar 0 annihilates to int 0
            if other == 1:
                return self
            assert other < 2**self.ring.smallest_prime  # large scaling not implemented
            lib.poly_scale(self.obj, self.obj, other)
        else:
            raise TypeError(f"cannot multiply Polynomial by {type(other)}")
        return self

    def __add__(self, other) -> Polynomial:
        if type(other) is int:
            if other == 0:
                return self.copy()
            res = Polynomial(self.ring)
            check(lib.poly_add_scalar(res.obj, self.obj, other))
            return res
        if self.domain != other.domain:
            self.to_eval()
            other.to_eval()
        res = Polynomial(self.ring.intersec(other.ring))
        res.add(self, other)
        return res

    def __radd__(self, other) -> Polynomial:
        return self.__add__(other)

    def __iadd__(self, other) -> Polynomial:
        if type(other) is int:
            if other == 0:
                return self
            check(lib.poly_add_scalar(self.obj, self.obj, other))
            return self
        if self.domain != other.domain:
            self.to_eval()
            other.to_eval()
        if type(other) is Polynomial:
            self.add(self, other)
            return self
        # Duck-typed higher-level element (e.g. RLWE): delegate to it.
        return other + self

    def __sub__(self, other) -> Polynomial:
        if type(other) is int:
            if other == 0:
                return self.copy()
            return self.__add__(-other)
        if self.domain != other.domain:
            self.to_eval()
            other.to_eval()
        res = Polynomial(self.ring.intersec(other.ring))
        res.sub(self, other)
        return res

    def __rsub__(self, other) -> Polynomial:
        return (-self) + other

    def __isub__(self, other) -> Polynomial:
        if type(other) is int:
            if other == 0:
                return self
            return self.__iadd__(-other)
        if self.domain != other.domain:
            self.to_eval()
            other.to_eval()
        self.sub(self, other)
        return self

    def __neg__(self) -> Polynomial:
        res = Polynomial(self.ring)
        res.negate(self)
        return res

    def __mod__(self, value: int) -> Polynomial:
        """Lift the residue of prime ``value`` into all limbs (COEFF result)."""
        self.to_coeff()
        res = Polynomial(self.ring)
        value_idx = self.ring.primes.index(value)
        check(
            lib.poly_lift_residue(res.obj, self.obj, self.ring.prime_indices[value_idx])
        )
        return res

    def __itruediv__(self, value) -> Polynomial:
        """In-place rounding division by a single limb prime ``value``."""
        self.to_coeff()
        value_idx = self.ring.primes.index(value)
        check(lib.poly_div_round(self.obj, 1 << self.ring.prime_indices[value_idx]))
        return self
