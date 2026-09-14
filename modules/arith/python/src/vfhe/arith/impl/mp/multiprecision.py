# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from math import ceil, log2, prod
from typing import TYPE_CHECKING, ClassVar

from vfhe.arith.registry import register
from vfhe.arith.spec import Capability, Constraints, Spec
from vfhe.engine import ffi, lib

if TYPE_CHECKING:
    from vfhe.arith.impl.rns.polynomial import RNSPolynomial


class Multiprecision:
    #: The (implementation, backend) this parent was built for.
    spec: ClassVar[Spec]

    def __init__(self) -> None:
        self.lib = lib
        try:
            self.vector_size = self.lib.get_mp_vector_size()
        except AttributeError:
            self.vector_size = 1

    # --- readers (reconstruct Python ints from the base-2^52 digit arrays) ---
    def scalar_digits(self, handle):
        s = ffi.cast("MPScalar", handle)
        # digits is an opaque mp_vector_t array; each entry holds vector_size lanes.
        lanes = ffi.cast("uint64_t *", s.digits)
        return [lanes[i] for i in range(self.vector_size * s.d)]

    def poly_to_list(self, handle):
        p = ffi.cast("MPPolynomial", handle)
        return [
            sum(p.coeffs[i][j] * (2 ** (52 * i)) for i in range(p.d))
            for j in range(p.N)
        ]

    # --- constructors ---
    def load(self, x):
        if type(x) is int:
            x = [(x >> (52 * i)) & ((1 << 52) - 1) for i in range(ceil(log2(x) / 52))]
        return self.lib.mp_load(ffi.new("uint64_t[]", list(x)), len(x))

    def load_small(self, x: int):
        return self.lib.load_m512(x)

    def compute_crt_consts(self, primes: list[int]) -> dict:
        """Barrett CRT-reconstruction constants for ``primes``.

        Every returned list is in the order ``primes`` is given in;
        `from_polynomial` is what matches them to a ring. ``hat_q`` is the
        per-prime factor to apply in RNS before reconstructing, and ``d`` the
        digit count the `MPPolynomial` accumulator needs. Raises `ValueError`
        when the moduli admit no single-digit Barrett reconstruction.
        """
        ell = len(primes)
        # Residues are one factor of an IFMA multiply, so they have to fit a
        # base-2^52 digit; a wider prime is silently truncated there.
        if max(primes) > 2**52:
            raise ValueError(
                f"primes up to {max(primes).bit_length()} bits are too wide to "
                "reconstruct from: a residue must fit 52 bits"
            )
        ql = prod(primes)
        Q = [ql // p for p in primes]
        hat_q = [pow(Q[i], -1, primes[i]) for i in range(ell)]

        # Reconstruction is sum_i (x_i * hat_q_i mod p_i) * Q_i, so the
        # accumulator stays under ell * ql and the quotient the reduction
        # estimates under ell. Folding hat_q into the multiplier instead --
        # sum_i x_i * (Q_i * hat_q_i mod ql) -- would put the quotient at
        # about the sum of the primes, past the 52 bits a Barrett step
        # multiplies by once the moduli grow.
        acc_max = sum((p - 1) * qi for p, qi in zip(primes, Q, strict=True))
        # Two things fix the digit count: the accumulator has to fit, and the
        # reduction reads its Barrett quotient out of digits d and d-1, which
        # needs k in [52*(d-1), 52*d - 1] while every d-digit operand it
        # scales is multiplied by a (d-1)-digit one.
        d = max(ceil(acc_max.bit_length() / 52), ceil(ql.bit_length() / 52) + 1)
        # Largest k with 2**k // ql < 2**52, so that m stays a single digit.
        k_max = (ql << 52).bit_length() - 1
        k = min(52 * d - 1, k_max)
        m_val = 2**k // ql
        # One Barrett pass plus a conditional subtraction reduces fully when
        # 2**k covers the accumulator and the quotient is a single digit.
        if k < 52 * (d - 1) or acc_max >= 2**k or acc_max // ql >= 2**52:
            raise ValueError(
                f"these {ell} primes admit no single-digit Barrett "
                f"reconstruction: k must be in [{52 * (d - 1)}, {52 * d - 1}], "
                f"at most {k_max}, and cover {acc_max.bit_length()} bits"
            )

        # ql and every Q carry exactly d-1 digits: the scaling kernels consume
        # that many and write the full d of their destination, so a shorter
        # operand is read past its end and leaves stale digits behind.
        digits = d - 1
        q = self.load(self.limbs(ql, digits))
        pw = [ffi.cast("MPScalar", self.load(self.limbs(v, digits))) for v in Q]
        m = self.load_small(m_val)

        return {
            "pw": pw,
            "q": q,
            "m": m,
            "k": k,
            "d": d,
            "hat_q": hat_q,
            "primes": list(primes),
        }

    @staticmethod
    def limbs(x: int, d: int) -> list[int]:
        """``x`` as exactly ``d`` base-2^52 digits."""
        return [(x >> (52 * i)) & ((1 << 52) - 1) for i in range(d)]

    def from_polynomial(self, poly: RNSPolynomial, crt_consts: dict):
        """Reconstruct ``poly``'s coefficients as multiprecision integers.

        ``crt_consts`` must have been built by `compute_crt_consts` over this
        polynomial's ring primes; the two orderings are matched up here, since
        neither of the ones in play is the caller's: the per-prime scaling
        takes its factors in `RNSRing.primes` order, and the native
        reconstruction walks the ring's mask, so in ascending base index.

        ``poly`` is left in the coefficient domain. The returned handle frees
        its native allocation when it is collected.
        """
        ring = poly.ring
        position = {p: i for i, p in enumerate(crt_consts["primes"])}
        if position.keys() != set(ring.primes):
            raise ValueError(
                "crt_consts were built for other primes than this polynomial's ring"
            )
        by_index = [
            p for _, p in sorted(zip(ring.prime_indices, ring.primes, strict=True))
        ]
        pw = ffi.new("MPScalar[]", [crt_consts["pw"][position[p]] for p in by_index])
        hat_q = [crt_consts["hat_q"][position[p]] for p in ring.primes]

        poly.to_coeff()
        # The reconstruction wants x_i * hat_q_i mod p_i, which is a per-prime
        # scaling in RNS; `poly` is the caller's, so it goes to a temporary.
        scaled = poly * hat_q
        # The native allocation is owned by the returned handle: nothing else
        # holds it, and a caller reconstructing in a loop has no other way to
        # release it.
        res = ffi.gc(
            self.lib.new_mp_polynomial(ring.N, crt_consts["d"]),
            self.lib.free_mp_polynomial,
        )
        self.lib.mp_polynomial_from_RNS(
            res,
            scaled.obj,
            pw,
            crt_consts["q"],
            crt_consts["m"],
            crt_consts["k"],
        )
        return res


#: Base-2^52 limb vectors: the exchange representation between RNS residues
#: and Python integers. A stateless service rather than a parent with its own
#: elements, so it registers no element class.
MP_LIMB = register(
    Spec(
        implementation="mp",
        backend="limb52",
        parent_cls=Multiprecision,
        element_cls=None,
        capabilities=Capability.CORE | Capability.EXACT | Capability.DOMAINS_COINCIDE,
        constraints=Constraints(),
    )
)

# `spec` binds to the class because each class here serves exactly one; a
# class serving several must set it per instance instead.
Multiprecision.spec = MP_LIMB
