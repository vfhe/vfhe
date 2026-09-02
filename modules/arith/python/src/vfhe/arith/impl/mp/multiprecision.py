# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from math import ceil, log2, prod
from typing import TYPE_CHECKING

from vfhe.arith.registry import register
from vfhe.arith.spec import Capability, Constraints, Spec
from vfhe.engine import ffi, lib

if TYPE_CHECKING:
    from vfhe.arith.impl.rns.polynomial import Polynomial


class Multiprecision:
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

    def compute_crt_consts(self, primes):
        ell = len(primes)
        ql = prod(primes)
        q = self.load(ql)
        prime_size = max(map(log2, primes))
        k = ceil(prime_size * (ell + 1))
        Q, hatQ, QhatQ = [], [], []
        for i in range(ell):
            Q.append(prod([primes[j] for j in range(ell) if i != j]))
            hatQ.append(pow(Q[-1], -1, primes[i]))
            QhatQ.append((Q[-1] * hatQ[-1]) % ql)

        pw = ffi.new(
            "MPScalar[]", [ffi.cast("MPScalar", self.load(int(v))) for v in QhatQ]
        )

        m_val = 2**k // ql
        if not (m_val < 2**52):
            raise ValueError("m_val < 2**52")
        m = self.load_small(m_val)

        return {"pw": pw, "q": q, "m": m, "k": k}

    def from_polynomial(self, poly: Polynomial, crt_consts):
        poly.to_coeff()
        res = self.lib.new_mp_polynomial(poly.ring.N, poly.ring.ell + 1)
        self.lib.mp_polynomial_from_RNS(
            res,
            poly.obj,
            crt_consts["pw"],
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
