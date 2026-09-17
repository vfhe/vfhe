# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Foldable codes over R_q for the basefold commitment.

The code is *interleaved*: it acts on a vector of ring elements
coefficient-slot-wise and per RNS prime, so each (prime, coefficient slot)
pair carries an independent codeword over Z_p, and a "scalar" is one
integer per RNS prime, applied through `Polynomial * list`.

The family is [ZCF24, Def. 5]'s foldable code, in vfhe's indexing: the
message splits into its even and odd entries (`m_even`, `m_odd`), the two
are encoded one level down, and two tables `T_l`, `T'_l` over the
half-length positions recombine them into adjacent pairs::

    encode_l(m)[2j]     = encode_{l-1}(m_even)[j] + encode_{l-1}(m_odd)[j] * T_l[j]
    encode_l(m)[2j + 1] = encode_{l-1}(m_even)[j] + encode_{l-1}(m_odd)[j] * T'_l[j]

With `T_l[j] != T'_l[j]` a codeword folds, pair by pair and with only the
tables known, into a codeword of `m_even + r * m_odd` for any challenge `r`::

    b[j]    = (word[2j] - word[2j + 1]) / (T_l[j] - T'_l[j])
    fold[j] = word[2j + 1] + b[j] * (-T'_l[j]) + b[j] * r

which is why the fold stores `1 / (T - T')` and `-T'` rather than the
tables themselves. The message is a monomial-basis coefficient vector
(LSB-first multilinear index order — `MLE.to_coefficients`), so folding the
codeword with r is exactly binding the first (LSB) variable of the MLE to r,
the same fold the sumcheck prover applies to the evaluation table.

Two instantiations share the class, selected by `instantiation`:

- ``"general"`` (the default): the tables are drawn from a seed, and the
  base code is a Reed-Solomon code on `n0` points — arith's negacyclic NTT
  when the ring's primes carry a root of unity of order `2 n0` (the `rs_*`
  kernels, `polycom/c/src/rscode_rns.c`), a Vandermonde product on seeded
  distinct points otherwise. No transform above the base, and no condition
  on the codeword length beyond the base's. The distance is [ZCF24,
  Thm. 2]'s bound, `foldable_relative_distance`.
- ``"rs"``: every level a Reed-Solomon code on roots of unity, the FRI
  structure [BBHR18], with one transform per level. `ntt_forward` is CT_NR
  (natural in, bit-reversed out), so position p of a length-n codeword holds
  P(psi^(2*brv(p)+1)) and the adjacent pairs are `(P(x_i), P(-x_i))` for
  `x_i = psi^(2*brv(i)+1)`: the tables are `T = x`, `T' = -x`, and the
  levels' points nest (`psi_{n/2} = psi_n^2`, since `ntt_new_plan` derives
  psi from the smallest quadratic non-residue, a choice independent of the
  length). Needs `2 n_d | p - 1`; faster, and with the exact distance
  `1 - 1/c + 1/n_d`.
"""

from __future__ import annotations

import contextlib
import math
from typing import TYPE_CHECKING, TypeVar

from vfhe.arith import Polynomial, RNSPolynomial, RNSRing
from vfhe.arith.mle import element_array, mark_ntt
from vfhe.crypto import hash_bytes, leaf_digest, seeded
from vfhe.engine import lib

if TYPE_CHECKING:
    from collections.abc import Callable

#: The seed a code's public tables are derived from when the caller names
#: none. Public and fixed: two parties building a code from the same
#: arguments hold the same code.
DEFAULT_SEED = b"vfhe.polycom foldable code"

#: The instantiations of the foldable family a code can be built as.
INSTANTIATIONS = ("general", "rs")

# How many seeds are tried for a table before giving up: a retry happens with
# probability about `n / p` per table, so a second one is already unlikely.
_MAX_SAMPLING_ATTEMPTS = 64

# The domain-separation tag of every seeded draw this module makes.
_CONTEXT = b"vfhe.polycom.foldable-code"

_T = TypeVar("_T")


def check_instantiation(name: str) -> str:
    """`name` if it is one of `INSTANTIATIONS`, else a ValueError."""
    if name not in INSTANTIATIONS:
        raise ValueError(f"instantiation must be one of {INSTANTIATIONS}, got {name!r}")
    return name


def derive_seed(seed: bytes, tag: bytes, attempt: int = 0) -> bytes:
    """The seed of one table: a digest of the code's seed, the table's tag
    and the attempt number, so every table of a code is an independent pure
    function of the seed and a rejected draw is redone under a fresh one."""
    return hash_bytes(
        _CONTEXT
        + attempt.to_bytes(2, "little")
        + len(tag).to_bytes(2, "little")
        + tag
        + seed
    )


def foldable_relative_distance(
    k0: int, c: int, d: int, field_bits: float, security_bits: int = 128
) -> float:
    """[ZCF24, Thm. 2]'s lower bound on the relative minimum distance of the
    general foldable code with base dimension `k0`, inverse rate `c`, depth
    `d` and random twists over a field of `2^field_bits` elements, in the
    recurrence form of [ZCF24, App. C]::

        Z_0 = 1 / c
        Z_i = Z_{i-1} + ((2 log2(n_{i-1}) + lambda) / n_i + 1.001 Z_{i-1} + 0.6)
                        / (field_bits - 1.001)
        distance >= 1 - Z_d

    which holds with probability at least `1 - d * 2^-security_bits` over the
    twists. `Z_0 = 1/c` is the base code's distance up to the `1/n_0` that
    an MDS code adds, so the base must be MDS (a Reed-Solomon code on
    distinct points) for the bound to apply. The value is clamped at 0: a 0
    means the theorem gives nothing at these parameters, not that the code
    has no distance.
    """
    if field_bits <= 1.001:
        raise ValueError(
            f"the field must have more than 2 elements, got 2^{field_bits}"
        )
    z = 1 / c
    for level in range(1, d + 1):
        n_below = c * k0 << (level - 1)
        z += (
            (2 * math.log2(n_below) + security_bits) / (2 * n_below) + 1.001 * z + 0.6
        ) / (field_bits - 1.001)
    return max(0.0, 1 - z)


def vandermonde_inverse(
    points: list[_T],
    one: _T,
    zero: _T,
    add: Callable[[_T, _T], _T],
    sub: Callable[[_T, _T], _T],
    mul: Callable[[_T, _T], _T],
    inv: Callable[[_T], _T],
) -> list[list[_T]]:
    """The inverse of the Vandermonde matrix on `points`, as rows:
    ``message[i] = sum_j inverse[i][j] * word[j]`` recovers the coefficients
    of the polynomial whose values at `points` are `word`.

    Row `i` holds coefficient `i` of every Lagrange basis polynomial, from
    the master polynomial `prod (X - x_t)` divided by each `(X - x_j)` and
    scaled by `1 / prod_{t != j} (x_j - x_t)`: `O(k^2)` operations in the
    callables, for `k` points that must be distinct.
    """
    k = len(points)
    master = [one]  # coefficients of prod (X - x_t), lowest first
    for x in points:
        lifted = [zero, *master]  # X * master
        for i, coefficient in enumerate(master):
            lifted[i] = sub(lifted[i], mul(x, coefficient))
        master = lifted
    inverse = [[zero] * k for _ in range(k)]
    for j, x in enumerate(points):
        # Synthetic division of the master polynomial by (X - x).
        quotient = [zero] * k
        quotient[k - 1] = master[k]
        for i in range(k - 1, 0, -1):
            quotient[i - 1] = add(master[i], mul(x, quotient[i]))
        denominator = zero
        for coefficient in reversed(quotient):  # Horner: quotient(x)
            denominator = add(mul(denominator, x), coefficient)
        scale = inv(denominator)
        for i in range(k):
            inverse[i][j] = mul(quotient[i], scale)
    return inverse


def bit_reverse(i: int, bits: int) -> int:
    """`i` with its low `bits` bits reversed — the index permutation
    `ntt_forward` (CT_NR) leaves in its output.

    Public because it is not this code family's: any code built on that
    transform needs it to say which evaluation point a position holds, and
    a second one carrying its own copy is a second place for the convention
    to drift from arith's.
    """
    out = 0
    for _ in range(bits):
        out = (out << 1) | (i & 1)
        i >>= 1
    return out


def bit_reverse_permutation(bits: int) -> list[int]:
    """``[bit_reverse(i, bits) for i in range(1 << bits)]``, in one pass.

    Each entry follows from the one at half its index -- the low bit becomes
    the high one, and the rest is the shorter reversal already computed -- so
    the whole permutation costs O(n) where calling `bit_reverse` per index
    costs O(n log n). That difference is the whole cost of a twist table once
    the modular exponentiations are gone.
    """
    if bits < 0:
        raise ValueError(f"bits must not be negative, got {bits}")
    n = 1 << bits
    rev = [0] * n
    for i in range(1, n):
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bits - 1))
    return rev


def _element_hash(element: RNSPolynomial):
    """The element's own digest, dispatched through its class."""
    return element.get_hash()


def pair_digest(pair: tuple[RNSPolynomial, RNSPolynomial]) -> bytes:
    """The Merkle leaf digest of one adjacent pair.

    Both ring elements are hashed with their own `get_hash` and the two
    digests hashed together, so the leaf binds the pair as an ordered unit.
    `get_hash` hashes the NTT form and leaves the entry alone, so hashing a
    codeword never disturbs it.
    """
    lo, hi = pair
    return hash_bytes(leaf_digest(lo, _element_hash) + leaf_digest(hi, _element_hash))


#: A per-position table of per-prime scalars: ``table[j][k]`` is the scalar
#: at position ``j`` for the ring's ``k``-th prime, the row `Polynomial *
#: list` takes.
Table = list[list[int]]


class FoldableRS:
    """A depth-d foldable code over `ring`, with base dimension k0 and
    inverse rate c: level l encodes k0 * 2^l ring elements into
    n_l = c * k0 * 2^l. `encode` infers the level from the message length;
    `fold` / `fold_at` implement the verifier-checkable fold taking the
    level-l codeword of P to the level-(l-1) codeword of P_even + r * P_odd
    (the module docstring has the construction and both instantiations).

    Everything is per RNS prime: the twists are per-prime integers applied
    through `Polynomial * list`, their inverses per-prime modular inverses,
    and there are no ring inversions anywhere.

    The ring's primes satisfy `p = 1 mod 2N/split_degree`, so a negacyclic
    transform of length n exists exactly when `n | N/split_degree`. The
    ``"rs"`` instantiation needs one of length `n_d`; the general code uses
    one of length `n0` when it exists and a Vandermonde product otherwise.

    `seed` is the public parameter every table of the general code is
    derived from; the ``"rs"`` instantiation ignores it.
    """

    def __init__(
        self,
        ring: RNSRing,
        k0: int,
        c: int,
        d: int,
        *,
        instantiation: str = "general",
        seed: bytes = DEFAULT_SEED,
    ):
        for name, value in (("k0", k0), ("c", c)):
            if value < 1 or value & (value - 1):
                raise ValueError(f"{name} must be a power of two, got {value}")
        if d < 1:
            raise ValueError(f"d must be at least 1, got {d}")
        self.instantiation = check_instantiation(instantiation)
        self.seed = bytes(seed)
        self.ring = ring
        self.k0 = k0
        self.c = c
        self.d = d
        self.n0 = c * k0
        self.k_d = k0 << d
        self.n_d = self.n0 << d
        # The four tables of every fold level, `[level][position][prime]`:
        # `twists` / `twists_odd` are `T` and `T'` of the fold from level l+1
        # to level l (what the encoder multiplies by), `_fold_scale` /
        # `_fold_shift` the `1 / (T - T')` and `-T'` the fold reads.
        self.twists: list[Table] = []
        self.twists_odd: list[Table] = []
        self._fold_scale: list[Table] = []
        self._fold_shift: list[Table] = []
        #: The roots of the transforms in use, `[level][prime]`: one row per
        #: level for the ``"rs"`` instantiation, the base transform's alone
        #: for the general code (empty when its base is a Vandermonde product).
        self.roots: list[list[int]] = []
        # One NTT-plan array per transform in use (indexed by global RNS prime
        # index, matching RNS_Polynomial.coeffs), owned by this object.
        # `rs_new_plans` sizes each array by `ring.rns_rows`, which is derived
        # from the ring's own mask and so is stable for the object's lifetime —
        # unlike the shared RNS base's prime count, which grows whenever another
        # ring of the same (N, split_degree) introduces a prime.
        self._plans: list = []
        if self.instantiation == "rs":
            self._init_rs()
        else:
            self._init_general()
        self._vandermonde_inverse: list[list[list[int]]] | None = None

    def _new_plans(self, size: int):
        plans = lib.rs_new_plans(self.ring.base, self.ring.mask, size)
        self._plans.append(plans)
        # Read back from the plans so the tables cannot drift from the
        # kernels' convention.
        self.roots.append(
            [lib.rs_plans_root(plans, idx) for idx in self.ring.prime_indices]
        )
        return plans

    def _transform_points(self, roots: list[int], n: int) -> Table:
        """The `n` evaluation points of a transform at `roots` (one per
        prime), in its output order: position `j` holds
        `psi^(2 brv(j) + 1)`."""
        bits = n.bit_length() - 1
        return [
            [
                pow(psi, 2 * bit_reverse(j, bits) + 1, p)
                for psi, p in zip(roots, self.ring.primes, strict=True)
            ]
            for j in range(n)
        ]

    def _init_rs(self) -> None:
        """Every level a transform; twists `x` and `-x` from its points."""
        limit = self.ring.N // self.ring.split_degree
        if self.n_d > limit:
            raise ValueError(
                f"codeword length {self.n_d} exceeds N/split_degree = {limit}: "
                "the ring's primes carry no root of unity of order 2 * n_d"
            )
        for level in range(self.d + 1):
            self._new_plans(self.n0 << level)
        self.base_points = self._transform_points(self.roots[0], self.n0)
        primes = self.ring.primes
        for level in range(self.d):
            n = self.n0 << level  # positions of the folded (level) codeword
            even = self._transform_points(self.roots[level + 1], n)
            self.twists.append(even)
            self.twists_odd.append(
                [
                    [(p - t) % p for t, p in zip(row, primes, strict=True)]
                    for row in even
                ]
            )
            self._fold_scale.append(
                [
                    [pow(2 * t, p - 2, p) for t, p in zip(row, primes, strict=True)]
                    for row in even
                ]
            )
            self._fold_shift.append(even)

    def _init_general(self) -> None:
        """A base code, encoded by a transform when the primes allow one of
        length `n0`, and seeded twists above it."""
        if self.n0 <= self.ring.N // self.ring.split_degree:
            self._new_plans(self.n0)
            self.base_points = self._transform_points(self.roots[0], self.n0)
        else:
            self.base_points = self._sample_points()
        primes = self.ring.primes
        for level in range(self.d):
            even, odd = self._sample_twists(level, self.n0 << level)
            self.twists.append(even)
            self.twists_odd.append(odd)
            self._fold_scale.append(
                [
                    [
                        pow((t - u) % p, p - 2, p)
                        for t, u, p in zip(row, row_odd, primes, strict=True)
                    ]
                    for row, row_odd in zip(even, odd, strict=True)
                ]
            )
            self._fold_shift.append(
                [[(p - u) % p for u, p in zip(row, primes, strict=True)] for row in odd]
            )

    def _sampled(self, n: int, tag: bytes, attempt: int) -> Table:
        """`n` positions of uniform per-prime scalars, a pure function of the
        code's seed, `tag` and `attempt`."""
        rows = [
            seeded.below_many(
                n, p, _CONTEXT, derive_seed(self.seed, tag + bytes([k]), attempt)
            )
            for k, p in enumerate(self.ring.primes)
        ]
        return [list(column) for column in zip(*rows, strict=True)]

    def _sample_twists(self, level: int, n: int) -> tuple[Table, Table]:
        """`(T, T')` for one level, resampled under a new seed until
        `T[j] != T'[j]` for every position and prime -- the condition the
        fold divides by."""
        tag = b"twists" + level.to_bytes(4, "little")
        for attempt in range(_MAX_SAMPLING_ATTEMPTS):
            even = self._sampled(n, tag + b"even", attempt)
            odd = self._sampled(n, tag + b"odd", attempt)
            if all(
                t != u
                for row, row_odd in zip(even, odd, strict=True)
                for t, u in zip(row, row_odd, strict=True)
            ):
                return even, odd
        raise RuntimeError(f"no distinct twist tables for level {level} found")

    def _sample_points(self) -> Table:
        """`n0` evaluation points for the base code, distinct modulo every
        prime, resampled under a new seed until they are."""
        for attempt in range(_MAX_SAMPLING_ATTEMPTS):
            points = self._sampled(self.n0, b"points", attempt)
            if all(len(set(column)) == self.n0 for column in zip(*points, strict=True)):
                return points
        raise RuntimeError(f"no {self.n0} distinct base points found")

    def relative_distance(self, security_bits: int = 128) -> float:
        """A lower bound on the relative minimum distance at every level, per
        RNS-prime component: the `delta` a soundness bound needs, which is a
        property of the code rather than of the protocol, which is why
        `Basefold.soundness_error` takes one instead of deriving it.

        For the ``"rs"`` instantiation every level is a Reed-Solomon code on
        distinct points, so level `l` has distance exactly
        `1 - k_l/n_l + 1/n_l`; the rate is `1/c` throughout, so the longest
        codeword's value is the bound and `security_bits` plays no part.

        For the general code it is [ZCF24, Thm. 2]'s recurrence
        (`foldable_relative_distance`) at the smallest prime, which holds
        with probability at least `1 - d * 2^-security_bits` over the twists.
        """
        if self.instantiation == "rs":
            return 1 - 1 / self.c + 1 / self.n_d
        return foldable_relative_distance(
            self.k0, self.c, self.d, math.log2(min(self.ring.primes)), security_bits
        )

    def __del__(self) -> None:
        # interpreter shutdown may already have torn the lib down
        with contextlib.suppress(Exception):
            for plans in self._plans:
                lib.rs_free_plans(plans, self.ring.rns_rows)

    def level_of(self, message: list) -> int:
        """The code level a message of this length belongs to."""
        level = (len(message) // self.k0).bit_length() - 1
        if self.k0 << level != len(message):
            raise ValueError(
                f"message length {len(message)} is not k0 * 2^l (k0 = {self.k0})"
            )
        if level > self.d:
            raise ValueError(
                f"message length {len(message)} exceeds the level-{self.d} "
                f"dimension {self.k_d}"
            )
        return level

    def _kernel_encode(self, message: list, size: int, plans) -> list:
        """The `rs_encode` kernel: the transform of `message` zero-padded to
        `size`."""
        # The kernel reads the entries' RNS coefficients directly, so they
        # must all be in NTT form (see MLE.to_NTT for why one may not be).
        for p in message:
            p.to_NTT()
        word = [Polynomial(self.ring) for _ in range(size)]
        lib.rs_encode(
            element_array(word), element_array(message), size, len(message), plans
        )
        mark_ntt(word)
        return word

    def _kernel_decode(self, word: list, degree: int, plans) -> tuple[bool, list]:
        """The `rs_decode` kernel: the inverse transform truncated to
        `degree`, and whether what was cut off was zero."""
        for p in word:
            p.to_NTT()
        message = [Polynomial(self.ring) for _ in range(degree)]
        ok = lib.rs_decode(
            element_array(message), element_array(word), len(word), degree, plans
        )
        mark_ntt(message)
        return bool(ok), message

    def encode(self, message: list) -> list:
        """The codeword of a coefficient vector (level inferred from its
        length)."""
        level = self.level_of(message)
        if self.instantiation == "rs":
            return self._kernel_encode(message, self.n0 << level, self._plans[level])
        return self._encode_general(message, level)

    def _encode_general(self, message: list, level: int) -> list:
        """The recursion of the module docstring: the two half-messages
        encoded one level down and recombined with this level's tables."""
        if level == 0:
            return self._base_encode(message)
        even = self._encode_general(message[0::2], level - 1)
        odd = self._encode_general(message[1::2], level - 1)
        word = []
        for a, b, t, u in zip(
            even, odd, self.twists[level - 1], self.twists_odd[level - 1], strict=True
        ):
            word.append(a + b * t)
            word.append(a + b * u)
        return word

    def _base_encode(self, message: list) -> list:
        """The base codeword of a `k0`-entry message: the transform when the
        code has one, else the polynomial's values at `base_points` by
        Horner's scheme."""
        if self._plans:
            return self._kernel_encode(message, self.n0, self._plans[0])
        word = [message[-1]] * self.n0
        for coefficient in reversed(message[:-1]):
            word = [
                w * x + coefficient for w, x in zip(word, self.base_points, strict=True)
            ]
        return word

    def decode(self, word: list) -> tuple[bool, list]:
        """`(is_codeword, message)` for a codeword: the encoding inverted,
        plus the check that rejects vectors outside the code (the level is
        inferred from the length)."""
        size = len(word)
        level = (size // self.n0).bit_length() - 1
        if self.n0 << level != size or level > self.d:
            raise ValueError(f"codeword length {size} is not n0 * 2^l, l <= d")
        if self.instantiation == "rs":
            return self._kernel_decode(word, self.k0 << level, self._plans[level])
        return self._decode_general(word, level)

    def _decode_general(self, word: list, level: int) -> tuple[bool, list]:
        """`_encode_general` inverted: per pair, `b = (lo - hi) / (T - T')`
        and `a = lo - b T` are the codewords of the odd and even
        half-messages, decoded one level down and interleaved back."""
        if level == 0:
            return self._base_decode(word)
        scale, twist = self._fold_scale[level - 1], self.twists[level - 1]
        odd = [
            (word[2 * j] - word[2 * j + 1]) * scale[j] for j in range(len(word) // 2)
        ]
        even = [word[2 * j] - b * twist[j] for j, b in enumerate(odd)]
        ok_even, m_even = self._decode_general(even, level - 1)
        ok_odd, m_odd = self._decode_general(odd, level - 1)
        message: list = [None] * (len(m_even) + len(m_odd))
        message[0::2] = m_even
        message[1::2] = m_odd
        return ok_even and ok_odd, message

    def _base_decode(self, word: list) -> tuple[bool, list]:
        """`_base_encode` inverted: the transform's inverse when the code has
        one, else the message interpolated from the first `k0` positions and
        checked by re-encoding."""
        if self._plans:
            return self._kernel_decode(word, self.k0, self._plans[0])
        message = []
        for row in self._lagrange_inverse():
            total = word[0] * row[0]
            for w, scalar in zip(word[1 : self.k0], row[1:], strict=True):
                total = total + w * scalar
            message.append(total)
        ok = all(a == b for a, b in zip(self._base_encode(message), word, strict=True))
        return ok, message

    def _lagrange_inverse(self) -> list[list[list[int]]]:
        """The inverse of the Vandermonde matrix on the first `k0` base
        points, `[i][j][prime]`: `message[i] = sum_j inverse[i][j] * word[j]`."""
        inverse = self._vandermonde_inverse
        if inverse is None:
            per_prime = [
                vandermonde_inverse(
                    [point[k] for point in self.base_points[: self.k0]],
                    1,
                    0,
                    lambda a, b, p=p: (a + b) % p,
                    lambda a, b, p=p: (a - b) % p,
                    lambda a, b, p=p: a * b % p,
                    lambda a, p=p: pow(a, p - 2, p),
                )
                for k, p in enumerate(self.ring.primes)
            ]
            inverse = self._vandermonde_inverse = [
                [[table[i][j] for table in per_prime] for j in range(self.k0)]
                for i in range(self.k0)
            ]
        return inverse

    def fold_pair(
        self, lo: RNSPolynomial, hi: RNSPolynomial, r: RNSPolynomial, level: int, i: int
    ) -> RNSPolynomial:
        """The folded value at position i of a level-`level` codeword, from
        that position's pair alone: `(lo, hi) = (word[2i], word[2i + 1])`.

        This is the form a Merkle verifier uses — it holds one authenticated
        pair per queried position, never a whole codeword.
        """
        coeff = (lo - hi) * self._fold_scale[level - 1][i]
        return hi + coeff * self._fold_shift[level - 1][i] + r * coeff

    def fold_pairs(self, los, his, r, level: int, indices) -> list:
        """`fold_pair` at many positions at once; see `FieldFoldableRS`, which
        has a vectorised one. Over a ring the entries are polynomials with
        their own arithmetic, so this is the loop a caller would write -- the
        uniform name is what lets a verifier fold a level without knowing
        which code family answers."""
        return [
            self.fold_pair(lo, hi, r, level, i)
            for lo, hi, i in zip(los, his, indices, strict=True)
        ]

    def pair_at(self, word: list, i: int) -> tuple[RNSPolynomial, RNSPolynomial]:
        """Position i's pair, `(word[2i], word[2i + 1])` — the unit the fold
        reads, and the Merkle leaf (see `pair_leaves`)."""
        return word[2 * i], word[2 * i + 1]

    def leaf_digests_of(self, pairs) -> bytes:
        """The leaf digests of `pairs`, packed; see `FieldFoldableRS`, which
        digests them together. Here a pair is digested at a time, as
        `leaf_digests` does for a whole codeword of ring elements."""
        return b"".join(self.leaf_digest(pair) for pair in pairs)

    def pair_leaves(self, word: list) -> list[tuple[RNSPolynomial, RNSPolynomial]]:
        """`word` as the leaf vector its Merkle tree commits to: one leaf per
        adjacent pair, so a single path authenticates both operands of a fold
        check ([ZCF24, Remark 9]'s packed leaves)."""
        return [self.pair_at(word, i) for i in range(len(word) // 2)]

    def leaf_digest(self, pair: tuple[RNSPolynomial, RNSPolynomial]) -> bytes:
        """The Merkle leaf digest of one adjacent pair (`pair_digest`)."""
        return pair_digest(pair)

    def leaf_digests(self, word: list) -> bytes | memoryview:
        """The leaf digests of every adjacent pair of `word`, packed: leaf `i`
        is ``digests[32 * i : 32 * (i + 1)]``.

        This family digests a pair at a time -- the entries are ring elements
        with their own hash -- so the leaves still cost an object each here,
        unlike the field codes, where a kernel writes the whole buffer. What
        the packing saves is the tree builder's half of that: it reads the
        buffer rather than walking a sequence.
        """
        return b"".join(pair_digest(pair) for pair in self.pair_leaves(word))

    def fold_at(
        self, word: list, r: RNSPolynomial, level: int, i: int
    ) -> RNSPolynomial:
        """Position i of the fold of the level-`level` codeword `word` with
        challenge r — the value the folded codeword must hold there. The pair
        is adjacent: `(word[2i], word[2i + 1])`."""
        return self.fold_pair(*self.pair_at(word, i), r, level, i)

    def fold(self, word: list, r: RNSPolynomial, level: int) -> list:
        """The full fold of a level-`level` codeword with challenge r (the
        level-(level-1) codeword of the r-folded message)."""
        return [self.fold_at(word, r, level, i) for i in range(len(word) // 2)]
