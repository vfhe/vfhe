# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Foldable Reed-Solomon codes over R_q for the basefold commitment.

The code is *interleaved*: it acts on a vector of ring elements
coefficient-slot-wise and per RNS prime, so each (prime, coefficient slot)
pair carries an independent RS codeword over Z_p. The transform is arith's
negacyclic NTT of the codeword length, applied by the `rs_*` C kernels
(`polycom/c/src/rscode.c`): with psi the 2n-th root of unity `ntt_new_proc`
picks, position p of a length-n codeword holds P(psi^(2*brv(p)+1)), where
brv reverses the log2(n) index bits — `ntt_forward` is CT_NR, natural in,
bit-reversed out.

That layout is foldable in the basefold sense [ZCF24, Def. 3.2] with every
level an RS code (the FRI folding structure), and the bit-reversal puts the
+/- pairs *adjacent*: positions 2i and 2i+1 hold P(x_i) and P(-x_i) for
x_i = psi^(2*brv(i)+1), so folding with challenge r gives the evaluations of
P_even + r * P_odd at the squared points — which are exactly the half-length
code's points, since `ntt_new_proc` derives psi from the smallest quadratic
non-residue mod p (a choice independent of the length) and therefore
psi_{n/2} = psi_n^2. The per-prime scalars are applied through
`Polynomial * list` (per-RNS-residue scaling), so a "scalar" here is one
integer per RNS prime.

The message is a monomial-basis coefficient vector (LSB-first multilinear
index order — `MLE.to_coefficients`), so folding the codeword with r is
exactly binding the first (LSB) variable of the MLE to r, the same fold the
sumcheck prover applies to the evaluation table.
"""

from __future__ import annotations

import contextlib

from vfhe.arith import Polynomial, Ring
from vfhe.misc.libvfhe import lib
from vfhe.piop.mle import handle_array, mark_ntt

# The shortest transform arith's kernels implement: the vectorized paths are
# guarded on `sub_n >= 16` (one AVX512 lane group per butterfly stage), and
# shorter lengths read past the buffer. Every level of the code is encoded, so
# the *base* length n0 is what has to clear the bar.
_MIN_CODEWORD = 16


def _bit_reverse(i: int, bits: int) -> int:
    """`i` with its low `bits` bits reversed — the index permutation
    `ntt_forward` (CT_NR) leaves in its output."""
    out = 0
    for _ in range(bits):
        out = (out << 1) | (i & 1)
        i >>= 1
    return out


class FoldableRS:
    """A depth-d foldable RS code over `ring`, with base dimension k0 and
    inverse rate c: level l encodes k0 * 2^l ring elements into
    n_l = c * k0 * 2^l. `encode` infers the level from the message length;
    `fold` / `fold_at` implement the verifier-checkable fold
    pi'[i] = pi[2i+1] + (t[i] + r) * (pi[2i] - pi[2i+1]) / (2 t[i])
    taking the level-l codeword of P to the level-(l-1) codeword of
    P_even + r * P_odd.

    The codeword length is bounded by the ring's primes: they satisfy
    p = 1 mod 2N/split_degree, and the negacyclic transform of length n_d
    needs 2*n_d | p - 1, so n_d must divide N/split_degree.
    """

    def __init__(self, ring: Ring, k0: int, c: int, d: int):
        for name, value in (("k0", k0), ("c", c)):
            if value < 1 or value & (value - 1):
                raise ValueError(f"{name} must be a power of two, got {value}")
        if d < 1:
            raise ValueError(f"d must be at least 1, got {d}")
        self.ring = ring
        self.k0 = k0
        self.c = c
        self.d = d
        self.n0 = c * k0
        self.k_d = k0 << d
        self.n_d = self.n0 << d
        limit = ring.N // ring.split_degree
        if self.n_d > limit:
            raise ValueError(
                f"codeword length {self.n_d} exceeds N/split_degree = {limit}: "
                "the ring's primes carry no root of unity of order 2 * n_d"
            )
        if self.n0 < _MIN_CODEWORD:
            raise ValueError(
                f"base codeword length n0 = c * k0 = {self.n0} is below "
                f"{_MIN_CODEWORD}: arith's NTT kernels vectorize over that many "
                "lanes and do not support shorter transforms"
            )

        # One NTT-processor array per level (indexed by global RNS prime
        # index, matching RNS_Polynomial.coeffs), owned by this object.
        # `rs_new_procs` sizes each array by the NTT's prime count *at this
        # moment*, and that count is not stable: the incNTT of an
        # (N, split_degree) pair is shared process-wide, so a later ring
        # adding a prime extends it in place and `ring._ntt_l()` grows under
        # us. The length is therefore recorded here and used to free — never
        # re-read from the ring, which would free past the end of the array.
        self._procs_l = ring._ntt_l()
        self._procs = [
            lib.rs_new_procs(ring.NTT, ring.mask, self.n0 << level)
            for level in range(d + 1)
        ]
        # roots[l][k] = psi for level l and the k-th active prime: the
        # 2*n_l-th root the level-l transform uses. Read back from the procs
        # so the twists below cannot drift from the kernels' convention.
        self.roots = [
            [lib.rs_procs_root(procs, idx) for idx in ring.prime_indices]
            for procs in self._procs
        ]
        # twists[l][i] = x_i = psi_{l+1}^(2*brv(i)+1), the evaluation point of
        # the pair (2i, 2i+1) folding level l+1 -> l, and twists2_inv their
        # (2 x_i)^-1 — per prime, so `Polynomial * list` applies them.
        self.twists: list[list[list[int]]] = []
        self.twists2_inv: list[list[list[int]]] = []
        for level in range(d):
            n = self.n0 << level  # positions of the folded (level) codeword
            bits = n.bit_length() - 1
            roots = self.roots[level + 1]
            row, row_inv = [], []
            for i in range(n):
                exponent = 2 * _bit_reverse(i, bits) + 1
                t = [
                    pow(psi, exponent, p)
                    for psi, p in zip(roots, ring.primes, strict=True)
                ]
                row.append(t)
                row_inv.append(
                    [
                        pow(2 * tj, p - 2, p)
                        for tj, p in zip(t, ring.primes, strict=True)
                    ]
                )
            self.twists.append(row)
            self.twists2_inv.append(row_inv)

    def __del__(self) -> None:
        # interpreter shutdown may already have torn the lib down
        with contextlib.suppress(Exception):
            for procs in self._procs:
                lib.rs_free_procs(procs, self._procs_l)

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

    def encode(self, message: list) -> list:
        """The codeword of a coefficient vector (level inferred from its
        length), computed by the `rs_encode` kernel."""
        level = self.level_of(message)
        size = self.n0 << level
        # The kernel reads the entries' RNS coefficients directly, so they
        # must all be in NTT form (see MLE.to_NTT for why one may not be).
        for p in message:
            p.to_NTT()
        word = [Polynomial(self.ring) for _ in range(size)]
        lib.rs_encode(
            handle_array(word),
            handle_array(message),
            size,
            len(message),
            self._procs[level],
        )
        mark_ntt(word)
        return word

    def decode(self, word: list) -> tuple[bool, list]:
        """`(is_codeword, message)` for a codeword, via the `rs_decode`
        kernel: the inverse transform plus the degree check that rejects
        vectors outside the code (the level is inferred from the length)."""
        size = len(word)
        level = (size // self.n0).bit_length() - 1
        if self.n0 << level != size or level > self.d:
            raise ValueError(f"codeword length {size} is not n0 * 2^l, l <= d")
        for p in word:
            p.to_NTT()
        degree = self.k0 << level
        message = [Polynomial(self.ring) for _ in range(degree)]
        ok = lib.rs_decode(
            handle_array(message),
            handle_array(word),
            size,
            degree,
            self._procs[level],
        )
        mark_ntt(message)
        return bool(ok), message

    def fold_pair(
        self, lo: Polynomial, hi: Polynomial, r: Polynomial, level: int, i: int
    ) -> Polynomial:
        """The folded value at position i of a level-`level` codeword, from
        that position's pair alone: `(lo, hi) = (P(x_i), P(-x_i))`.

        This is the form a Merkle verifier uses — it holds one authenticated
        pair per queried position, never a whole codeword.
        """
        coeff = (lo - hi) * self.twists2_inv[level - 1][i]
        return hi + coeff * self.twists[level - 1][i] + r * coeff

    def pair_at(self, word: list, i: int) -> tuple[Polynomial, Polynomial]:
        """Position i's `±x` pair, `(word[2i], word[2i + 1])` — the unit the
        fold reads, and the Merkle leaf (see `pair_leaves`)."""
        return word[2 * i], word[2 * i + 1]

    def pair_leaves(self, word: list) -> list[tuple[Polynomial, Polynomial]]:
        """`word` as the leaf vector its Merkle tree commits to: one leaf per
        `±x` pair, so a single path authenticates both operands of a fold
        check ([ZCF24, Remark 9]'s packed leaves)."""
        return [self.pair_at(word, i) for i in range(len(word) // 2)]

    def fold_at(self, word: list, r: Polynomial, level: int, i: int) -> Polynomial:
        """Position i of the fold of the level-`level` codeword `word` with
        challenge r — the value the folded codeword must hold there. The pair
        is adjacent: word[2i] = P(x), word[2i + 1] = P(-x)."""
        return self.fold_pair(*self.pair_at(word, i), r, level, i)

    def fold(self, word: list, r: Polynomial, level: int) -> list:
        """The full fold of a level-`level` codeword with challenge r (the
        level-(level-1) codeword of the r-folded message)."""
        return [self.fold_at(word, r, level, i) for i in range(len(word) // 2)]
