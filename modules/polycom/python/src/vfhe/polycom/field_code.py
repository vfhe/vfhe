# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Foldable Reed-Solomon codes over a finite field for the basefold commitment.

The field counterpart of `code.FoldableRS`, with the same interface, so
`Basefold` runs over either. A codeword is a `vfhe.arith.FieldVector`, and
every whole-codeword operation (encoding, folding, leaf hashing) is a fixed
number of vector operations rather than a loop over entries.

The transform is arith's negacyclic NTT of the codeword length over the
field's prime, applied to each coefficient plane of the vector by the
`rs_field_*` kernels (`polycom/c/src/rscode_field.c`): the evaluation points
lie in F_p, so encoding is F_p-linear and the extension structure never
enters. The evaluation-point layout, and therefore the fold, is exactly the
one `code.FoldableRS` documents: position p holds P(psi^(2*brv(p)+1)), the
+/- pairs are adjacent, and psi_{n/2} = psi_n^2 across levels. The twists
are the same integers, now lifted to field elements (and held once more as
vectors, for folding a whole codeword).

Only `ExtensionField` is served: arith offers no transform over its
pseudo-Mersenne field yet, so constructing a code over one raises.
"""

from __future__ import annotations

import contextlib

from vfhe.arith import ExtensionField, Field, FieldVector
from vfhe.engine import ffi, lib

from .code import _bit_reverse


class FieldFoldableRS:
    """A depth-d foldable RS code over `field`, with base dimension k0 and
    inverse rate c: level l encodes k0 * 2^l field elements into
    n_l = c * k0 * 2^l. `encode` infers the level from the message length;
    `fold` / `fold_at` implement the verifier-checkable fold
    pi'[i] = pi[2i+1] + (t[i] + r) * (pi[2i] - pi[2i+1]) / (2 t[i])
    taking the level-l codeword of P to the level-(l-1) codeword of
    P_even + r * P_odd.

    The codeword length is bounded by the prime's 2-adicity: the negacyclic
    transform of length n_d needs 2 * n_d | p - 1, i.e. log2(n_d) + 1 at most
    `field.two_adicity`.
    """

    def __init__(self, field: Field, k0: int, c: int, d: int):
        for name, value in (("k0", k0), ("c", c)):
            if value < 1 or value & (value - 1):
                raise ValueError(f"{name} must be a power of two, got {value}")
        if d < 1:
            raise ValueError(f"d must be at least 1, got {d}")
        if not isinstance(field, ExtensionField):
            raise NotImplementedError(
                f"no Reed-Solomon transform over {type(field).__name__}: arith "
                "provides an NTT for ExtensionField only"
            )
        self.field = field
        self.k0 = k0
        self.c = c
        self.d = d
        self.n0 = c * k0
        self.k_d = k0 << d
        self.n_d = self.n0 << d
        if self.n_d.bit_length() > field.two_adicity:
            raise ValueError(
                f"codeword length {self.n_d} needs a root of unity of order "
                f"{2 * self.n_d}, but p - 1 has 2-adicity {field.two_adicity}"
            )
        # One plan per level, over the field's modulus (borrowed, so the
        # field must outlive this object).
        self._plans = []
        for level in range(d + 1):
            plan = lib.rs_field_new_plan(self.n0 << level, field.mod)
            if plan == ffi.NULL:
                raise RuntimeError(
                    f"arith rejected a transform of length {self.n0 << level}"
                )
            self._plans.append(plan)
        # roots[l] = psi, the 2*n_l-th root the level-l transform uses, read
        # back from the plan so the twists cannot drift from the kernel.
        self.roots = [lib.rs_field_plan_root(plan) for plan in self._plans]
        # twists[l][i] = x_i = psi_{l+1}^(2*brv(i)+1), the evaluation point of
        # the pair (2i, 2i+1) folding level l+1 -> l, and twists2_inv their
        # (2 x_i)^-1, as integers mod p; `_twist_*` hold them as field
        # elements (for one pair) and as vectors (for a whole codeword).
        p = field.prime
        self.twists: list[list[int]] = []
        self.twists2_inv: list[list[int]] = []
        for level in range(d):
            n = self.n0 << level  # positions of the folded (level) codeword
            bits = n.bit_length() - 1
            psi = self.roots[level + 1]
            row = [pow(psi, 2 * _bit_reverse(i, bits) + 1, p) for i in range(n)]
            self.twists.append(row)
            self.twists2_inv.append([pow(2 * t, p - 2, p) for t in row])
        self._twist_elements = [[self._element(t) for t in row] for row in self.twists]
        self._twist2_inv_elements = [
            [self._element(t) for t in row] for row in self.twists2_inv
        ]
        self._twist_vectors = [FieldVector(field, row) for row in self.twists]
        self._twist2_inv_vectors = [FieldVector(field, row) for row in self.twists2_inv]

    def __del__(self) -> None:
        # interpreter shutdown may already have torn the lib down
        with contextlib.suppress(Exception):
            for plan in self._plans:
                lib.rs_field_free_plan(plan)

    def _element(self, value: int):
        """`value` as an element of the field (a constant of F_p)."""
        return type(self.field.one)(self.field, value)

    def _vector(self, values) -> FieldVector:
        """`values` as a vector over the field, adopted when it is one."""
        if isinstance(values, FieldVector):
            if values.field is not self.field:
                raise ValueError("vector belongs to a different field")
            return values
        return FieldVector(self.field, list(values))

    def level_of(self, message) -> int:
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

    def encode(self, message) -> FieldVector:
        """The codeword of a coefficient vector (level inferred from its
        length), computed by the `rs_field_encode` kernel."""
        message = self._vector(message)
        level = self.level_of(message)
        word = FieldVector(self.field, self.n0 << level)
        lib.rs_field_encode(word._struct, message._struct, self._plans[level])
        return word

    def decode(self, word) -> tuple[bool, FieldVector]:
        """`(is_codeword, message)` for a codeword, via the `rs_field_decode`
        kernel: the inverse transform plus the degree check that rejects
        vectors outside the code (the level is inferred from the length)."""
        word = self._vector(word)
        size = len(word)
        level = (size // self.n0).bit_length() - 1
        if self.n0 << level != size or level > self.d:
            raise ValueError(f"codeword length {size} is not n0 * 2^l, l <= d")
        message = FieldVector(self.field, self.k0 << level)
        ok = lib.rs_field_decode(message._struct, word._struct, self._plans[level])
        return bool(ok), message

    def fold_pair(self, lo, hi, r, level: int, i: int):
        """The folded value at position i of a level-`level` codeword, from
        that position's pair alone: `(lo, hi) = (P(x_i), P(-x_i))`.

        This is the form a Merkle verifier uses — it holds one authenticated
        pair per queried position, never a whole codeword.
        """
        coeff = (lo - hi) * self._twist2_inv_elements[level - 1][i]
        return hi + coeff * self._twist_elements[level - 1][i] + coeff * r

    def pair_at(self, word: FieldVector, i: int) -> tuple:
        """Position i's `±x` pair, `(word[2i], word[2i + 1])` — the unit the
        fold reads, and the Merkle leaf (see `leaf_digest`)."""
        return word[2 * i], word[2 * i + 1]

    def pair_leaves(self, word: FieldVector) -> list[tuple]:
        """`word` as the list of `±x` pairs its Merkle tree commits to."""
        return [self.pair_at(word, i) for i in range(len(word) // 2)]

    def leaf_digest(self, pair: tuple) -> bytes:
        """The Merkle leaf digest of one `±x` pair: the vector digest of the
        two elements in order, so it equals the matching entry of
        `leaf_digests` and binds the pair as an ordered unit."""
        return FieldVector(self.field, list(pair)).hash()

    def leaf_digests(self, word: FieldVector) -> list[bytes]:
        """The leaf digests of every `±x` pair of `word`, in one pass over
        the codeword (`FieldVector.hash_elements` with adjacent windows)."""
        return word.hash_elements(group=2, stride=2)

    def fold_at(self, word: FieldVector, r, level: int, i: int):
        """Position i of the fold of the level-`level` codeword `word` with
        challenge r — the value the folded codeword must hold there."""
        return self.fold_pair(*self.pair_at(word, i), r, level, i)

    def fold(self, word: FieldVector, r, level: int) -> FieldVector:
        """The full fold of a level-`level` codeword with challenge r (the
        level-(level-1) codeword of the r-folded message), as whole-vector
        operations over the even (`P(x_i)`) and odd (`P(-x_i)`) halves."""
        lo, hi = word.split_even_odd()
        coeff = (lo - hi) * self._twist2_inv_vectors[level - 1]
        return hi + coeff * self._twist_vectors[level - 1] + coeff * r
