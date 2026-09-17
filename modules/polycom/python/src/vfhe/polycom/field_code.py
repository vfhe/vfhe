# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Foldable Reed-Solomon codes over a finite field for the basefold commitment.

The field counterpart of `code.FoldableRS`, with the same interface, so
`Basefold` runs over either. A codeword is a `vfhe.arith.FieldVector`, and
every whole-codeword operation (encoding, folding, leaf hashing) is a fixed
number of vector operations rather than a loop over entries.

The transform is a negacyclic NTT of the codeword length, one per level,
provided by whichever of `_ExtensionTransforms` / `_PseudoMersenneTransforms`
the field selects: over an extension field the `rs_field_*` kernels
(`polycom/c/src/rscode_field.c`) run arith's NTT over the field's modulus once
per coefficient plane -- the evaluation points lie in F_p, so encoding is
F_p-linear and the extension never enters -- and over a pseudo-Mersenne field
arith's own `PseudoMersenneNTT` transforms the vector whole. Both share the
basis and output order `code.FoldableRS` documents: position p holds
P(psi^(2*brv(p)+1)), the +/- pairs are adjacent, and psi_{n/2} = psi_n^2 across
levels. The twists are the same integers, now lifted to field elements (and
held once more as vectors, for folding a whole codeword).
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, cast

from vfhe.arith import (
    ExtensionField,
    Field,
    FieldVector,
    PseudoMersenneField,
    PseudoMersenneVector,
)
from vfhe.engine import ffi, lib

from .code import bit_reverse_permutation

if TYPE_CHECKING:
    from vfhe.arith import FieldElement


def _pmf_vector(vector: FieldVector) -> PseudoMersenneVector:
    """`vector` as the pseudo-Mersenne vector it is over such a field.

    The transforms below take the field's own plans, which only accept its
    own vectors; a code is built over one field, so every codeword reaching
    them is one. Stated here rather than at each call.
    """
    return cast("PseudoMersenneVector", vector)


class _ExtensionTransforms:
    """The level transforms of a code over an extension field: one arith
    `NTT_Plan` per level, driven by the `rs_field_*` kernels.

    The plans borrow the field's modulus, so the field must outlive this
    object.
    """

    def __init__(self, field: ExtensionField, lengths: list[int]):
        self.field = field
        self.lengths = lengths
        self._plans = []
        for size in lengths:
            plan = lib.rs_field_new_plan(size, field.mod)
            if plan == ffi.NULL:
                raise RuntimeError(f"arith rejected a transform of length {size}")
            self._plans.append(plan)

    def __del__(self) -> None:
        # interpreter shutdown may already have torn the lib down
        with contextlib.suppress(Exception):
            for plan in self._plans:
                lib.rs_field_free_plan(plan)

    def root(self, level: int) -> int:
        """psi, the 2*n_level-th root the level's transform evaluates at."""
        return lib.rs_field_plan_root(self._plans[level])

    def points(self, root: int, n: int) -> FieldVector:
        """The `n` evaluation points of a transform at `root`, in its output
        order -- arith's running product, not `n` exponentiations."""
        return self.field.ntt_points(root, n)

    def encode(self, message: FieldVector, level: int) -> FieldVector:
        """The forward transform of `message`, zero-padded to the level's
        codeword length by the kernel."""
        word = FieldVector(self.field, self.lengths[level])
        lib.rs_field_encode(word._struct, message._struct, self._plans[level])
        return word

    def decode(
        self, word: FieldVector, level: int, degree: int
    ) -> tuple[bool, FieldVector]:
        """The inverse transform of `word`, truncated to `degree`, and whether
        what was cut off was zero (the degree check)."""
        message = FieldVector(self.field, degree)
        ok = lib.rs_field_decode(message._struct, word._struct, self._plans[level])
        return bool(ok), message


class _PseudoMersenneTransforms:
    """The level transforms of a code over a pseudo-Mersenne field: the
    field's own `ntt_plan` per level, applied to the codeword as a whole.

    The plans are the field's (memoized there), so several codes over one
    field share them.
    """

    def __init__(self, field: PseudoMersenneField, lengths: list[int]):
        self.field = field
        self.lengths = lengths
        self._plans = [field.ntt_plan(size) for size in lengths]

    def root(self, level: int) -> int:
        """psi, the 2*n_level-th root the level's transform evaluates at."""
        return int(self._plans[level].root_of_unity)

    def points(self, root: int, n: int) -> FieldVector:
        """The `n` evaluation points of a transform at `root`, in its output
        order.

        The same running product as the extension provider's, written here in
        Python: this field has no kernel for it, and its codes are short --
        the 2-adicity that bounds a codeword bounds this table with it.
        """
        prime = self.field.prime
        reversed_index = bit_reverse_permutation(n.bit_length() - 1)
        row = [0] * n
        step = root * root % prime
        power = root % prime
        for j in range(n):
            row[reversed_index[j]] = power
            power = power * step % prime
        return FieldVector(self.field, row)

    def encode(self, message: FieldVector, level: int) -> FieldVector:
        """The forward transform of `message` zero-padded to the level's
        codeword length. The padding leaves the operand untouched, so the
        transform runs in place on the padded copy."""
        padding = FieldVector(self.field, self.lengths[level] - len(message))
        padded = type(message).concat([message, padding])
        return self._plans[level].forward(_pmf_vector(padded), in_place=True)

    def decode(
        self, word: FieldVector, level: int, degree: int
    ) -> tuple[bool, FieldVector]:
        """The inverse transform of `word`, truncated to `degree`, and whether
        what was cut off was zero (the degree check)."""
        coefficients = self._plans[level].inverse(_pmf_vector(word))
        size = len(coefficients)
        message = coefficients.query(range(degree))
        tail = coefficients.query(range(degree, size))
        return tail == FieldVector(self.field, size - degree), message


def _transforms(field: Field, lengths: list[int]):
    """The transform provider for `field`, or a NotImplementedError naming the
    implementations that have one."""
    if isinstance(field, ExtensionField):
        return _ExtensionTransforms(field, lengths)
    if isinstance(field, PseudoMersenneField):
        return _PseudoMersenneTransforms(field, lengths)
    raise NotImplementedError(
        f"no Reed-Solomon transform over {type(field).__name__}: arith provides "
        "an NTT for ExtensionField and PseudoMersenneField only"
    )


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
        # One transform per level, over the field's prime.
        self._transforms = _transforms(
            field, [self.n0 << level for level in range(d + 1)]
        )
        # roots[l] = psi, the 2*n_l-th root the level-l transform uses, read
        # back from the plan so the twists cannot drift from the kernel.
        self.roots = [self._transforms.root(level) for level in range(d + 1)]
        # twists[l][i] = x_i = psi_{l+1}^(2*brv(i)+1), the evaluation point of
        # the pair (2i, 2i+1) folding level l+1 -> l, and twists2_inv their
        # (2 x_i)^-1, as integers mod p; `_twist_*` hold them as field
        # elements (for one pair) and as vectors (for a whole codeword).
        # x_i and (2 x_i)^-1 per level, as vectors and nothing else. The tables
        # are Theta(n) entries, so anything per entry -- an exponentiation, an
        # inversion, a Python object -- is Theta(n) interpreted work and used to
        # dominate a large instance. The points come from the transform's own
        # running product, and the inverses from one batch inversion.
        self._twist_vectors: list[FieldVector] = []
        self._twist2_inv_vectors: list[FieldVector] = []
        for level in range(d):
            n = self.n0 << level  # positions of the folded (level) codeword
            vector = self._transforms.points(self.roots[level + 1], n)
            self._twist_vectors.append(vector)
            # Montgomery's trick: one inversion and three multiplications per
            # entry, in place of n Fermat exponentiations.
            self._twist2_inv_vectors.append(vector.scale(2).inverse())
        self._twist_ints: list[list[list[int]] | None] = [None, None]

    def _as_ints(self, which: int) -> list[list[int]]:
        """One of the twist tables as integers, materialized on first use.

        The folds read these as vectors; only a caller that wants the integers
        pays for n Python objects per level.
        """
        cached = self._twist_ints[which]
        if cached is None:
            vectors = (self._twist_vectors, self._twist2_inv_vectors)[which]
            cached = [[int(v) for v in row] for row in vectors]
            self._twist_ints[which] = cached
        return cached

    @property
    def twists(self) -> list[list[int]]:
        """``x_i = psi_{l+1}^(2 brv(i) + 1)`` per level, as integers."""
        return self._as_ints(0)

    @property
    def twists2_inv(self) -> list[list[int]]:
        """``(2 x_i)^-1`` per level, as integers."""
        return self._as_ints(1)

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
        length): the level's transform of the zero-padded message."""
        message = self._vector(message)
        return self._transforms.encode(message, self.level_of(message))

    def decode(self, word) -> tuple[bool, FieldVector]:
        """`(is_codeword, message)` for a codeword: the level's inverse
        transform plus the degree check that rejects vectors outside the code
        (the level is inferred from the length)."""
        word = self._vector(word)
        size = len(word)
        level = (size // self.n0).bit_length() - 1
        if self.n0 << level != size or level > self.d:
            raise ValueError(f"codeword length {size} is not n0 * 2^l, l <= d")
        return self._transforms.decode(word, level, self.k0 << level)

    def fold_pair(self, lo, hi, r, level: int, i: int):
        """The folded value at position i of a level-`level` codeword, from
        that position's pair alone: `(lo, hi) = (P(x_i), P(-x_i))`.

        This is the form a Merkle verifier uses — it holds one authenticated
        pair per queried position, never a whole codeword.
        """
        coeff = (lo - hi) * self._twist2_inv_vectors[level - 1][i]
        return hi + coeff * self._twist_vectors[level - 1][i] + coeff * r

    def fold_pairs(self, los, his, r, level: int, indices) -> list:
        """`fold_pair` at many positions at once.

        `los[k]`, `his[k]` are the pair at `indices[k]`, and the result is
        that position's folded value -- exactly what `fold_pair` returns for
        it, one entry per position.

        A verifier holds one pair per queried position and folds it at every
        level, so the scalar form costs a handful of element operations per
        (query, level), each its own crossing into C. Here a level's positions
        are one set of whole-vector operations instead, and the twist tables
        are read with one gather rather than an index at a time.
        """
        positions = list(indices)
        twist = self._twist_vectors[level - 1].query(positions)
        coeff = FieldVector(self.field, list(los))
        highs = FieldVector(self.field, list(his))
        coeff.sub(highs, out=coeff)
        coeff.mul(self._twist2_inv_vectors[level - 1].query(positions), out=coeff)
        # `twist` is this call's own, so it can hold the running result.
        folded = highs.fma(coeff, twist, out=twist)
        return folded.fma(coeff, r, out=folded).to_list()

    def pair_at(self, word: FieldVector, i: int) -> tuple[FieldElement, FieldElement]:
        """Position i's `±x` pair, `(word[2i], word[2i + 1])` — the unit the
        fold reads, and the Merkle leaf (see `leaf_digest`)."""
        return word[2 * i], word[2 * i + 1]

    def leaf_digests_of(self, pairs) -> memoryview:
        """The leaf digests of `pairs`, packed -- `leaf_digest` for a set of
        pairs rather than one.

        The pairs go into one vector and are digested with the adjacent
        windows `leaf_digests` uses on a codeword, so entry `k` is
        `leaf_digest(pairs[k])` and the lane-parallel hashing applies to a
        handful of pairs as much as to a whole codeword.
        """
        flat = FieldVector(self.field, [e for pair in pairs for e in pair])
        return flat.hash_elements(group=2, stride=2)

    def pair_leaves(self, word: FieldVector) -> list[tuple]:
        """`word` as the list of `±x` pairs its Merkle tree commits to."""
        return [self.pair_at(word, i) for i in range(len(word) // 2)]

    def leaf_digest(self, pair: tuple) -> bytes:
        """The Merkle leaf digest of one `±x` pair: the vector digest of the
        two elements in order, so it equals the matching entry of
        `leaf_digests` and binds the pair as an ordered unit.

        A verifier recomputes this once per query, where the vector it used
        to build was most of the cost and none of the hashing."""
        return self.field.digest_elements(pair)

    def leaf_digests(self, word: FieldVector) -> memoryview:
        """The leaf digests of every `±x` pair of `word`, packed, in one pass
        over the codeword (`FieldVector.hash_elements` with adjacent
        windows). Leaf `i` is ``digests[32 * i : 32 * (i + 1)]``, and equals
        `leaf_digest` of `pair_at(word, i)`.

        A read-only view of the kernel's buffer, not a copy of it -- so a
        codeword reaches its root without its digests being duplicated on the
        way. `bytes(...)` is how a caller takes a copy."""
        return word.hash_elements(group=2, stride=2)

    def fold_at(self, word: FieldVector, r, level: int, i: int):
        """Position i of the fold of the level-`level` codeword `word` with
        challenge r — the value the folded codeword must hold there."""
        return self.fold_pair(*self.pair_at(word, i), r, level, i)

    def fold(self, word: FieldVector, r, level: int) -> FieldVector:
        """The full fold of a level-`level` codeword with challenge r: the
        level-(level-1) codeword of the r-folded message."""
        return word.fold_twisted(
            self._twist2_inv_vectors[level - 1], self._twist_vectors[level - 1], r
        )
