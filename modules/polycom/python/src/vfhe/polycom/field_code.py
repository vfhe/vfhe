# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Foldable codes over a finite field for the basefold commitment.

The field counterpart of `code.FoldableRS`, with the same interface, so
`Basefold` runs over either. A codeword is a `vfhe.arith.FieldVector`, and
every whole-codeword operation (encoding, folding, leaf hashing) is a fixed
number of vector operations rather than a loop over entries.

Two instantiations of the foldable family [ZCF24, Def. 5] share the class;
`code.FoldableRS` describes both and the fold they share. The default,
``"general"``, draws the twist table `T` of every level from a seed (with
`T' = -T`, as [ZCF24]'s random foldable code does) and needs nothing of the
field beyond a base code, so any field serves at any codeword length: level
`l` is `FieldVector.fma_interleave` of the two half-messages' level-(l-1)
codewords with `T` and `-T`, bottoming out in a
Reed-Solomon code on `n0` points. The base encoder is arith's negacyclic NTT
where the field has a `2 n0`-th root of unity -- `_ExtensionTransforms` runs
the `rs_field_*` kernels (`polycom/c/src/rscode_field.c`) over the field's
prime once per coefficient plane, `_PseudoMersenneTransforms` arith's own
`PseudoMersenneNTT` on the vector whole -- and a Vandermonde product on
seeded distinct points otherwise; the two are encoders of one code, so a
codeword does not depend on which one produced it. Above the base there is
no transform.

The ``"rs"`` instantiation makes every level a Reed-Solomon code on roots of
unity: one transform per level, from the same providers, in the basis and
output order `code.FoldableRS` documents (position p holds
P(psi^(2*brv(p)+1)), so the pairs are `(P(x), P(-x))` and the twists are
`x` and `-x`). It needs `2 n_d | p - 1`.
"""

from __future__ import annotations

import contextlib
import math
import operator
from typing import TYPE_CHECKING, cast

from vfhe.arith import (
    ExtensionField,
    Field,
    FieldVector,
    PseudoMersenneField,
    PseudoMersenneVector,
)
from vfhe.engine import ffi, lib

from .code import (
    DEFAULT_SEED,
    bit_reverse_permutation,
    check_instantiation,
    derive_seed,
    foldable_relative_distance,
    vandermonde_inverse,
)

if TYPE_CHECKING:
    from vfhe.arith import FieldElement

# How many seeds are tried for a table before giving up: a retry happens with
# probability about `n / |F|` per table, so a second one is already unlikely.
_MAX_SAMPLING_ATTEMPTS = 64


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


def _tile(vector: FieldVector, copies: int) -> FieldVector:
    """`copies` copies of `vector` end to end."""
    return FieldVector.concat([vector] * copies) if copies > 1 else vector


def _repeat_each(vector: FieldVector, copies: int) -> FieldVector:
    """Every element of `vector` repeated `copies` times in place: the
    vector read as a column and written as that many equal columns.

    `copies` is a power of two, so doubling by `interleave` gets there in
    `log2(copies)` passes whose lengths sum to twice the result's.
    """
    out = vector
    target = len(vector) * copies
    while len(out) < target:
        out = FieldVector.interleave(out, out)
    return out


def _window(vector: FieldVector, start: int, length: int) -> FieldVector:
    """`vector[start : start + length]`, sharing the buffer where the
    kernels' alignment allows a view and copied where it does not."""
    unit = vector.padding_unit
    if start % unit == 0 and (length % unit == 0 or start + length == len(vector)):
        return vector.view(start, length)
    return vector.query(range(start, start + length))


def _halves(vector: FieldVector) -> tuple[FieldVector, FieldVector]:
    """The first and second half of `vector`, as views where aligned."""
    half = len(vector) // 2
    if half % vector.padding_unit == 0:
        return vector.view(0, half), vector.view(half, half)
    return vector.split_even_odd(block=half)


def _columns(matrix: FieldVector, width: int) -> list[FieldVector]:
    """The columns of `matrix`, a row-major table `width` elements wide, in
    column order: `log2(width)` deinterleaving passes, each halving the
    width of every table it is applied to."""
    if width == 1:
        return [matrix]
    even, odd = matrix.split_even_odd()
    evens = _columns(even, width // 2)
    odds = _columns(odd, width // 2)
    return [column for pair in zip(evens, odds, strict=True) for column in pair]


class FieldFoldableRS:
    """A depth-d foldable code over `field`, with base dimension k0 and
    inverse rate c: level l encodes k0 * 2^l field elements into
    n_l = c * k0 * 2^l. `encode` infers the level from the message length;
    `fold` / `fold_at` implement the verifier-checkable fold that takes the
    level-l codeword of P to the level-(l-1) codeword of P_even + r * P_odd
    (see `code.FoldableRS` for the construction and the tables).

    `instantiation` selects the code (the module docstring describes both):

    - ``"general"`` (the default): seeded twists above a Reed-Solomon base
      code, [ZCF24]'s construction. Any field, any length; the base encoder
      is a transform when the field has a `2 n0`-th root of unity and a
      Vandermonde product otherwise.
    - ``"rs"``: every level a Reed-Solomon code on roots of unity, encoded
      by one transform per level. Needs `2 n_d | p - 1`, i.e.
      ``log2(n_d) + 1 <= field.two_adicity``; faster, and with the exact
      distance `relative_distance` reports for it.

    `seed` is the public parameter every table of the general code is
    derived from -- two parties building the code from the same arguments
    hold the same code. It is ignored by the ``"rs"`` instantiation.
    """

    def __init__(
        self,
        field: Field,
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
        self.field = field
        self.k0 = k0
        self.c = c
        self.d = d
        self.n0 = c * k0
        self.k_d = k0 << d
        self.n_d = self.n0 << d
        # The two tables of every fold level, as vectors over the codeword
        # positions they act on: `_twist_vectors[l]` is `T` of the fold from
        # level l+1 to level l (the encoder multiplies by `T` and `-T`, and
        # the fold's `-T'` is `T` again), `_fold_scale_vectors[l]` the
        # `1 / (2 T)` the fold reads. Theta(n) entries in all, so nothing here
        # is built an entry at a time.
        self._twist_vectors: list[FieldVector] = []
        self._fold_scale_vectors: list[FieldVector] = []
        #: The roots of the transforms in use: one per level for the ``"rs"``
        #: instantiation, the base transform's alone for the general code
        #: (empty when its base is a Vandermonde product).
        self.roots: list[int] = []
        self._transforms: _ExtensionTransforms | _PseudoMersenneTransforms | None = None
        if self.instantiation == "rs":
            self._init_rs()
        else:
            self._init_general()
        self._twist_elements: list[list[list[FieldElement]] | None] = [None, None]
        self._vandermonde_inverse: list[list[FieldElement]] | None = None

    def _init_rs(self) -> None:
        """Every level a transform; twists `x` and `-x` from its points."""
        if self.n_d.bit_length() > self.field.two_adicity:
            raise ValueError(
                f"codeword length {self.n_d} needs a root of unity of order "
                f"{2 * self.n_d}, but p - 1 has 2-adicity {self.field.two_adicity}"
            )
        self._transforms = _transforms(
            self.field, [self.n0 << level for level in range(self.d + 1)]
        )
        # Read back from the plans so the twists cannot drift from the kernel.
        self.roots = [self._transforms.root(level) for level in range(self.d + 1)]
        self.base_points = self._transforms.points(self.roots[0], self.n0)
        for level in range(self.d):
            n = self.n0 << level  # positions of the folded (level) codeword
            # x_i = psi_{l+1}^(2 brv(i) + 1) from the transform's own running
            # product, and (2x)^-1 by one batch inversion (Montgomery's trick).
            x = self._transforms.points(self.roots[level + 1], n)
            self._twist_vectors.append(x)
            self._fold_scale_vectors.append(x.scale(2).inverse())

    def _level_transforms(self):
        """The per-level transforms of the ``"rs"`` instantiation."""
        if self._transforms is None:
            raise RuntimeError("only the 'rs' instantiation transforms every level")
        return self._transforms

    def _init_general(self) -> None:
        """A base code, encoded by a transform when the field has one of
        length `n0`, and seeded twists above it."""
        if self.n0.bit_length() <= self.field.two_adicity:
            with contextlib.suppress(NotImplementedError):
                self._transforms = _transforms(self.field, [self.n0])
        if self._transforms is not None:
            self.roots = [self._transforms.root(0)]
            self.base_points = self._transforms.points(self.roots[0], self.n0)
        else:
            self.base_points = self._sample_points()
        for level in range(self.d):
            twist, scale = self._sample_twists(level, self.n0 << level)
            self._twist_vectors.append(twist)
            self._fold_scale_vectors.append(scale)

    def _sampled(self, n: int, tag: bytes, attempt: int) -> FieldVector:
        """`n` uniform elements, a pure function of the code's seed, `tag`
        and `attempt`."""
        vector = FieldVector(self.field, n)
        vector.sample_random(derive_seed(self.seed, tag, attempt))
        return vector

    def _sample_twists(self, level: int, n: int) -> tuple[FieldVector, FieldVector]:
        """`(T, 1 / (2 T))` for one level: `T` uniform, resampled under a new
        seed until no entry is zero -- the condition `T != -T` the fold
        divides by -- and the inverses in one batch (Montgomery's trick)."""
        tag = b"twists" + level.to_bytes(4, "little")
        for attempt in range(_MAX_SAMPLING_ATTEMPTS):
            twist = self._sampled(n, tag, attempt)
            try:
                scale = twist.scale(2).inverse()
            except ValueError:
                continue
            return twist, scale
        raise RuntimeError(f"no nonzero twist table for level {level} found")

    def _sample_points(self) -> FieldVector:
        """`n0` distinct evaluation points for the base code, resampled under
        a new seed until they are distinct."""
        for attempt in range(_MAX_SAMPLING_ATTEMPTS):
            points = self._sampled(self.n0, b"points", attempt)
            digests = points.hash_elements()
            if len({bytes(digests[32 * i : 32 * (i + 1)]) for i in range(self.n0)}) == (
                self.n0
            ):
                return points
        raise RuntimeError(f"no {self.n0} distinct base points found")

    def relative_distance(
        self, security_bits: int = 128, bound: str = "cccfgs26"
    ) -> float:
        """A lower bound on the relative minimum distance at every level: the
        `delta` a soundness bound needs, which is a property of the code
        rather than of the protocol, which is why `Basefold.soundness_error`
        takes one instead of deriving it.

        For the ``"rs"`` instantiation every level is a Reed-Solomon code on
        distinct points, so level `l` has distance exactly
        `1 - k_l/n_l + 1/n_l`; the rate is `1/c` throughout, so the longest
        codeword's value is the bound and the other arguments play no part.

        For the general code it is `code.foldable_relative_distance` --
        [CCCFGS26]'s bound unless `bound` names [ZCF24]'s -- which holds
        with probability at least `1 - d * 2^-security_bits` over the twists;
        the field size in it is this field's order, twists being drawn from
        the whole field.
        """
        if self.instantiation == "rs":
            return 1 - 1 / self.c + 1 / self.n_d
        return foldable_relative_distance(
            self.k0, self.c, self.d, math.log2(self.field.order), security_bits, bound
        )

    def _as_elements(self, which: int) -> list[list[FieldElement]]:
        """One of the twist tables as elements, materialized on first use.

        The folds read the vectors; only a caller that wants the elements pays
        for n Python objects per level.
        """
        cached = self._twist_elements[which]
        if cached is None:
            cached = [
                (row if which == 0 else -row).to_list() for row in self._twist_vectors
            ]
            self._twist_elements[which] = cached
        return cached

    @property
    def twists(self) -> list[list[FieldElement]]:
        """``T_l[i]``, the even-position twist of the pair `i` folding level
        `l+1` to `l`, per level; `x_i = psi_{l+1}^(2 brv(i) + 1)` for the
        ``"rs"`` instantiation."""
        return self._as_elements(0)

    @property
    def twists_odd(self) -> list[list[FieldElement]]:
        """``T'_l[i] = -T_l[i]``, the odd-position twist, per level; `-x_i` for
        the ``"rs"`` instantiation. Derived, not stored."""
        return self._as_elements(1)

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
        length). The operand is left untouched."""
        message = self._vector(message)
        level = self.level_of(message)
        if self.instantiation == "rs":
            return self._level_transforms().encode(message, level)
        return self._encode_general(message, level)

    def _encode_general(self, message: FieldVector, level: int) -> FieldVector:
        """The recursion `encode_l(m) = fma_interleave(encode_{l-1}(m_even),
        encode_{l-1}(m_odd), T_l, -T_l)` unrolled into whole-vector passes.

        The 2^level base messages are encoded together, and every level
        above is one `fma_interleave` over all of that level's codewords at
        once, laid end to end with the even half-messages' codewords first --
        so the two operands are the two halves of the previous pass, and the
        tables are that level's, tiled over the codewords they apply to.
        """
        if level == 0 and self._transforms is not None:
            return self._transforms.encode(message, 0)
        word = self._base_encode_all(message, level)
        for depth in range(level - 1, -1, -1):
            even, odd = _halves(word)
            twist = _tile(self._twist_vectors[level - depth - 1], 1 << depth)
            word = FieldVector.fma_interleave(even, odd, twist, -twist)
        return word

    def _base_encode_all(self, message: FieldVector, level: int) -> FieldVector:
        """The base codewords of the 2^level base messages of a level-`level`
        message, end to end (base message `b` is `message[b::2^level]`).

        A Vandermonde product on `base_points` as a Horner scheme over whole
        vectors: `k0` fused steps, each over the whole result, with the
        message's blocks repeated across the codeword positions they feed.
        """
        copies = 1 << level
        points = _tile(self.base_points, copies)
        blocks = [
            _window(message, i * copies, copies) for i in range(self.k0)
        ]  # block i holds coefficient i of every base message
        acc = _repeat_each(blocks[-1], self.n0)
        for i in range(self.k0 - 2, -1, -1):
            term = _repeat_each(blocks[i], self.n0)
            acc = term.fma(acc, points, out=term)
        return acc

    def decode(self, word) -> tuple[bool, FieldVector]:
        """`(is_codeword, message)` for a codeword: the level's inverse
        transform plus the degree check that rejects vectors outside the code
        (the level is inferred from the length)."""
        word = self._vector(word)
        size = len(word)
        level = (size // self.n0).bit_length() - 1
        if self.n0 << level != size or level > self.d:
            raise ValueError(f"codeword length {size} is not n0 * 2^l, l <= d")
        if self.instantiation == "rs":
            return self._level_transforms().decode(word, level, self.k0 << level)
        return self._decode_general(word, level)

    def _decode_general(
        self, word: FieldVector, level: int
    ) -> tuple[bool, FieldVector]:
        """`_encode_general` inverted: per level, the pair `(lo, hi)` gives
        `b = (lo - hi) / (2 T)` and `a = lo - b T`, the codewords of the odd
        and even half-messages, until only base codewords are left."""
        if level == 0 and self._transforms is not None:
            return self._transforms.decode(word, 0, self.k0)
        for depth in range(level):
            tables = level - depth - 1
            copies = 1 << depth
            lo, hi = word.split_even_odd()
            odd = lo - hi
            odd.mul(_tile(self._fold_scale_vectors[tables], copies), out=odd)
            product = odd * _tile(self._twist_vectors[tables], copies)
            even = lo.sub(product, out=lo)
            word = FieldVector.concat([even, odd])
        return self._base_decode_all(word, level)

    def _base_decode_all(
        self, words: FieldVector, level: int
    ) -> tuple[bool, FieldVector]:
        """`_base_encode_all` inverted: the message interpolated from the
        first `k0` positions of every base codeword, and whether re-encoding
        it gives back `words`."""
        columns = _columns(words, self.n0)
        inverse = self._lagrange_inverse()
        blocks = []
        for row in inverse:
            acc = columns[0].scale(row[0])
            for j in range(1, self.k0):
                acc = acc.fma(columns[j], row[j], out=acc)
            blocks.append(acc)
        message = FieldVector.concat(blocks)
        return self._base_encode_all(message, level) == words, message

    def _lagrange_inverse(self) -> list[list[FieldElement]]:
        """The inverse of the Vandermonde matrix on the first `k0` base
        points: `message[i] = sum_j inverse[i][j] * word[j]`."""
        if self._vandermonde_inverse is None:
            self._vandermonde_inverse = vandermonde_inverse(
                self.base_points.query(range(self.k0)).to_list(),
                self.field.one,
                self.field.zero,
                operator.add,
                operator.sub,
                operator.mul,
                lambda element: element.inverse(),
            )
        return self._vandermonde_inverse

    def fold_pair(self, lo, hi, r, level: int, i: int):
        """The folded value at position i of a level-`level` codeword, from
        that position's pair alone: `(lo, hi) = (word[2i], word[2i + 1])`.

        This is the form a Merkle verifier uses — it holds one authenticated
        pair per queried position, never a whole codeword.
        """
        coeff = (lo - hi) * self._fold_scale_vectors[level - 1][i]
        return hi + coeff * self._twist_vectors[level - 1][i] + coeff * r

    def fold_pairs(self, los, his, r, level: int, indices) -> list:
        """`fold_pair` at many positions at once.

        `los[k]`, `his[k]` are the pair at `indices[k]`, and the result is
        that position's folded value -- exactly what `fold_pair` returns for
        it, one entry per position.

        A verifier holds one pair per queried position and folds it at every
        level, so the scalar form costs a handful of element operations per
        (query, level), each its own crossing into C. Here a level's positions
        are one set of whole-vector operations instead, and the tables are
        read with one gather rather than an index at a time.
        """
        positions = list(indices)
        shift = self._twist_vectors[level - 1].query(positions)
        coeff = FieldVector(self.field, list(los))
        highs = FieldVector(self.field, list(his))
        coeff.sub(highs, out=coeff)
        coeff.mul(self._fold_scale_vectors[level - 1].query(positions), out=coeff)
        # `shift` is this call's own, so it can hold the running result.
        folded = highs.fma(coeff, shift, out=shift)
        return folded.fma(coeff, r, out=folded).to_list()

    def pair_at(self, word: FieldVector, i: int) -> tuple[FieldElement, FieldElement]:
        """Position i's pair, `(word[2i], word[2i + 1])` — the unit the fold
        reads, and the Merkle leaf (see `leaf_digest`)."""
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
        """`word` as the list of adjacent pairs its Merkle tree commits to."""
        return [self.pair_at(word, i) for i in range(len(word) // 2)]

    def leaf_digest(self, pair: tuple) -> bytes:
        """The Merkle leaf digest of one pair: the vector digest of the two
        elements in order, so it equals the matching entry of `leaf_digests`
        and binds the pair as an ordered unit.

        A verifier recomputes this once per query, where the vector it used
        to build was most of the cost and none of the hashing."""
        return self.field.digest_elements(pair)

    def leaf_digests(self, word: FieldVector) -> memoryview:
        """The leaf digests of every adjacent pair of `word`, packed, in one
        pass over the codeword (`FieldVector.hash_elements` with adjacent
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
        level-(level-1) codeword of the r-folded message, as one call
        (`FieldVector.fold_twisted` with this level's two tables)."""
        return word.fold_twisted(
            self._fold_scale_vectors[level - 1], self._twist_vectors[level - 1], r
        )
