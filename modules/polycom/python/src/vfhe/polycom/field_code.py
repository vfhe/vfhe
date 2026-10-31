# SPDX-FileCopyrightText: 2026 The vFHE Authors
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
Reed-Solomon code on `n0` points. The base encoder is a negacyclic NTT where
the field has a `2 n0`-th root of unity, which transforms every base
message of a level-`l` message in one call -- over an extension field
arith's `ExtensionFieldNTT` (`_BatchedTransforms`), with its root in `F_p`
(or, when asked with `root_field="field"`, in the extension where `F_p` has
none); over a pseudo-Mersenne field arith's own `PseudoMersenneNTT`
(`_PseudoMersenneBatchedTransforms`) -- and a Vandermonde product on seeded
distinct points otherwise; the two are encoders of one code, so a codeword
does not depend on which one produced it. Above the base there is no
transform.

The ``"rs"`` instantiation makes every level a Reed-Solomon code on roots of
unity: one transform per level -- the `rs_field_*` kernels
(`polycom/c/src/rscode_field.c`, `_ExtensionTransforms`) over an extension
field, the field's own plans over a pseudo-Mersenne one -- in the basis and
output order `code.FoldableRS` documents (position p holds
P(psi^(2*brv(p)+1)), so the pairs are `(P(x), P(-x))` and the twists are
`x` and `-x`). It needs `2 n_d | p - 1`.
"""

from __future__ import annotations

import contextlib
import copy
import math
import operator
from array import array
from typing import TYPE_CHECKING, cast

from vfhe.arith import (
    ExtensionField,
    Field,
    FieldVector,
    PseudoMersenneField,
    PseudoMersenneVector,
)
from vfhe.util.bindings import ffi, lib

from .code import (
    DEFAULT_SEED,
    _check_whole_leaves,
    bit_reverse_permutation,
    check_instantiation,
    derive_seed,
    foldable_relative_distance,
    vandermonde_inverse,
)

if TYPE_CHECKING:
    from vfhe.arith import FieldElement

    from .code import FoldableRS

#: Where the general code's twists may be drawn from: the whole field, or its
#: prime subfield.
TWIST_FIELDS = ("field", "prime")

#: Where the general code's base transform may find its root of unity: in the
#: prime subfield only, or in the whole field.
ROOT_FIELDS = ("prime", "field")

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


def _pmf_points(field: PseudoMersenneField, root: int, n: int) -> FieldVector:
    """The `n` evaluation points of a pseudo-Mersenne transform at `root`, in
    its output order: position `j` holds `root^(2 brv(j) + 1)`.

    The same running product as `ExtensionField.ntt_points`, written here in
    Python: this field has no kernel for it, and its codes are short -- the
    2-adicity that bounds a codeword bounds this table with it.
    """
    prime = field.prime
    reversed_index = bit_reverse_permutation(n.bit_length() - 1)
    row = [0] * n
    step = root * root % prime
    power = root % prime
    for j in range(n):
        row[reversed_index[j]] = power
        power = power * step % prime
    return FieldVector(field, row)


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
        order."""
        return _pmf_points(self.field, root, n)

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


class _BatchedTransforms:
    """The base transform of the general code over an extension field:
    arith's `ExtensionFieldNTT` of length `n0`, in whichever domain the field
    supplies a `2 n0`-th root of unity, run over all the base messages of a
    level-`l` message in one call.

    The code lays its base codewords out by blocks -- base message `b`'s
    codeword at `b * n0` -- and a level-`l` message zero-padded to the
    codeword length is the same batch interleaved (coefficient `i` of base
    message `b` at `i * 2^l + b`). The plan takes and returns its batch in one
    of those two layouts, which one depending on its domain, so `plan.pack`
    and `plan.unpack` bring the message in and the codewords out: exactly one
    of the two is a transpose pass, the other nothing, in either domain.
    """

    def __init__(self, field: ExtensionField, n: int, domain: str = "auto"):
        self.field = field
        self.n = n
        # ValueError when the domain asked for has no root
        self.plan = field.ntt_plan(n, domain)
        self.root = self.plan.root_of_unity
        #: Where the root lives: ``'base'`` (`F_p`) or ``'extension'``.
        self.domain: str = self.plan.domain

    def points(self) -> FieldVector:
        """The `n` evaluation points, in the transform's output order: position
        `j` holds `psi^(2 brv(j) + 1)`."""
        if self.domain == "base":
            return self.field.ntt_points(int(self.root), self.n)
        # An extension root: the same running product, as elements, once.
        reversed_index = bit_reverse_permutation(self.n.bit_length() - 1)
        row: list = [None] * self.n
        step = self.root * self.root
        power = self.root
        for j in range(self.n):
            row[reversed_index[j]] = power
            power = power * step
        return FieldVector(self.field, row)

    def encode_all(self, message: FieldVector, level: int) -> FieldVector:
        """The base codewords of the `2^level` base messages of `message`,
        end to end."""
        blocks = 1 << level
        padding = FieldVector(self.field, blocks * self.n - len(message))
        padded = FieldVector.concat([message, padding])
        work = self.plan.pack(padded, blocks, layout="interleaved")
        self._forward(work, blocks)
        return self.plan.unpack(work, blocks)

    def decode_all(
        self, words: FieldVector, level: int, degree: int
    ) -> tuple[bool, FieldVector]:
        """`encode_all` inverted: the message of `degree` coefficients, and
        whether every coefficient the padding should have left zero is."""
        blocks = 1 << level
        # Into a destination of its own: `pack` hands back `words` itself
        # where nothing needs converting, and the inverse runs in place.
        work = self.plan.pack(words, blocks, out=FieldVector(self.field, len(words)))
        self._inverse(work, blocks)
        padded = self.plan.unpack(work, blocks, layout="interleaved")
        tail = _window(padded, degree, len(padded) - degree)
        return tail == FieldVector(self.field, len(tail)), _window(padded, 0, degree)

    def _forward(self, work: FieldVector, blocks: int) -> None:
        self.plan.forward(work, blocks)

    def _inverse(self, work: FieldVector, blocks: int) -> None:
        self.plan.inverse(work, blocks)


class _PseudoMersenneBatchedTransforms(_BatchedTransforms):
    """The base transform of the general code over a pseudo-Mersenne field:
    the field's own `PseudoMersenneNTT` of length `n0`, run over all the base
    messages of a level-`l` message in one call.

    `encode_all` is the plan's `encode_blocks`, which takes the interleaved
    message as it is and pads each base message itself; `decode_all`
    transposes the coefficients out, as `_BatchedTransforms` describes for a
    plan that takes its batch by blocks. The root is in the field, which is
    prime.
    """

    def __init__(self, field: PseudoMersenneField, n: int):
        self.field = field
        self.n = n
        self.plan = field.ntt_plan(n)
        self.root = self.plan.root_of_unity
        self.domain = "base"

    def points(self) -> FieldVector:
        return _pmf_points(self.field, int(self.root), self.n)

    def encode_all(self, message: FieldVector, level: int) -> FieldVector:
        return self.plan.encode_blocks(_pmf_vector(message), 1 << level)

    def _forward(self, work: FieldVector, blocks: int) -> None:
        self.plan.forward(_pmf_vector(work), in_place=True, blocks=blocks)

    def _inverse(self, work: FieldVector, blocks: int) -> None:
        self.plan.inverse(_pmf_vector(work), in_place=True, blocks=blocks)


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

    `twist_field` says where the general code's twists are drawn from:
    ``"field"`` (the default), the whole field, or ``"prime"``, its prime
    subfield `F_p`. Prime-subfield twists make every lift and fold multiply
    plane-wise -- `d` prime-field products per element in place of a full
    extension product -- at the price of the distance bound running on
    `log2 p` bits rather than `log2 p^d`, which is more queries for the same
    soundness. Over a prime field the two coincide, and the ``"rs"``
    instantiation's twists are roots of unity in `F_p` whatever is asked.

    `root_field` says where the general code's base transform may find its
    `2 n0`-th root of unity: ``"prime"`` (the default), only in `F_p`, so a
    field whose `p - 1` lacks the 2-adicity gets the Vandermonde base; or
    ``"field"``, in `F_p` where it has one and in the extension otherwise,
    for a transform whose butterflies are extension products rather than
    plane-wise products in `F_p`. Both encode the same code on their points.
    It plays no part over a prime field or in the ``"rs"`` instantiation,
    whose transforms always have their roots in `F_p`.
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
        twist_field: str = "field",
        root_field: str = "prime",
    ):
        for name, value in (("k0", k0), ("c", c)):
            if value < 1 or value & (value - 1):
                raise ValueError(f"{name} must be a power of two, got {value}")
        if d < 1:
            raise ValueError(f"d must be at least 1, got {d}")
        if twist_field not in TWIST_FIELDS:
            raise ValueError(
                f"twist_field must be one of {TWIST_FIELDS}, got {twist_field!r}"
            )
        if root_field not in ROOT_FIELDS:
            raise ValueError(
                f"root_field must be one of {ROOT_FIELDS}, got {root_field!r}"
            )
        self._init_shape(field, k0, c, d, instantiation, seed, twist_field, root_field)
        if self.instantiation == "rs":
            self._init_rs()
        else:
            self._init_general()
        #: 1/2 in the field, the constant term of the fold's per-pair factor.
        self._half = (field.one + field.one).inverse()

    @classmethod
    def residue_code(cls, code: FoldableRS, index: int) -> FieldFoldableRS:
        """The code that `code`, a foldable code over `R_q`, induces on the
        residue field `R_q/P_0` at the prime ``p = code.ring.primes[index]``
        (`RNSRing.residue_field`): the code whose codewords are the CRT
        components at p (`RNSRing.crt_components`) of `code`'s codewords.

        A ring code with twists in the residue fields (``twist_field =
        "field"``) is built from one such code per prime, and that code is
        returned.

        A ring code with integer twists applies the same `F_p`-linear maps to
        every coordinate of its symbols, since its twists and base points at
        p are integers mod p, and the CRT components at p are ring
        homomorphisms that fix `F_p`. So component s of `code.encode(m)` is
        the encoding of component s of m under the same tables, and the code
        returned has `code`'s instantiation and parameters, its twists and
        base points at p as tables over the prime subfield (``twist_field =
        "prime"``), and a fold that commutes with `code.fold` component by
        component. Its base level is encoded by the field's transform when
        that transform's root in `F_p` is the ring transform's, and by a
        Vandermonde product on the same points otherwise. For the ``"rs"``
        instantiation every level's transform must have the ring code's root;
        a `ValueError` is raised when it does not.
        """
        if code.residue_codes:
            # A ring code with full-field twists is defined by these.
            return code.residue_codes[index]
        ring = code.ring
        field = ring.residue_field(index)
        out = cls.__new__(cls)
        out._init_shape(
            field,
            code.k0,
            code.c,
            code.d,
            code.instantiation,
            code.seed,
            "prime",
            "prime",
        )

        def at_prime(table) -> list[int]:
            return [row[index] for row in table]

        out.base_points = FieldVector(field, at_prime(code.base_points))
        for twist in code.twists:
            vector = FieldVector(out.twists_domain, at_prime(twist))
            out._twist_vectors.append(vector)
            out._fold_scale_vectors.append(vector.scale(2).inverse())
        ring_roots = [row[index] for row in code.roots]
        if out.instantiation == "rs":
            transforms = _ExtensionTransforms(
                field, [out.n0 << level for level in range(out.d + 1)]
            )
            roots = [transforms.root(level) for level in range(out.d + 1)]
            if roots != ring_roots:
                raise ValueError(
                    "the residue field's transforms evaluate at other roots of "
                    f"unity ({roots}) than the ring code's ({ring_roots})"
                )
            out._transforms, out.roots = transforms, roots
        elif ring_roots:
            with contextlib.suppress(ValueError):
                batched = _BatchedTransforms(field, out.n0)
                if batched.plan.domain == "base" and int(batched.root) == ring_roots[0]:
                    out._batched, out.roots = batched, ring_roots
        out._half = (field.one + field.one).inverse()
        return out

    def _init_shape(
        self,
        field: Field,
        k0: int,
        c: int,
        d: int,
        instantiation: str,
        seed: bytes,
        twist_field: str,
        root_field: str,
    ) -> None:
        """The code's parameters and the state every instantiation starts
        from, before its tables are built."""
        self.instantiation = check_instantiation(instantiation)
        self.seed = bytes(seed)
        self.field = field
        self.twist_field = twist_field
        self.root_field = root_field
        #: The field the general code's twists are drawn from: `field`, or its
        #: prime subfield as a degree-1 `ExtensionField` when `twist_field` is
        #: ``"prime"`` and `field` is a proper extension.
        self.twists_domain: Field = field
        if twist_field == "prime" and isinstance(field, ExtensionField) and field.d > 1:
            self.twists_domain = ExtensionField(field.prime, 1)
        self.k0 = k0
        self.c = c
        self.d = d
        self.n0 = c * k0
        self.k_d = k0 << d
        self.n_d = self.n0 << d
        # The two tables of every level, as vectors over the codeword
        # positions they act on: `_twist_vectors[l]` is `T` between level l
        # and level l+1 (the encoder multiplies by `T` and `-T`),
        # `_fold_scale_vectors[l]` the `1 / (2 T)`, the one table the fold
        # reads. Both over `twists_domain`, which the vector kernels take as a
        # table either way. Theta(n) entries in all,
        # so nothing here is built an entry at a time.
        self._twist_vectors: list[FieldVector] = []
        self._fold_scale_vectors: list[FieldVector] = []
        #: The roots in `F_p` of the transforms in use: one per level for the
        #: ``"rs"`` instantiation, the base transform's alone for the general
        #: code -- and none when that root lies outside `F_p` or the base is a
        #: Vandermonde product; `base_points` is the table either way.
        self.roots: list[int] = []
        self._transforms: _ExtensionTransforms | _PseudoMersenneTransforms | None = None
        self._batched: _BatchedTransforms | None = None
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

    def truncated(self, depth: int) -> FieldFoldableRS:
        """The depth-`depth` code with this code's field, `k0`, `c`, seed and
        options: its levels 0 to `depth`.

        A foldable code's level `l` is defined by the levels below it and
        level `l`'s twists [ZCF24, Def. 5], and here a level's twists are
        drawn from the seed, the level and its length alone, as the base
        code is from the seed and `n0`. So the code of a smaller depth is
        this one's first levels, and it is returned sharing their tables and
        transforms rather than sampling them again; it encodes, folds and
        hashes exactly as the code built with `depth` does.
        """
        if not 1 <= depth <= self.d:
            raise ValueError(f"depth must be in [1, {self.d}], got {depth}")
        out = copy.copy(self)
        out.d = depth
        out.k_d = self.k0 << depth
        out.n_d = self.n0 << depth
        out._twist_vectors = self._twist_vectors[:depth]
        out._fold_scale_vectors = self._fold_scale_vectors[:depth]
        out._twist_elements = [None, None]
        if self.instantiation == "rs":
            out.roots = self.roots[: depth + 1]
        return out

    def _level_transforms(self):
        """The per-level transforms of the ``"rs"`` instantiation."""
        if self._transforms is None:
            raise RuntimeError("only the 'rs' instantiation transforms every level")
        return self._transforms

    def _init_general(self) -> None:
        """A base code, encoded by a batched transform when the field has one
        of length `n0`, and seeded twists above it.

        Over an extension field the transform is arith's batched one, in the
        base domain (root in `F_p`) where `2 n0 | p - 1`, and with
        ``root_field="field"`` in the extension otherwise; over a
        pseudo-Mersenne field it is the field's own, where `2 n0 | p - 1`.
        With no transform the base is a Vandermonde product on seeded points.
        """
        if isinstance(self.field, ExtensionField):
            domain = "auto" if self.root_field == "field" else "base"
            with contextlib.suppress(ValueError):
                self._batched = _BatchedTransforms(self.field, self.n0, domain)
        elif (
            isinstance(self.field, PseudoMersenneField)
            and self.n0.bit_length() <= self.field.two_adicity
        ):
            self._batched = _PseudoMersenneBatchedTransforms(self.field, self.n0)
        if self._batched is not None:
            if self._batched.domain == "base":
                self.roots = [int(self._batched.root)]
            self.base_points = self._batched.points()
        else:
            self.base_points = self._sample_points()
        for level in range(self.d):
            twist, scale = self._sample_twists(level, self.n0 << level)
            self._twist_vectors.append(twist)
            self._fold_scale_vectors.append(scale)

    def _sampled(self, n: int, tag: bytes, attempt: int, domain=None) -> FieldVector:
        """`n` uniform elements of `domain` (the field unless given), a pure
        function of the code's seed, `tag` and `attempt`."""
        vector = FieldVector(self.field if domain is None else domain, n)
        vector.sample_random(derive_seed(self.seed, tag, attempt))
        return vector

    def _sample_twists(self, level: int, n: int) -> tuple[FieldVector, FieldVector]:
        """`(T, 1 / (2 T))` for one level: `T` uniform, resampled under a new
        seed until no entry is zero -- the condition `T != -T` the fold
        divides by -- and the inverses in one batch (Montgomery's trick)."""
        tag = b"twists" + level.to_bytes(4, "little")
        for attempt in range(_MAX_SAMPLING_ATTEMPTS):
            twist = self._sampled(n, tag, attempt, self.twists_domain)
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
        distance Delta that bounds the proximity parameter `delta` of
        `BasefoldEval.soundness_error`. It is a property of the code rather
        than of the protocol.

        For the ``"rs"`` instantiation every level is a Reed-Solomon code on
        distinct points, so level `l` has distance exactly
        `1 - k_l/n_l + 1/n_l`; the rate is `1/c` throughout, so the longest
        codeword's value is the bound and the other arguments play no part.

        For the general code it is `code.foldable_relative_distance` --
        [CCCFGS26]'s bound unless `bound` names [ZCF24]'s -- which holds
        with probability at least `1 - d * 2^-security_bits` over the twists;
        the field size in it is the order of `twists_domain`, the field the
        twists are drawn from.
        """
        if self.instantiation == "rs":
            return 1 - 1 / self.c + 1 / self.n_d
        return foldable_relative_distance(
            self.k0,
            self.c,
            self.d,
            math.log2(self.twists_domain.order),
            security_bits,
            bound,
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
        `l+1` to `l`, per level, as elements of `twists_domain`;
        `x_i = psi_{l+1}^(2 brv(i) + 1)` for the ``"rs"`` instantiation."""
        return self._as_elements(0)

    @property
    def twists_odd(self) -> list[list[FieldElement]]:
        """``T'_l[i] = -T_l[i]``, the odd-position twist, per level; `-x_i` for
        the ``"rs"`` instantiation. Derived, not stored."""
        return self._as_elements(1)

    def _element(self, value: int):
        """`value` as an element of the field (a constant of F_p)."""
        return type(self.field.one)(self.field, value)

    def _scalar(self, entry):
        """A table entry as an element of the field: itself when the tables
        are over the field, lifted from the prime subfield otherwise."""
        return entry if self.twists_domain is self.field else self._element(int(entry))

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

    def encode_batch(self, messages, count: int) -> FieldVector:
        """The codewords of `count` messages of one level, in one pass.

        `messages` holds them interleaved -- coefficient `i` of message `c`
        at ``i * count + c`` -- and the result holds their codewords end to
        end, message `c`'s at ``c * n_l``. `count` is a power of two.

        For the general code the interleaved messages are one message of
        ``log2(count)`` more levels, whose base messages are those of all
        `count` messages. Encoding it but stopping after the lifts of the
        messages' own levels leaves exactly their codewords, in this order.
        For the ``"rs"`` instantiation each message is transformed on its
        own.
        """
        messages = self._vector(messages)
        if count < 1 or count & (count - 1) or len(messages) % count:
            raise ValueError(
                f"count must be a power of two dividing the length, got {count}"
            )
        length = len(messages) // count
        level = self.level_of(range(length))
        if count == 1:
            return self.encode(messages)
        if self.instantiation == "rs":
            transforms = self._level_transforms()
            return FieldVector.concat(
                [transforms.encode(m, level) for m in _columns(messages, count)]
            )
        return self._encode_general(
            messages, level + count.bit_length() - 1, lifts=level
        )

    def _encode_general(
        self, message: FieldVector, level: int, lifts: int | None = None
    ) -> FieldVector:
        """The recursion `encode_l(m) = fma_interleave(encode_{l-1}(m_even),
        encode_{l-1}(m_odd), T_l, -T_l)` unrolled into whole-vector passes.

        The 2^level base messages are encoded together -- one batched
        transform where the field has one, a Horner scheme over the whole
        result otherwise -- and every level above is one `lift_twisted` over
        all of that level's codewords at once, laid end to end with the even
        half-messages' codewords first: the two operands are the two halves
        of the previous pass, and the level's table is read cyclically over
        the codewords it applies to. The passes alternate between two
        buffers, since a level's input is dead once its output is written.

        `lifts` stops the recursion after that many passes (all `level` by
        default), leaving the codewords of level `lifts` end to end.
        """
        lifts = level if lifts is None else lifts
        if self._batched is not None:
            word = self._batched.encode_all(message, level)
        else:
            word = self._base_encode_all(message, level)
        spare = FieldVector(self.field, len(word)) if lifts else None
        for table in range(lifts):
            even, odd = _halves(word)
            lifted = FieldVector.lift_twisted(
                even, odd, self._twist_vectors[table], out=spare
            )
            word, spare = lifted, word
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
        """`_base_encode_all` inverted -- by the batched transform where the
        field has one, else the message interpolated from the first `k0`
        positions of every base codeword and checked by re-encoding."""
        if self._batched is not None:
            return self._batched.decode_all(words, level, self.k0 << level)
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
        scale = self._scalar(self._fold_scale_vectors[level - 1][i])
        return hi + (lo - hi) * (scale * r + self._half)

    def fold_pairs(self, los, his, r, level: int, indices) -> list:
        """`fold_pair` at many positions at once.

        `los[k]`, `his[k]` are the pair at `indices[k]`, and the result is
        that position's folded value -- exactly what `fold_pair` returns for
        it, one entry per position.

        A verifier holds one pair per queried position and folds it at every
        level, so the scalar form costs a handful of element operations per
        (query, level), each its own crossing into C. Here a level's pairs are
        one vector folded by one `FieldVector.fold_twisted` call, and the
        table is read with one gather rather than an index at a time.
        """
        pairs = FieldVector(
            self.field, [e for pair in zip(los, his, strict=True) for e in pair]
        )
        return self.fold_fibers(pairs, [r], level, indices).to_list()

    def fold_batch(self, words, challenges, level: int) -> FieldVector:
        """`fold` of ``len(challenges)`` level-`level` codewords laid end to
        end, codeword `c` with ``challenges[c]``; the folded codewords come
        out end to end in the same order. It is one `fold_twisted_blocks`
        call, in which every codeword reads the level's table. `challenges`
        is a vector over the field or a sequence of its elements."""
        words = self._vector(words)
        challenges = self._vector(challenges)
        n = self.n0 << level
        if len(words) != n * len(challenges):
            raise ValueError(
                f"{len(words)} symbols are not {len(challenges)} codewords of {n}"
            )
        return words.fold_twisted_blocks(
            self._fold_scale_vectors[level - 1], challenges
        )

    def fold_fibers_batch(self, values, challenges, level: int, indices) -> FieldVector:
        """`fold_fibers` for ``count`` codewords at once, at the same
        positions: `values` holds `count` blocks end to end, block `c` the
        fibers of codeword `c` above `indices` (as `fold_fibers` takes them),
        and ``challenges[k]`` holds the `count` challenges of fold `k`, one per
        codeword. The result holds, block by block, `fold_fibers` of each.
        Each fold is one `fold_twisted_blocks` call, with the level's table
        gathered at the fibers' pairs."""
        values = self._vector(values)
        steps = len(challenges)
        if not 1 <= steps <= level:
            raise ValueError(f"{steps} folds from level {level}")
        count = len(challenges[0])
        if len(values) != count * (len(indices) << steps):
            raise ValueError(
                f"{len(values)} symbols are not {count} blocks of {len(indices)} "
                f"fibers of {1 << steps}"
            )
        for k, r in enumerate(challenges):
            half = 1 << (steps - k - 1)
            positions = array("Q", [j * half + i for j in indices for i in range(half)])
            values = values.fold_twisted_blocks(
                self._fold_scale_vectors[level - k - 1].query(positions),
                self._vector(r),
            )
        return values

    def fold_fibers(self, values, challenges, level: int, indices) -> FieldVector:
        """Fold fibers of a level-`level` codeword `s = len(challenges)` times,
        one challenge per fold in order, down to the one symbol each
        determines.

        The fiber above position `j` of the codeword `s` folds below is the
        `2^s` adjacent symbols ``word[j 2^s : (j + 1) 2^s]``, since every fold
        sends pair ``(2i, 2i + 1)`` to position `i`. `values` holds fibers end
        to end and `indices[k]` is the position below fiber `k`; the result
        has one symbol per fiber, entry `k` the value at `indices[k]` of the
        level-``(level - s)`` codeword. Each fold is one
        `FieldVector.fold_twisted` call over all the fibers, with the level's
        table gathered at their pairs' positions. With one challenge it is
        `fold_pairs` on the pairs of `values`.
        """
        values = self._vector(values)
        steps = len(challenges)
        if not 1 <= steps <= level:
            raise ValueError(f"{steps} folds from level {level}")
        if len(values) != len(indices) << steps:
            raise ValueError(
                f"{len(values)} symbols are not {len(indices)} fibers of {1 << steps}"
            )
        for k, r in enumerate(challenges):
            half = 1 << (steps - k - 1)
            positions = array("Q", [j * half + i for j in indices for i in range(half)])
            values = values.fold_twisted(
                self._fold_scale_vectors[level - k - 1].query(positions), r
            )
        return values

    def fiber_at(self, word: FieldVector, i: int, size: int):
        """The `size` adjacent symbols ``word[i size : (i + 1) size]``: the
        Merkle leaf `i` of a tree over `size`-symbol fibers. A pair is
        `pair_at`'s tuple, a larger fiber a vector."""
        if size == 2:
            return self.pair_at(word, i)
        return word.query(array("Q", range(i * size, (i + 1) * size)))

    def join_fibers(self, fibers, size: int) -> FieldVector:
        """`fibers`, as `fiber_at` returns them, end to end in one vector.

        Raises `ValueError` when an entry is not a fiber of `size` symbols of
        this code's field.
        """
        if size == 2:
            if any(not isinstance(pair, tuple) or len(pair) != 2 for pair in fibers):
                raise ValueError("a pair is a tuple of two field elements")
            try:
                return FieldVector(self.field, [e for pair in fibers for e in pair])
            except (TypeError, ValueError) as error:
                raise ValueError(f"a pair is not over this field: {error}") from error
        for fiber in fibers:
            if (
                not isinstance(fiber, FieldVector)
                or fiber.field is not self.field
                or len(fiber) != size
            ):
                raise ValueError(f"a fiber is a vector of {size} elements of the field")
        return FieldVector.concat(list(fibers))

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

    def leaf_digests(self, word: FieldVector, size: int = 2) -> memoryview:
        """The leaf digests of every adjacent pair of `word`, packed, in one
        pass over the codeword (`FieldVector.hash_elements` with adjacent
        windows). Leaf `i` is ``digests[32 * i : 32 * (i + 1)]``, and equals
        `leaf_digest` of `pair_at(word, i)`. With `size`, the leaves are the
        fibers of that many symbols instead, `fiber_at(word, i, size)`.
        Raises `ValueError` when `size` does not divide the length of `word`.

        A read-only view of the kernel's buffer, not a copy of it -- so a
        codeword reaches its root without its digests being duplicated on the
        way. `bytes(...)` is how a caller takes a copy."""
        _check_whole_leaves(len(word), size)
        return word.hash_elements(group=size, stride=size)

    def fold_at(self, word: FieldVector, r, level: int, i: int):
        """Position i of the fold of the level-`level` codeword `word` with
        challenge r — the value the folded codeword must hold there."""
        return self.fold_pair(*self.pair_at(word, i), r, level, i)

    def fold(self, word: FieldVector, r, level: int) -> FieldVector:
        """The full fold of a level-`level` codeword with challenge r: the
        level-(level-1) codeword of the r-folded message, as one call
        (`FieldVector.fold_twisted` with this level's `1 / (2 T)` table)."""
        return word.fold_twisted(self._fold_scale_vectors[level - 1], r)
