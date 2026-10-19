# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Sparse index oracles committed through one dense oracle (piop.md §4, §7).

A relation whose index holds a family of sparse tables -- the wiring
predicates of a circuit -- is written against them as *restrictions* of one
sparse oracle P, the family's master over the slot variables T and the table
variables X: member i is P(bin(i), X), a `VirtualOracle.restriction` of P
whose slot is a variable map and nothing more. The master is never held by
the verifier: it is the head of a statement of `Relation_SpEval`, "P is the
sparse oracle that the dense index oracle P_dense encodes", whose
composable protocol reduces the evaluation claims on P to claims on
P_dense, a table committed with a polynomial commitment scheme. The
framework fixes neither the encoding nor that protocol: a *realization* is a
subclass of `Relation_SpEval` with its own `compute_head` (the decoding),
decorated with `with_start` and with a protocol registered against its
`Start`.

`FamilyShape` is the shape of such a family: the slots in order, their
widths, which sparse index fields they fill; it builds the master's handle
and its restrictions. `dense_relation(inner)` derives from a relation with
sparse index fields (`Relation_XSparseIndex`, declaring
`sparse_index_fields`) the relation `Relation_X` whose index holds P_dense,
the shape and the realization in their place, and `dense_layering` is its
protocol as a two-layer composition (`composer.py`): the inner relation over
restrictions of a fresh master, then the realization's statement headed at
that master.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vfhe.arith import MLE_Variable

from .piop import Implicit, Relation, Statement, _chain, _map_oracles, element_digest
from .virtual import VirtualOracle, resolve_table

if TYPE_CHECKING:
    from collections.abc import Mapping


class FamilyShape:
    """The shape of a family of sparse tables packed into one master oracle.

    `widths` maps each sparse index field of the inner relation to the width
    (number of variables) of the table it holds, or to the tuple of widths
    of the tables it holds; the slots are those tables in that order, `a` of
    them. The master P is an oracle over `selectors + xs`: `ceil(log2 a)`
    slot variables T (none for a single slot) followed by `max width`
    table variables X, with the index bits of a nonzero `i || k` read
    LSB-first -- slot `i` in the low bits, the table's own index `k` above
    them (`index_of`). Member `i` is the restriction of P at `T = bin(i)`
    (bit `j` of `i` at `T_j`), the coordinates of X past its width fixed to
    0, the rest renamed to variables of its own.

    The shape sits in the index of the dense relation, as data: it exposes
    no `variables` attribute, so `Relation.index_oracles` does not take it
    for an oracle, and `digest()` is by the widths.
    """

    def __init__(self, widths: Mapping[str, int | tuple[int, ...]]) -> None:
        self.widths: dict[str, int | tuple[int, ...]] = {
            name: w if isinstance(w, int) else tuple(w) for name, w in widths.items()
        }
        # (field, position in the field's tuple or None, width), per slot.
        self.slots: list[tuple[str, int | None, int]] = []
        for name, w in self.widths.items():
            if isinstance(w, int):
                self.slots.append((name, None, w))
            else:
                self.slots.extend((name, k, wk) for k, wk in enumerate(w))
        if not self.slots:
            raise ValueError("a family has at least one table")
        self.width = max(w for _, _, w in self.slots)
        bits = (self.count - 1).bit_length()
        self.selectors = [MLE_Variable(f"T{j}") for j in range(bits)]
        self.xs = [MLE_Variable(f"X{k}") for k in range(self.width)]

    @classmethod
    def of(cls, index: Mapping, fields: tuple[str, ...]) -> FamilyShape:
        """The shape of the tables an index holds in its sparse `fields`."""
        widths: dict[str, int | tuple[int, ...]] = {}
        for name in fields:
            entry = index[name]
            if hasattr(entry, "variables"):
                widths[name] = entry.num_vars
            else:
                widths[name] = tuple(t.num_vars for t in entry)
        return cls(widths)

    @property
    def count(self) -> int:
        """The number of slots, a."""
        return len(self.slots)

    def implicit(self) -> Implicit:
        """A fresh handle for the master, over `selectors + xs`."""
        return Implicit(self.selectors + self.xs)

    def slot_bits(self, i: int) -> dict:
        """`T = bin(i)`: the slot variables fixed to the bits of `i`."""
        if not 0 <= i < self.count:
            raise ValueError(f"slot {i} out of range for {self.count} slots")
        return {t: (i >> j) & 1 for j, t in enumerate(self.selectors)}

    def index_of(self, i: int, k: int) -> int:
        """The master's index of entry `k` of the table in slot `i`."""
        return i | (k << len(self.selectors))

    def restriction(self, P, i: int) -> VirtualOracle:
        """Member `i` of the family as a restriction of `P` (the master's
        handle, or its table): `T = bin(i)`, the coordinates of X past the
        slot's width fixed to 0, the rest renamed to variables of its own."""
        bits = self.slot_bits(i)  # range-checked
        _, _, width = self.slots[i]
        fixed = {**bits, **dict.fromkeys(self.xs[width:], 0)}
        variables = [MLE_Variable(f"s{i}_{k}") for k in range(width)]
        return VirtualOracle.restriction(P, variables, fixed)

    def place(self, P) -> dict:
        """The sparse index fields filled with the restrictions of `P`, in
        the shape of the family: one restriction, or a tuple of them."""
        out: dict = {}
        for i, (name, position, _) in enumerate(self.slots):
            r = self.restriction(P, i)
            if position is None:
                out[name] = r
            else:
                out.setdefault(name, []).append(r)
        return {n: tuple(v) if isinstance(v, list) else v for n, v in out.items()}

    def digest(self) -> bytes:
        return _chain([b"family-shape", element_digest(self.widths)])

    def __repr__(self) -> str:
        return f"FamilyShape({self.widths!r})"


def _nonzeros(table) -> dict:
    """The nonzero entries of a sparse table, zero values dropped."""
    return {k: v for k, v in table.nonzeros() if v != 0}


class Relation_SpEval(Relation):
    """The sparse-evaluation relation: the sparse oracle `"sparse"` (P, the
    head) is what the dense index oracle `"dense"` (P_dense) encodes, for a
    family of the given `"shape"`; no instance fields.

    Abstract: the decoding is `compute_head`, which a *realization*
    supplies. A realization subclasses this relation, implements
    `compute_head` (the sparse table over the head's variables, from the
    dense table and the shape), holds any parameter of its own as a class
    attribute -- the dense layering builds it over the three index fields
    and nothing else -- is decorated with `with_start` itself (the registry
    keys on the exact class), and registers against its `Start` a
    sequentially composable protocol: one that takes any claims on P,
    never queries P, draws every challenge itself, and ends in evaluation
    claims on P_dense -- carried in `oracles`, so that the driver bundles
    them for the commitment scheme, or terminal when P_dense is a plain
    table of the ideal model. The generic decider compares the nonzeros of
    the `"sparse"` entry, a table in the ideal model, with those of the
    decoding.
    """

    name = "speval"
    fields = ()
    index_fields = ("sparse", "dense", "shape")
    head_index = "sparse"

    def compute_head(self, statement: Statement):
        """The decoding: P's sparse table from P_dense and the shape."""
        raise NotImplementedError(
            f"{type(self).__name__} is abstract; a realization decodes the master"
        )

    def check(self, statement: Statement) -> bool:
        P = self.index["sparse"]
        decoded = self.compute_head(statement)
        if list(decoded.variables) != list(P.variables):
            return False
        return _nonzeros(decoded) == _nonzeros(P)


def dense_relation(inner_cls: type[Relation]) -> type[Relation]:
    """The relation with a dense index derived from `inner_cls`, a relation
    whose index holds a family of sparse tables in the fields it declares
    as `sparse_index_fields`: the same instance fields and head, the index
    fields of `inner_cls` that are not sparse, then `"dense"` (P_dense, a
    table), `"shape"` (a `FamilyShape`) and `"speval"` (the realization, a
    `Relation_SpEval` subclass). Its decider decodes once through the
    realization and decides `inner_cls` with the slices in the sparse
    fields' place; `compute_head` is `inner_cls`'s, over the index with the
    sparse fields holding restrictions of the master's handle (the head of a
    relation does not depend on its sparse index oracles). The class is
    returned undecorated: apply `with_start` when `inner_cls` has a head.
    """
    sparse = tuple(getattr(inner_cls, "sparse_index_fields", ()))
    if not sparse:
        raise TypeError(f"{inner_cls.__name__} declares no sparse_index_fields")
    rest = tuple(n for n in inner_cls.index_fields if n not in sparse)

    class Dense(Relation):
        name = inner_cls.name + "_dense"
        fields = inner_cls.fields
        index_fields = (*rest, "dense", "shape", "speval")
        head_index = inner_cls.head_index
        inner = inner_cls

        def _inner_fields(self, statement: Statement) -> dict:
            return {
                n: statement.fields[n]
                for n in inner_cls.fields
                if n in statement.fields
            }

        def decode(self):
            """The master's sparse table, as the realization decodes it from
            the dense index oracle."""
            shape, realization = self.index["shape"], self.index["speval"]
            relation = realization(
                index={
                    "sparse": shape.implicit(),
                    "dense": self.index["dense"],
                    "shape": shape,
                }
            )
            return relation.compute_head(Statement(relation))

        def check(self, statement: Statement) -> bool:
            placed = _map_oracles(
                self.index["shape"].place(self.decode()), resolve_table
            )
            inner = inner_cls(index={**{n: self.index[n] for n in rest}, **placed})
            return Statement(inner, **self._inner_fields(statement)).check()

        def compute_head(self, statement: Statement):
            shape = self.index["shape"]
            placed = shape.place(shape.implicit())
            inner = inner_cls(index={**{n: self.index[n] for n in rest}, **placed})
            return inner.compute_head(statement)

    Dense.__name__ = "Relation_" + inner_cls.__name__.removeprefix(
        "Relation_"
    ).removesuffix("SparseIndex")
    Dense.__qualname__ = Dense.__name__
    Dense.__doc__ = (
        f"{inner_cls.__name__} with its sparse index oracles replaced by one dense "
        "index oracle, its shape and its realization (`dense_relation`)."
    )
    return Dense


def dense_layering(root: Statement) -> list[list[Statement]]:
    """The protocol of a dense-index relation as a layered reduction: `L_0`
    the inner relation over the root's instance fields, its sparse index
    fields holding restrictions of a fresh master P, headed at the root's
    head when the root is started (the Composer then starts it with the
    root's claims); `L_1` the realization over `(P, P_dense, shape)`, headed
    at P through its index. Rule 1: P heads `L_1`; rule 2: `L_0` reaches P
    through its index, `L_1` mentions only P_dense beside its head. No
    message, no challenge, soundness error 0. Register
    `compose(dense_layering)` against the dense relation and its `Start`.
    """
    inner_cls = root.relation.inner
    shape = root.index["shape"]
    P = shape.implicit()
    sparse = inner_cls.sparse_index_fields
    rest = {n: root.index[n] for n in inner_cls.index_fields if n not in sparse}
    inner = inner_cls(index={**rest, **shape.place(P)})
    realization = root.index["speval"](
        index={"sparse": P, "dense": root.index["dense"], "shape": shape}
    )
    fields = {n: root.fields[n] for n in inner_cls.fields if n in root.fields}
    return [
        [root.reduce_to(inner, head=root.head, **fields)],
        [root.reduce_to(realization)],
    ]
