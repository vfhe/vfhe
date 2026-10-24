# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The prover's forms of a virtual oracle (piop.md section 4).

A sumcheck over a virtual oracle runs on its products form, never on a
materialized table of its hypercube values. `prover_view` resolves every
constituent to a concrete table and folds those, with one of two
*strategies* for the round messages, both producing the same messages: the
dense one (`_ProverVirtual`) enumerates the hypercube of the remaining
variables; the two-phase one (`_LibraProver`, [XZZPS19, §3.3, Algs. 4-6],
generalized to arbitrary predicates as in [LXZ21; ZLWZS21]) keeps the
sparse predicates sparse and builds, per phase, dense bookkeeping tables in
time linear in their nonzeros, so that every round is a product of dense
tables of degree at most two. Both answer the round loop of
`_SumcheckRounds` with `variables`, `degree_in`, `round_evals`, `evaluate`
and `constant`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from vfhe.arith.mle import MLE, SparseMLE, native_table, pair_indices
from vfhe.piop.virtual import (
    VirtualOracle,
    combine_terms,
    expand_table,
    gather_bits,
    is_variable,
    resolve_table,
    scaled,
    table_like,
)

from .rounds import prod2_round_evals_native, round_evals


def _one_of(tables) -> Any:
    """The multiplicative identity of the tables' domain (1 for plain values)."""
    for t in tables:
        if getattr(t, "ring", None) is not None:
            from vfhe.arith import Polynomial

            return Polynomial(t.ring).from_array([1])
        if getattr(t, "field", None) is not None:
            return t.field.one
    return 1


def _eq_weights(values: list, one) -> list:
    """The table of eq~(values, b) over b in {0,1}^len(values), LSB-first:
    the weight of a hypercube point against fixed coordinates ([XZZPS19]'s
    precomputed G table; a scalar `MLE.eq` that also works on plain values)."""
    table = [one]
    for c in values:
        nc = one - c
        table = [t * nc for t in table] + [t * c for t in table]
    return table


def _mul(a, b) -> Any:
    """a * b, skipping the multiplication by an integer 1."""
    if isinstance(a, int) and a == 1:
        return b
    if isinstance(b, int) and b == 1:
        return a
    return a * b


def _scatter(k: int, bits: list[int], positions: list[int]) -> int:
    """The bits of k at positions `bits`, moved to `positions`."""
    out = 0
    for bit, pos in zip(bits, positions, strict=True):
        out |= ((k >> bit) & 1) << pos
    return out


class _ProverVirtual:
    """The dense round-message strategy: a products form over tables that
    all span the same variables (`expand_table`ed on construction), so a round
    over the variable at position `idx` reads the table pairs
    `pair_indices(size, idx)`, and a factor that never read the variable
    is simply constant across each pair.

    `round_evals(var)` gives the round polynomial at the nodes
    `0..degree_in(var)`; the degree counts the factors that *read* the
    variable in the original definition (`reads`), not the tables that now
    span it. `evaluate` folds every table by the challenge."""

    owned = True  # fresh tables: the round loop may fold them in place

    def __init__(self, variables: list, factors: list, terms: list):
        self.variables = list(variables)
        self.reads = [list(f.variables) for f in factors]
        self.tables = [expand_table(f, self.variables) for f in factors]
        self.terms = [(coeff, tuple(idxs)) for coeff, idxs in terms]

    @property
    def num_vars(self) -> int:
        return len(self.variables)

    def degree_in(self, var) -> int:
        """The degree bound in `var` (at least 1, so a message always has
        the two nodes the verifier's `g(0) + g(1)` check reads)."""
        per_term = (
            sum(1 for j in idxs if var in self.reads[j]) for _, idxs in self.terms
        )
        return max(1, max(per_term))

    def evaluate(self, point: dict | list, in_place: bool = True) -> _ProverVirtual:
        if isinstance(point, list):
            point = dict(zip(self.variables, point, strict=True))
        tables = [t.evaluate(point, in_place=in_place) for t in self.tables]
        variables = [v for v in self.variables if v not in point]
        if in_place:
            self.tables, self.variables = tables, variables
            return self
        view = _ProverVirtual.__new__(_ProverVirtual)
        view.variables, view.reads, view.tables, view.terms = (
            variables,
            self.reads,
            tables,
            self.terms,
        )
        return view

    def constant(self):
        assert self.num_vars == 0, "constant() needs a fully-evaluated oracle"
        return combine_terms(self.terms, [t.constant() for t in self.tables])

    def round_evals(self, var=None) -> tuple:
        """sum_terms coeff * (sum_b prod_j table_j(t, b)) at t = 0..degree,
        b over the cube of the other variables — each term summed first,
        then scaled once per node."""
        var = self.variables[0] if var is None else var
        idx = self.variables.index(var)
        nodes = self.degree_in(var) + 1
        total = None
        for coeff, idxs in self.terms:
            evals = self._term_round_evals(
                [self.tables[j] for j in idxs], var, idx, nodes
            )
            evals = [scaled(coeff, e) for e in evals]
            total = (
                evals
                if total is None
                else [a + b for a, b in zip(total, evals, strict=True)]
            )
        if total is None:
            raise ValueError(
                "a products-form oracle with no terms has no round message"
            )
        return tuple(total)

    @staticmethod
    def _term_round_evals(tables: list, var, idx: int, nodes: int) -> list:
        """One product's node sums; the native kernels for the shapes they
        cover (one table, or two ring tables at three nodes), Python pairs
        otherwise. Every factor is multilinear, so its value at node t is
        `lo + t * (hi - lo)` from the pair."""
        if len(tables) == 1 and nodes == 2:
            return list(round_evals(tables[0], var))
        if len(tables) == 2 and nodes == 3 and all(native_table(t) for t in tables):
            return list(prod2_round_evals_native(tables[0], tables[1], var))
        size = 1 << tables[0].num_vars
        evals = []
        for t in range(nodes):
            total = None
            for lo, hi in pair_indices(size, idx):
                prod = None
                for table in tables:
                    a, b = table.table[lo], table.table[hi]
                    v = a if t == 0 else b if t == 1 else a + t * (b - a)
                    prod = v if prod is None else prod * v
                total = prod if total is None else total + prod
            evals.append(total)
        return evals


class _Unsupported(Exception):
    """The oracle's shape is not one the two-phase strategy handles."""


@dataclass
class _Predicate:
    """A sparse predicate of a term, with its nonzero indices' bits sorted
    by where the variable they encode lives: in the X block, in the Y
    block, or fixed to a constant (a bound z coordinate, or a 0/1 in the
    map). `fixed_weights[gather(k, fixed_bits)]` is eq~(constants, k)."""

    sparse: SparseMLE
    x_bits: list[int]
    x_vars: list
    y_bits: list[int]
    y_positions: list[int]  # positions in the Y block of the y variables read
    fixed_bits: list[int]
    fixed_weights: list

    def weight(self, k: int):
        return self.fixed_weights[gather_bits(k, self.fixed_bits)]

    def x_index(self, k: int) -> int:
        return gather_bits(k, self.x_bits)

    def y_index(self, k: int) -> int:
        return gather_bits(k, self.y_bits)

    def y_block_index(self, k: int) -> int:
        return _scatter(k, self.y_bits, self.y_positions)


@dataclass
class _Term:
    coeff: object
    x_factors: list[int]  # constituent indices of the dense factors over X
    y_factors: list[int]  # ... over Y
    constants: list[int]  # ... with no variables left
    predicate: _Predicate | None


class _LibraProver:
    """The two-phase round-message strategy [XZZPS19, §3.3, Algs. 4-6; LXZ21,
    §4] for `sum_{x,y} sum_terms coeff * pred(z,x,y) * A(x) * B(y)`, z fixed:

    - **phase 1**, the rounds over X: the term is `A(x) * h(x)` with the
      bookkeeping table `h(x) = sum_y pred(z,x,y) B(y)`, filled from the
      predicate's nonzeros in O(nnz) — one eq~ weight lookup for the fixed
      coordinates and one B lookup per nonzero;
    - **phase 2**, the rounds over Y, X bound to r: the term is
      `A(r) * p(y) * B(y)` with `p(y) = pred(z, r, y)`, filled the same way
      with the extra weight eq~(r, x).

    Each phase is a dense products form (`_ProverVirtual`) of degree at most
    two, and its messages are those of the full polynomial — the dense
    strategy's transcript, at the cost of the wires and the block sizes
    instead of the hypercube of (x, y).

    Shape: the summed variables form two contiguous blocks X then Y (a
    variable read by no dense factor joins the block of its position),
    every dense factor lives inside one block, and a term has at most one
    sparse predicate. `build` returns None for anything else.
    """

    owned = True

    def __init__(self, oracle: VirtualOracle, tables: dict, sparse: dict):
        self.xs, self.ys = self._blocks(oracle.variables, tables)
        self.one = _one_of([*tables.values(), *sparse.values()])
        self.tables = tables
        self.terms = [
            self._term(coeff, idxs, oracle, tables, sparse)
            for coeff, idxs in oracle.terms
        ]
        # The Y-side factors over the whole Y block, read by the phase-1
        # bookkeeping and handed to phase 2.
        self.y_tables = {
            j: expand_table(tables[j], self.ys) for t in self.terms for j in t.y_factors
        }
        self.x_point: dict = {}
        self.in_phase_two = False
        self.phase = self._phase_one()

    @classmethod
    def build(cls, oracle: VirtualOracle, tables: dict, sparse: dict):
        """The strategy for `oracle`, or None when its shape does not fit."""
        try:
            return cls(oracle, tables, sparse)
        except _Unsupported:
            return None

    # -- shape ---------------------------------------------------------------

    @staticmethod
    def _blocks(variables: list, tables: dict) -> tuple[list, list]:
        """Split the round order into the X and Y blocks from the dense
        factors' variable sets."""
        sets: list[list] = []
        for t in tables.values():
            if t.num_vars and not any(
                set(map(id, t.variables)) == set(map(id, s)) for s in sets
            ):
                sets.append(list(t.variables))
        if not sets or len(sets) > 2:
            raise _Unsupported
        first = next(i for i, v in enumerate(variables) if any(v in s for s in sets))
        x_set = next(s for s in sets if variables[first] in s)
        y_set = next((s for s in sets if s is not x_set), [])
        y_start = next(
            (i for i, v in enumerate(variables) if v in y_set), len(variables)
        )
        xs, ys = variables[:y_start], variables[y_start:]
        if any(v in y_set for v in xs) or any(v in x_set for v in ys):
            raise _Unsupported  # the blocks are not contiguous
        return xs, ys

    def _term(self, coeff, idxs, oracle, tables, sparse) -> _Term:
        preds = [j for j in idxs if j in sparse]
        if len(preds) > 1:
            raise _Unsupported
        term = _Term(coeff, [], [], [], None)
        for j in idxs:
            if j in sparse:
                term.predicate = self._predicate(sparse[j], oracle.maps[j])
            elif tables[j].num_vars == 0:
                term.constants.append(j)
            elif all(v in self.xs for v in tables[j].variables):
                term.x_factors.append(j)
            elif all(v in self.ys for v in tables[j].variables):
                term.y_factors.append(j)
            else:
                raise _Unsupported
        return term

    def _predicate(self, sparse: SparseMLE, mapping: dict) -> _Predicate:
        x_bits, x_vars, y_bits, y_positions, fixed_bits, fixed = [], [], [], [], [], []
        for bit, cv in enumerate(sparse.variables):
            src = mapping[cv]
            if is_variable(src, self.xs):
                x_bits.append(bit)
                x_vars.append(src)
            elif is_variable(src, self.ys):
                y_bits.append(bit)
                y_positions.append(self.ys.index(src))
            else:
                fixed_bits.append(bit)
                fixed.append(src)
        return _Predicate(
            sparse,
            x_bits,
            x_vars,
            y_bits,
            y_positions,
            fixed_bits,
            _eq_weights(fixed, self.one),
        )

    # -- the two phases -------------------------------------------------------

    def _phase_one(self) -> _ProverVirtual:
        factors: list = []
        terms: list = []
        self.slot: dict[int, int] = {}  # constituent -> its phase-1 table
        for term in self.terms:
            start = len(factors)
            for j in term.x_factors + term.constants:
                self.slot.setdefault(j, len(factors))
                factors.append(self.tables[j])
            factors.append(self._bookkeeping_x(term))
            terms.append((term.coeff, range(start, len(factors))))
        return _ProverVirtual(self.xs, factors, terms)

    def _phase_two(self) -> _ProverVirtual:
        factors: list = []
        terms: list = []
        for term in self.terms:
            scale = term.coeff
            for j in term.x_factors + term.constants:
                scale = _mul(scale, self.phase.tables[self.slot[j]].constant())
            start = len(factors)
            factors += [self.y_tables[j] for j in term.y_factors]
            if term.predicate is not None:
                factors.append(self._bookkeeping_y(term.predicate))
            terms.append((scale, range(start, len(factors))))
        return _ProverVirtual(self.ys, factors, terms)

    def _bookkeeping_x(self, term: _Term) -> MLE:
        """h(x) = sum_y pred(z, x, y) * prod B(y), over the predicate's X
        variables; without a predicate, the constant sum_y prod B(y)."""
        b_tables = [self.y_tables[j] for j in term.y_factors]
        pred = term.predicate
        if pred is None:
            total = None
            for idx in range(1 << len(self.ys)):
                prod = self.one
                for b in b_tables:
                    prod = _mul(prod, b.table[idx])
                total = prod if total is None else total + prod
            return table_like(list(self.tables.values()), [], [total])
        # Y positions the predicate does not read are summed over: every
        # completion of its y bits contributes (a factor 2 each without B).
        free = [p for p in range(len(self.ys)) if p not in pred.y_positions]
        completions = [
            sum(1 << p for i, p in enumerate(free) if (m >> i) & 1)
            for m in range(1 << len(free))
        ]
        acc: dict[int, Any] = {}
        for k, value in pred.sparse.nonzeros():
            weight = pred.weight(k)
            if isinstance(weight, int) and weight == 0:
                continue
            base = pred.y_block_index(k)
            b_sum = None
            for mask in completions:
                prod = 1
                for b in b_tables:
                    prod = _mul(prod, b.table[base | mask])
                b_sum = prod if b_sum is None else b_sum + prod
            entry = _mul(_mul(weight, value), b_sum)
            i = pred.x_index(k)
            acc[i] = entry if i not in acc else acc[i] + entry
        table = [acc.get(i, 0) for i in range(1 << len(pred.x_vars))]
        return table_like(list(self.tables.values()), pred.x_vars, table)

    def _bookkeeping_y(self, pred: _Predicate) -> MLE:
        """p(y) = pred(z, r, y) over the predicate's Y variables, r the
        bound X point."""
        x_weights = _eq_weights([self.x_point[v] for v in pred.x_vars], self.one)
        acc: dict[int, Any] = {}
        for k, value in pred.sparse.nonzeros():
            weight = _mul(pred.weight(k), x_weights[pred.x_index(k)])
            entry = _mul(weight, value)
            i = pred.y_index(k)
            acc[i] = entry if i not in acc else acc[i] + entry
        y_vars = [self.ys[p] for p in pred.y_positions]
        table = [acc.get(i, 0) for i in range(1 << len(y_vars))]
        return table_like(list(self.tables.values()), y_vars, table)

    # -- the oracle surface the sumcheck drives -------------------------------

    @property
    def variables(self) -> list:
        return list(self.phase.variables) + ([] if self.in_phase_two else list(self.ys))

    @property
    def num_vars(self) -> int:
        return len(self.variables)

    def degree_in(self, var) -> int:
        return self.phase.degree_in(var)

    def round_evals(self, var=None) -> tuple:
        var = self.variables[0] if var is None else var
        if var not in self.phase.variables:
            raise ValueError(
                "the two-phase prover binds the X block before the Y block"
            )
        return self.phase.round_evals(var)

    def evaluate(self, point: dict | list, in_place: bool = True) -> _LibraProver:
        if isinstance(point, list):
            point = dict(zip(self.variables, point, strict=True))
        if any(v not in self.phase.variables for v in point):
            raise ValueError(
                "the two-phase prover binds the X block before the Y block"
            )
        target = self if in_place else copy.copy(self)
        target.phase = self.phase.evaluate(point, in_place=in_place)
        if not target.in_phase_two:
            target.x_point = {**self.x_point, **point}
            if target.phase.num_vars == 0 and self.ys:
                target.phase = target._phase_two()
                target.in_phase_two = True
        return target

    def constant(self):
        assert self.num_vars == 0, "constant() needs a fully-evaluated oracle"
        return self.phase.constant()


def prover_view(oracle, witnesses: dict | None = None, strategy: str = "auto") -> Any:
    """The prover's working form of an oracle. For a table, committed or
    implicit oracle it is the table behind it (`resolve_table`). For a
    virtual oracle it is one of the two strategies over the resolved
    constituents -- a constituent being sparse when its *resolved* table is:
    a sparse predicate, a restriction of a sparse table, an implicit oracle
    whose stored table is sparse -- chosen by `strategy`: `"auto"` takes the
    two-phase one whenever the oracle has sparse constituents and the shape
    it needs (`_LibraProver`), the dense one (`_ProverVirtual`, every
    constituent materialized) otherwise; `"dense"` / `"libra"` force one, the
    latter raising when the shape does not fit. Both send identical
    messages.
    """
    if not isinstance(oracle, VirtualOracle):
        return resolve_table(oracle, witnesses)
    tables: dict[int, MLE] = {}
    sparse: dict[int, SparseMLE] = {}
    for j, c in enumerate(oracle.constituents):
        table = resolve_table(c, witnesses)
        if isinstance(table, SparseMLE):
            if strategy != "dense":
                sparse[j] = table
                continue
            table = table.materialize()
        tables[j] = oracle.dense_constituent(j, table)
    if sparse:
        libra = _LibraProver.build(oracle, tables, sparse)
        if libra is not None:
            return libra
        if strategy == "libra":
            raise ValueError(
                "this oracle's shape is not one the two-phase prover handles"
            )
        for j, table in sparse.items():
            tables[j] = oracle.dense_constituent(j, table.materialize())
    ordered = [tables[j] for j in range(len(oracle.constituents))]
    return _ProverVirtual(oracle.variables, ordered, oracle.terms)
