# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Defined oracles: virtual oracles and sum definitions (piop.md §4).

An oracle the verifier has neither as a table nor as a commitment may still
be *defined* in terms of other oracles:

- a **virtual oracle** [CBBZ23, §2.2; BCRSW19, Def. 4.6] is a local definition,
  `p*(x) = ev(p_1(h_1(x)), ..., p_k(h_k(x)))`: a query to it is answered by
  one query to each constituent. Here the query maps `h_j` are variable
  maps (a renaming of the constituent's variables, possibly fixing some to
  constants) and `ev` is a sum of products of the constituents — the shape
  every sumcheck-style relation needs, and the one whose round messages the
  prover can compute. An evaluation claim on a virtual oracle reduces, with
  no challenge and no soundness loss, to one claim per non-public
  constituent plus a local check (`VirtualEval`).
- a **sum definition** (`Relation_SumDef`) is a definition with a hypercube
  sum, `L(z) = sum_{b} G(z, b)` for a virtual oracle `G` over the free
  variables z and the summed variables b — a layer of a layered circuit in
  terms of the layer below it [Tha22, §4.6], a grand-product level [Tha13,
  §5.3.1]. It is a *relation* with L as its head: in a layered reduction
  (`composer.py`) L is an implicit oracle, `Implicit`, whose table the
  prover computes (`compute_head`), and making its value at a point
  explicit takes a sumcheck: `SumDefStart`, the composable protocol of the
  relation, turns the claims on L into one Sum claim over the definition
  (the claims combined by a random linear combination with independent
  challenges [Sor22, Lem. 11; CCCFGS25, Eqs. (19)-(20)]), which the
  registered sumcheck then reduces to claims on the constituents.

A virtual oracle is an ordinary oracle to the rest of the framework: it has
`variables`, binds them with `evaluate`, and yields a `constant()` once
fully bound, so the ideal deciders of the relations work on it
(recursively, and exponentially — they are the reference semantics, not
the fast path). Binding is *symbolic*: a bound variable becomes a constant
in the query maps and no constituent is touched, so claims derived later
are about the original constituents at explicit points. The prover's fast
path is `prover_view`, which resolves every constituent to a concrete table
(`Prover.witnesses` for implicit constituents) and folds those — with one
of two *strategies* for the round messages, both producing the same
messages: the dense one (`_ProverVirtual`) enumerates the hypercube of the
remaining variables; the two-phase one (`_LibraProver`, [XZZPS19, §3.3,
Algs. 4-6], generalized to arbitrary predicates as in [LXZ21; ZLWZS21])
keeps the sparse predicates sparse and builds, per phase, dense
bookkeeping tables in time linear in their nonzeros, so that every round
is a product of dense tables of degree at most two.

Naming hazard: Jolt's code calls the implicit oracles "virtual
polynomials"; this module's "virtual" is the HyperPlonk sense.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from vfhe.arith.mle import (
    MLE,
    MLE_Basis,
    MLE_Variable,
    SparseMLE,
    native_table,
    pair_indices,
)

from .piop import (
    IOP,
    OracleKind,
    Protocol,
    Prover,
    Rejection,
    Relation,
    Relation_Eval,
    Relation_Sum,
    Statement,
    Verifier,
    _chain,
    _constant,
    _hypercube,
    element_digest,
    with_start,
)


def _is_var(src, variables: list) -> bool:
    """Whether a map source is one of `variables` (by identity: a constant
    source may be a ring element, whose `==` is not for this)."""
    return any(src is v for v in variables)


def _scale(coeff, value):
    """coeff * value, skipping the multiplication by the integer 1."""
    if isinstance(coeff, int) and coeff == 1:
        return value
    return coeff * value


def _product(values: list):
    prod = None
    for v in values:
        prod = v if prod is None else prod * v
    return prod


def _combine(terms: list, values: list):
    """ev in products form: sum_terms coeff * prod_{j in term} values[j]."""
    total = None
    for coeff, idxs in terms:
        t = _scale(coeff, _product([values[j] for j in idxs]))
        total = t if total is None else total + t
    return total


def resolve_table(oracle, witnesses: dict | None = None) -> Any:
    """The concrete table behind an oracle on the prover's side: the
    witness recorded for it (`Prover.witnesses[oracle]`), the oracle itself
    when it is a table, dense or sparse (a sparse table is returned as it
    is, never materialized), or, for a *restriction*
    (`VirtualOracle.restriction`), the restriction of its constituent's
    resolved table renamed to the restriction's variables: the slice of a
    sparse table (`SparseMLE.restrict`), the partial evaluation of a dense
    one. Any other virtual oracle has no table of its own and raises."""
    if witnesses is not None and oracle in witnesses:
        return witnesses[oracle]
    if isinstance(oracle, SparseMLE) or hasattr(oracle, "table"):
        return oracle
    if isinstance(oracle, VirtualOracle):
        if not oracle.is_restriction():
            raise TypeError(
                f"no table for {oracle!r}: a virtual oracle resolves to a table "
                "only when it is a restriction of one constituent"
            )
        (constituent,) = oracle.constituents
        (mapping,) = oracle.maps
        table = resolve_table(constituent, witnesses)
        fixed = {
            cv: src for cv, src in mapping.items() if not _is_var(src, oracle.variables)
        }
        renames = {
            cv: src for cv, src in mapping.items() if _is_var(src, oracle.variables)
        }
        if isinstance(table, SparseMLE):
            sliced = table.restrict(fixed)
            sliced.variables = [renames[v] for v in sliced.variables]
            return sliced
        if not fixed:
            return table.rename(renames)
        table = table.evaluate(fixed, in_place=False)
        table.variables = [renames[v] for v in table.variables]
        return table
    raise TypeError(f"no table for oracle {oracle!r}")


class VirtualOracle:
    """`p*(x) = sum_terms coeff * prod_j p_j(h_j(x))` over constituents p_j.

    `variables` are the virtual oracle's own; `maps[j]` sends each variable
    of constituent j to its source: one of `variables` (a renaming) or a
    constant (a fixed coordinate of the query). `terms` is the products
    form, `[(coeff, (j, ...)), ...]` over constituent indices; a
    constituent may appear in several terms and the same oracle may be a
    constituent several times under different maps (W(x) and W(y)).
    """

    kind = OracleKind.virtual

    def __init__(self, variables: list, constituents: list, terms: list, maps=None):
        self.variables = list(variables)
        self.constituents = list(constituents)
        self.terms = [(coeff, tuple(idxs)) for coeff, idxs in terms]
        if maps is None:
            maps = [{v: v for v in c.variables} for c in self.constituents]
        self.maps = [dict(m) for m in maps]
        if len(self.maps) != len(self.constituents):
            raise ValueError("one variable map per constituent")
        for c, m in zip(self.constituents, self.maps, strict=True):
            if any(v not in m for v in c.variables):
                raise ValueError("a constituent variable has no source in its map")
            for src in m.values():
                if isinstance(src, MLE_Variable) and not _is_var(src, self.variables):
                    raise ValueError(
                        f"map source {src!r} is neither a variable of this oracle "
                        "nor a constant"
                    )
        for _, idxs in self.terms:
            if not idxs or any(not 0 <= j < len(self.constituents) for j in idxs):
                raise ValueError("a term must index at least one constituent")

    @classmethod
    def product(cls, constituents: list, variables: list | None = None, coeff=1):
        """A single product `coeff * prod_j p_j` over the union of the
        constituents' variables (identity maps)."""
        if variables is None:
            variables = []
            for c in constituents:
                variables += [v for v in c.variables if v not in variables]
        return cls(variables, constituents, [(coeff, range(len(constituents)))])

    @classmethod
    def restriction(cls, constituent, variables: list, fixed: dict) -> VirtualOracle:
        """The restriction of `constituent` to the slice `fixed`, a map from
        some of its variables to the constants 0 or 1, the remaining ones
        renamed, in order, to `variables`: the virtual oracle with the one
        constituent, one term of coefficient 1 and that map -- how one
        member of a family of tables packed into one oracle over slot
        variables and table variables is read, the slot being a variable
        map and nothing more. `resolve_table` resolves it to a slice of the
        constituent's table; `is_restriction` recognises the form."""
        free = [v for v in constituent.variables if v not in fixed]
        if len(free) != len(variables):
            raise ValueError(
                f"the constituent has {len(free)} free variables, "
                f"{len(variables)} names were given"
            )
        for bit in fixed.values():
            if not isinstance(bit, int) or bit not in (0, 1):
                raise ValueError("a restriction fixes variables to 0 or 1")
        renames = dict(zip(free, variables, strict=True))
        mapping = {cv: fixed.get(cv, renames.get(cv)) for cv in constituent.variables}
        return cls(variables, [constituent], [(1, (0,))], [mapping])

    def is_restriction(self) -> bool:
        """Whether this oracle has the form `restriction` builds: one
        constituent, one term of coefficient 1, every map source a variable
        of this oracle or the constant 0 or 1."""
        if len(self.constituents) != 1 or len(self.terms) != 1:
            return False
        coeff, idxs = self.terms[0]
        if idxs != (0,) or not (isinstance(coeff, int) and coeff == 1):
            return False
        return all(
            _is_var(src, self.variables) or (isinstance(src, int) and src in (0, 1))
            for src in self.maps[0].values()
        )

    @classmethod
    def linear(cls, coeffs: list, oracles: list[VirtualOracle]) -> VirtualOracle:
        """`sum_i coeffs[i] * oracles[i]`, one virtual oracle over the union
        of their variables — the constituent lists concatenated, every term
        scaled by its oracle's coefficient."""
        variables: list = []
        constituents: list = []
        maps: list = []
        terms: list = []
        for coeff, g in zip(coeffs, oracles, strict=True):
            variables += [v for v in g.variables if v not in variables]
            offset = len(constituents)
            constituents += g.constituents
            maps += g.maps
            terms += [
                (_scale(coeff, c), tuple(j + offset for j in idxs))
                for c, idxs in g.terms
            ]
        return cls(variables, constituents, terms, maps)

    @property
    def num_vars(self) -> int:
        return len(self.variables)

    @property
    def public(self) -> bool:
        """Public when every constituent is: the verifier can then evaluate
        it itself."""
        return all(getattr(c, "public", False) for c in self.constituents)

    def reads(self, j: int, var) -> bool:
        """Whether constituent j reads the variable `var` through its map."""
        return any(src is var for src in self.maps[j].values())

    def degree_in(self, var) -> int:
        """The degree bound in `var`: the most factors reading it in one term."""
        return max(sum(1 for j in idxs if self.reads(j, var)) for _, idxs in self.terms)

    @property
    def degree(self) -> int:
        """The per-variable degree bound over all variables (1 for a
        multilinear oracle; 0 with no variables)."""
        return max((self.degree_in(v) for v in self.variables), default=0)

    def dependencies(self) -> list:
        return list(self.constituents)

    def query_point(self, j: int, point: dict | None = None) -> dict:
        """The point constituent j is queried at, for a query at `point`
        (which may leave variables free): its map with the bound sources
        replaced by their values."""
        point = point or {}
        out = {}
        for cv, src in self.maps[j].items():
            if _is_var(src, self.variables):
                if src in point:
                    out[cv] = point[src]
            else:
                out[cv] = src
        return out

    def evaluate(self, point: dict | list, in_place: bool = False) -> VirtualOracle:
        """Bind variables symbolically: each becomes a constant in the maps
        of the constituents reading it; the constituents are untouched."""
        if isinstance(point, list):
            point = dict(zip(self.variables, point, strict=True))
        target = self if in_place else self.copy()
        for var, val in point.items():
            if var not in target.variables:
                continue
            for m in target.maps:
                for cv, src in list(m.items()):
                    if src is var:
                        m[cv] = val
            target.variables.remove(var)
        return target

    def copy(self) -> VirtualOracle:
        return VirtualOracle(self.variables, self.constituents, self.terms, self.maps)

    def constituent_values(self, point: dict | None = None) -> list:
        """Every constituent evaluated at its query point (the ideal decider's
        queries)."""
        return [
            _constant(c.evaluate(self.query_point(j, point), in_place=False))
            for j, c in enumerate(self.constituents)
        ]

    def combine(self, values: list):
        """ev applied to constituent values, one per constituent index."""
        return _combine(self.terms, values)

    def constant(self):
        """The value of a fully-bound virtual oracle."""
        assert self.num_vars == 0, "constant() needs a fully-evaluated oracle"
        return self.combine(self.constituent_values())

    def claims(self, point: dict) -> list[tuple[object, dict, list[int]]]:
        """The constituent queries a query at `point` becomes: one entry
        `(constituent, point, indices)` per distinct (constituent, point)
        pair, public constituents excluded (the verifier evaluates those
        itself). Deterministic in the constituent order."""
        out: list[tuple[object, dict, list[int]]] = []
        for j, c in enumerate(self.constituents):
            if getattr(c, "public", False):
                continue
            pt = self.query_point(j, point)
            for entry in out:
                if entry[0] is c and _same_point(entry[1], pt):
                    entry[2].append(j)
                    break
            else:
                out.append((c, pt, [j]))
        return out

    # Which round-message strategy `prover_view` builds: "auto" takes the
    # two-phase one whenever the oracle has sparse constituents and the shape
    # it needs (§`_LibraProver`), the dense one otherwise; "dense" / "libra"
    # force one (the latter raising when the shape does not fit). A class
    # attribute so tests can pin it; both strategies send identical messages.
    strategy = "auto"

    def _dense_constituent(self, j: int, table):
        """Constituent j as the prover reads it: fixed coordinates bound
        (out of place), variables renamed to this oracle's. Not yet a copy
        of its own: the strategies expand every factor into a fresh table."""
        m = self.maps[j]
        fixed = {cv: src for cv, src in m.items() if not _is_var(src, self.variables)}
        renames = {cv: src for cv, src in m.items() if _is_var(src, self.variables)}
        if fixed:
            table = table.evaluate(fixed, in_place=False)
            table.variables = [renames.get(v, v) for v in table.variables]
            return table
        return table.rename(renames)

    def prover_view(
        self, witnesses: dict | None = None, strategy: str | None = None
    ) -> Any:
        """The prover's concrete form: every constituent resolved to a table
        (`resolve_table`), a constituent being sparse when its *resolved*
        table is -- a sparse predicate, a restriction of a sparse table, an
        implicit oracle whose stored table is sparse -- and kept sparse for
        the two-phase strategy, materialized for the dense one; the
        round-message strategy is chosen by `strategy` (default: the class
        attribute)."""
        strategy = self.strategy if strategy is None else strategy
        tables: dict[int, MLE] = {}
        sparse: dict[int, SparseMLE] = {}
        for j, c in enumerate(self.constituents):
            table = resolve_table(c, witnesses)
            if isinstance(table, SparseMLE):
                if strategy != "dense":
                    sparse[j] = table
                    continue
                table = table.materialize()
            tables[j] = self._dense_constituent(j, table)
        if sparse:
            libra = _LibraProver.build(self, tables, sparse)
            if libra is not None:
                return libra
            if strategy == "libra":
                raise ValueError(
                    "this oracle's shape is not one the two-phase prover handles"
                )
            for j, table in sparse.items():
                tables[j] = self._dense_constituent(j, table.materialize())
        ordered = [tables[j] for j in range(len(self.constituents))]
        return _ProverVirtual(self.variables, ordered, self.terms)

    def digest(self) -> bytes:
        parts = [b"virtual"]
        parts += [element_digest(v) for v in self.variables]
        for c, m in zip(self.constituents, self.maps, strict=True):
            parts.append(element_digest(c))
            for cv in c.variables:
                src = m[cv]
                parts += [element_digest(cv), element_digest(src)]
        for coeff, idxs in self.terms:
            parts += [element_digest(coeff), element_digest(list(idxs))]
        return _chain(parts)

    def __repr__(self) -> str:
        return (
            f"VirtualOracle(vars={self.num_vars}, constituents="
            f"{len(self.constituents)}, terms={len(self.terms)})"
        )


def _same_point(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    return all(bool(a[k] == b[k]) for k in a)


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


def _gather(k: int, bits: list[int]) -> int:
    """The bits of k at positions `bits`, packed in that order."""
    out = 0
    for i, bit in enumerate(bits):
        out |= ((k >> bit) & 1) << i
    return out


def _scatter(k: int, bits: list[int], positions: list[int]) -> int:
    """The bits of k at positions `bits`, moved to `positions`."""
    out = 0
    for bit, pos in zip(bits, positions, strict=True):
        out |= ((k >> bit) & 1) << pos
    return out


def _expand(table, variables: list) -> MLE:
    """`table` as a fresh dense table over `variables` — a superset of its
    own, in any order — holding the same polynomial (constant in the added
    variables). Evaluation basis only: the entries are read as values."""
    if getattr(table, "basis", MLE_Basis.eval) is not MLE_Basis.eval:
        raise NotImplementedError("defined oracles need evaluation-basis tables")
    own = [variables.index(v) for v in table.variables]
    entries = [table.table[_gather(idx, own)] for idx in range(1 << len(variables))]
    return _table_like([table], variables, entries)


class _ProverVirtual:
    """The dense round-message strategy: a products form over tables that
    all span the same variables (`_expand`ed on construction), so a round
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
        self.tables = [_expand(f, self.variables) for f in factors]
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
        return _combine(self.terms, [_constant(t) for t in self.tables])

    def at(self, point: dict):
        """The value at a hypercube point of the remaining variables."""
        idx = 0
        for i, v in enumerate(self.variables):
            idx |= point[v] << i
        return _combine(self.terms, [t.table[idx] for t in self.tables])

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
            evals = [_scale(coeff, e) for e in evals]
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
        from .sumcheck import Sumcheck, SumcheckProd

        if len(tables) == 1 and nodes == 2:
            return list(Sumcheck.round_evals(tables[0], var))
        if len(tables) == 2 and nodes == 3 and all(native_table(t) for t in tables):
            return list(
                SumcheckProd.prod2_round_evals_native(tables[0], tables[1], var)
            )
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
        return self.fixed_weights[_gather(k, self.fixed_bits)]

    def x_index(self, k: int) -> int:
        return _gather(k, self.x_bits)

    def y_index(self, k: int) -> int:
        return _gather(k, self.y_bits)

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
            j: _expand(tables[j], self.ys) for t in self.terms for j in t.y_factors
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
            if _is_var(src, self.xs):
                x_bits.append(bit)
                x_vars.append(src)
            elif _is_var(src, self.ys):
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
                scale = _mul(scale, _constant(self.phase.tables[self.slot[j]]))
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
            return _table_like(list(self.tables.values()), [], [total])
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
        return _table_like(list(self.tables.values()), pred.x_vars, table)

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
        return _table_like(list(self.tables.values()), y_vars, table)

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


def _table_like(tables: list, variables: list, evaluations: list) -> MLE:
    """A dense table over `variables` in the domain of `tables`."""
    ring = next((t.ring for t in tables if getattr(t, "ring", None) is not None), None)
    field = next(
        (t.field for t in tables if getattr(t, "field", None) is not None), None
    )
    return MLE(ring=ring, field=field, variables=variables, evaluations=evaluations)


class VirtualEval(Protocol):
    """`Eval(p*, x, v)` on a virtual oracle -> one `Eval(p_j, h_j(x), y_j)`
    per distinct non-public constituent query (piop.md §4).

    The prover sends the constituent values y_j; the verifier evaluates
    the public constituents itself, checks `ev(y, x) == v`, and emits the
    claims. No challenge: each y_j is bound by its own claim and the check
    pins v, so the step has soundness error 0.
    """

    name = "virtual"
    reduces_from: type[Relation] = Relation_Eval
    reduces_to: tuple[type[Relation], ...] = (Relation_Eval,)

    def _outputs(self, statement: Statement, claims: list, values: list):
        return [
            self.reduce([statement], oracles=[c], point=pt, value=y)
            for (c, pt, _), y in zip(claims, values, strict=True)
        ]

    async def prove(
        self, prover: Prover, statements: list[Statement]
    ) -> list[Statement]:
        (statement,) = statements
        await statement.resolved()
        iop = _iop_of(prover)
        (g,) = statement.oracles
        point = dict(statement.point)
        claims = g.claims(point)
        values = [
            _constant(resolve_table(c, prover.witnesses).evaluate(pt, in_place=False))
            for c, pt, _ in claims
        ]
        iop.transcript.write(f"{self.name}{statement.path}/vals", tuple(values))
        return self._outputs(statement, claims, values)

    async def verify(
        self, verifier: Verifier, statements: list[Statement]
    ) -> list[Statement]:
        (statement,) = statements
        await statement.resolved()
        iop = _iop_of(verifier)
        (g,) = statement.oracles
        point = dict(statement.point)
        claims = g.claims(point)
        label = f"{self.name}{statement.path}"
        values = await iop.transcript.read(f"{label}/vals")
        if len(values) != len(claims):
            raise Rejection(f"{label}: wrong number of constituent values")
        by_index: dict = {}
        for (_, _, idxs), y in zip(claims, values, strict=True):
            for j in idxs:
                by_index[j] = y
        for j, c in enumerate(g.constituents):
            if j not in by_index:  # public: evaluate it here
                by_index[j] = _constant(
                    c.evaluate(g.query_point(j, point), in_place=False)
                )
        combined = g.combine([by_index[j] for j in range(len(g.constituents))])
        if not (combined == statement.value):
            raise Rejection(f"{label}: constituent values do not combine to the claim")
        return self._outputs(statement, claims, list(values))


@with_start
class Relation_SumDef(Relation):
    """`L(z) = sum_{b in {0,1}^m} G(z, b)` on the hypercube: a *sum definition*
    of the oracle L, the head, by the virtual oracle G over L's variables z
    and the summed variables b, the others of G, in G's order: a layer of a
    layered circuit in terms of the layer below it [Tha22, §4.6], a
    grand-product level [Tha13, §5.3.1]. A statement headed at L is how an
    oracle that is defined and never committed enters a layered reduction:
    it sits in the layer of L, and `SumDefStart`, registered against
    `Relation_SumDef.Start`, is its composable protocol.
    """

    name = "sumdef"
    fields = ("oracles",)
    head_index = 0

    @staticmethod
    def summed_of(L, G) -> list:
        """The summed variables: G's that are not L's."""
        return [v for v in G.variables if not any(v is u for u in L.variables)]

    @classmethod
    def summed(cls, statement: Statement) -> list:
        L, G = statement.oracles
        return cls.summed_of(L, G)

    def check(self, statement: Statement) -> bool:
        L, G = statement.oracles
        summed = self.summed(statement)
        for z in _hypercube(L.variables):
            total = None
            for b in _hypercube(summed):
                e = _constant(G.evaluate({**z, **b}, in_place=False))
                total = e if total is None else total + e
            if _constant(L.evaluate(z, in_place=False)) != total:
                return False
        return True

    def compute_head(self, statement: Statement):
        """L's table, the hypercube sums of G over the summed variables, from
        G's constituent tables (the prover view): the generic, exponential
        route. A relation that knows G's structure (a circuit layer) computes
        the table directly instead."""
        L, G = statement.oracles
        _defining(G)
        summed = self.summed(statement)
        view = G.prover_view(None, strategy="dense")
        evaluations = []
        for z in _hypercube(L.variables):
            total = None
            for b in _hypercube(summed):
                e = view.at({**z, **b})
                total = e if total is None else total + e
            evaluations.append(total)
        return _table_like(view.tables, L.variables, evaluations)


class SumDefStart(Protocol):
    """The composable protocol of a sum definition: the started statement's
    claims `L(r_i) = y_i` become one `Sum(sum_i alpha_i G(r_i, .), sum_i
    alpha_i y_i)` over the summed variables, the alphas independent challenges
    (none for a single claim); the registered sumcheck then reduces it to one
    evaluation claim on that virtual oracle, and `VirtualEval` that claim to
    claims on G's constituents, both run by the Composer inside the layer (or
    by the driver, for a started statement run on its own). Zero
    communication; soundness error (k-1)/|A| for k claims [Sor22, Lem. 11;
    CCCFGS25, Eqs. (19)-(20)].
    """

    name = "sumdef"
    reduces_from: type[Relation] = Relation_SumDef.Start
    reduces_to: tuple[type[Relation], ...] = (Relation_Sum,)

    def soundness_error(self, statement: Statement, domain) -> float | None:
        from .sumcheck import _exceptional_set_size

        size = _exceptional_set_size(domain)
        if size is None:
            return None
        return (len(statement.points) - 1) / size

    async def _combine(self, party, statements: list[Statement]) -> list[Statement]:
        iop = _iop_of(party)
        (s,) = statements
        await s.resolved()
        L, G = s.oracles
        _defining(G)
        points, values = list(s.points), list(s.values)
        if not points or len(points) != len(values):
            raise ValueError("a started sum definition carries k >= 1 paired claims")
        label = f"{self.name}{s.path}"
        if len(points) == 1:
            alphas = [1]
        else:
            alphas = [
                iop.verifier.challenge(f"{label}/a{i}") for i in range(len(points))
            ]
        bodies = [
            G.evaluate({v: p[v] for v in L.variables}, in_place=False) for p in points
        ]
        combined = VirtualOracle.linear(alphas, bodies)
        value = None
        for alpha, y in zip(alphas, values, strict=True):
            t = _scale(alpha, y)
            value = t if value is None else value + t
        return [self.reduce([s], Relation_Sum, oracles=[combined], value=value)]

    async def prove(
        self, prover: Prover, statements: list[Statement]
    ) -> list[Statement]:
        return await self._combine(prover, statements)

    async def verify(
        self, verifier: Verifier, statements: list[Statement]
    ) -> list[Statement]:
        return await self._combine(verifier, statements)


def _defining(G) -> None:
    if not isinstance(G, VirtualOracle):
        raise TypeError("the defining oracle of a sum definition is a virtual oracle")


def _iop_of(party) -> IOP:
    iop = party.iop
    if iop is None:
        raise RuntimeError("this party is not bound to an IOP")
    return iop
