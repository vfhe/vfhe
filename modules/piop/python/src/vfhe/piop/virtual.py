# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Virtual oracles (piop.md section 4).

An oracle the verifier has neither as a table nor as a commitment may still
be *defined* in terms of other oracles: a **virtual oracle** [CBBZ23, §2.2;
BCRSW19, Def. 4.6] is a local definition,
`p*(x) = ev(p_1(h_1(x)), ..., p_k(h_k(x)))`: a query to it is answered by
one query to each constituent. Here the query maps `h_j` are variable maps
(a renaming of the constituent's variables, possibly fixing some to
constants) and `ev` is a sum of products of the constituents -- the shape
every sumcheck-style relation needs, and the one whose round messages the
prover can compute. An evaluation claim on a virtual oracle reduces, with
no challenge and no soundness loss, to one claim per non-public
constituent plus a local check (`protocols.VirtualEval`).

A virtual oracle is an ordinary oracle to the rest of the framework: it has
`variables`, binds them with `evaluate`, and yields a `constant()` once
fully bound, so the ideal deciders of the relations work on it
(recursively, and exponentially -- they are the reference semantics, not
the fast path). Binding is *symbolic*: a bound variable becomes a constant
in the query maps and no constituent is touched, so claims derived later
are about the original constituents at explicit points. On the prover's
side every constituent resolves to a concrete table (`resolve_table`;
`Prover.witnesses` for implicit constituents), and `expanded_tables` is the
definition as a products form over equal-length dense tables -- what a
relation computing its head reads, and what the sumcheck prover's
round-message strategies (`protocols/prover_view.py`) are built on.

Naming hazard: Jolt's code calls the implicit oracles "virtual
polynomials"; this module's "virtual" is the HyperPlonk sense.
"""

from __future__ import annotations

from typing import Any

from vfhe.arith.mle import MLE, MLE_Basis, MLE_Variable, SparseMLE
from vfhe.crypto import chain_digest

from .digest import element_digest
from .oracle import Oracle, OracleKind, TableOracle


def is_variable(src, variables: list) -> bool:
    """Whether a map source is one of `variables` (by identity: a constant
    source may be a ring element, whose `==` is not for this)."""
    return any(src is v for v in variables)


def scaled(coeff, value):
    """coeff * value, skipping the multiplication by the integer 1."""
    if isinstance(coeff, int) and coeff == 1:
        return value
    return coeff * value


def _product(values: list):
    prod = None
    for v in values:
        prod = v if prod is None else prod * v
    return prod


def combine_terms(terms: list, values: list):
    """ev in products form: sum_terms coeff * prod_{j in term} values[j]."""
    total = None
    for coeff, idxs in terms:
        t = scaled(coeff, _product([values[j] for j in idxs]))
        total = t if total is None else total + t
    return total


def resolve_table(oracle, witnesses: dict | None = None) -> Any:
    """The concrete table behind an oracle on the prover's side: the
    witness recorded for it (`Prover.witnesses[oracle]`: the table of a
    committed or an implicit oracle), the table a `TableOracle` holds,
    dense or sparse (a sparse table is returned as it is, never
    materialized), or, for a *restriction* (`VirtualOracle.restriction`),
    the restriction of its constituent's resolved table renamed to the
    restriction's variables: the slice of a sparse table
    (`SparseMLE.restrict`), the partial evaluation of a dense one. Any other
    oracle has no table on this side and raises."""
    if witnesses is not None and oracle in witnesses:
        return witnesses[oracle]
    if isinstance(oracle, TableOracle):
        return oracle.polynomial
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
            cv: src
            for cv, src in mapping.items()
            if not is_variable(src, oracle.variables)
        }
        renames = {
            cv: src for cv, src in mapping.items() if is_variable(src, oracle.variables)
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
    raise TypeError(
        f"no table for {oracle!r}: not a table oracle, and no witness stored for it"
    )


class VirtualOracle(Oracle):
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
        for c in self.constituents:
            if not isinstance(c, Oracle):
                raise TypeError(
                    f"a constituent is an oracle, not a table: got {type(c).__name__}; "
                    "wrap a table in TableOracle"
                )
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
                if isinstance(src, MLE_Variable) and not is_variable(
                    src, self.variables
                ):
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
            is_variable(src, self.variables) or (isinstance(src, int) and src in (0, 1))
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
                (scaled(coeff, c), tuple(j + offset for j in idxs))
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
        return all(c.public for c in self.constituents)

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

    @property
    def domain(self):
        """The one domain of the constituents that know theirs (`Oracle.domain`);
        None when none does. Constituents over different domains raise."""
        found = None
        for c in self.constituents:
            domain = c.domain
            if domain is None:
                continue
            if found is None:
                found = domain
            elif domain != found:
                raise ValueError(
                    f"the constituents of {self!r} are over different domains"
                )
        return found

    def dependencies(self) -> list:
        return list(self.constituents)

    def query_point(self, j: int, point: dict | None = None) -> dict:
        """The point constituent j is queried at, for a query at `point`
        (which may leave variables free): its map with the bound sources
        replaced by their values."""
        point = point or {}
        out = {}
        for cv, src in self.maps[j].items():
            if is_variable(src, self.variables):
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
            c.query(self.query_point(j, point)) for j, c in enumerate(self.constituents)
        ]

    def combine(self, values: list):
        """ev applied to constituent values, one per constituent index."""
        return combine_terms(self.terms, values)

    def query(self, point: dict) -> Any:
        """The value at a full point: one query per constituent, combined."""
        return self.combine(self.constituent_values(point))

    def constant(self):
        """The value of a fully-bound virtual oracle."""
        assert self.num_vars == 0, "constant() needs a fully-evaluated oracle"
        return self.query({})

    def claims(self, point: dict) -> list[tuple[object, dict, list[int]]]:
        """The constituent queries a query at `point` becomes: one entry
        `(constituent, point, indices)` per distinct (constituent, point)
        pair, public constituents excluded (the verifier evaluates those
        itself). Deterministic in the constituent order."""
        out: list[tuple[object, dict, list[int]]] = []
        for j, c in enumerate(self.constituents):
            if c.public:
                continue
            pt = self.query_point(j, point)
            for entry in out:
                if entry[0] is c and _same_point(entry[1], pt):
                    entry[2].append(j)
                    break
            else:
                out.append((c, pt, [j]))
        return out

    def dense_constituent(self, j: int, table):
        """Constituent j as the prover reads it: fixed coordinates bound
        (out of place), variables renamed to this oracle's. Not yet a copy
        of its own: the strategies expand every factor into a fresh table."""
        m = self.maps[j]
        fixed = {
            cv: src for cv, src in m.items() if not is_variable(src, self.variables)
        }
        renames = {cv: src for cv, src in m.items() if is_variable(src, self.variables)}
        if fixed:
            table = table.evaluate(fixed, in_place=False)
            table.variables = [renames.get(v, v) for v in table.variables]
            return table
        return table.rename(renames)

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
        return chain_digest(parts)

    def __repr__(self) -> str:
        return (
            f"VirtualOracle(vars={self.num_vars}, constituents="
            f"{len(self.constituents)}, terms={len(self.terms)})"
        )


def _same_point(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    return all(bool(a[k] == b[k]) for k in a)


def gather_bits(k: int, bits: list[int]) -> int:
    """The bits of k at positions `bits`, packed in that order."""
    out = 0
    for i, bit in enumerate(bits):
        out |= ((k >> bit) & 1) << i
    return out


def expand_table(table, variables: list) -> MLE:
    """`table` as a fresh dense table over `variables` — a superset of its
    own, in any order — holding the same polynomial (constant in the added
    variables). Evaluation basis only: the entries are read as values."""
    if getattr(table, "basis", MLE_Basis.eval) is not MLE_Basis.eval:
        raise NotImplementedError("defined oracles need evaluation-basis tables")
    own = [variables.index(v) for v in table.variables]
    if own == list(range(len(variables))):  # the same variables, in order
        same = table.copy()
        same.variables = list(variables)
        return same
    entries = [table.table[gather_bits(idx, own)] for idx in range(1 << len(variables))]
    return table_like([table], variables, entries)


def table_like(tables: list, variables: list, evaluations: list) -> MLE:
    """A dense table over `variables` in the domain of `tables`."""
    ring = next((t.ring for t in tables if getattr(t, "ring", None) is not None), None)
    field = next(
        (t.field for t in tables if getattr(t, "field", None) is not None), None
    )
    return MLE(ring=ring, field=field, variables=variables, evaluations=evaluations)


def expanded_tables(oracle: VirtualOracle, witnesses: dict | None = None) -> list:
    """Every constituent of `oracle` as a fresh dense table over all of the
    oracle's variables, in its order: resolved on the prover's side
    (`resolve_table`, a sparse table materialized), fixed coordinates bound
    and variables renamed (`dense_constituent`), then expanded (`expand_table`).
    The definition is then a products form over equal-length tables, read
    index for index: entry `i` of each is its constituent's value at the
    `i`-th hypercube point (`hypercube_index`)."""
    tables = []
    for j, c in enumerate(oracle.constituents):
        table = resolve_table(c, witnesses)
        if isinstance(table, SparseMLE):
            table = table.materialize()
        tables.append(
            expand_table(oracle.dense_constituent(j, table), oracle.variables)
        )
    return tables


def hypercube_index(variables: list, point: dict) -> int:
    """The index of a hypercube `point` in a table over `variables`, LSB
    first: bit `i` of the index is the point's value at variable `i`."""
    index = 0
    for i, v in enumerate(variables):
        index |= point[v] << i
    return index
