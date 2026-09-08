# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Defined oracles: virtual and implicit (piop.md §4).

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
- an **implicit oracle** is a definition with a hypercube sum,
  `L(z) = sum_{b} G(z, b)` for a virtual oracle `G` over the free variables
  z and the summed variables b — a layer of a layered circuit in terms of
  the layer below it [Tha22, §4.6], a grand-product level [Tha13, §5.3.1].
  Making its value at a point explicit takes a sumcheck: `ImplicitEval`
  turns every pending claim on it into one Sum claim over the definition
  (the claims combined by a random linear combination with independent
  challenges [Sor22, Lem. 11; CCCFGS25, Eqs. (19)-(20)]), which the
  registered sumcheck then reduces to claims on the constituents.

Both are ordinary oracles to the rest of the framework: they have
`variables`, bind them with `evaluate`, and yield a `constant()` once fully
bound, so the ideal deciders of the relations work on them (recursively,
and exponentially — they are the reference semantics, not the fast path).
Binding is *symbolic*: a bound variable becomes a constant in the query
maps and no constituent is touched, so claims derived later are about the
original constituents at explicit points. The prover's fast path is
`prover_view`, which resolves every constituent to a concrete table
(`Prover.witnesses` for implicit constituents) and folds those.

Naming hazard: Jolt's code calls the implicit kind "virtual polynomials";
this module's "virtual" is the HyperPlonk sense.
"""

from __future__ import annotations

from .mle import MLE, MLE_Basis
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


def resolve_table(oracle, witnesses: dict | None = None):
    """The concrete table behind an oracle on the prover's side: the
    witness recorded for it (`Prover.witnesses[oracle]`), the oracle itself
    when it is a table, else its materialization (defined oracles)."""
    if witnesses is not None and oracle in witnesses:
        return witnesses[oracle]
    if hasattr(oracle, "table"):
        return oracle
    materialize = getattr(oracle, "materialize", None)
    if callable(materialize):
        return materialize(witnesses)
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

    @property
    def degree(self) -> int:
        """The per-variable degree bound: the longest product."""
        return max(len(idxs) for _, idxs in self.terms)

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
        queries; exponential through implicit constituents)."""
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

    def prover_view(self, witnesses: dict | None = None) -> _ProverVirtual:
        """The prover's concrete form: every constituent resolved to a table
        (`resolve_table`), its fixed coordinates bound, its variables
        renamed to this oracle's — each an independent copy, so the round
        loop may fold them in place."""
        tables = []
        for j, c in enumerate(self.constituents):
            t = resolve_table(c, witnesses)
            m = self.maps[j]
            fixed = {
                cv: src for cv, src in m.items() if not _is_var(src, self.variables)
            }
            renames = {cv: src for cv, src in m.items() if _is_var(src, self.variables)}
            t = t.evaluate(fixed, in_place=False) if fixed else t.copy()
            tables.append(t.rename(renames))
        return _ProverVirtual(self.variables, tables, self.terms)

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


class _ProverVirtual:
    """A virtual oracle with concrete constituent tables, for the prover's
    sumcheck rounds: `round_evals` gives the round polynomial at the nodes
    0..degree, `evaluate` folds the tables holding the round variable."""

    def __init__(self, variables: list, tables: list, terms: list):
        self.variables = list(variables)
        self.tables = tables
        self.terms = terms

    @property
    def num_vars(self) -> int:
        return len(self.variables)

    @property
    def degree(self) -> int:
        return max(len(idxs) for _, idxs in self.terms)

    def evaluate(self, point: dict | list, in_place: bool = True) -> _ProverVirtual:
        if isinstance(point, list):
            point = dict(zip(self.variables, point, strict=True))
        tables = [t.evaluate(point, in_place=in_place) for t in self.tables]
        variables = [v for v in self.variables if v not in point]
        if in_place:
            self.tables, self.variables = tables, variables
            return self
        return _ProverVirtual(variables, tables, self.terms)

    def constant(self):
        assert self.num_vars == 0, "constant() needs a fully-evaluated oracle"
        return _combine(self.terms, [_constant(t) for t in self.tables])

    def round_evals(self, var=None) -> tuple:
        """The round message over `var` (default: the first variable): the
        evaluations at t = 0..degree of g(t) = sum_b p*(t, b), b over the
        hypercube of the other variables. Every factor is multilinear, so
        its value at node t is `lo + t * (hi - lo)` from its two hypercube
        neighbours; a factor without the round variable is constant in t.
        """
        var = self.variables[0] if var is None else var
        rest = [v for v in self.variables if v is not var]
        nodes = self.degree + 1
        evals: list = [None] * nodes
        lookups = [_Lookup(t) for t in self.tables]
        for b in _hypercube(rest):
            lo = [lk.at({**b, var: 0}) for lk in lookups]
            hi = [lk.at({**b, var: 1}) for lk in lookups]
            for t in range(nodes):
                vals = [
                    lo_j if t == 0 else hi_j if t == 1 else lo_j + t * (hi_j - lo_j)
                    for lo_j, hi_j in zip(lo, hi, strict=True)
                ]
                total = _combine(self.terms, vals)
                evals[t] = total if evals[t] is None else evals[t] + total
        return tuple(evals)


class _Lookup:
    """Hypercube reads of one factor table: an index computation for a dense
    evaluation-basis MLE, a fold for anything else."""

    def __init__(self, table):
        self.table = table
        self.direct = isinstance(table, MLE) and table.basis is MLE_Basis.eval
        self.positions = {v: i for i, v in enumerate(table.variables)}

    def at(self, b: dict):
        if self.direct:
            idx = 0
            for v, i in self.positions.items():
                idx |= b[v] << i
            return self.table.table[idx]
        return _constant(self.table.evaluate(b, in_place=False))


class ImplicitOracle:
    """`L(z) = sum_{b in {0,1}^m} body(z, b)`: the free variables `variables`
    (z), the summed ones `summed` (b), and the `body`, a virtual oracle over
    both. The verifier has nothing of L but this definition; a claim
    `L(r) = v` is the Sum claim `sum_b body(r, b) = v` (`instantiate`)."""

    kind = OracleKind.implicit
    public = False

    def __init__(self, variables: list, body: VirtualOracle, summed: list):
        self.variables = list(variables)
        self.body = body
        self.summed = list(summed)
        expected = set(map(id, self.variables)) | set(map(id, self.summed))
        if expected != set(map(id, body.variables)) or len(expected) != len(
            body.variables
        ):
            raise ValueError("body variables must be the free and summed ones")

    @property
    def num_vars(self) -> int:
        return len(self.variables)

    def dependencies(self) -> list:
        return self.body.dependencies()

    def instantiate(self, point: dict) -> VirtualOracle:
        """The summed oracle of the Sum claim `L(point) = v`: the body with
        the free variables bound (symbolically), over the summed ones."""
        if any(v not in point for v in self.variables):
            raise ValueError("instantiate needs every free variable bound")
        g = self.body.evaluate({v: point[v] for v in self.variables}, in_place=False)
        g.variables = [v for v in self.body.variables if v in self.summed]
        return g

    def evaluate(self, point: dict | list, in_place: bool = False) -> ImplicitOracle:
        """Bind free variables symbolically (a partially evaluated L)."""
        if isinstance(point, list):
            point = dict(zip(self.variables, point, strict=True))
        point = {v: val for v, val in point.items() if v in self.variables}
        body = self.body.evaluate(point, in_place=in_place)
        variables = [v for v in self.variables if v not in point]
        if in_place:
            self.body, self.variables = body, variables
            return self
        return ImplicitOracle(variables, body, self.summed)

    def constant(self):
        """The value of a fully-bound L: the hypercube sum of its body (the
        ideal decider — exponential)."""
        assert self.num_vars == 0, "constant() needs a fully-evaluated oracle"
        total = None
        for b in _hypercube(self.summed):
            e = self.body.evaluate(b, in_place=False).constant()
            total = e if total is None else total + e
        return total

    def materialize(self, witnesses: dict | None = None) -> MLE:
        """L's hypercube table from concrete constituent tables — the
        generic (slow) prover-side route; a protocol that knows the
        structure (a circuit) computes the table directly instead."""
        view = self.body.prover_view(witnesses)
        lookups = [_Lookup(t) for t in view.tables]
        evaluations = []
        for z in _hypercube(self.variables):
            total = None
            for b in _hypercube(self.summed):
                vals = [lk.at({**z, **b}) for lk in lookups]
                e = _combine(view.terms, vals)
                total = e if total is None else total + e
            evaluations.append(total)
        return _table_like(view.tables, self.variables, evaluations)

    def digest(self) -> bytes:
        parts = [b"implicit"]
        parts += [element_digest(v) for v in self.variables]
        parts += [element_digest(v) for v in self.summed]
        parts.append(self.body.digest())
        return _chain(parts)

    def __repr__(self) -> str:
        return f"ImplicitOracle(vars={self.num_vars}, summed={len(self.summed)})"


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


class ImplicitEval(Protocol):
    """Every pending `Eval(L, r_i, v_i)` on one implicit oracle -> one
    `Sum(sum_i alpha_i body(r_i, .), sum_i alpha_i v_i)` (piop.md §4).

    A bundle protocol: the driver parks claims on L until nothing else on
    the frontier can produce one, then hands them over together. The
    verifier draws one independent challenge per claim (none when there is
    a single claim), the combined summed oracle is itself a virtual oracle
    (`VirtualOracle.linear` of the instantiated definitions), and the
    registered sumcheck does the rest — ending in an Eval claim on that
    virtual oracle, which `VirtualEval` turns into claims on L's
    constituents. Zero communication; soundness error (k-1)/|A| for k
    claims [Sor22, Lem. 11; CCCFGS25, Eqs. (19)-(20)].
    """

    name = "implicit"
    reduces_from: type[Relation] = Relation_Eval
    reduces_to: tuple[type[Relation], ...] = (Relation_Sum,)
    batching = True

    def soundness_error(self, statements: list[Statement], domain) -> float | None:
        from .sumcheck import _exceptional_set_size

        size = _exceptional_set_size(domain)
        if size is None:
            return None
        return (len(statements) - 1) / size

    async def _combine(self, party, statements: list[Statement]) -> list[Statement]:
        iop = _iop_of(party)
        for s in statements:
            await s.resolved()
        first = statements[0]
        oracle = first.oracles[0]
        if any(s.oracles[0] is not oracle for s in statements):
            raise ValueError("a bundle of implicit claims must share the oracle")
        label = f"{self.name}{first.path}"
        if len(statements) == 1:
            alphas = [1]
        else:
            alphas = [
                iop.verifier.challenge(f"{label}/a{i}") for i in range(len(statements))
            ]
        bodies = [oracle.instantiate(dict(s.point)) for s in statements]
        combined = VirtualOracle.linear(alphas, bodies)
        value = None
        for alpha, s in zip(alphas, statements, strict=True):
            t = _scale(alpha, s.value)
            value = t if value is None else value + t
        return [self.reduce(statements, Relation_Sum, oracles=[combined], value=value)]

    async def prove(
        self, prover: Prover, statements: list[Statement]
    ) -> list[Statement]:
        return await self._combine(prover, statements)

    async def verify(
        self, verifier: Verifier, statements: list[Statement]
    ) -> list[Statement]:
        return await self._combine(verifier, statements)


def _iop_of(party) -> IOP:
    iop = party.iop
    if iop is None:
        raise RuntimeError("this party is not bound to an IOP")
    return iop
