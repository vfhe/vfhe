# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Multilinear extensions (piop.md §7).

`MLE` is the dense table protocols work with; `SparseMLE` is an independent
bookkeeping form (a sparse map of hypercube evaluations).
"""

from __future__ import annotations

import operator
from enum import Enum

from vfhe.arith import Polynomial, Ring, repr
from vfhe.engine import ffi, lib

# Which basis a dense table is written in — a property of the table, not of
# its domain class (piop.md §7).
MLE_Basis = Enum("MLE_Basis", ["eval", "coeff"])


class MLE_Variable:
    """A named variable identifier for MLEs, compared by identity.

    Any hashable works as an MLE variable; this class is the default used
    when only `num_vars` is given. It is a plain name — protocol futures
    (`piop.Variable`) are a different, unrelated object.
    """

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"MLE_Variable({self.name!r})"


def _default_variables(num_vars: int) -> list[MLE_Variable]:
    """The anonymous variables used when only a count is given."""
    return [MLE_Variable(f"var_{i}") for i in range(num_vars)]


def _pair_indices(size: int, idx: int):
    """The (lo, hi) index pairs of the variable at position `idx` in a table
    of `size` entries: the entries differing only in bit `idx`, yielded in
    the folded table's output order. This is the index arithmetic the
    strided C kernels use; the first (LSB) and last (MSB) variables have the
    cheaper layouts `(2i, 2i+1)` and `(i, i + size/2)`."""
    stride = 1 << idx
    for i in range(size // 2):
        lo = (i & (stride - 1)) + ((i >> idx) << (idx + 1))
        yield lo, lo + stride


def element_array(polys: list):
    """C array of `ArithElement` for a list of ring elements.

    A view: each entry points at the element's own storage, so the array must
    not outlive them. It also carries the domain each element is in, which the
    kernels check, so `stamp_domains` has to run after anything that changes a
    representation without rebuilding the array.
    """
    array = ffi.new("ArithElement[]", len(polys))
    for slot, poly in zip(array, polys, strict=True):
        slot.handle = poly.obj
        slot.domain = domain_of(poly.repr)
    return array


def stamp_domains(array, polys: list) -> None:
    """Refresh an element array's domains from the elements it views."""
    for slot, poly in zip(array, polys, strict=True):
        slot.domain = domain_of(poly.repr)


def mark_ntt(polys: list) -> None:
    """Flag fresh kernel outputs as NTT-form.

    The kernels are RNS routines, so with NTT-form inputs — which
    `MLE.to_NTT()` guarantees before every call — their outputs are NTT-form
    too; but the fresh `Polynomial` wrappers they filled still carry the
    default `empty` flag. Without this, later arithmetic on the entries
    converts domains wrongly (`Polynomial.__mul__`'s `to_NTT()` would
    transform already-transformed data and yield ~q-sized junk). Stamp the
    output rather than copying a source entry's flag, which may describe a
    different representation than the kernel just wrote.
    """
    for p in polys:
        p.repr = repr.ntt


def native_table(f) -> bool:
    """Whether the C kernels can take `f` directly.

    They can when it is a dense `MLE` of `vfhe.arith.Polynomial` entries (a
    `ring`) in the **evaluation** basis: the kernels are RNS_Polynomial
    routines, and the binding ones interpolate (`lo + a*(hi - lo)`), which
    is the evaluation-basis fold. Coefficient-basis tables, tables over
    plain Python values, and `SparseMLE` take the pure-Python path.
    Protocols gate their native delegation on this (piop.md §5).
    """
    return isinstance(f, MLE) and f.ring is not None and f.basis is MLE_Basis.eval


def vector_table(f) -> bool:
    """Whether `f` is a field-backed table in the evaluation basis.

    Such a table is one `vfhe.arith.FieldVector`, so a protocol computes a
    round message with a fixed number of whole-vector operations (split,
    multiply, sum) instead of a loop over entries. Like `native_table`, the
    evaluation basis is part of the contract: the round-message shortcuts
    interpolate the (lo, hi) pairs, which is the evaluation-basis fold.
    """
    return isinstance(f, MLE) and f.field is not None and f.basis is MLE_Basis.eval


class MLE:
    """A multilinear extension as a dense table of 2^n coefficients.

    Two properties of the table, deliberately not subclasses, because they
    vary independently:

    - **`basis`** — `MLE_Basis.eval`, the hypercube evaluations (pass
      `evaluations=`), or `MLE_Basis.coeff`, the monomial coefficients (pass
      `coefficients=`, entry `b` multiplying prod_{i in bits(b)} x_i, i.e. a
      multilinear *polynomial* rather than an extension table). Both bind a
      variable by folding (lo, hi) pairs; only the fold differs —
      interpolation `lo + r*(hi - lo)` in the evaluation basis, Horner
      `c_lo + r*c_hi` in the monomial one. `to_coefficients()` converts.
    - **coefficient type** — with a `ring`, entries are
      `vfhe.arith.Polynomial` over that `Ring` and the C kernels
      (`mle_dense_poly_*`) do the work; with a `field`, the table is one
      `vfhe.arith.FieldVector` over that `Field` and every fold is a
      handful of whole-vector operations; without either, entries are plain
      Python values (any type with `+` and `*`, e.g. ints) folded in Python.
      The kernels additionally require the evaluation basis, so
      `native_table()` / `vector_table()`, not `isinstance`, is what
      protocols gate on.

    Variable order is generic: any variable may be bound at any position, in
    any order (`evaluate`). `_bind` dispatches on the variable's *position*
    to the layout implemented most efficiently: adjacent entries for the
    first (LSB) variable, the two table halves for the last (MSB) one, and
    stride-computed pairs for anything in between.

    `public` marks a table both parties hold — a wiring predicate, an eq~
    table, a public output — so that an evaluation claim on it is decided
    by the verifier's own evaluation and never becomes an opening claim;
    every derived table (`copy`, out-of-place `evaluate`, `scale`, `+`/`-`
    of two public tables) inherits the mark. Anything else is a witness
    oracle: what happens to a claim on it is decided by the protocol
    registered for its kind (piop.md §5).

    `table` holds the entries (also keeping them alive across C calls) and
    `table_ptr` is the array of their handles the kernels take (None without
    a ring) — the two are always views of the same entries, replaced
    together by `_set_table`. Over a field, `table` is the `FieldVector`
    itself (indexable and iterable like the list it replaces).
    """

    def __init__(
        self,
        ring: Ring | None = None,
        variables: list | None = None,
        evaluations: list | None = None,
        coefficients: list | None = None,
        num_vars: int | None = None,
        field: Field | None = None,
        public: bool = False,
    ):
        if variables is not None:
            self.variables = list(variables)
        elif num_vars is not None:
            self.variables = _default_variables(num_vars)
        else:
            raise ValueError("Either variables or num_vars must be provided")
        if evaluations is not None and coefficients is not None:
            raise TypeError("pass evaluations or coefficients, not both")
        if ring is not None and field is not None:
            raise TypeError("pass ring or field, not both")
        self.ring = ring
        self.field = field
        self.public = public
        self.basis = MLE_Basis.coeff if coefficients is not None else MLE_Basis.eval

        size = 1 << self.num_vars
        entries = coefficients if coefficients is not None else evaluations
        if entries is None:
            if ring is not None:
                entries = [Polynomial(ring) for _ in range(size)]
            elif field is not None:
                entries = FieldVector(field, size)
            else:
                entries = [0] * size
        else:
            assert len(entries) == size, f"table length must be {size}"
            if ring is not None:
                entries = [self._entry(e) for e in entries]
            elif field is None:
                entries = list(entries)
        self._set_table(entries)

    @property
    def num_vars(self) -> int:
        """The number of free variables; derived, so it cannot go stale."""
        return len(self.variables)

    @property
    def _ring(self) -> Ring:
        """The ring of a ring-backed table; the kernel paths require one."""
        if self.ring is None:
            raise ValueError("this MLE has no ring backing")
        return self.ring

    def _entry(self, item) -> Polynomial:
        """A ring-backed table entry: Polynomials are adopted as they are (so
        kernels can fill a freshly allocated table), integers are lifted to
        constants."""
        if isinstance(item, Polynomial):
            return item
        if isinstance(item, int):
            return Polynomial(self._ring).from_array([item])
        raise TypeError("Entries of a ring-backed table must be Polynomial or integer")

    def _set_table(self, entries) -> None:
        """Install a table and the handle array the kernels take with it.

        Over a field the table is a `FieldVector`; a list of entries (elements
        or ints) is packed into one."""
        if self.field is not None and not isinstance(entries, FieldVector):
            entries = FieldVector(self.field, list(entries))
        self.table = entries
        self.table_ptr = element_array(entries) if self.ring is not None else None

    @classmethod
    def _like(cls, src: MLE, entries: list, basis=None) -> MLE:
        """A table with `src`'s variables and ring, holding `entries` (in
        `src`'s basis unless another is given)."""
        basis = src.basis if basis is None else basis
        domain = {"ring": src.ring, "field": src.field, "public": src.public}
        if basis is MLE_Basis.coeff:
            return cls(variables=src.variables, coefficients=entries, **domain)
        return cls(variables=src.variables, evaluations=entries, **domain)

    @classmethod
    def eq(cls, domain, point: list, variables: list | None = None) -> MLE:
        """The multilinear equality polynomial eq~(point, .) as a dense table:
        table[b] = prod_i (point_i * b_i + (1 - point_i) * (1 - b_i)), the
        chi_w Lagrange basis of [Tha22, section 3.5]. `domain` is the Ring
        or Field of the coefficients; `point` entries are its elements (or
        ints, lifted to constants); entry i pairs with variable i (the
        table's LSB-first order)."""
        if variables is None:
            variables = _default_variables(len(point))
        if isinstance(domain, Field):
            one = domain.one
            table = FieldVector(domain, [one])
            for z in point:
                # Same doubling as below, on whole vectors.
                table = type(table).concat([table * (one - z), table * z])
            return cls(
                field=domain, variables=variables, evaluations=table, public=True
            )
        ring = domain
        one = Polynomial(ring).from_array([1])
        table = [one]
        for z in point:
            if not isinstance(z, Polynomial):
                z = Polynomial(ring).from_array([int(z)])
            nz = one - z
            # Appending variable i doubles the table: bit i = 0 keeps the
            # (1 - z_i) branch, bit i = 1 (the new MSB half) the z_i branch.
            table = [t * nz for t in table] + [t * z for t in table]
        return cls(ring=ring, variables=variables, evaluations=table, public=True)

    def to_NTT(self) -> None:
        """Put every entry in NTT (RNS) form, the representation the C
        kernels read; a no-op on a table of plain Python values.

        The kernels read `coeffs` directly and fold the whole table as if it
        were uniform, so a table holding a mix of representations would give
        silent garbage; this establishes their precondition instead. Cheap when
        the table is already normalized: each entry's `to_NTT()` is then just a
        flag check. (Entries no longer *become* mixed just from being read --
        arith's readers convert a copy -- but a table can still be assembled
        from coefficient-form elements.)
        """
        if self.ring is None:
            return
        for p in self.table:
            p.to_NTT()
        # the array the kernels read carries the domain too, and they refuse
        # to combine elements that disagree
        stamp_domains(self.table_ptr, self.table)

    def _entries_copy(self) -> list:
        """A table whose entries can be reassigned without touching self's."""
        if self.ring is not None:
            return [p.copy() for p in self.table]
        if self.field is not None:
            return self.table.copy()
        return list(self.table)

    def to_coefficients(self) -> MLE:
        """This MLE in the monomial basis, index-aligned with the evaluation
        table (coefficient of prod_{i in bits(b)} x_i at position b;
        LSB-first variable order). Per variable the butterfly is
        c_hi = e_hi - e_lo, c_lo = e_lo; self is left untouched."""
        if self.basis is MLE_Basis.coeff:
            return self.copy()
        if self.field is not None:
            # One round per variable on the whole vector: the LSB butterfly
            # (even, odd - even), then the two halves concatenated -- which
            # moves that variable to the MSB, so the next round's LSB is the
            # next variable, and after num_vars rounds the order is restored.
            table = self.table
            for _ in range(self.num_vars):
                even, odd = table.split_even_odd()
                table = type(table).concat([even, odd - even])
            return MLE._like(self, table, basis=MLE_Basis.coeff)
        coeffs = self._entries_copy()
        for i in range(self.num_vars):
            for lo, hi in _pair_indices(len(coeffs), i):
                coeffs[hi] = coeffs[hi] - coeffs[lo]
        return MLE._like(self, coeffs, basis=MLE_Basis.coeff)

    def _check_compatible(self, other, op: str) -> None:
        if not isinstance(other, MLE):
            raise TypeError(f"can only {op} MLE with MLE")
        assert self.variables == other.variables, "Variables must match"
        assert self.ring == other.ring, "Rings must match"
        assert self.field == other.field, "Fields must match"
        assert self.basis is other.basis, "Bases must match"

    def _elementwise(self, kernel, *args) -> MLE:
        """A same-shape table filled by an elementwise kernel, whose
        signature is `(out, in, *args, size)`. Elementwise kernels are
        linear, so they apply in either basis."""
        self.to_NTT()
        size = 1 << self.num_vars
        new_table = [Polynomial(self._ring) for _ in range(size)]
        res = MLE._like(self, new_table)
        kernel(self.ring.arith_ring, res.table_ptr, self.table_ptr, *args, size)
        mark_ntt(new_table)
        return res

    def _combine(self, other, op, kernel) -> MLE:
        self._check_compatible(other, op.__name__)
        if self.ring is not None:
            other.to_NTT()
            res = self._elementwise(kernel, other.table_ptr)
        elif self.field is not None:
            res = MLE._like(self, op(self.table, other.table))
        else:
            res = MLE._like(
                self, [op(a, b) for a, b in zip(self.table, other.table, strict=True)]
            )
        res.public = self.public and other.public
        return res

    def __add__(self, other):
        return self._combine(other, operator.add, lib.mle_dense_poly_add)

    def __sub__(self, other):
        return self._combine(other, operator.sub, lib.mle_dense_poly_sub)

    def scale(self, factor):
        if self.field is not None:
            return MLE._like(self, self.table * factor)
        if self.ring is None:
            return MLE._like(self, [c * factor for c in self.table])
        if isinstance(factor, Polynomial):
            factor.to_NTT()
            return self._elementwise(lib.mle_dense_poly_scale, factor.as_element())
        return self._elementwise(lib.mle_dense_poly_scale_scalar, int(factor))

    def __mul__(self, other):
        if isinstance(other, (MLE, SparseMLE)):
            raise TypeError(
                "MLE * MLE is not defined; product claims are Relation_SumProd"
            )
        return self.scale(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def evaluate(self, point: dict | list, in_place: bool = True) -> MLE:
        """Bind the variables in `point` to concrete values; variables not
        in `point` stay free. Returns the folded MLE."""
        if isinstance(point, dict):
            bindings = point
        elif isinstance(point, list):
            bindings = dict(zip(self.variables, point, strict=True))
        else:
            raise TypeError("point must be a list or a dict")

        target = self if in_place else self.copy()
        for var, val in bindings.items():
            if var in target.variables:
                target._bind(var, val)
        return target

    def _bind(self, var, val) -> None:
        """Bind one variable to `val` in place, dispatching on its position
        (see the class docstring) and dropping it from `variables`."""
        idx = self.variables.index(var)
        if idx == 0:  # also the num_vars == 1 case, where the layouts agree
            self._bind_pairs(val)
        elif idx == self.num_vars - 1:
            self._bind_halves(val)
        else:
            self._bind_generic(val, idx)
        del self.variables[idx]

    def _fold(self, lo, hi, val):
        """One (lo, hi) pair bound to `val`: interpolation in the evaluation
        basis, Horner in the monomial one."""
        if self.basis is MLE_Basis.eval:
            return lo + val * (hi - lo)
        return lo + val * hi

    def _python_entries(self) -> list:
        """The table as a Python list, the operand of the pure-Python fold.
        A field table is unpacked, so a layout the vector cannot express
        (halves, strided pairs) still folds correctly, one element at a
        time."""
        if self.field is not None:
            return self.table.to_list()
        return self.table

    def _run_bind(self, poly_kernel, scalar_kernel, val, *tail) -> None:
        """Fold into a freshly allocated half-size table and install it. The
        kernel pair differs only in the type of `val` (a ring element or a
        small integer); `tail` completes the chosen kernel's signature."""
        self.to_NTT()
        new_table = [Polynomial(self._ring) for _ in range(1 << (self.num_vars - 1))]
        new_ptr = handle_array(new_table)
        if isinstance(val, Polynomial):
            val.to_NTT()
            poly_kernel(
                self.ring.arith_ring, new_ptr, self.table_ptr, val.as_element(), *tail
            )
        else:
            scalar_kernel(
                self.ring.arith_ring, new_ptr, self.table_ptr, int(val), *tail
            )
        mark_ntt(new_table)
        self.table, self.table_ptr = new_table, new_ptr

    def _bind_pairs(self, val) -> None:
        if native_table(self):
            self._run_bind(
                lib.mle_dense_poly_evaluate_pairs,
                lib.mle_dense_poly_evaluate_pairs_scalar,
                val,
                1 << (self.num_vars - 1),
            )
            return
        if self.field is not None:
            if self.basis is MLE_Basis.eval:
                # One fused kernel pass: lo + val * (hi - lo) over the pairs.
                self._set_table(self.table.fold(val))
            else:
                even, odd = self.table.split_even_odd()
                self._set_table(even + odd * val)
            return
        t = self.table
        self._set_table(
            [self._fold(lo, hi, val) for lo, hi in zip(t[0::2], t[1::2], strict=True)]
        )

    def _bind_halves(self, val) -> None:
        if native_table(self):
            self._run_bind(
                lib.mle_dense_poly_evaluate_halves,
                lib.mle_dense_poly_evaluate_halves_scalar,
                val,
                1 << (self.num_vars - 1),
            )
            return
        t = self._python_entries()
        half = len(t) // 2
        self._set_table(
            [self._fold(lo, hi, val) for lo, hi in zip(t[:half], t[half:], strict=True)]
        )

    def _bind_generic(self, val, idx: int) -> None:
        if native_table(self):
            self._run_bind(
                lib.mle_dense_poly_evaluate,
                lib.mle_dense_poly_evaluate_scalar,
                val,
                self.num_vars,
                idx,
            )
            return
        t = self._python_entries()
        self._set_table(
            [self._fold(t[lo], t[hi], val) for lo, hi in _pair_indices(len(t), idx)]
        )

    def constant(self):
        """The single value of a fully-evaluated (0-variable) MLE."""
        assert self.num_vars == 0, "constant() needs a fully-evaluated MLE"
        return self.table[0]

    def copy(self) -> MLE:
        return MLE._like(self, self._entries_copy())

    def rename(self, mapping: dict) -> MLE:
        """A copy of this table with its variables relabelled through
        `mapping` (old variable -> new variable; variables absent from the
        mapping keep their name). The same polynomial read under other
        variable names — how one oracle appears twice in a product, as
        W(x) and W(y)."""
        res = self.copy()
        res.variables = [mapping.get(v, v) for v in self.variables]
        if len(set(map(id, res.variables))) != len(res.variables):
            raise ValueError("rename would give two variables the same name")
        return res


class SparseMLE:
    """A sparse map of hypercube evaluations: `evaluations[index] = value`
    for the nonzero entries, `index` packing the variables LSB-first like a
    dense table's position. The carrier of wiring predicates and other
    mostly-zero public tables.

    Independent of `MLE` on purpose — it supports the linear operations
    (add / sub / scale) and evaluation at a *full* point (`evaluate`, the
    sum over the nonzeros of value times the eq~ factors, O(nnz * vars)),
    but none of the per-variable folding a dense table exists for: binding
    a variable at a time would need the dense form, so a partial point
    raises. Defined oracles (`virtual.py`) bind symbolically and only ever
    query a constituent at a full point, which is what makes a sparse
    constituent sufficient there; `materialize()` gives the dense `MLE` when
    a fold is really wanted. Values are non-negative ints or elements of the
    domain (`ring=` / `field=`, or plain Python values without either).
    """

    def __init__(
        self,
        variables: list | None = None,
        evaluations: dict | None = None,
        num_vars: int | None = None,
        ring: Ring | None = None,
        field: Field | None = None,
        public: bool = False,
    ):
        if variables is not None:
            self.variables = list(variables)
        elif num_vars is not None:
            self.variables = _default_variables(num_vars)
        else:
            raise ValueError("Either variables or num_vars must be provided")
        if ring is not None and field is not None:
            raise TypeError("pass ring or field, not both")
        self.ring = ring
        self.field = field
        self.public = public
        self.evaluations = dict(evaluations) if evaluations is not None else {}
        size = 1 << self.num_vars
        if any(not 0 <= k < size for k in self.evaluations):
            raise ValueError(f"sparse index out of range for {self.num_vars} variables")

    @property
    def num_vars(self) -> int:
        """The number of free variables; derived, so it cannot go stale."""
        return len(self.variables)

    def _like(self, evaluations: dict, variables: list | None = None) -> SparseMLE:
        return SparseMLE(
            variables=self.variables if variables is None else variables,
            evaluations=evaluations,
            ring=self.ring,
            field=self.field,
            public=self.public,
        )

    def _one(self):
        if self.ring is not None:
            return Polynomial(self.ring).from_array([1])
        if self.field is not None:
            return self.field.one
        return 1

    def nonzeros(self):
        """The (index, value) pairs, in insertion order."""
        return self.evaluations.items()

    def evaluate(self, point: dict | list, in_place: bool = False) -> SparseMLE:
        """The value at a full point, as a 0-variable SparseMLE (so that
        `constant()` reads it, like a folded dense table):
        sum_{k nonzero} value_k * prod_i (point_i if k_i else 1 - point_i).
        A partial point raises: this form has no per-variable fold."""
        if isinstance(point, list):
            point = dict(zip(self.variables, point, strict=True))
        if any(v not in point for v in self.variables):
            raise NotImplementedError(
                "SparseMLE evaluates at full points only; materialize() to fold"
            )
        one = self._one()
        factors = [(one - point[v], point[v]) for v in self.variables]
        total = None
        for index, value in self.evaluations.items():
            prod = None
            for i, pair in enumerate(factors):
                f = pair[(index >> i) & 1]
                prod = f if prod is None else prod * f
            if prod is None:  # no variables: the single entry
                term = value
            elif isinstance(value, int) and value == 1:
                term = prod
            else:
                term = prod * value
            total = term if total is None else total + term
        if total is None:
            total = one * 0
        return self._like({0: total}, variables=[])

    def constant(self):
        """The single value of a fully-evaluated (0-variable) SparseMLE."""
        assert self.num_vars == 0, "constant() needs a fully-evaluated SparseMLE"
        return self.evaluations.get(0, self._one() * 0)

    def materialize(self, *_ignored) -> MLE:
        """The dense table with the same evaluations, in the same domain."""
        table = [0] * (1 << self.num_vars)
        for k, v in self.evaluations.items():
            table[k] = v
        return MLE(
            ring=self.ring,
            field=self.field,
            variables=self.variables,
            evaluations=table,
            public=self.public,
        )

    def _combine(self, other, op) -> SparseMLE:
        if not isinstance(other, SparseMLE):
            raise TypeError(f"can only {op.__name__} SparseMLE with SparseMLE")
        assert self.variables == other.variables, "Variables must match"
        # Insertion order, not set order, so the result is deterministic.
        keys = dict.fromkeys([*self.evaluations, *other.evaluations])
        new_evals = {}
        for k in keys:
            res = op(self.evaluations.get(k, 0), other.evaluations.get(k, 0))
            if res != 0:
                new_evals[k] = res
        res_mle = self._like(new_evals)
        res_mle.public = self.public and other.public
        return res_mle

    def __add__(self, other):
        return self._combine(other, operator.add)

    def __sub__(self, other):
        return self._combine(other, operator.sub)

    def scale(self, factor):
        new_evals = {}
        for k, v in self.evaluations.items():
            res = v * factor
            if res != 0:
                new_evals[k] = res
        return self._like(new_evals)

    def __mul__(self, other):
        if isinstance(other, (MLE, SparseMLE)):
            raise TypeError(
                "MLE * MLE is not defined; product claims are Relation_SumProd"
            )
        return self.scale(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def copy(self) -> SparseMLE:
        return self._like(self.evaluations)
