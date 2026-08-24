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
from vfhe.misc.libvfhe import ffi, lib

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


def handle_array(polys: list):
    """C array of RNS_Polynomial handles for a list of Polynomials."""
    return ffi.new("void*[]", [p.obj for p in polys])


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
      (`mle_dense_poly_*`) do the work; without one, entries are plain
      Python values (any type with `+` and `*`, e.g. ints) folded in Python.
      The kernels additionally require the evaluation basis, so
      `native_table()`, not `isinstance`, is what protocols gate on.

    Variable order is generic: any variable may be bound at any position, in
    any order (`evaluate`). `_bind` dispatches on the variable's *position*
    to the layout implemented most efficiently: adjacent entries for the
    first (LSB) variable, the two table halves for the last (MSB) one, and
    stride-computed pairs for anything in between.

    `table` holds the entries (also keeping them alive across C calls) and
    `table_ptr` is the array of their handles the kernels take (None without
    a ring) — the two are always views of the same entries, replaced
    together by `_set_table`.
    """

    def __init__(
        self,
        ring: Ring | None = None,
        variables: list | None = None,
        evaluations: list | None = None,
        coefficients: list | None = None,
        num_vars: int | None = None,
    ):
        if variables is not None:
            self.variables = list(variables)
        elif num_vars is not None:
            self.variables = _default_variables(num_vars)
        else:
            raise ValueError("Either variables or num_vars must be provided")
        if evaluations is not None and coefficients is not None:
            raise TypeError("pass evaluations or coefficients, not both")
        self.ring = ring
        self.basis = MLE_Basis.coeff if coefficients is not None else MLE_Basis.eval

        size = 1 << self.num_vars
        entries = coefficients if coefficients is not None else evaluations
        if entries is None:
            entries = [Polynomial(ring) for _ in range(size)] if ring else [0] * size
        else:
            assert len(entries) == size, f"table length must be {size}"
            entries = [self._entry(e) for e in entries] if ring else list(entries)
        self._set_table(entries)

    @property
    def num_vars(self) -> int:
        """The number of free variables; derived, so it cannot go stale."""
        return len(self.variables)

    def _entry(self, item) -> Polynomial:
        """A ring-backed table entry: Polynomials are adopted as they are (so
        kernels can fill a freshly allocated table), integers are lifted to
        constants."""
        if isinstance(item, Polynomial):
            return item
        if isinstance(item, int):
            return Polynomial(self.ring).from_array([item])
        raise TypeError("Entries of a ring-backed table must be Polynomial or integer")

    def _set_table(self, entries: list) -> None:
        """Install a table and the handle array the kernels take with it."""
        self.table = entries
        self.table_ptr = handle_array(entries) if self.ring is not None else None

    @classmethod
    def _like(cls, src: MLE, entries: list, basis=None) -> MLE:
        """A table with `src`'s variables and ring, holding `entries` (in
        `src`'s basis unless another is given)."""
        basis = src.basis if basis is None else basis
        key = "coefficients" if basis is MLE_Basis.coeff else "evaluations"
        return cls(ring=src.ring, variables=src.variables, **{key: entries})

    @classmethod
    def eq(cls, ring: Ring, point: list, variables: list | None = None) -> MLE:
        """The multilinear equality polynomial eq~(point, .) as a dense table:
        table[b] = prod_i (point_i * b_i + (1 - point_i) * (1 - b_i)), the
        chi_w Lagrange basis of [Tha22, section 3.5]. `point` entries are
        Polynomials (or ints, lifted to constants); entry i pairs with
        variable i (the table's LSB-first order)."""
        one = Polynomial(ring).from_array([1])
        table = [one]
        for z in point:
            if not isinstance(z, Polynomial):
                z = Polynomial(ring).from_array([int(z)])
            nz = one - z
            # Appending variable i doubles the table: bit i = 0 keeps the
            # (1 - z_i) branch, bit i = 1 (the new MSB half) the z_i branch.
            table = [t * nz for t in table] + [t * z for t in table]
        if variables is None:
            variables = _default_variables(len(point))
        return cls(ring=ring, variables=variables, evaluations=table)

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

    def _entries_copy(self) -> list:
        """A table whose entries can be reassigned without touching self's."""
        if self.ring is not None:
            return [p.copy() for p in self.table]
        return list(self.table)

    def to_coefficients(self) -> MLE:
        """This MLE in the monomial basis, index-aligned with the evaluation
        table (coefficient of prod_{i in bits(b)} x_i at position b;
        LSB-first variable order). Per variable the butterfly is
        c_hi = e_hi - e_lo, c_lo = e_lo; self is left untouched."""
        if self.basis is MLE_Basis.coeff:
            return self.copy()
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
        assert self.basis is other.basis, "Bases must match"

    def _elementwise(self, kernel, *args) -> MLE:
        """A same-shape table filled by an elementwise kernel, whose
        signature is `(out, in, *args, size)`. Elementwise kernels are
        linear, so they apply in either basis."""
        self.to_NTT()
        size = 1 << self.num_vars
        new_table = [Polynomial(self.ring) for _ in range(size)]
        res = MLE._like(self, new_table)
        kernel(res.table_ptr, self.table_ptr, *args, size)
        mark_ntt(new_table)
        return res

    def _combine(self, other, op, kernel) -> MLE:
        self._check_compatible(other, op.__name__)
        if self.ring is not None:
            other.to_NTT()
            return self._elementwise(kernel, other.table_ptr)
        return MLE._like(
            self, [op(a, b) for a, b in zip(self.table, other.table, strict=True)]
        )

    def __add__(self, other):
        return self._combine(other, operator.add, lib.mle_dense_poly_add)

    def __sub__(self, other):
        return self._combine(other, operator.sub, lib.mle_dense_poly_sub)

    def scale(self, factor):
        if self.ring is None:
            return MLE._like(self, [c * factor for c in self.table])
        if isinstance(factor, Polynomial):
            factor.to_NTT()
            return self._elementwise(lib.mle_dense_poly_scale, factor.obj)
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

    def _run_bind(self, poly_kernel, scalar_kernel, val, *tail) -> None:
        """Fold into a freshly allocated half-size table and install it. The
        kernel pair differs only in the type of `val` (a ring element or a
        small integer); `tail` completes the chosen kernel's signature."""
        self.to_NTT()
        new_table = [Polynomial(self.ring) for _ in range(1 << (self.num_vars - 1))]
        new_ptr = handle_array(new_table)
        if isinstance(val, Polynomial):
            val.to_NTT()
            poly_kernel(new_ptr, self.table_ptr, val.obj, *tail)
        else:
            scalar_kernel(new_ptr, self.table_ptr, int(val), *tail)
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
        t = self.table
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
        t = self.table
        self._set_table(
            [self._fold(t[lo], t[hi], val) for lo, hi in _pair_indices(len(t), idx)]
        )

    def constant(self):
        """The single value of a fully-evaluated (0-variable) MLE."""
        assert self.num_vars == 0, "constant() needs a fully-evaluated MLE"
        return self.table[0]

    def copy(self) -> MLE:
        return MLE._like(self, self._entries_copy())


class SparseMLE:
    """A sparse map of hypercube evaluations: a bookkeeping form, not a
    dense `MLE`.

    Independent of `MLE` on purpose — it supports the linear operations
    (add / sub / scale) but none of the folding a dense table exists for, so
    inheriting `evaluate` would only promise what it cannot do. Convert to
    an `MLE` to evaluate.
    """

    def __init__(
        self,
        variables: list | None = None,
        evaluations: dict | None = None,
        num_vars: int | None = None,
    ):
        if variables is not None:
            self.variables = list(variables)
        elif num_vars is not None:
            self.variables = _default_variables(num_vars)
        else:
            raise ValueError("Either variables or num_vars must be provided")
        self.evaluations = dict(evaluations) if evaluations is not None else {}

    @property
    def num_vars(self) -> int:
        """The number of free variables; derived, so it cannot go stale."""
        return len(self.variables)

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
        return SparseMLE(variables=self.variables, evaluations=new_evals)

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
        return SparseMLE(variables=self.variables, evaluations=new_evals)

    def __mul__(self, other):
        if isinstance(other, (MLE, SparseMLE)):
            raise TypeError(
                "MLE * MLE is not defined; product claims are Relation_SumProd"
            )
        return self.scale(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def evaluate(self, point: dict | list, in_place: bool = True):
        raise NotImplementedError(
            "SparseMLE cannot be evaluated; build a dense MLE to fold variables"
        )

    def constant(self):
        """The single value of a fully-evaluated (0-variable) MLE."""
        assert self.num_vars == 0, "constant() needs a fully-evaluated MLE"
        return self.evaluations.get(0, 0)

    def copy(self) -> SparseMLE:
        return SparseMLE(variables=self.variables, evaluations=self.evaluations)
