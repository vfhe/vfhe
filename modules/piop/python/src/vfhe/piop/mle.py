# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Multilinear extensions (piop.md §7).

This layer is pure math and asyncio-free: evaluation points are always
concrete values. Unresolved protocol values (challenges not drawn yet) live
at the Transcript / Statement level in piop.py — `Statement.resolved()`
awaits them before any decider evaluates an MLE.
"""

from __future__ import annotations

from vfhe.arith import Polynomial, Ring
from vfhe.misc.libvfhe import ffi, lib


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


def _handle_array(polys: list):
    """C array of RNS_Polynomial handles for a list of Polynomials."""
    return ffi.new("void*[]", [p.obj for p in polys])


def _sync_repr(new_polys: list, source: Polynomial):
    """The mle_dense_poly_* kernels are elementwise and leave their outputs
    in the inputs' representation, but the fresh Polynomial wrappers still
    carry the default (empty) repr flag; mirror the source table's flag so
    later Polynomial arithmetic on the entries converts domains correctly."""
    for p in new_polys:
        p.repr = source.repr


class MLE:
    def __init__(self, variables: list | None = None, num_vars: int | None = None):
        if variables is not None:
            self.variables = list(variables)
            self.num_vars = len(self.variables)
        elif num_vars is not None:
            self.variables = [MLE_Variable(f"var_{i}") for i in range(num_vars)]
            self.num_vars = num_vars
        else:
            raise ValueError("Either variables or num_vars must be provided")

    def __add__(self, other):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    def scale(self, factor):
        raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, MLE):
            raise TypeError(
                "MLE * MLE is not defined; product claims are Relation_SumProd"
            )
        return self.scale(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def evaluate(self, point: dict | list, in_place: bool = True):
        """Bind the variables in `point` to concrete values; variables not
        in `point` stay free. Returns the folded MLE."""
        if isinstance(point, list):
            normalized_point = dict(zip(self.variables, point, strict=True))
        elif isinstance(point, dict):
            normalized_point = point.copy()
        else:
            raise TypeError("point must be a list or a dict")
        return self._evaluate_sync(normalized_point, in_place=in_place)

    def _evaluate_sync(self, point: dict, in_place: bool = True):
        raise NotImplementedError

    def constant(self):
        """The single value of a fully-evaluated (0-variable) MLE."""
        raise NotImplementedError


class ML_Polynomial(MLE):
    def __init__(
        self,
        variables: list | None = None,
        coefficients: list | None = None,
        num_vars: int | None = None,
    ):
        super().__init__(variables=variables, num_vars=num_vars)
        if coefficients is not None:
            self.coefficients = list(coefficients)
        else:
            self.coefficients = [0] * (1 << self.num_vars)
        assert len(self.coefficients) == (1 << self.num_vars), (
            "coefficients list length must be 2^num_vars"
        )

    def __add__(self, other):
        if not isinstance(other, ML_Polynomial):
            raise TypeError("Can only add ML_Polynomial to ML_Polynomial")
        assert self.variables == other.variables, "Variables must match"
        new_coeffs = [
            c1 + c2
            for c1, c2 in zip(self.coefficients, other.coefficients, strict=True)
        ]
        return ML_Polynomial(variables=self.variables, coefficients=new_coeffs)

    def __sub__(self, other):
        if not isinstance(other, ML_Polynomial):
            raise TypeError("Can only subtract ML_Polynomial from ML_Polynomial")
        assert self.variables == other.variables, "Variables must match"
        new_coeffs = [
            c1 - c2
            for c1, c2 in zip(self.coefficients, other.coefficients, strict=True)
        ]
        return ML_Polynomial(variables=self.variables, coefficients=new_coeffs)

    def scale(self, factor):
        new_coeffs = [c * factor for c in self.coefficients]
        return ML_Polynomial(variables=self.variables, coefficients=new_coeffs)

    def _evaluate_sync(self, point: dict, in_place: bool = True):
        target = self if in_place else self.copy()

        for var, val in point.items():
            if var in target.variables:
                idx = target.variables.index(var)
                stride = 1 << idx
                new_size = 1 << (len(target.variables) - 1)
                new_coeffs = [None] * new_size
                for i in range(new_size):
                    i_low = i & (stride - 1)
                    i_high = i >> idx
                    idx0 = i_low + (i_high << (idx + 1))
                    idx1 = idx0 + stride
                    new_coeffs[i] = (
                        target.coefficients[idx0] + val * target.coefficients[idx1]
                    )
                target.coefficients = new_coeffs
                target.variables = target.variables[:idx] + target.variables[idx + 1 :]
                target.num_vars = len(target.variables)
        return target

    def constant(self):
        assert self.num_vars == 0, "constant() needs a fully-evaluated MLE"
        return self.coefficients[0]

    def copy(self) -> ML_Polynomial:
        return ML_Polynomial(
            variables=list(self.variables), coefficients=list(self.coefficients)
        )


class MLE_Sparse(MLE):
    def __init__(
        self,
        variables: list | None = None,
        evaluations: dict | None = None,
        num_vars: int | None = None,
    ):
        super().__init__(variables=variables, num_vars=num_vars)
        if evaluations is not None:
            self.evaluations = dict(evaluations)
        else:
            self.evaluations = {}

    def __add__(self, other):
        if not isinstance(other, MLE_Sparse):
            raise TypeError("Can only add MLE_Sparse to MLE_Sparse")
        assert self.variables == other.variables, "Variables must match"
        new_evals = {}
        all_keys = set(self.evaluations.keys()) | set(other.evaluations.keys())
        for k in all_keys:
            v1 = self.evaluations.get(k, 0)
            v2 = other.evaluations.get(k, 0)
            res = v1 + v2
            if res != 0:
                new_evals[k] = res
        return MLE_Sparse(variables=self.variables, evaluations=new_evals)

    def __sub__(self, other):
        if not isinstance(other, MLE_Sparse):
            raise TypeError("Can only subtract MLE_Sparse from MLE_Sparse")
        assert self.variables == other.variables, "Variables must match"
        new_evals = {}
        all_keys = set(self.evaluations.keys()) | set(other.evaluations.keys())
        for k in all_keys:
            v1 = self.evaluations.get(k, 0)
            v2 = other.evaluations.get(k, 0)
            res = v1 - v2
            if res != 0:
                new_evals[k] = res
        return MLE_Sparse(variables=self.variables, evaluations=new_evals)

    def scale(self, factor):
        new_evals = {}
        for k, v in self.evaluations.items():
            res = v * factor
            if res != 0:
                new_evals[k] = res
        return MLE_Sparse(variables=self.variables, evaluations=new_evals)

    def _evaluate_sync(self, point: dict, in_place: bool = True):
        raise NotImplementedError(
            "Evaluation is only supported for MLE_Dense and ML_Polynomial"
        )

    def constant(self):
        assert self.num_vars == 0, "constant() needs a fully-evaluated MLE"
        return self.evaluations.get(0, 0)

    def copy(self) -> MLE_Sparse:
        return MLE_Sparse(
            variables=list(self.variables), evaluations=dict(self.evaluations)
        )


class MLE_Dense(MLE):
    def __init__(
        self,
        ring: Ring,
        variables: list | None = None,
        evaluations: list | None = None,
        num_vars: int | None = None,
    ):
        super().__init__(variables=variables, num_vars=num_vars)
        self.ring = ring

        size = 1 << self.num_vars
        if evaluations is not None:
            assert len(evaluations) == size, f"evaluations length must be {size}"
            self.py_refs = []
            for item in evaluations:
                if isinstance(item, Polynomial):
                    self.py_refs.append(item)
                else:
                    p = Polynomial(ring)
                    if isinstance(item, int):
                        p.from_array([item])
                    else:
                        raise TypeError("Evaluations must be Polynomial or integer")
                    self.py_refs.append(p)
        else:
            self.py_refs = [Polynomial(ring) for _ in range(size)]

        self.data_ptr = _handle_array(self.py_refs)

    @classmethod
    def _from_polys(cls, ring: Ring, variables: list, polys: list) -> MLE_Dense:
        """Wrap an existing Polynomial table without allocating a fresh one
        (the public constructor always allocates 2^n polynomials)."""
        res = cls.__new__(cls)
        res.variables = list(variables)
        res.num_vars = len(res.variables)
        res.ring = ring
        res.py_refs = polys
        res.data_ptr = _handle_array(polys)
        return res

    def __add__(self, other):
        if not isinstance(other, MLE_Dense):
            raise TypeError("Can only add MLE_Dense to MLE_Dense")
        assert self.variables == other.variables, "Variables must match"
        assert self.ring == other.ring, "Rings must match"

        size = 1 << self.num_vars
        new_polys = [Polynomial(self.ring) for _ in range(size)]
        res = MLE_Dense._from_polys(self.ring, self.variables, new_polys)

        lib.mle_dense_poly_add(res.data_ptr, self.data_ptr, other.data_ptr, size)
        _sync_repr(new_polys, self.py_refs[0])
        return res

    def __sub__(self, other):
        if not isinstance(other, MLE_Dense):
            raise TypeError("Can only subtract MLE_Dense from MLE_Dense")
        assert self.variables == other.variables, "Variables must match"
        assert self.ring == other.ring, "Rings must match"

        size = 1 << self.num_vars
        new_polys = [Polynomial(self.ring) for _ in range(size)]
        res = MLE_Dense._from_polys(self.ring, self.variables, new_polys)

        lib.mle_dense_poly_sub(res.data_ptr, self.data_ptr, other.data_ptr, size)
        _sync_repr(new_polys, self.py_refs[0])
        return res

    def scale(self, factor):
        size = 1 << self.num_vars
        new_polys = [Polynomial(self.ring) for _ in range(size)]
        res = MLE_Dense._from_polys(self.ring, self.variables, new_polys)

        if isinstance(factor, Polynomial):
            lib.mle_dense_poly_scale(res.data_ptr, self.data_ptr, factor.obj, size)
        else:
            lib.mle_dense_poly_scale_scalar(
                res.data_ptr, self.data_ptr, int(factor), size
            )
        _sync_repr(new_polys, self.py_refs[0])
        return res

    def _evaluate_sync(self, point: dict, in_place: bool = True):
        target = self if in_place else self.copy()

        for var, val in point.items():
            if var in target.variables:
                idx = target.variables.index(var)

                new_num_vars = len(target.variables) - 1
                new_size = 1 << new_num_vars

                new_polys = [Polynomial(target.ring) for _ in range(new_size)]
                new_data_ptr = _handle_array(new_polys)

                if isinstance(val, Polynomial):
                    lib.mle_dense_poly_evaluate(
                        new_data_ptr,
                        target.data_ptr,
                        val.obj,
                        len(target.variables),
                        idx,
                    )
                else:
                    lib.mle_dense_poly_evaluate_scalar(
                        new_data_ptr,
                        target.data_ptr,
                        int(val),
                        len(target.variables),
                        idx,
                    )

                _sync_repr(new_polys, target.py_refs[0])
                target.data_ptr = new_data_ptr
                target.py_refs = new_polys
                target.variables = target.variables[:idx] + target.variables[idx + 1 :]
                target.num_vars = new_num_vars

        return target

    def constant(self):
        assert self.num_vars == 0, "constant() needs a fully-evaluated MLE"
        return self.py_refs[0]

    def copy(self) -> MLE_Dense:
        return MLE_Dense._from_polys(
            self.ring, self.variables, [p.copy() for p in self.py_refs]
        )
