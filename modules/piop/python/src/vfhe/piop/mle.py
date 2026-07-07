import asyncio

from vfhe.arith import Polynomial, Ring
from vfhe.misc.libvfhe import ffi, lib

from .piop import IOPValue, IOPVariable


def _handle_array(polys):
    """C array of RNS_Polynomial handles for a list of Polynomials."""
    return ffi.new("void*[]", [p.obj for p in polys])


class MLE:
    def __init__(self, variables=None, num_vars=None, iop=None):
        if variables is not None:
            self.variables = list(variables)
            self.num_vars = len(self.variables)
        elif num_vars is not None:
            self.variables = []
            for i in range(num_vars):
                name = f"var_{i}"
                if iop is not None:
                    name = f"iop_var_{id(iop)}_{i}"
                self.variables.append(IOPVariable(name))
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
        return self.scale(other)

    def __rmul__(self, other):
        return self.scale(other)

    def evaluate(self, point, in_place=True):
        if isinstance(point, list):
            normalized_point = {var: val for var, val in zip(self.variables, point)}
        elif isinstance(point, dict):
            normalized_point = point.copy()
        else:
            raise TypeError("point must be a list or a dict")

        has_futures = False
        for val in normalized_point.values():
            if isinstance(val, asyncio.Future) and not val.done():
                has_futures = True
                break

        if has_futures:
            loop = asyncio.get_running_loop()
            res_future = IOPValue(loop=loop)

            async def wait_and_eval():
                try:
                    resolved = {}
                    for var, val in normalized_point.items():
                        if isinstance(val, asyncio.Future):
                            resolved[var] = await val
                        else:
                            resolved[var] = val
                    eval_result = self.evaluate(resolved, in_place=in_place)
                    res_future.set_result(eval_result)
                except Exception as e:
                    res_future.set_exception(e)

            asyncio.create_task(wait_and_eval())
            return res_future
        else:
            resolved = {}
            for var, val in normalized_point.items():
                if isinstance(val, asyncio.Future):
                    resolved[var] = val.result()
                else:
                    resolved[var] = val
            return self._evaluate_sync(resolved, in_place=in_place)

    def _evaluate_sync(self, point: dict, in_place=True):
        raise NotImplementedError


class ML_Polynomial(MLE):
    def __init__(self, variables=None, coefficients=None, num_vars=None, iop=None):
        super().__init__(variables=variables, num_vars=num_vars, iop=iop)
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
        new_coeffs = [c1 + c2 for c1, c2 in zip(self.coefficients, other.coefficients)]
        return ML_Polynomial(variables=self.variables, coefficients=new_coeffs)

    def __sub__(self, other):
        if not isinstance(other, ML_Polynomial):
            raise TypeError("Can only subtract ML_Polynomial from ML_Polynomial")
        assert self.variables == other.variables, "Variables must match"
        new_coeffs = [c1 - c2 for c1, c2 in zip(self.coefficients, other.coefficients)]
        return ML_Polynomial(variables=self.variables, coefficients=new_coeffs)

    def scale(self, factor):
        new_coeffs = [c * factor for c in self.coefficients]
        return ML_Polynomial(variables=self.variables, coefficients=new_coeffs)

    def _evaluate_sync(self, point: dict, in_place=True):
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

    def copy(self):
        return ML_Polynomial(
            variables=list(self.variables), coefficients=list(self.coefficients)
        )


class MLE_Sparse(MLE):
    def __init__(self, variables=None, evaluations=None, num_vars=None, iop=None):
        super().__init__(variables=variables, num_vars=num_vars, iop=iop)
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

    def _evaluate_sync(self, point: dict, in_place=True):
        raise NotImplementedError(
            "Evaluation is only supported for MLE_Dense and ML_Polynomial"
        )

    def copy(self):
        return MLE_Sparse(
            variables=list(self.variables), evaluations=dict(self.evaluations)
        )


class MLE_Dense(MLE):
    def __init__(
        self, ring: Ring, variables=None, evaluations=None, num_vars=None, iop=None
    ):
        super().__init__(variables=variables, num_vars=num_vars, iop=iop)
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

    def __add__(self, other):
        if not isinstance(other, MLE_Dense):
            raise TypeError("Can only add MLE_Dense to MLE_Dense")
        assert self.variables == other.variables, "Variables must match"
        assert self.ring == other.ring, "Rings must match"

        size = 1 << self.num_vars
        new_polys = [Polynomial(self.ring) for _ in range(size)]
        new_data_ptr = _handle_array(new_polys)

        lib.mle_dense_poly_add(new_data_ptr, self.data_ptr, other.data_ptr, size)

        res = MLE_Dense(ring=self.ring, variables=self.variables)
        res.py_refs = new_polys
        res.data_ptr = new_data_ptr
        return res

    def __sub__(self, other):
        if not isinstance(other, MLE_Dense):
            raise TypeError("Can only subtract MLE_Dense from MLE_Dense")
        assert self.variables == other.variables, "Variables must match"
        assert self.ring == other.ring, "Rings must match"

        size = 1 << self.num_vars
        new_polys = [Polynomial(self.ring) for _ in range(size)]
        new_data_ptr = _handle_array(new_polys)

        lib.mle_dense_poly_sub(new_data_ptr, self.data_ptr, other.data_ptr, size)

        res = MLE_Dense(ring=self.ring, variables=self.variables)
        res.py_refs = new_polys
        res.data_ptr = new_data_ptr
        return res

    def scale(self, factor):
        size = 1 << self.num_vars
        new_polys = [Polynomial(self.ring) for _ in range(size)]
        new_data_ptr = _handle_array(new_polys)

        if isinstance(factor, Polynomial):
            lib.mle_dense_poly_scale(new_data_ptr, self.data_ptr, factor.obj, size)
        else:
            lib.mle_dense_poly_scale_scalar(
                new_data_ptr, self.data_ptr, int(factor), size
            )

        res = MLE_Dense(ring=self.ring, variables=self.variables)
        res.py_refs = new_polys
        res.data_ptr = new_data_ptr
        return res

    def _evaluate_sync(self, point: dict, in_place=True):
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

                target.data_ptr = new_data_ptr
                target.py_refs = new_polys
                target.variables = target.variables[:idx] + target.variables[idx + 1 :]
                target.num_vars = new_num_vars

        return target

    def copy(self):
        new_polys = [p.copy() for p in self.py_refs]
        new_data_ptr = _handle_array(new_polys)

        res = MLE_Dense(ring=self.ring, variables=list(self.variables))
        res.py_refs = new_polys
        res.data_ptr = new_data_ptr
        return res
