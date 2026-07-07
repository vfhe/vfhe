# SPDX-License-Identifier: Apache-2.0
"""Characterization tests for vfhe.piop: pure-Python ML_Polynomial / MLE_Sparse
arithmetic + evaluation, and the C-backed MLE_Dense (mle_dense_poly_* kernels)
over the cffi boundary, including async evaluation on resolved IOP futures.
"""

import asyncio

from vfhe.arith import Polynomial, Ring
from vfhe.piop import IOPValue, IOPVariable, ML_Polynomial, MLE_Dense, MLE_Sparse


def test_ml_polynomial_arithmetic_and_eval():
    v = [IOPVariable("x0"), IOPVariable("x1")]
    p1 = ML_Polynomial(variables=v, coefficients=[1, 2, 3, 4])
    p2 = ML_Polynomial(variables=v, coefficients=[10, 20, 30, 40])
    assert (p1 + p2).coefficients == [11, 22, 33, 44]
    assert (p2 - p1).coefficients == [9, 18, 27, 36]
    assert p1.scale(5).coefficients == [5, 10, 15, 20]
    # evaluate() may return an IOPValue (async path); here it is synchronous.
    ex0 = p1.evaluate({v[0]: 2}, in_place=False)
    ex1 = p1.evaluate({v[1]: 3}, in_place=False)
    exf = p1.evaluate({v[0]: 2, v[1]: 3}, in_place=False)
    assert isinstance(ex0, ML_Polynomial) and ex0.coefficients == [5, 11]
    assert isinstance(ex1, ML_Polynomial) and ex1.coefficients == [10, 14]
    assert isinstance(exf, ML_Polynomial) and exf.coefficients == [38]


def test_mle_sparse_arithmetic():
    v = [IOPVariable("x0"), IOPVariable("x1")]
    s1 = MLE_Sparse(variables=v, evaluations={0: 5, 2: 12})
    s2 = MLE_Sparse(variables=v, evaluations={1: 7, 2: 3})
    assert (s1 + s2).evaluations == {0: 5, 1: 7, 2: 15}
    assert (s1 - s2).evaluations == {0: 5, 1: -7, 2: 9}
    assert s1.scale(2).evaluations == {0: 10, 2: 24}


def test_mle_dense_arithmetic_and_eval():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    v = [IOPVariable("x0"), IOPVariable("x1")]
    d1 = MLE_Dense(ring=ring, variables=v, evaluations=[1, 2, 3, 4])
    d2 = MLE_Dense(ring=ring, variables=v, evaluations=[10, 20, 30, 40])

    d3 = d1 + d2
    assert d3.py_refs[0].get_polynomial()[0] == 11
    assert d3.py_refs[3].get_polynomial()[0] == 44

    d4 = d2 - d1
    assert d4.py_refs[0].get_polynomial()[0] == 9
    assert d4.py_refs[3].get_polynomial()[0] == 36

    d5 = d1.scale(5)
    assert d5.py_refs[0].get_polynomial()[0] == 5
    assert d5.py_refs[3].get_polynomial()[0] == 20

    d6 = d1.scale(Polynomial(ring).from_array([10]))
    assert d6.py_refs[0].get_polynomial()[0] == 10
    assert d6.py_refs[3].get_polynomial()[0] == 40

    eval_x0 = d1.evaluate({v[0]: 2}, in_place=False)
    assert isinstance(eval_x0, MLE_Dense)
    assert eval_x0.num_vars == 1
    assert eval_x0.py_refs[0].get_polynomial()[0] == 3
    assert eval_x0.py_refs[1].get_polynomial()[0] == 5


def test_async_evaluation():
    async def run():
        ring = Ring(1024, prime_size=[49], split_degree=4)
        v = [IOPVariable("x0"), IOPVariable("x1")]
        d1 = MLE_Dense(ring=ring, variables=v, evaluations=[1, 2, 3, 4])
        fut = IOPValue()
        fut2 = IOPValue()
        res = d1.evaluate({v[0]: fut, v[1]: fut2}, in_place=False)
        assert isinstance(res, asyncio.Future) and not res.done()
        fut.set_result(2)
        fut2.set_result(3)
        out = await res
        # Full evaluation over both resolved futures -> a single (num_vars==0) poly.
        assert out.num_vars == 0
        return out.py_refs[0].get_polynomial()[0]

    assert asyncio.run(run()) == 9
