# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""CKKS and BFV schemes, ciphertexts and keys, the bootstrapping key
containers and CKKS linear transforms through vfhe.io."""

import random
from typing import cast

import pytest
from vfhe.arith import Polynomial, Ring
from vfhe.crypto import entropy
from vfhe.fhe import (
    CGGI16,
    GP25,
    BFV_Scheme,
    CGGI16_Key,
    CKKS_Ciphertext,
    CKKS_LinearTransform,
    CKKS_Scheme,
)
from vfhe.io import Serializer
from vfhe.mlwe import MLWE_Scheme, MLWE_Set

N = 64
PROFILES = ["default", "compact", "fast"]


def _ckks():
    scheme = CKKS_Scheme(
        Ring(N, prime_size=[60, 40, 40, 40, 60], split_degree=1),
        scaling_factor=2**40,
        special_primes=1,
    )
    return scheme, scheme.key_gen_sparse(N // 8, 3.2)


def _values(n, seed=1):
    rng = random.Random(seed)  # noqa: S311 - test data, not a key
    return [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(n)]


def _close(a, b, tol=1e-6):
    return max(abs(x - y) for x, y in zip(a, b, strict=True)) < tol


def _same(c1, c2):
    return c1.get_b_poly() == c2.get_b_poly() and all(
        c1.get_a_poly(j) == c2.get_a_poly(j) for j in range(c1.r)
    )


@pytest.mark.parametrize("profile", PROFILES)
def test_ckks_ciphertexts_keep_their_scaling_factor(profile):
    scheme, key = _ckks()
    scheme.rlk = scheme.gen_rlk(key, key)
    x, y = _values(N // 2, 1), _values(N // 2, 2)
    cx = scheme.encrypt(scheme.encode(x), key)
    cy = scheme.encrypt(scheme.encode(y), key)
    prod = cx * cy  # relinearized and rescaled: a lower level, another delta
    s = Serializer(profile)
    got = s.loads(s.dumps({"x": cx, "prod": prod}), schemes=scheme)
    assert isinstance(got["prod"], CKKS_Ciphertext)
    assert got["prod"].delta == prod.delta and got["prod"].lvl == prod.lvl
    assert got["x"].delta == cx.delta
    assert _same(got["prod"], prod)
    assert _close(scheme.decode(scheme.decrypt(got["x"], key)), x)
    expected = [a * b for a, b in zip(x, y, strict=True)]
    assert _close(
        scheme.decode(scheme.decrypt(got["prod"], key), got["prod"].delta),
        expected,
        1e-4,
    )


def test_ckks_rotation_keys_by_step():
    scheme, key = _ckks()
    keys = {k: scheme.gen_rotation_key(key, k) for k in (1, 3)}
    s = Serializer("compact")
    got = s.loads(s.dumps(keys), schemes=scheme)
    assert sorted(got) == [1, 3]
    x = _values(N // 2)
    c = scheme.encrypt(scheme.encode(x), key)
    for k in (1, 3):
        assert _same(scheme.rotate(c, k, keys[k]), scheme.rotate(c, k, got[k]))


def test_ckks_scheme_rebuilt_with_its_scaling_factor():
    scheme, key = _ckks()
    x = _values(N // 2)
    c = scheme.encrypt(scheme.encode(x), key)
    s = Serializer()
    got, got_key = s.loads(s.dumps_secret([c, key]))
    rebuilt = got.scheme
    assert isinstance(rebuilt, CKKS_Scheme) and rebuilt is not scheme
    assert rebuilt.scaling_factor == scheme.scaling_factor
    assert _close(rebuilt.decode(rebuilt.decrypt(got, got_key)), x)
    # Another scaling factor is another scheme: it does not bind.
    other = CKKS_Scheme(
        scheme.rings,
        scaling_factor=2**30,
        special_primes=1,
        special_rings=scheme.special_rings,
    )
    with pytest.warns(UserWarning, match="matches none"):
        assert s.loads(s.dumps(c), schemes=other).scheme is not other


@pytest.mark.parametrize("profile", PROFILES)
def test_bfv(profile):
    ring = Ring(N, prime_size=[17, 45, 45, 45, 50], split_degree=1)
    scheme = BFV_Scheme(ring, special_primes=1)
    key = scheme.key_gen_sparse(N // 8, 3.2)
    values = [entropy.below(scheme.t) - scheme.t // 2 for _ in range(scheme.n_slots)]
    ct = scheme.encrypt(scheme.encode(values), key)
    s = Serializer(profile)
    assert s.loads(s.dumps(ct), schemes=scheme).scheme is scheme
    got, got_key = s.loads(s.dumps_secret([ct, key]))
    rebuilt = got.scheme
    assert isinstance(rebuilt, BFV_Scheme) and rebuilt.t == scheme.t
    assert rebuilt.decode(rebuilt.decrypt(got, got_key)) == scheme.decode(
        scheme.decrypt(ct, key)
    )


def test_linear_transform_keeps_its_encoding():
    scheme, key = _ckks()
    n = 8
    rng = random.Random(3)  # noqa: S311 - test data, not a key
    diagonals = {
        d: [complex(rng.uniform(-1, 1), 0) for _ in range(n)] for d in (0, 1, 5)
    }
    lt = CKKS_LinearTransform(scheme, diagonals, scale=2**30)
    got = Serializer().loads(Serializer().dumps(lt), schemes=scheme)
    assert got.rotations == lt.rotations and got.baby_steps == lt.baby_steps
    keys = {k: scheme.gen_rotation_key(key, k) for k in lt.rotations}
    c = scheme.encrypt(scheme.encode(_values(N // 2)), key)
    assert _same(lt.apply(c, keys), got.apply(c, keys))


def test_cggi16_bootstrap_key():
    Rq = Ring(256, prime_size=[50, 50, 50], split_degree=1)
    in_ring = Rq.quotient_ring(ell=2)
    in_scheme = MLWE_Scheme(in_ring, special_primes=0, module_rank=1)
    out_scheme = MLWE_Scheme(Rq, special_primes=1, module_rank=1, max_lvl=1)
    input_key = in_scheme.key_gen_sparse(17, 3.2, ternary=False)
    output_key = out_scheme.key_gen_sparse(64, 3.2, ternary=False)
    cggi16 = CGGI16(out_scheme)
    bk = cggi16.generate_bootstrap_key(input_key.extract_lwe_key(), output_key)
    bk.b_prec = 5
    s = Serializer("compact")
    data = s.dumps(bk)
    got = s.loads(data, schemes=[out_scheme, cggi16.mgsw_scheme])
    assert got.b_prec == 5 and len(got.bk) == len(bk.bk)
    assert (got.n, got.unfolding, got.variant, got.combination) == (
        256,
        1,
        "bmmp18",
        "evaluation",
    )
    assert all(g.scheme is cggi16.mgsw_scheme for g in got.bk)
    for g, o in zip(got.bk, bk.bk, strict=True):
        assert all(_same(a, b) for a, b in zip(g.obj, o.obj, strict=True))
    assert len(data) < 0.55 * len(Serializer("compact", seeded=False).dumps(bk))


def test_cggi16_unfolded_bootstrap_key_layout():
    bk = CGGI16_Key()
    bk.n, bk.unfolding, bk.variant, bk.combination = 7, 3, "zyl17", "coefficient"
    got = Serializer().loads(Serializer().dumps(bk))
    assert (got.n, got.unfolding, got.variant, got.combination, got.bk) == (
        7,
        3,
        "zyl17",
        "coefficient",
        [],
    )


def test_gp25_sab_key():
    from vfhe.fhe.gp25 import SAB_Key

    Rq = Ring(256, prime_size=[50, 50, 50], split_degree=1)
    out_scheme = MLWE_Scheme(Rq, special_primes=1, module_rank=1, max_lvl=1)
    output_key = out_scheme.key_gen_sparse(64, 3.2, ternary=True)
    gp25 = GP25(out_scheme)
    sab = SAB_Key()
    sab.h, sab.b_prec, sab.r_prec = 3, 5, 2
    one = Polynomial(gp25.mgsw_scheme.ring).from_array([1])
    sab.s = [[[gp25.mgsw_scheme.encrypt(one, output_key)]]]
    sab.s_sign = [[gp25.mgsw_scheme.encrypt(one, output_key)]]
    packing_key = cast("MLWE_Set", out_scheme.gen_ksk(output_key, output_key, lvl=0))
    sab.packing_key = packing_key
    got = Serializer().loads(
        Serializer().dumps(sab), schemes=[out_scheme, gp25.mgsw_scheme]
    )
    assert (got.h, got.b_prec, got.r_prec) == (3, 5, 2)
    assert got.hw_reducing_key is None and got.trace_repack_key is None
    assert all(
        _same(a, b) for a, b in zip(got.s[0][0][0].obj, sab.s[0][0][0].obj, strict=True)
    )
    got_samples, samples = got.packing_key.mlwe[0], packing_key.mlwe[0]
    assert got_samples is not None and samples is not None
    assert all(_same(a, b) for a, b in zip(got_samples, samples, strict=True))
