# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""CKKS and BFV schemes, ciphertexts and keys, the bootstrapping key
containers and CKKS linear transforms through vfhe.util.io."""

import io
import random

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
from vfhe.fhe.codec import (
    BFVSchemeCodec,
    CGGI16KeyCodec,
    CKKSCiphertextCodec,
    CKKSSchemeCodec,
    LinearTransformCodec,
    SABKeyCodec,
)
from vfhe.mlwe import MLWE_Scheme
from vfhe.mlwe.codec import (
    KeySwitchKeyCodec,
    MGSWCodec,
    MGSWSchemeCodec,
    MLWEKeyCodec,
    MLWESchemeCodec,
    SampleCodec,
)
from vfhe.util.io import DictCodec, IntCodec, ListCodec, TupleCodec


def _mod_switch(v: int, q: int, p: int) -> int:
    """``v`` scaled from ``Z_q`` to ``Z_p``, rounded half up."""
    return ((2 * v * p + q) // (2 * q)) % p


N = 64
PACKINGS = ["word", "tight"]


def dumps(codec, obj) -> bytes:
    buf = io.BytesIO()
    codec.encode(obj, buf)
    return buf.getvalue()


def loads(codec, data):
    return codec.decode(io.BytesIO(data))


def roundtrip(codec, obj):
    return loads(codec, dumps(codec, obj))


def _sab_codec(io_scheme, rotation, gp25):
    """GP25's key: MGSW ciphertexts over the rotation scheme, key-switch keys over either scheme."""
    schemes = MLWESchemeCodec(io_scheme, rotation)
    mgsw = MGSWCodec(
        MGSWSchemeCodec(gp25.mgsw_scheme, inner=schemes), SampleCodec(schemes)
    )
    return SABKeyCodec(mgsw, KeySwitchKeyCodec(SampleCodec(schemes)))


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


@pytest.mark.parametrize("packing", PACKINGS)
def test_ckks_ciphertexts_keep_their_scaling_factor(packing):
    scheme, key = _ckks()
    scheme.rlk = scheme.gen_rlk(key, key)
    x, y = _values(N // 2, 1), _values(N // 2, 2)
    cx = scheme.encrypt(scheme.encode(x), key)
    cy = scheme.encrypt(scheme.encode(y), key)
    prod = cx * cy  # relinearized and rescaled: a lower level, another delta
    ct = CKKSCiphertextCodec(CKKSSchemeCodec(scheme), packing=packing)
    got = dict(
        zip(("x", "prod"), roundtrip(TupleCodec(ct, ct), (cx, prod)), strict=True)
    )
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


def test_a_ckks_ciphertext_or_scheme_needs_its_own_codec():
    scheme, key = _ckks()
    c = scheme.encrypt(scheme.encode(_values(N // 2)), key)
    with pytest.raises(TypeError, match="CKKS_Ciphertext needs its own codec"):
        dumps(SampleCodec(CKKSSchemeCodec(scheme)), c)
    with pytest.raises(TypeError, match="CKKS_Scheme needs its own codec"):
        dumps(MLWESchemeCodec(scheme), scheme)


def test_ckks_rotation_keys_by_step():
    scheme, key = _ckks()
    keys = {k: scheme.gen_rotation_key(key, k) for k in (1, 3)}
    codec = DictCodec(
        IntCodec(),
        ListCodec(
            KeySwitchKeyCodec(SampleCodec(CKKSSchemeCodec(scheme), packing="tight"))
        ),
    )
    got = roundtrip(codec, keys)
    assert sorted(got) == [1, 3]
    x = _values(N // 2)
    c = scheme.encrypt(scheme.encode(x), key)
    for k in (1, 3):
        assert _same(scheme.rotate(c, k, keys[k]), scheme.rotate(c, k, got[k]))


def test_ckks_scheme_rebuilt_with_its_scaling_factor():
    scheme, key = _ckks()
    x = _values(N // 2)
    c = scheme.encrypt(scheme.encode(x), key)
    writer = TupleCodec(
        CKKSCiphertextCodec(CKKSSchemeCodec(scheme)),
        MLWEKeyCodec(CKKSSchemeCodec(scheme)),
    )
    schemes = CKKSSchemeCodec()
    reader = TupleCodec(CKKSCiphertextCodec(schemes), MLWEKeyCodec(schemes))
    got, got_key = loads(reader, dumps(writer, (c, key)))
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
        data = dumps(CKKSCiphertextCodec(CKKSSchemeCodec(scheme)), c)
        assert (
            loads(CKKSCiphertextCodec(CKKSSchemeCodec(other)), data).scheme is not other
        )


@pytest.mark.parametrize("packing", PACKINGS)
def test_bfv(packing):
    ring = Ring(N, prime_size=[17, 45, 45, 45, 50], split_degree=1)
    scheme = BFV_Scheme(ring, special_primes=1)
    key = scheme.key_gen_sparse(N // 8, 3.2)
    values = [entropy.below(scheme.t) - scheme.t // 2 for _ in range(scheme.n_slots)]
    ct = scheme.encrypt(scheme.encode(values), key)
    assert (
        roundtrip(SampleCodec(BFVSchemeCodec(scheme), packing=packing), ct).scheme
        is scheme
    )
    writer = TupleCodec(
        SampleCodec(BFVSchemeCodec(scheme), packing=packing),
        MLWEKeyCodec(BFVSchemeCodec(scheme)),
    )
    schemes = BFVSchemeCodec()
    reader = TupleCodec(SampleCodec(schemes), MLWEKeyCodec(schemes))
    got, got_key = loads(reader, dumps(writer, (ct, key)))
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
    got = roundtrip(LinearTransformCodec(CKKSSchemeCodec(scheme)), lt)
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
    codec = CGGI16KeyCodec(
        MGSWCodec(
            MGSWSchemeCodec(cggi16.mgsw_scheme),
            SampleCodec(MLWESchemeCodec(out_scheme), packing="tight"),
        )
    )
    data = dumps(codec, bk)
    got = loads(codec, data)
    assert len(got.bk) == len(bk.bk)
    assert (got.n, got.unfolding) == (256, 1)
    assert all(g.scheme is cggi16.mgsw_scheme for g in got.bk)
    for g, o in zip(got.bk, bk.bk, strict=True):
        assert all(_same(a, b) for a, b in zip(g.obj, o.obj, strict=True))
    unseeded = CGGI16KeyCodec(
        MGSWCodec(MGSWSchemeCodec(), SampleCodec(seeded=False, packing="tight"))
    )
    assert len(data) < 0.55 * len(dumps(unseeded, bk))


def test_cggi16_unfolded_bootstrap_key_layout():
    bk = CGGI16_Key()
    bk.n, bk.unfolding = 7, 3
    got = roundtrip(CGGI16KeyCodec(MGSWCodec()), bk)
    assert (got.n, got.unfolding, got.bk) == (7, 3, [])


@pytest.mark.parametrize("n", [N, 2 * N])
def test_gp25_bootstrap_key(deterministic_prng, n):
    # A loaded key bootstraps like the one it was saved from.
    deterministic_prng(0x5AB00004)
    io_ring = Ring(max(n, N), prime_size=[50, 50, 50, 50], split_degree=1)
    rotation_ring = Ring(min(n, N), primes=list(io_ring.primes), split_degree=1)
    if n < N:
        io_ring, rotation_ring = rotation_ring, io_ring
    io = MLWE_Scheme(io_ring, special_primes=1)
    rotation = MLWE_Scheme(rotation_ring, special_primes=1, max_lvl=1)
    input_key = GP25.sample_input_key(io, 3, 7, 3.2)
    output_key = io.key_gen_sparse(16, 3.2)
    rotation_key = rotation.key_gen_sparse(16, 3.2)
    gp25 = GP25(rotation)
    sab = gp25.generate_bootstrap_key(input_key, rotation_key, output_key, 3, 7)
    got = roundtrip(_sab_codec(io, rotation, gp25), sab)
    assert (got.n, got.h, got.gap_bits) == (n, 3, 7)
    assert (got.hw_reducing_lvl, got.output_lvl) == (len(io.rings) - 1, 0)
    assert all(
        _same(a, b)
        for a, b in zip(got.gaps[0][0][0].obj, sab.gaps[0][0][0].obj, strict=True)
    )
    assert got.trace_repack_key is not None and got.output_switch_key is not None

    q = rotation.rings[0].q_l
    tv = gp25.test_vector([_mod_switch(t, 16, q) for t in range(8)])
    msg = [k % 8 for k in range(n)]
    c = io.sample(
        Polynomial(io.rings[0]).from_bigint_array(
            [_mod_switch(x, 16, io.rings[0].q_l) for x in msg]
        ),
        output_key,
    ).round_division(lvl=len(io.rings) - 1)
    for key in (sab, got):
        out = gp25.bootstrap(c, tv, key)
        d = io.linear_decrypt(out, output_key).get_polynomial()
        assert [_mod_switch(v, q, 16) for v in d] == msg


def test_gp25_bootstrap_key_with_two_bit_layers(deterministic_prng):
    deterministic_prng(0x5AB0000E)
    ring = Ring(N, prime_size=[50, 50, 50], split_degree=1)
    io = MLWE_Scheme(ring, special_primes=1)
    rotation = MLWE_Scheme(ring, special_primes=1, max_lvl=1)
    input_key = GP25.sample_input_key(io, 3, 7, 3.2)
    output_key = io.key_gen_sparse(16, 3.2)
    rotation_key = rotation.key_gen_sparse(16, 3.2)
    gp25 = GP25(rotation, bits_per_layer=2)
    sab = gp25.generate_bootstrap_key(input_key, rotation_key, output_key, 3, 7)
    got = roundtrip(_sab_codec(io, rotation, gp25), sab)
    assert (sab.bits_per_layer, got.bits_per_layer) == (2, 2)
    assert len(got.gaps[0][0]) == 3 * 3 + 1  # three pairs and a single bit
    q = rotation.rings[0].q_l
    tv = gp25.test_vector([_mod_switch(t, 16, q) for t in range(8)])
    msg = [k % 8 for k in range(N)]
    c = io.sample(
        Polynomial(io.rings[0]).from_bigint_array(
            [_mod_switch(x, 16, io.rings[0].q_l) for x in msg]
        ),
        output_key,
    ).round_division(lvl=len(io.rings) - 1)
    out = gp25.bootstrap(c, tv, got)
    d = io.linear_decrypt(out, output_key).get_polynomial()
    assert [_mod_switch(v, q, 16) for v in d] == msg


def test_gp25_bootstrap_key_without_an_output_key(deterministic_prng):
    deterministic_prng(0x5AB0000B)
    ring = Ring(N, prime_size=[50, 50, 50], split_degree=1)
    io = MLWE_Scheme(ring, special_primes=1)
    rotation = MLWE_Scheme(ring, special_primes=1, max_lvl=1)
    input_key = GP25.sample_input_key(io, 3, 7, 3.2)
    rotation_key = rotation.key_gen_sparse(16, 3.2)
    gp25 = GP25(rotation)
    sab = gp25.generate_bootstrap_key(input_key, rotation_key, None, 3, 7)
    got = roundtrip(_sab_codec(io, rotation, gp25), sab)
    assert (got.hw_reducing_lvl, got.output_lvl) == (None, None)
    assert got.hw_reducing_key is None and got.output_switch_key is None

    q = rotation.rings[0].q_l
    tv = gp25.test_vector([_mod_switch(t, 16, q) for t in range(8)])
    msg = [k % 8 for k in range(N)]
    c = io.sample(
        Polynomial(io.rings[0]).from_bigint_array(
            [_mod_switch(x, 16, io.rings[0].q_l) for x in msg]
        ),
        input_key,
    )
    (packed,) = gp25.repack(gp25.blind_rotate(c, tv, got), got)
    d = rotation.linear_decrypt(packed, rotation_key).get_polynomial()
    assert [_mod_switch(v, q, 16) for v in d] == msg
