# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""MLWE schemes, samples, keys and key-switch keys through vfhe.util.io, and the
seeded samples that let a fresh one be written as its seed and body."""

import io
import json
import os
import subprocess
import sys
import textwrap
import warnings

import pytest
from _vfhe_proto.vfhe.mlwe.io.v1 import io_pb2 as pb
from google.protobuf import proto
from vfhe.arith import Polynomial, Ring
from vfhe.arith import repr as Repr
from vfhe.arith.codec import RingCodec
from vfhe.mlwe import (
    LWE,
    MLWE,
    GadgetParams,
    LWE_Key,
    MGSW_Scheme,
    MLWE_Key,
    MLWE_Scheme,
)
from vfhe.mlwe.codec import (
    KeySwitchKeyCodec,
    LWECodec,
    LWEKeyCodec,
    MGSWCodec,
    MGSWSchemeCodec,
    MLWEKeyCodec,
    MLWESchemeCodec,
    SampleCodec,
)
from vfhe.util.io import ListCodec, TupleCodec

N = 256
PACKINGS = ["word", "tight"]


def dumps(codec, obj) -> bytes:
    buf = io.BytesIO()
    codec.encode(obj, buf)
    return buf.getvalue()


def loads(codec, data):
    return codec.decode(io.BytesIO(data))


def roundtrip(codec, obj):
    return loads(codec, dumps(codec, obj))


@pytest.fixture
def ghs():
    Rq = Ring(N, prime_size=[45, 45, 28, 50], split_degree=1)
    Rp = Rq.quotient_ring(ell=1)
    scheme = MLWE_Scheme(Rq, special_primes=1, module_rank=1)
    return Rp, scheme, scheme.key_gen_sparse(N // 8, 3.2)


def enc(scheme, Rp, m, key, lvl=0, seeded=None):
    delta = scheme.rings[lvl].modulus_ratio(Rp, return_pointer=True)
    return scheme.sample(
        m.scaled_lift(scheme.rings[lvl], delta=delta), key, lvl=lvl, seeded=seeded
    )


def same(c1, c2):
    if c1.r != c2.r or c1.ring is not c2.ring or c1.lvl != c2.lvl:
        return False
    return c1.get_b_poly() == c2.get_b_poly() and all(
        c1.get_a_poly(j) == c2.get_a_poly(j) for j in range(c1.r)
    )


def _sample(data: bytes) -> pb.Sample:
    """The Sample message of a sample's stream, after its scheme."""
    f = io.BytesIO(data)
    proto.parse_length_prefixed(pb.Scheme, f)
    return proto.parse_length_prefixed(pb.Sample, f)


def body_rows_bytes(ring):
    return N * sum(4 if q <= 2**32 else 8 for q in ring.primes)


@pytest.mark.parametrize("packing", PACKINGS)
@pytest.mark.parametrize("seeded", [True, False])
@pytest.mark.parametrize("held", [Repr.coeff, Repr.ntt])
def test_sample_round_trip(ghs, packing, seeded, held):
    Rp, scheme, key = ghs
    m = Rp.random_element()
    c = enc(scheme, Rp, m, key, seeded=seeded)
    assert (c.seed is not None) == seeded
    if held == Repr.ntt:
        c.to_NTT()
    back = roundtrip(SampleCodec(MLWESchemeCodec(scheme), packing=packing), c)
    assert back.scheme is scheme
    assert back.repr == held
    assert same(back, c)
    assert scheme.linear_decrypt(back, key).round_division(Rp) == m
    assert (back.seed is not None) == seeded


def test_seeded_sample_is_half_the_bytes(ghs):
    Rp, scheme, key = ghs
    c = enc(scheme, Rp, Rp.random_element(), key)
    seeded = _sample(dumps(SampleCodec(), c))
    full = _sample(dumps(SampleCodec(seeded=False), c))
    rows = body_rows_bytes(scheme.rings[0])
    assert len(seeded.seed) == 32 and not seeded.a and len(seeded.b) == rows
    assert not full.seed and [len(a) for a in full.a] == [rows]


def test_an_unknown_seed_expansion_is_refused(ghs):
    Rp, scheme, key = ghs
    f = io.BytesIO(dumps(SampleCodec(), enc(scheme, Rp, Rp.random_element(), key)))
    scheme_message = proto.parse_length_prefixed(pb.Scheme, f)
    sample = proto.parse_length_prefixed(pb.Sample, f)
    assert sample.seed_expansion == pb.SEED_EXPANSION_AES128_CTR
    sample.seed_expansion = 7  # pyright: ignore[reportAttributeAccessIssue] - a value this vfhe lacks
    out = io.BytesIO()
    proto.serialize_length_prefixed(scheme_message, out)
    proto.serialize_length_prefixed(sample, out)
    with pytest.raises(ValueError, match="unknown seed expansion"):
        loads(SampleCodec(MLWESchemeCodec(scheme)), out.getvalue())


def test_a_changed_mask_is_written_in_full(ghs):
    Rp, scheme, key = ghs
    m0, m1 = Rp.random_element(), Rp.random_element()
    c0, c1 = enc(scheme, Rp, m0, key), enc(scheme, Rp, m1, key)
    c0 += c1  # in place: the mask is no longer c0's seed's expansion
    assert c0.seed is not None
    data = dumps(SampleCodec(), c0)
    assert c0.seed is None  # the writer found it stale and dropped it
    assert len(data) == len(dumps(SampleCodec(seeded=False), c0))
    back = loads(SampleCodec(MLWESchemeCodec(scheme)), data)
    assert scheme.linear_decrypt(back, key).round_division(Rp) == m0 + m1


def test_changing_only_the_body_or_the_level_keeps_the_seed(ghs):
    Rp, scheme, key = ghs
    m = Rp.random_element()
    c = enc(scheme, Rp, m, key)
    pt = Polynomial(scheme.rings[0]).from_bigint_array([0] * N)
    pt.to_coeff()
    c.add_poly(c, pt)  # b only
    c.mod_reduce(lvl=1)  # drops a prime in place: the kept rows still match
    data = dumps(SampleCodec(), c)
    assert c.seed is not None
    assert len(data) < len(dumps(SampleCodec(seeded=False), c))
    back = loads(SampleCodec(MLWESchemeCodec(scheme)), data)
    assert back.lvl == 1 and back.ring is scheme.rings[1]
    assert same(back, c)


def test_copies_keep_the_seed_and_new_samples_do_not(ghs):
    Rp, scheme, key = ghs
    c = enc(scheme, Rp, Rp.random_element(), key)
    assert c.copy().seed == c.seed
    assert (c + c).seed is None
    assert c.new_like().seed is None


@pytest.mark.parametrize("r", [2, 3])
def test_module_rank(r):
    Rq = Ring(128, prime_size=[45, 45, 50], split_degree=1)
    Rp = Rq.quotient_ring(ell=1)
    scheme = MLWE_Scheme(Rq, special_primes=1, module_rank=r)
    key = scheme.key_gen_sparse(32, 3.2)
    m = Rp.random_element()
    delta = scheme.rings[0].modulus_ratio(Rp, return_pointer=True)
    c = scheme.sample(m.scaled_lift(scheme.rings[0], delta=delta), key)
    for packing in PACKINGS:
        back = roundtrip(SampleCodec(MLWESchemeCodec(scheme), packing=packing), c)
        assert same(back, c)
        assert scheme.linear_decrypt(back, key).round_division(Rp) == m


@pytest.mark.parametrize("radix", [None, 20])
@pytest.mark.parametrize("packing", PACKINGS)
def test_key_switch_key(ghs, radix, packing):
    Rp, scheme, key = ghs
    key2 = scheme.key_gen_sparse(N // 8, 3.2)
    ksk = scheme.gen_ksk(key2, key, gadget_params=GadgetParams(radix))
    codec = ListCodec(
        KeySwitchKeyCodec(SampleCodec(MLWESchemeCodec(scheme), packing=packing))
    )
    data = dumps(codec, ksk)
    back = loads(codec, data)
    m = Rp.random_element()
    c = enc(scheme, Rp, m, key)
    out, out_back = scheme.keyswitch(c, ksk), scheme.keyswitch(c, back)
    assert same(out, out_back)
    assert scheme.linear_decrypt(out_back, key2).round_division(Rp) == m
    # One seed per gadget sample instead of its mask.
    unseeded = dumps(
        ListCodec(KeySwitchKeyCodec(SampleCodec(seeded=False, packing=packing))), ksk
    )
    assert len(data) < 0.55 * len(unseeded)


def test_a_key_writes_its_scheme_once(ghs):
    _Rp, scheme, key = ghs
    key2 = scheme.key_gen_sparse(N // 8, 3.2)
    ksk = scheme.gen_ksk(key2, key, lvl=0)
    f = io.BytesIO(dumps(KeySwitchKeyCodec(SampleCodec(MLWESchemeCodec(scheme))), ksk))
    header = proto.parse_length_prefixed(pb.KeySwitchKey, f)
    proto.parse_length_prefixed(pb.Scheme, f)
    for _ in range(sum(header.lengths)):
        proto.parse_length_prefixed(pb.Sample, f)
    assert f.read() == b""


def test_key_switch_key_without_the_special_primes(ghs):
    Rp, scheme, key = ghs
    key2 = scheme.key_gen_sparse(N // 8, 3.2)
    ksk = scheme.gen_ksk(key2, key, lvl=0, gadget_params=GadgetParams(4), hybrid=False)
    back = roundtrip(KeySwitchKeyCodec(SampleCodec(MLWESchemeCodec(scheme))), ksk)
    m = Rp.random_element()
    c = enc(scheme, Rp, m, key)
    assert same(scheme.keyswitch(c, ksk), scheme.keyswitch(c, back))


def test_key_switch_key_with_fewer_digits(ghs):
    _Rp, scheme, key = ghs
    key2 = scheme.key_gen_sparse(N // 8, 3.2)
    last = len(scheme.rings) - 1
    ksk = scheme.gen_ksk(
        key2,
        key,
        lvl=last,
        gadget_params=GadgetParams(4, 8),
    )
    back = roundtrip(KeySwitchKeyCodec(SampleCodec(MLWESchemeCodec(scheme))), ksk)
    assert back.gadget_params.digits == 8
    ring = scheme.rings[last]
    c = scheme.sample(ring.random_element(), key.at_ring(ring), lvl=last)
    assert same(scheme.keyswitch(c, ksk), scheme.keyswitch(c, back))


def test_relinearization_key_keeps_its_pass_through_slots(ghs):
    Rp, scheme, key = ghs
    s_0 = key.poly[0]
    rlk = scheme.gen_rlk(key, [-(s_0 * s_0)])  # one set per level
    back = roundtrip(
        ListCodec(KeySwitchKeyCodec(SampleCodec(MLWESchemeCodec(scheme)))), rlk
    )
    assert len(back) == len(rlk)
    for got, orig in zip(back, rlk, strict=True):
        assert [c is None for c in got.mlwe] == [c is None for c in orig.mlwe]
    m1, m2 = Rp.random_element(), Rp.random_element()
    c1, c2 = enc(scheme, Rp, m1, key), enc(scheme, Rp, m2, key)
    assert same(scheme.multiply(c1, c2, rlk), scheme.multiply(c1, c2, back))


def test_nested_key_switch_keys(ghs):
    _Rp, scheme, key = ghs
    keys = scheme.gen_ksk_trace(key, key, gens=[5, 25], lvl=0)
    assert keys.dim == 3
    back = roundtrip(KeySwitchKeyCodec(SampleCodec(MLWESchemeCodec(scheme))), keys)
    assert back.dim == 3 and len(back._children) == 2
    for orig, got in zip(keys._children, back._children, strict=True):
        assert all(
            same(a, b)
            for ca, cb in zip(orig.mlwe, got.mlwe, strict=True)
            for a, b in zip(ca, cb, strict=True)
        )


def test_mgsw(ghs):
    Rp, scheme, key = ghs
    mgsw_scheme = MGSW_Scheme(scheme)
    ct_id = mgsw_scheme.encrypt(Polynomial(Rp).from_array([1] + [0] * (N - 1)), key)
    back = roundtrip(
        MGSWCodec(
            MGSWSchemeCodec(mgsw_scheme),
            SampleCodec(MLWESchemeCodec(scheme), packing="tight"),
        ),
        ct_id,
    )
    assert back.scheme is mgsw_scheme
    m = Rp.random_element()
    c = enc(scheme, Rp, m, key)
    assert same(ct_id.external_product(c), back.external_product(c))


def test_mgsw_scheme_with_fewer_digits():
    scheme = MLWE_Scheme(Ring(N, prime_size=[50, 50], split_degree=1), special_primes=1)
    key = scheme.key_gen_sparse(N // 8, 3.2)
    mgsw_scheme = MGSW_Scheme(
        scheme,
        gadget_params=GadgetParams(10, 3),
    )
    (one,) = mgsw_scheme.encrypt_constants([1], key)
    back = roundtrip(MGSWCodec(MGSWSchemeCodec(inner=MLWESchemeCodec(scheme))), one)
    assert back.scheme.gadget_params.digits == 3
    other = MGSW_Scheme(scheme)  # the default gadget: another MGSW scheme
    with pytest.warns(UserWarning, match="MGSW_Scheme.*matches none"):
        roundtrip(MGSWCodec(MGSWSchemeCodec(other, inner=MLWESchemeCodec(scheme))), one)
    c = scheme.sample(scheme.rings[0].random_element(), key)
    assert same(one.external_product(c), back.external_product(c))


def test_grouped_digits_travel_with_keys_and_mgsw_schemes():
    scheme = MLWE_Scheme(
        Ring(N, prime_size=[45] * 3 + [50] * 2, split_degree=1), special_primes=2
    )
    key, key2 = (scheme.key_gen_sparse(N // 8, 3.2) for _ in range(2))
    params = GadgetParams(primes_per_digit=2)
    ksk = scheme.gen_ksk(key2, key, lvl=0, gadget_params=params)
    mgsw_scheme = MGSW_Scheme(scheme, gadget_params=params)
    (one,) = mgsw_scheme.encrypt_constants([1], key)
    ksk_back = roundtrip(KeySwitchKeyCodec(SampleCodec(MLWESchemeCodec(scheme))), ksk)
    one_back = roundtrip(MGSWCodec(MGSWSchemeCodec(inner=MLWESchemeCodec(scheme))), one)
    assert ksk_back.gadget_params == params
    assert one_back.scheme.gadget_params == params
    c = scheme.sample(scheme.rings[0].random_element(), key)
    assert same(scheme.keyswitch(c, ksk), scheme.keyswitch(c, ksk_back))
    assert same(one.external_product(c), one_back.external_product(c))


def test_secret_key_round_trip(ghs):
    Rp, scheme, key = ghs
    back = roundtrip(MLWEKeyCodec(MLWESchemeCodec(scheme)), key)
    assert isinstance(back, MLWE_Key)
    assert back.key == key.key and back.sigma_err == key.sigma_err
    m = Rp.random_element()
    assert scheme.linear_decrypt(enc(scheme, Rp, m, key), back).round_division(Rp) == m


def test_lwe_and_its_key():
    Ring(N, prime_size=[45, 45], split_degree=1)
    ring = Ring(N, prime_size=[21, 45, 45], split_degree=1)
    key = LWE_Key(ring, sec_sigma=3.2, err_sigma=3.2)
    m = [p // 4 for p in ring.primes]
    sample = LWE(ring=ring, m=m, key=key)
    rings = RingCodec(ring)
    got_sample, got_key = roundtrip(
        TupleCodec(LWECodec(rings), LWEKeyCodec(rings)), (sample, key)
    )
    assert got_sample.get_a() == sample.get_a() and got_sample.get_b() == sample.get_b()
    assert got_key.get_s() == key.get_s()
    assert got_key.err_sigma == 3.2
    assert got_sample.linear_decrypt(got_key) == sample.linear_decrypt(key)
    # a key without a noise parameter stays without one
    bare = LWE_Key(ring, key=key.get_s())
    got_bare = roundtrip(LWEKeyCodec(rings), bare)
    assert got_bare.err_sigma is None


def test_rebuilt_scheme_works_without_the_original(ghs):
    Rp, scheme, key = ghs
    m = Rp.random_element()
    c = enc(scheme, Rp, m, key)
    writer = TupleCodec(
        SampleCodec(MLWESchemeCodec(scheme)), MLWEKeyCodec(MLWESchemeCodec(scheme))
    )
    schemes = MLWESchemeCodec()  # knows no scheme: it rebuilds one, shared by both
    reader = TupleCodec(SampleCodec(schemes), MLWEKeyCodec(schemes))
    got_c, got_key = loads(reader, dumps(writer, (c, key)))
    assert got_c.scheme is not scheme and got_c.scheme is got_key.scheme
    rebuilt = got_c.scheme
    assert [r.primes for r in rebuilt.rings] == [r.primes for r in scheme.rings]
    assert rebuilt.special_primes == scheme.special_primes
    assert got_c.ring is rebuilt.rings[0]
    Rp2 = rebuilt.rings[0].quotient_ring(ell=1)
    assert rebuilt.linear_decrypt(got_c, got_key).round_division(
        Rp2
    ).get_polynomial() == (m.get_polynomial())


def test_a_scheme_that_does_not_match_warns(ghs):
    Rp, scheme, key = ghs
    other = MLWE_Scheme(Ring(N, prime_size=[45, 50], split_degree=1), special_primes=1)
    data = dumps(
        SampleCodec(MLWESchemeCodec(scheme)), enc(scheme, Rp, Rp.random_element(), key)
    )
    with pytest.warns(UserWarning, match="matches none"):
        loads(SampleCodec(MLWESchemeCodec(other)), data)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loads(SampleCodec(), data)


_CHILD = textwrap.dedent(
    """
    import json, sys
    from vfhe.arith import Ring
    from vfhe.mlwe.codec import KeySwitchKeyCodec, MLWEKeyCodec, SampleCodec, MLWESchemeCodec
    from vfhe.util.io import ListCodec, TupleCodec
    primes = json.loads(sys.argv[1])
    Ring(256, prime_size=[33, 47], split_degree=1)
    for p in reversed(primes):
        Ring(256, primes=[p], prime_size=[p.bit_length()], split_degree=1)
    schemes = MLWESchemeCodec()
    samples = SampleCodec(schemes)
    codec = TupleCodec(samples, ListCodec(KeySwitchKeyCodec(samples)), MLWEKeyCodec(schemes), MLWEKeyCodec(schemes))
    with open(sys.argv[2], "rb") as f:
        c, ksk, key, key2 = codec.decode(f)
    scheme = key.scheme
    c = scheme.keyswitch(c, ksk)
    Rp = scheme.rings[0].quotient_ring(ell=1)
    out = scheme.linear_decrypt(c, key2).round_division(Rp)
    print(json.dumps(out.get_polynomial()))
    """
)


@pytest.mark.parametrize("packing", PACKINGS)
def test_another_process_with_another_base_order(ghs, tmp_path, packing):
    Rp, scheme, key = ghs
    key2 = scheme.key_gen_sparse(N // 8, 3.2)
    m = Rp.random_element()
    stuff = (enc(scheme, Rp, m, key), scheme.gen_ksk(key2, key), key, key2)
    schemes = MLWESchemeCodec(scheme)
    samples = SampleCodec(schemes, packing=packing)
    codec = TupleCodec(
        samples,
        ListCodec(KeySwitchKeyCodec(samples)),
        MLWEKeyCodec(schemes),
        MLWEKeyCodec(schemes),
    )
    path = tmp_path / "keys.vfhe"
    with open(path, "wb") as f:
        codec.encode(stuff, f)
    primes = sorted({p for r in scheme.special_rings for p in r.primes})
    # The child picks its own engine: under an emulator it runs on the bare CPU.
    env = {k: v for k, v in os.environ.items() if k != "VFHE_ENGINE"}
    out = subprocess.run(  # noqa: S603 - this interpreter, a fixed script
        [sys.executable, "-W", "ignore", "-c", _CHILD, json.dumps(primes), str(path)],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert json.loads(out.stdout) == m.get_polynomial()


def test_unseeded_encryption_on_request(ghs):
    Rp, scheme, key = ghs
    scheme.seeded_encryption = False
    try:
        c = enc(scheme, Rp, Rp.random_element(), key)
        assert c.seed is None
    finally:
        del scheme.seeded_encryption
    assert enc(scheme, Rp, Rp.random_element(), key).seed is not None
    assert MLWE(scheme).seed is None
