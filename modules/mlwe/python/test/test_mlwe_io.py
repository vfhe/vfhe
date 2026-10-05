# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""MLWE schemes, samples, keys and key-switch keys through vfhe.io, and the
seeded samples that let a fresh one be written as its seed and body."""

import json
import os
import subprocess
import sys
import textwrap
import warnings

import pytest
from vfhe.arith import Polynomial, Ring
from vfhe.arith import repr as Repr
from vfhe.io import Serializer
from vfhe.mlwe import LWE, MLWE, LWE_Key, MGSW_Scheme, MLWE_Key, MLWE_Scheme

N = 256
PROFILES = ["default", "compact", "fast"]


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


def body_rows_bytes(ring):
    return N * sum(4 if q <= 2**32 else 8 for q in ring.primes)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("seeded", [True, False])
@pytest.mark.parametrize("held", [Repr.coeff, Repr.ntt])
def test_sample_round_trip(ghs, profile, seeded, held):
    Rp, scheme, key = ghs
    m = Rp.random_element()
    c = enc(scheme, Rp, m, key, seeded=seeded)
    assert (c.seed is not None) == seeded
    if held == Repr.ntt:
        c.to_NTT()
    s = Serializer(profile)
    back = s.loads(s.dumps(c), schemes=scheme)
    assert back.scheme is scheme
    assert back.repr == held
    assert same(back, c)
    assert scheme.linear_decrypt(back, key).round_division(Rp) == m
    assert (back.seed is not None) == (seeded and s.options.seeded)


def test_seeded_sample_is_half_the_bytes(ghs):
    Rp, scheme, key = ghs
    c = enc(scheme, Rp, Rp.random_element(), key)
    seeded = len(Serializer(checksum=False).dumps(c))
    full = len(Serializer(checksum=False, seeded=False).dumps(c))
    rows = body_rows_bytes(scheme.rings[0])
    assert full - seeded == rows - 32


def test_a_changed_mask_is_written_in_full(ghs):
    Rp, scheme, key = ghs
    m0, m1 = Rp.random_element(), Rp.random_element()
    c0, c1 = enc(scheme, Rp, m0, key), enc(scheme, Rp, m1, key)
    c0 += c1  # in place: the mask is no longer c0's seed's expansion
    assert c0.seed is not None
    s = Serializer(checksum=False)
    data = s.dumps(c0)
    assert c0.seed is None  # the writer found it stale and dropped it
    assert len(data) == len(Serializer(checksum=False, seeded=False).dumps(c0))
    back = s.loads(data, schemes=scheme)
    assert scheme.linear_decrypt(back, key).round_division(Rp) == m0 + m1


def test_changing_only_the_body_or_the_level_keeps_the_seed(ghs):
    Rp, scheme, key = ghs
    m = Rp.random_element()
    c = enc(scheme, Rp, m, key)
    pt = Polynomial(scheme.rings[0]).from_bigint_array([0] * N)
    pt.to_coeff()
    c.add_poly(c, pt)  # b only
    c.mod_reduce(lvl=1)  # drops a prime in place: the kept rows still match
    s = Serializer(checksum=False)
    data = s.dumps(c)
    assert c.seed is not None
    assert len(data) < len(Serializer(checksum=False, seeded=False).dumps(c))
    back = s.loads(data, schemes=scheme)
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
    for profile in PROFILES:
        s = Serializer(profile)
        back = s.loads(s.dumps(c), schemes=scheme)
        assert same(back, c)
        assert scheme.linear_decrypt(back, key).round_division(Rp) == m


@pytest.mark.parametrize("radix", [None, 20])
@pytest.mark.parametrize("profile", PROFILES)
def test_key_switch_key(ghs, radix, profile):
    Rp, scheme, key = ghs
    key2 = scheme.key_gen_sparse(N // 8, 3.2)
    ksk = scheme.gen_ksk(key2, key, radix_log_base=radix)
    s = Serializer(profile)
    data = s.dumps(ksk)
    back = s.loads(data, schemes=scheme)
    m = Rp.random_element()
    c = enc(scheme, Rp, m, key)
    out, out_back = scheme.keyswitch(c, ksk), scheme.keyswitch(c, back)
    assert same(out, out_back)
    assert scheme.linear_decrypt(out_back, key2).round_division(Rp) == m
    if profile != "fast":
        # One seed per gadget sample instead of its mask.
        unseeded = Serializer(profile, seeded=False).dumps(ksk)
        assert len(data) < 0.55 * len(unseeded)


def test_relinearization_key_keeps_its_pass_through_slots(ghs):
    Rp, scheme, key = ghs
    s_0 = key.poly[0]
    rlk = scheme.gen_rlk(key, [-(s_0 * s_0)])  # one set per level
    back = Serializer().loads(Serializer().dumps(rlk), schemes=scheme)
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
    back = Serializer().loads(Serializer().dumps(keys), schemes=scheme)
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
    back = Serializer("compact").loads(
        Serializer("compact").dumps(ct_id), schemes=[scheme, mgsw_scheme]
    )
    assert back.scheme is mgsw_scheme
    m = Rp.random_element()
    c = enc(scheme, Rp, m, key)
    assert same(ct_id.external_product(c), back.external_product(c))


def test_secret_keys_only_through_dump_secret(ghs):
    Rp, scheme, key = ghs
    s = Serializer()
    with pytest.raises(TypeError, match="dump_secret"):
        s.dumps({"key": key})
    back = s.loads(s.dumps_secret(key), schemes=scheme)
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
    s = Serializer()
    got_sample, got_key = s.loads(s.dumps_secret([sample, key]), rings=[ring])
    assert got_sample.get_a() == sample.get_a() and got_sample.get_b() == sample.get_b()
    assert got_key.get_s() == key.get_s()
    assert got_key.err_sigma == 3.2
    assert got_sample.linear_decrypt(got_key) == sample.linear_decrypt(key)
    with pytest.raises(TypeError, match="dump_secret"):
        s.dumps(key)
    # a key without a noise parameter stays without one
    bare = LWE_Key(ring, key=key.get_s())
    (got_bare,) = s.loads(s.dumps_secret([bare]), rings=[ring])
    assert got_bare.err_sigma is None


def test_rebuilt_scheme_works_without_the_original(ghs):
    Rp, scheme, key = ghs
    m = Rp.random_element()
    c = enc(scheme, Rp, m, key)
    s = Serializer()
    got_c, got_key = s.loads(s.dumps_secret([c, key]))
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
    data = Serializer().dumps(enc(scheme, Rp, Rp.random_element(), key))
    with pytest.warns(UserWarning, match="matches none"):
        Serializer().loads(data, schemes=other)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Serializer().loads(data)


_CHILD = textwrap.dedent(
    """
    import json, sys
    from vfhe.arith import Ring
    from vfhe.io import Serializer
    primes = json.loads(sys.argv[1])
    Ring(256, prime_size=[33, 47], split_degree=1)
    for p in reversed(primes):
        Ring(256, primes=[p], prime_size=[p.bit_length()], split_degree=1)
    with open(sys.argv[2], "rb") as f:
        stuff = Serializer().load(f)
    scheme, key = stuff["key"].scheme, stuff["key"]
    c = scheme.keyswitch(stuff["c"], stuff["ksk"])
    Rp = scheme.rings[0].quotient_ring(ell=1)
    out = scheme.linear_decrypt(c, stuff["key2"]).round_division(Rp)
    print(json.dumps(out.get_polynomial()))
    """
)


@pytest.mark.parametrize("profile", PROFILES)
def test_another_process_with_another_base_order(ghs, tmp_path, profile):
    Rp, scheme, key = ghs
    key2 = scheme.key_gen_sparse(N // 8, 3.2)
    m = Rp.random_element()
    stuff = {
        "c": enc(scheme, Rp, m, key),
        "ksk": scheme.gen_ksk(key2, key),
        "key": key,
        "key2": key2,
    }
    path = tmp_path / "keys.vfhe"
    with open(path, "wb") as f:
        Serializer(profile).dump_secret(stuff, f)
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


def test_unseeded_encryption_is_still_available(ghs):
    Rp, scheme, key = ghs
    scheme.seeded_encryption = False
    try:
        c = enc(scheme, Rp, Rp.random_element(), key)
        assert c.seed is None
    finally:
        del scheme.seeded_encryption
    assert enc(scheme, Rp, Rp.random_element(), key).seed is not None
    assert MLWE(scheme).seed is None
