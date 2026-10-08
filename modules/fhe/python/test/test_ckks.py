# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Characterization tests for the (reverted) vfhe.fhe CKKS scheme over cffi.

Encode/decode, encrypt/decrypt, slot rotation, and ciphertext multiplication
(ciphertext*ciphertext with relinearization+rescale, and ciphertext*plaintext).
"""

import cmath
import math

import pytest
import vfhe.engine as engine
from vfhe.arith import Ring
from vfhe.arith.residue_selection import search_log_residues_minq0
from vfhe.crypto import entropy
from vfhe.engine import ffi
from vfhe.fhe import CKKS_Ciphertext, CKKS_Scheme

N = 256

# Module ranks above 1, paired with a ring dimension that keeps the lattice
# dimension N*r (and the runtime) in line with the rank-1 tests above.
RANK_DIMS = [(2, 128), (4, 64)]


def _subring(Rq, *indices):
    """Quotient ring of ``Rq`` keeping the primes at the given positions."""
    mask = 0
    for i in indices:
        mask |= 1 << Rq.prime_indices[i]
    return Rq.quotient_ring(mask=mask)


def rand_values(n):
    draw = [-2, -1, 0, 1, 2]
    return [complex(draw[entropy.below(5)], draw[entropy.below(5)]) for _ in range(n)]


def test_encode_decode():
    scheme = CKKS_Scheme(
        Ring(N, 300, split_degree=1), scaling_factor=2**25, special_primes=1
    )
    values = rand_values(N // 2)
    dec = scheme.decode(scheme.encode(values))
    assert all(abs(v - dv) < 1e-3 for v, dv in zip(values, dec, strict=False))


@pytest.mark.parametrize("n", [1, 4, 32, N // 2])
def test_sparse_encoding(n):
    scheme = CKKS_Scheme(
        Ring(N, 300, split_degree=1), scaling_factor=2**25, special_primes=1
    )
    key = scheme.key_gen_sparse(N // 8, 3.2)
    values = rand_values(n)
    pt = scheme.encode(values)
    # Repeated slots are a polynomial in X^(N/2n): every other coefficient is 0.
    step = N // (2 * n)
    coeffs = pt.get_polynomial(signed=True)
    assert all(c == 0 for k, c in enumerate(coeffs) if k % step)
    assert all(
        abs(v - d) < 1e-3
        for v, d in zip(values, scheme.decode(pt, slots=n), strict=True)
    )
    full = scheme.decode(pt)
    assert all(abs(full[i] - values[i % n]) < 1e-3 for i in range(N // 2))
    dec = scheme.decode(scheme.decrypt(scheme.encrypt(pt, key), key), slots=n)
    assert all(abs(v - d) < 0.05 for v, d in zip(values, dec, strict=True))
    with pytest.raises(ValueError, match="dividing"):
        scheme.encode(rand_values(3))
    with pytest.raises(ValueError, match="slots must divide"):
        scheme.decode(pt, slots=3)


def test_encode_batch_matches_encode():
    scheme = CKKS_Scheme(
        Ring(N, 300, split_degree=1), scaling_factor=2**25, special_primes=1
    )
    ring = scheme.rings[1]
    batch = [rand_values(n) for n in (1, 4, N // 2, 32, N // 2)]
    engine.set_num_threads(8)
    try:
        polys = scheme.encode_batch(batch, ring=ring, scale=2**20, n_threads=3)
    finally:
        engine.set_num_threads()
    for values, poly in zip(batch, polys, strict=True):
        expected = scheme.encode(values, ring=ring, scale=2**20)
        assert poly.get_polynomial(signed=True) == expected.get_polynomial(signed=True)
    assert scheme.encode_batch([]) == []
    with pytest.raises(ValueError, match="dividing"):
        scheme.encode_batch([rand_values(4), rand_values(3)])
    with pytest.raises(NotImplementedError, match="str"):
        scheme.encode(["1"] * 4)  # pyright: ignore[reportArgumentType]


def test_encode_reads_complex_buffers():
    np = pytest.importorskip("numpy")
    scheme = CKKS_Scheme(
        Ring(N, 300, split_degree=1), scaling_factor=2**25, special_primes=1
    )
    values = rand_values(N // 2)
    expected = scheme.encode(values).get_polynomial(signed=True)
    for buffer in (
        np.array(values),  # read in place
        np.repeat(np.array(values), 2)[::2],  # not contiguous: converted
    ):
        assert scheme.encode(buffer).get_polynomial(signed=True) == expected
    reals = [v.real for v in values]
    assert scheme.encode(np.array(reals)).get_polynomial(signed=True) == scheme.encode(
        reals
    ).get_polynomial(signed=True)


def test_encrypt_decrypt():
    scheme = CKKS_Scheme(
        Ring(N, 300, split_degree=1), scaling_factor=2**25, special_primes=1
    )
    key = scheme.key_gen_sparse(N // 8, 3.2)
    values = rand_values(N // 2)
    poly = scheme.encode(values)
    ct = scheme.encrypt(poly, key)
    dec = scheme.decode(scheme.decrypt(ct, key))
    assert all(abs(v - dv) < 0.05 for v, dv in zip(values, dec, strict=False))


@pytest.mark.parametrize("k", [1, 3, 7])
def test_rotation(k):
    # Rotation is an automorphism followed by a GHS hybrid key-switch, so it
    # needs special primes (special_primes=1) to keep the noise decodable.
    scheme = CKKS_Scheme(
        Ring(N, 300, split_degree=1), scaling_factor=2**25, special_primes=1
    )
    key = scheme.key_gen_sparse(N // 8, 3.2)
    values = rand_values(N // 2)
    ct = scheme.encrypt(scheme.encode(values), key)
    ksk = scheme.gen_rotation_key(key, k)
    dec_rot = scheme.decode(scheme.decrypt(scheme.rotate(ct, k, ksk), key))
    M = N // 2
    assert all(abs(dec_rot[i] - values[(i + k) % M]) < 0.05 for i in range(M))


def test_hoisted_rotations():
    # Several rotations of one ciphertext share its decomposition; CKKS
    # decode does not strip noise, so this also bounds what hoisting adds.
    scheme = CKKS_Scheme(
        Ring(N, 300, split_degree=1), scaling_factor=2**25, special_primes=1
    )
    key = scheme.key_gen_sparse(N // 8, 3.2)
    values = rand_values(N // 2)
    ct = scheme.encrypt(scheme.encode(values), key)
    ks = [1, 3, 7, N // 2 - 1]
    gens = [pow(5, k, 2 * N) for k in ks]
    ksks = [scheme.gen_rotation_key(key, k) for k in ks]
    M = N // 2
    for k, out in zip(ks, scheme.automorphisms(ct, gens, ksks), strict=True):
        assert isinstance(out, CKKS_Ciphertext) and out.delta == ct.delta
        dec = scheme.decode(scheme.decrypt(out, key))
        assert all(abs(dec[i] - values[(i + k) % M]) < 0.05 for i in range(M))


def test_keyswitch_ghs():
    # Direct GHS key switching: re-encrypt a ciphertext under a fresh key. Unlike
    # the MLWE-level test, CKKS decode does not strip noise, so this only works
    # with the special-prime (GHS) hybrid key-switch.
    scheme = CKKS_Scheme(
        Ring(N, 300, split_degree=1), scaling_factor=2**25, special_primes=1
    )
    key_in = scheme.key_gen_sparse(N // 8, 3.2)
    key_out = scheme.key_gen_sparse(N // 8, 3.2)
    values = rand_values(N // 2)
    ct = scheme.encrypt(scheme.encode(values), key_in)

    ksk = scheme.gen_ksk(key_out, key_in)
    ct_switched = scheme.keyswitch(ct, ksk)

    dec = scheme.decode(scheme.decrypt(ct_switched, key_out))
    assert all(abs(v - d) < 0.05 for v, d in zip(values, dec, strict=False))


@pytest.mark.parametrize("special_primes", [0, 1])
def test_ciphertext_multiplication(special_primes):
    # Relinearization is a GHS hybrid key-switch. BV (special_primes=0) also
    # works here because the CKKS product is immediately rescaled, which divides
    # out the relinearization noise.
    scheme = CKKS_Scheme(
        Ring(N, 300, split_degree=1),
        scaling_factor=2**49,
        special_primes=special_primes,
    )
    key = scheme.key_gen_sparse(N // 8, 3.2)
    s_0 = key.poly[0]
    scheme.rlk = scheme.gen_rlk(key, [-(s_0 * s_0)])

    v1 = [complex(0.5, 0.5) if i == 0 else 0 for i in range(N // 2)]
    v2 = [complex(0.4, -0.4) if i == 0 else 0 for i in range(N // 2)]
    c1 = scheme.encrypt(scheme.encode(v1), key)
    c2 = scheme.encrypt(scheme.encode(v2), key)

    c_mul = c1 * c2
    dec = scheme.decode(scheme.decrypt(c_mul, key), scaling_factor=c_mul.delta)
    expected = [a * b for a, b in zip(v1, v2, strict=False)]
    assert all(abs(e - d) < 0.05 for e, d in zip(expected, dec, strict=False))


def test_ciphertext_plaintext_multiplication():
    scheme = CKKS_Scheme(
        Ring(N, 300, split_degree=1), scaling_factor=2**49, special_primes=0
    )
    key = scheme.key_gen_sparse(N // 8, 3.2)

    v1 = [complex(0.5, 0.5) if i == 0 else 0 for i in range(N // 2)]
    v2 = [complex(0.4, -0.4) if i == 0 else 0 for i in range(N // 2)]
    poly2 = scheme.encode(v2)
    c1 = scheme.encrypt(scheme.encode(v1), key)

    c_mul = c1 * poly2
    dec = scheme.decode(scheme.decrypt(c_mul, key), scaling_factor=c_mul.delta)
    expected = [a * b for a, b in zip(v1, v2, strict=False)]
    assert all(abs(e - d) < 0.05 for e, d in zip(expected, dec, strict=False))


def test_encode_at_a_ring_and_scale():
    # Plaintexts for a ciphertext further down the chain are encoded in that
    # ciphertext's ring, at whatever scale the computation calls for.
    scheme = CKKS_Scheme(
        Ring(N, 300, split_degree=1), scaling_factor=2**40, special_primes=0
    )
    key = scheme.key_gen_sparse(N // 8, 3.2)
    v, w = rand_values(N // 2), rand_values(N // 2)
    ring = scheme.rings[1]
    pt = scheme.encode(w, ring=ring, scale=2**30)
    assert pt.ring == ring and pt.rns_mask == ring.mask
    assert all(
        abs(a - b) < 1e-6
        for a, b in zip(w, scheme.decode(pt, scaling_factor=2**30), strict=True)
    )
    ct = scheme.sample(scheme.encode(v, ring=ring), key, lvl=1)
    assert isinstance(ct, CKKS_Ciphertext)
    prod = scheme.multiply_plain(ct, pt, scale=2**30)
    assert prod.lvl == 1 and prod.delta == 2**40 * 2**30
    dec = scheme.decode(scheme.decrypt(prod, key), scaling_factor=prod.delta)
    assert all(abs(a * b - d) < 0.05 for a, b, d in zip(v, w, dec, strict=True))


def test_plaintext_products_summed_before_one_rescale():
    scheme = CKKS_Scheme(
        Ring(N, 300, split_degree=1), scaling_factor=2**49, special_primes=0
    )
    key = scheme.key_gen_sparse(N // 8, 3.2)
    v, w, u = (rand_values(N // 2) for _ in range(3))
    ct = scheme.encrypt(scheme.encode(v), key)

    # An unscaled plaintext leaves the ciphertext's scale where it was.
    one = scheme.multiply_plain(ct, scheme.encode([1] * (N // 2), scale=1), scale=1)
    assert one.delta == ct.delta and one.lvl == ct.lvl
    dec = scheme.decode(scheme.decrypt(one, key), scaling_factor=one.delta)
    assert all(abs(a - d) < 0.05 for a, d in zip(v, dec, strict=True))

    acc = scheme.multiply_plain(ct, scheme.encode(w))
    acc += scheme.multiply_plain(ct, scheme.encode(u))
    acc = scheme.rescale(acc)
    assert acc.lvl == 1
    dec = scheme.decode(scheme.decrypt(acc, key), scaling_factor=acc.delta)
    expected = [a * (b + c) for a, b, c in zip(v, w, u, strict=True)]
    assert all(abs(e - d) < 0.05 for e, d in zip(expected, dec, strict=True))


def test_rational_rescale_shared_primes():
    # Level 0 = primes {0,1,2}, level 1 = primes {0,1,3}: level 1 is NOT a
    # quotient of level 0 (they diverge in the third prime), so a plain rescale
    # cannot bridge them. rational_rescale lifts to the union ring {0,1,2,3} and
    # rounds away the level-0-only prime. The scheme is built from an explicit
    # per-level ring chain (index 4 is the special prime).
    Rq = Ring(N, split_degree=1, prime_size=[50, 50, 50, 30, 50])
    rings = [_subring(Rq, 0, 1, 2), _subring(Rq, 0, 1, 3)]
    special_rings = [_subring(Rq, 0, 1, 2, 4), _subring(Rq, 0, 1, 3, 4)]
    scheme = CKKS_Scheme(
        rings,
        scaling_factor=2**50,
        special_primes=1,
        special_rings=special_rings,
    )
    key = scheme.key_gen_sparse(N // 8, 3.2)
    values = rand_values(N // 2)

    ct = scheme.encrypt(scheme.encode(values), key)
    assert ct.lvl == 0

    rescaled = scheme.rational_rescale(ct)
    assert rescaled.lvl == 1

    dec = scheme.decode(scheme.decrypt(rescaled, key), scaling_factor=rescaled.delta)
    assert all(abs(v - d) < 0.05 for v, d in zip(values, dec, strict=False))


def test_rational_rescale_disjoint_primes():
    # Level 0 = primes {2,3}, level 1 = primes {0,1}: the two levels share no
    # primes, so the union ring holds all four and rational_rescale divides out
    # both level-0 primes at once (index 4 is the special prime).
    Rq = Ring(N, split_degree=1, prime_size=[40, 40, 51, 58, 50])
    rings = [_subring(Rq, 2, 3), _subring(Rq, 0, 1), _subring(Rq, 2)]
    special_rings = [_subring(Rq, 2, 3, 4), _subring(Rq, 0, 1, 4), _subring(Rq, 2, 4)]
    scheme = CKKS_Scheme(
        rings,
        scaling_factor=2**50,
        special_primes=1,
        special_rings=special_rings,
    )
    key = scheme.key_gen_sparse(N // 8, 3.2)
    values = rand_values(N // 2)

    ct = scheme.encrypt(scheme.encode(values), key)
    assert ct.lvl == 0

    rescaled = scheme.rational_rescale(ct)
    assert rescaled.lvl == 1

    dec = scheme.decode(scheme.decrypt(rescaled, key), scaling_factor=rescaled.delta)
    assert all(abs(v - d) < 0.05 for v, d in zip(values, dec, strict=False))


def test_rational_rescale_from_residue_selection():
    # Derive the RNS prime chain from a target scaling-factor chain, then rescale
    # once through the selected residues (no special prime here, so no
    # special_rings are needed).
    log_scaling_factor_chain = [29, 29, 29]
    log_top_residues, residue_indices_chain = search_log_residues_minq0(
        log_scaling_factor_chain=log_scaling_factor_chain,
        logr_min=40,
        logr_max=64,
        max_modulus=200,
    )
    Rq = Ring(N, split_degree=1, prime_size=log_top_residues)
    rings = [_subring(Rq, *indices) for indices in residue_indices_chain]
    scheme = CKKS_Scheme(
        rings,
        scaling_factor=2 ** (log_scaling_factor_chain[0] + log_scaling_factor_chain[1]),
        special_primes=0,
    )
    key = scheme.key_gen_sparse(N // 8, 3.2)
    values = rand_values(N // 2)

    ct = scheme.encrypt(scheme.encode(values), key)
    assert ct.lvl == 0

    rescaled = scheme.rational_rescale(ct)
    assert rescaled.lvl == 1

    dec = scheme.decode(scheme.decrypt(rescaled, key), scaling_factor=rescaled.delta)
    assert all(abs(v - d) < 0.05 for v, d in zip(values, dec, strict=False))


def test_multiplication_with_rational_rescale():
    # Mix multiplication and rational rescaling: a non-nested level chain (from
    # residue selection) makes every ciphertext product's rescale a *rational*
    # rescale (see CKKS_Scheme.rescale routing). We evaluate (v0*v1)*(v2*v3) --
    # multiplication depth 2, consuming two levels. Relinearization uses a
    # special prime (GHS); at this scale (2**29) the BV variant's relin noise
    # would swamp the signal.
    log_top_residues, residue_indices_chain = search_log_residues_minq0(
        log_scaling_factor_chain=[29, 29, 29],
        logr_min=40,
        logr_max=64,
        max_modulus=200,
    )
    # Append one special prime (last index) for the GHS hybrid key-switch.
    Rq = Ring(N, split_degree=1, prime_size=[*log_top_residues, 50])
    special_index = len(log_top_residues)
    rings = [_subring(Rq, *indices) for indices in residue_indices_chain]
    special_rings = [
        _subring(Rq, *indices, special_index) for indices in residue_indices_chain
    ]
    scheme = CKKS_Scheme(
        rings,
        scaling_factor=2**29,
        special_primes=1,
        special_rings=special_rings,
    )
    # Consecutive levels are non-nested, so each rescale is a rational rescale.
    assert not rings[1].is_quotient_ring(rings[0])
    assert not rings[2].is_quotient_ring(rings[1])

    key = scheme.key_gen_sparse(N // 8, 3.2)
    s_0 = key.poly[0]
    scheme.rlk = scheme.gen_rlk(key, [-(s_0 * s_0)])

    v = [rand_values(N // 2) for _ in range(4)]
    c = [scheme.encrypt(scheme.encode(vi), key) for vi in v]

    prod01 = c[0] * c[1]  # level 0 -> 1, rational rescale
    prod23 = c[2] * c[3]  # level 0 -> 1, rational rescale
    assert prod01.lvl == 1 and prod23.lvl == 1

    result = prod01 * prod23  # level 1 -> 2, rational rescale
    assert result.lvl == 2

    dec = scheme.decode(scheme.decrypt(result, key), scaling_factor=result.delta)
    expected = [a * b * cc * d for a, b, cc, d in zip(*v, strict=False)]
    assert all(abs(e - d) < 0.05 for e, d in zip(expected, dec, strict=False))


def test_operations_preserve_ciphertext_type():
    # Operations CKKS does not override (automorphism/rotation, key-switching,
    # copy, add, sub) must hand back a CKKS_Ciphertext carrying the *actual*
    # scaling factor of their input, not a plain MLWE and not a ciphertext reset
    # to scheme.scaling_factor. They allocate through MLWE.new_like, which keeps
    # the concrete class and copies subclass metadata (delta) over.
    scheme = CKKS_Scheme(
        Ring(N, 300, split_degree=1), scaling_factor=2**25, special_primes=1
    )
    key = scheme.key_gen_sparse(N // 8, 3.2)
    key_out = scheme.key_gen_sparse(N // 8, 3.2)
    poly = scheme.encode(rand_values(N // 2))
    ct = scheme.encrypt(poly, key)
    # Sentinel: a ciphertext that has been rescaled no longer sits at the
    # scheme's scaling factor, so derived ciphertexts must inherit this value.
    ct.delta = 12345.0

    derived = {
        "copy": ct.copy(),
        "add": ct + ct,
        "sub": ct - ct,
        "automorphism": scheme.automorphism(
            ct, 5, scheme.gen_ksk_automorphism(key, key, 5)
        ),
        "rotate": scheme.rotate(ct, 1, scheme.gen_rotation_key(key, 1)),
        "keyswitch": scheme.keyswitch(ct, scheme.gen_ksk(key_out, key)),
    }
    for name, out in derived.items():
        assert isinstance(out, CKKS_Ciphertext), f"{name} returned {type(out).__name__}"
        assert out.delta == ct.delta, f"{name} lost delta"

    # A freshly sampled ciphertext has no source to inherit from: it takes the
    # scheme's scaling factor, but is still a CKKS_Ciphertext.
    fresh = scheme.sample(poly, key)
    assert isinstance(fresh, CKKS_Ciphertext)
    assert fresh.delta == scheme.scaling_factor


def _rank_scheme(N_r, r, scaling_factor):
    """CKKS scheme of module rank ``r`` over a ring of dimension ``N_r``."""
    return CKKS_Scheme(
        Ring(N_r, 300, split_degree=1),
        scaling_factor=scaling_factor,
        module_rank=r,
        special_primes=1,
    )


@pytest.mark.parametrize("r, N_r", RANK_DIMS)
def test_encrypt_decrypt_module_rank(r, N_r):
    scheme = _rank_scheme(N_r, r, 2**25)
    key = scheme.key_gen_sparse(N_r * r // 8, 3.2)
    values = rand_values(N_r // 2)
    ct = scheme.encrypt(scheme.encode(values), key)
    assert ct.r == r
    dec = scheme.decode(scheme.decrypt(ct, key))
    assert all(abs(v - d) < 0.05 for v, d in zip(values, dec, strict=False))


@pytest.mark.parametrize("r, N_r", RANK_DIMS)
def test_rotation_module_rank(r, N_r):
    # The automorphism permutes all r "a" components plus b, then key-switches
    # with an r-component rotation key.
    k = 1
    scheme = _rank_scheme(N_r, r, 2**25)
    key = scheme.key_gen_sparse(N_r * r // 8, 3.2)
    values = rand_values(N_r // 2)
    ct = scheme.encrypt(scheme.encode(values), key)
    ksk = scheme.gen_rotation_key(key, k)
    dec_rot = scheme.decode(scheme.decrypt(scheme.rotate(ct, k, ksk), key))
    M = N_r // 2
    assert all(abs(dec_rot[i] - values[(i + k) % M]) < 0.05 for i in range(M))


@pytest.mark.parametrize("r, N_r", RANK_DIMS)
def test_ciphertext_plaintext_multiplication_module_rank(r, N_r):
    scheme = _rank_scheme(N_r, r, 2**49)
    key = scheme.key_gen_sparse(N_r * r // 8, 3.2)
    v1 = [complex(0.5, 0.5) if i == 0 else 0 for i in range(N_r // 2)]
    v2 = [complex(0.4, -0.4) if i == 0 else 0 for i in range(N_r // 2)]
    c1 = scheme.encrypt(scheme.encode(v1), key)

    c_mul = c1 * scheme.encode(v2)
    dec = scheme.decode(scheme.decrypt(c_mul, key), scaling_factor=c_mul.delta)
    expected = [a * b for a, b in zip(v1, v2, strict=False)]
    assert all(abs(e - d) < 0.05 for e, d in zip(expected, dec, strict=False))


@pytest.mark.parametrize("r, N_r", RANK_DIMS)
def test_ciphertext_multiplication_module_rank(r, N_r):
    scheme = _rank_scheme(N_r, r, 2**49)
    key = scheme.key_gen_sparse(N_r * r // 8, 3.2)
    # One relinearization key per quadratic pair -(s_i*s_j), i <= j.
    scheme.rlk = scheme.gen_rlk(key, key)

    v1 = [complex(0.5, 0.5) if i == 0 else 0 for i in range(N_r // 2)]
    v2 = [complex(0.4, -0.4) if i == 0 else 0 for i in range(N_r // 2)]
    c1 = scheme.encrypt(scheme.encode(v1), key)
    c2 = scheme.encrypt(scheme.encode(v2), key)

    c_mul = c1 * c2
    dec = scheme.decode(scheme.decrypt(c_mul, key), scaling_factor=c_mul.delta)
    expected = [a * b for a, b in zip(v1, v2, strict=False)]
    assert all(abs(e - d) < 0.05 for e, d in zip(expected, dec, strict=False))


@pytest.fixture
def threads():
    """Lets the test's parallel calls use up to 8 threads (vfhe defaults to 1)."""
    engine.set_num_threads(8)
    yield
    engine.set_num_threads()


def _product_scheme(n_levels, n=N):
    scheme = CKKS_Scheme(
        Ring(n, prime_size=[60] + [40] * n_levels + [60], split_degree=1),
        scaling_factor=2**40,
        special_primes=1,
    )
    key = scheme.key_gen_sparse(n // 8, 3.2)
    s_0 = key.poly[0]
    scheme.rlk = scheme.gen_rlk(key, [-(s_0 * s_0)])
    return scheme, key


def _unit_values(n):
    # On the unit circle, so a product of many stays the same size.
    return [
        cmath.exp(2j * cmath.pi * entropy.below(1 << 20) / (1 << 20)) for _ in range(n)
    ]


def _check_product(scheme, key, n_factors, n_threads=0):
    values = [_unit_values(scheme.N // 2) for _ in range(n_factors)]
    cts = [scheme.encrypt(scheme.encode(v), key) for v in values]
    out = scheme.product(cts, n_threads)
    assert out.lvl == (n_factors - 1).bit_length()
    dec = scheme.decode(scheme.decrypt(out, key), scaling_factor=out.delta)
    expected = [math.prod(v[i] for v in values) for i in range(scheme.N // 2)]
    assert all(abs(e - d) < 1e-3 for e, d in zip(expected, dec, strict=True))


@pytest.mark.parametrize("n_factors", [1, 2, 3, 4, 5, 6, 7, 8])
def test_product_of_ciphertexts(n_factors):
    scheme, key = _product_scheme(3)
    _check_product(scheme, key, n_factors)


@pytest.mark.usefixtures("threads")
def test_product_on_several_threads():
    scheme, key = _product_scheme(3, n=256)
    for n_threads in (2, 8, 0):
        _check_product(scheme, key, 8, n_threads)


def test_product_refusals():
    scheme, key = _product_scheme(2)
    ct = scheme.encrypt(scheme.encode(_unit_values(N // 2)), key)
    with pytest.raises(ValueError, match="levels"):
        scheme.product([ct] * 8)
    scheme.rlk = None
    with pytest.raises(ValueError, match="rlk"):
        scheme.product([ct, ct])


@pytest.mark.parametrize("fused", [False, True])
def test_multiply_batch_with_a_shared_operand(fused):
    scheme, key = _product_scheme(1)
    x, y = (_unit_values(N // 2) for _ in range(2))
    cx, cy = (scheme.encrypt(scheme.encode(v), key) for v in (x, y))
    if fused:
        outs = scheme.multiply_batch([cx, cx], [cx, cy], scheme.rlk, lvl=1)
    else:
        outs = scheme.rescale_batch(
            scheme.multiply_batch([cx, cx], [cx, cy], scheme.rlk)
        )
    dropped = scheme.rings[0].primes[-1]
    for out, expected in zip(
        outs,
        ([a * a for a in x], [a * b for a, b in zip(x, y, strict=True)]),
        strict=True,
    ):
        assert out.lvl == 1
        assert out.delta == cx.delta * cx.delta / dropped
        dec = scheme.decode(scheme.decrypt(out, key), scaling_factor=out.delta)
        assert all(abs(e - d) < 1e-3 for e, d in zip(expected, dec, strict=True))


def test_product_over_a_rational_rescale_chain():
    log_top_residues, residue_indices_chain = search_log_residues_minq0(
        log_scaling_factor_chain=[29, 29, 29],
        logr_min=40,
        logr_max=64,
        max_modulus=200,
    )
    Rq = Ring(N, split_degree=1, prime_size=[*log_top_residues, 50])
    special_index = len(log_top_residues)
    rings = [_subring(Rq, *indices) for indices in residue_indices_chain]
    special_rings = [
        _subring(Rq, *indices, special_index) for indices in residue_indices_chain
    ]
    scheme = CKKS_Scheme(
        rings, scaling_factor=2**29, special_primes=1, special_rings=special_rings
    )
    assert not rings[1].is_quotient_ring(rings[0])
    key = scheme.key_gen_sparse(N // 8, 3.2)
    s_0 = key.poly[0]
    scheme.rlk = scheme.gen_rlk(key, [-(s_0 * s_0)])
    _check_product(scheme, key, 4)
    # An odd factor would have to drop a level, and these levels do not nest.
    ct = scheme.encrypt(scheme.encode(_unit_values(N // 2)), key)
    with pytest.raises(ValueError, match="nested"):
        scheme.product([ct, ct, ct])


def test_multiply_carries_the_product_of_the_scales():
    scheme, key = _product_scheme(1)
    x, y = (_unit_values(N // 2) for _ in range(2))
    # Unequal scales, so a delta taken from one operand would show.
    cx = scheme.encrypt(scheme.encode(x), key)
    cy = scheme.encrypt(scheme.encode(y, scale=2**39), key)
    cy.delta = 2.0**39
    out = scheme.multiply(cx, cy, scheme.rlk)
    assert out.delta == 2.0**79
    out = scheme.rescale(out)
    dec = scheme.decode(scheme.decrypt(out, key), scaling_factor=out.delta)
    expected = [a * b for a, b in zip(x, y, strict=True)]
    assert all(abs(e - d) < 1e-3 for e, d in zip(expected, dec, strict=True))


@pytest.mark.parametrize("ntt", [False, True])
def test_mod_reduce_drops_a_level_and_keeps_the_value(ntt):
    scheme, key = _product_scheme(2)
    x, y = (_unit_values(N // 2) for _ in range(2))
    cx = scheme.encrypt(scheme.encode(x), key)
    if ntt:
        cx.to_NTT()
    else:
        cx.to_coeff()
    assert cx.mod_reduce(lvl=1) is cx
    assert cx.lvl == 1 and cx.ring == scheme.rings[1] and cx.delta == 2.0**40
    assert ffi.cast("MLWE", cx.obj).ring == scheme.rings[1].arith_ring
    dec = scheme.decode(scheme.decrypt(cx, key), scaling_factor=cx.delta)
    assert all(abs(a - d) < 1e-3 for a, d in zip(x, dec, strict=True))
    # and it multiplies with a ciphertext that reached level 1 by a rescale
    ones = scheme.encrypt(scheme.encode([1] * (N // 2)), key)
    cy = scheme.encrypt(scheme.encode(y), key) * ones
    assert cy.lvl == 1
    out = cx * cy
    dec = scheme.decode(scheme.decrypt(out, key), scaling_factor=out.delta)
    expected = [a * b for a, b in zip(x, y, strict=True)]
    assert all(abs(e - d) < 1e-3 for e, d in zip(expected, dec, strict=True))
    with pytest.raises(ValueError, match="quotient"):
        cx.mod_reduce(lvl=0)


def test_decrypt_drops_to_the_lowest_level_that_holds_the_message():
    scheme, key = _product_scheme(3)
    x, y = (_unit_values(N // 2) for _ in range(2))
    cx, cy = (scheme.encrypt(scheme.encode(v), key) for v in (x, y))

    # At the scheme's scale the last level holds the message.
    pt = scheme.decrypt(cx, key)
    assert pt.ring == scheme.rings[-1]
    dec = scheme.decode(pt, cx.delta)
    assert all(abs(a - d) < 1e-3 for a, d in zip(x, dec, strict=True))
    full = scheme.decrypt(cx, key, drop=False)
    assert full.ring == cx.ring

    # An unrescaled product sits at delta^2: it needs more primes, and gets them.
    product = scheme.multiply(cx, cy, scheme.rlk)
    pt = scheme.decrypt(product, key)
    assert pt.ring != scheme.rings[-1] and pt.ring.is_quotient_ring(product.ring)
    dec = scheme.decode(pt, product.delta)
    expected = [a * b for a, b in zip(x, y, strict=True)]
    assert all(abs(e - d) < 1e-3 for e, d in zip(expected, dec, strict=True))

    # A larger bound on the message keeps more primes.
    assert scheme.decrypt(cx, key, message_bound=2.0**60).ring.ell > 1

    # The key over the target ring is built once and kept.
    assert key.at_ring(scheme.rings[-1]) is key.at_ring(scheme.rings[-1])
    with pytest.raises(ValueError, match="quotient"):
        scheme.linear_decrypt(scheme.rescale(cx * cy), key, ring=scheme.rings[0])
