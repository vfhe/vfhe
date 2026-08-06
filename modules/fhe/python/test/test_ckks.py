# SPDX-License-Identifier: Apache-2.0
"""Characterization tests for the (reverted) vfhe.fhe CKKS scheme over cffi.

Encode/decode, encrypt/decrypt, slot rotation, and ciphertext multiplication
(ciphertext*ciphertext with relinearization+rescale, and ciphertext*plaintext).
"""

import secrets

import pytest
from vfhe.arith import Ring
from vfhe.arith.residue_selection import search_log_residues_minq0
from vfhe.fhe import CKKS_Ciphertext, CKKS_Scheme

N = 256
rng = secrets.SystemRandom()

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
    return [
        complex(rng.choice([-2, -1, 0, 1, 2]), rng.choice([-2, -1, 0, 1, 2]))
        for _ in range(n)
    ]


def test_encode_decode():
    scheme = CKKS_Scheme(
        Ring(N, 300, split_degree=1), scaling_factor=2**25, special_primes=1
    )
    values = rand_values(N // 2)
    dec = scheme.decode(scheme.encode(values))
    assert all(abs(v - dv) < 1e-3 for v, dv in zip(values, dec))


def test_encrypt_decrypt():
    scheme = CKKS_Scheme(
        Ring(N, 300, split_degree=1), scaling_factor=2**25, special_primes=1
    )
    key = scheme.key_gen_sparse(N // 8, 3.2)
    values = rand_values(N // 2)
    poly = scheme.encode(values)
    ct = scheme.encrypt(poly, key)
    dec = scheme.decode(scheme.decrypt(ct, key))
    assert all(abs(v - dv) < 0.05 for v, dv in zip(values, dec))


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
    assert all(abs(v - d) < 0.05 for v, d in zip(values, dec))


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
    expected = [a * b for a, b in zip(v1, v2)]
    assert all(abs(e - d) < 0.05 for e, d in zip(expected, dec))


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
    expected = [a * b for a, b in zip(v1, v2)]
    assert all(abs(e - d) < 0.05 for e, d in zip(expected, dec))


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
    assert all(abs(v - d) < 0.05 for v, d in zip(values, dec))


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
    assert all(abs(v - d) < 0.05 for v, d in zip(values, dec))


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
    assert all(abs(v - d) < 0.05 for v, d in zip(values, dec))


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
    Rq = Ring(N, split_degree=1, prime_size=log_top_residues + [50])
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
    expected = [a * b * cc * d for a, b, cc, d in zip(*v)]
    assert all(abs(e - d) < 0.05 for e, d in zip(expected, dec))


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
    assert all(abs(v - d) < 0.05 for v, d in zip(values, dec))


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
    expected = [a * b for a, b in zip(v1, v2)]
    assert all(abs(e - d) < 0.05 for e, d in zip(expected, dec))


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
    expected = [a * b for a, b in zip(v1, v2)]
    assert all(abs(e - d) < 0.05 for e, d in zip(expected, dec))
