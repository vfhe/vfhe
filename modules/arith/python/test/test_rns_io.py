# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""RNS rings and polynomials through vfhe.io, and polynomials from a seed."""

import json
import os
import struct
import subprocess
import sys
import textwrap

import pytest
from vfhe.arith import Polynomial, Ring
from vfhe.arith import repr as Repr
from vfhe.engine import ffi, lib
from vfhe.io import Serializer

# Narrow (stored 32-bit), a prime between 2^30 and 2^32 (stored 64-bit,
# written as 4-byte words), and wide primes up to the 62-bit limit.
PRIME_SIZES = [[28, 50], [31, 32, 40], [60, 61, 49]]


def _ring(N, sizes, split):
    return Ring(N, prime_size=sizes, split_degree=split)


def _values(ring, seed=1):
    return [((i * 7919 + seed) * 104729) % 2**40 - 2**39 for i in range(ring.N)]


@pytest.mark.parametrize("sizes", PRIME_SIZES)
@pytest.mark.parametrize("split", [1, 4])
@pytest.mark.parametrize("packing", ["word", "tight"])
@pytest.mark.parametrize("domain", ["held", "mul", "canonical"])
@pytest.mark.parametrize("held", [Repr.coeff, Repr.ntt])
def test_polynomial_round_trip(sizes, split, packing, domain, held):
    ring = _ring(256, sizes, split)
    p = Polynomial(ring).from_bigint_array(_values(ring))
    p.to_repr(held)
    s = Serializer(packing=packing, domain=domain)
    back = s.loads(s.dumps(p), rings=[ring])
    assert back.ring is ring
    expected = {"held": held, "mul": Repr.ntt, "canonical": Repr.coeff}[domain]
    assert back.repr == expected
    assert back == p
    assert p.repr == held  # writing never converts the object written


def test_tight_packing_is_smaller_by_the_bit_count():
    ring = _ring(1024, [28, 50], 1)
    p = Polynomial(ring).from_bigint_array(_values(ring))
    word = len(Serializer(packing="word", checksum=False).dumps(p))
    tight = len(Serializer(packing="tight", checksum=False).dumps(p))
    assert word - tight == 1024 * (4 + 8) - 1024 * (28 + 50) // 8


def test_ring_rebuilt_from_its_primes():
    ring = _ring(128, [40, 45, 50], 2)
    p = Polynomial(ring).from_bigint_array(_values(ring))
    back = Serializer().loads(Serializer().dumps(p))
    assert back.ring is not ring
    assert back.ring.primes == ring.primes  # the ring's own order, kept
    assert back.ring.mask == ring.mask
    assert back == p


def _rows(data: bytes):
    pos, out = 16, []
    while pos < len(data):
        _, tag_len, meta_len, size = struct.unpack_from("<B3xIQQ", data, pos)
        pos += 24
        tag = data[pos : pos + tag_len].decode()
        pos += tag_len
        meta_at = pos
        pos += meta_len
        if size:
            pos += -pos % 64
        out.append((tag, meta_at, meta_len, pos, size))
        pos += size + (32 if data[8] & 1 else 0)
    return out


def test_a_residue_above_its_prime_is_refused():
    ring = _ring(64, [50], 1)
    p = Polynomial(ring).from_bigint_array(_values(ring))
    for packing in ["word", "tight"]:
        s = Serializer(packing=packing, checksum=False)
        data = bytearray(s.dumps(p))
        _, _, _, offset, _ = next(
            r for r in _rows(bytes(data)) if r[0] == "arith.rns_polynomial"
        )
        data[offset : offset + 7] = b"\xff" * 7  # >= 2^50 in the first residue
        with pytest.raises(ValueError, match="not below"):
            s.loads(bytes(data))
        unchecked = Serializer(packing=packing, checksum=False, validate=False)
        unchecked.loads(bytes(data))


def test_ntt_rows_need_the_same_roots():
    ring = _ring(64, [50], 1)
    p = Polynomial(ring).from_bigint_array(_values(ring))
    p.to_NTT()
    s = Serializer(checksum=False)
    data = s.dumps(p)
    _, at, n, _, _ = next(r for r in _rows(data) if r[0] == "arith.rns_ring")
    meta = json.loads(data[at : at + n])
    root = str(meta["roots"][0]).encode()
    # Another value of the same length, so the framing does not move.
    other = str(int(root) - 1).encode().rjust(len(root), b"1")
    tampered = data[:at] + data[at : at + n].replace(root, other) + data[at + n :]
    with pytest.raises(ValueError, match="other roots of unity"):
        s.loads(tampered)
    # The coefficient domain does not depend on the roots.
    s.loads(Serializer(domain="canonical", checksum=False).dumps(p))


def test_seeded_polynomial_is_its_rows_expansion():
    ring = _ring(256, [28, 50, 60], 4)
    a = Polynomial(ring)
    lib.polynomial_RNS_expand_seeded(a.obj, b"seed", 4, 3)
    a.repr = Repr.ntt
    again = Polynomial(ring)
    lib.polynomial_RNS_expand_seeded(again.obj, b"seed", 4, 3)
    again.repr = Repr.ntt
    assert a == again
    other = Polynomial(ring)
    lib.polynomial_RNS_expand_seeded(other.obj, b"seed", 4, 4)
    other.repr = Repr.ntt
    assert a != other
    rows = a.get_coeff_matrix(repr=Repr.ntt)
    for q, row in zip(ring.primes, rows, strict=True):
        assert all(0 <= v < q for v in row)
    for idx in ring.prime_indices:
        assert lib.polynomial_RNS_matches_seeded(a.obj, False, idx, b"seed", 4, 3)
        assert not lib.polynomial_RNS_matches_seeded(a.obj, False, idx, b"seed", 4, 2)
    a.to_coeff()
    for idx in ring.prime_indices:
        assert lib.polynomial_RNS_matches_seeded(a.obj, True, idx, b"seed", 4, 3)


def test_seeded_rows_are_named_by_prime():
    # The same prime in two rings, at different base positions: same row.
    big = _ring(256, [50, 51, 52], 1)
    small = Ring(256, primes=[big.primes[2]], prime_size=[52], split_degree=1)
    x, y = Polynomial(big), Polynomial(small)
    for p in (x, y):
        lib.polynomial_RNS_expand_seeded(p.obj, b"k", 1, 0)
        p.repr = Repr.ntt
    assert x.get_coeff_matrix(repr=Repr.ntt)[2] == y.get_coeff_matrix(repr=Repr.ntt)[0]


_CHILD = textwrap.dedent(
    """
    import json, sys
    from vfhe.arith import Polynomial, Ring
    from vfhe.arith import repr as Repr
    from vfhe.io import Serializer
    primes = json.loads(sys.argv[1])
    # Populate the base with the primes in reverse, and another prime first,
    # so every prime sits at another base index than in the writer.
    Ring(256, prime_size=[45], split_degree=1)
    for p in reversed(primes):
        Ring(256, primes=[p], prime_size=[p.bit_length()], split_degree=1)
    with open(sys.argv[2], "rb") as f:
        p = Serializer().load(f)
    print(json.dumps({"ntt": p.repr == Repr.ntt,
                      "rows": p.get_coeff_matrix(repr=Repr.ntt),
                      "primes": p.ring.primes}))
    """
)


def test_another_process_with_another_base_order_reads_the_same(tmp_path):
    ring = _ring(256, [50, 40, 60], 1)
    ring.quotient_ring(ell=1)
    p = Polynomial(ring).from_bigint_array(_values(ring))
    p.to_NTT()
    path = tmp_path / "p.vfhe"
    with open(path, "wb") as f:
        Serializer().dump(p, f)
    # The child picks its own engine: under an emulator it runs on the bare CPU.
    env = {k: v for k, v in os.environ.items() if k != "VFHE_ENGINE"}
    out = subprocess.run(  # noqa: S603 - this interpreter, a fixed script
        [
            sys.executable,
            "-W",
            "ignore",
            "-c",
            _CHILD,
            json.dumps(ring.primes),
            str(path),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    got = json.loads(out.stdout)
    assert got["ntt"]
    assert got["primes"] == ring.primes
    assert got["rows"] == p.get_coeff_matrix(repr=Repr.ntt)


def test_expansion_matches_crypto_definition():
    ring = _ring(64, [50], 1)
    a = Polynomial(ring)
    lib.polynomial_RNS_expand_seeded(a.obj, b"s", 1, 9)
    a.repr = Repr.ntt
    key = ffi.new("uint8_t[16]")
    label = ffi.new("uint64_t[]", [9, ring.primes[0]])
    lib.prng_expand_key(
        key,
        b"vfhe 2026-10-04 RNS polynomial mul-domain row from a seed",
        b"s",
        1,
        label,
        2,
    )
    out = ffi.new("uint64_t[]", 64)
    lib.prng_expand_below(out, 64, 0, ring.primes[0], key)
    assert a.get_coeff_matrix(repr=Repr.ntt)[0] == list(out)
