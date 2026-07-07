#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""vfhe smoke test — an end-to-end CKKS computation on encrypted vectors.

Builds a CKKS scheme, encrypts two complex vectors, evaluates them
homomorphically (encrypt/decrypt, add, multiply), decrypts, and checks each
result against plaintext arithmetic within CKKS's approximate tolerance. Prints
a readable trace and exits non-zero on any mismatch.

This is both the post-install smoke check (the native extension and proto
bindings must load for any of it to run) and the integration test / usage demo:
one script that proves the whole stack actually computes, not just imports. Used
by CI's package-smoke job (run against the sdist install) and pytest.

Runs against an installed ``vfhe`` or, inside the source tree, the
``modules/*/python/src`` working copy plus the ``.generated`` extension.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_importable() -> None:
    """Resolve ``vfhe`` from an install, else from the source-tree build."""
    try:
        import vfhe.fhe  # noqa: F401

        return
    except ModuleNotFoundError:
        root = Path(__file__).resolve().parent.parent
        generated = root / ".generated"
        if not generated.exists():
            raise  # not installed and not built — nothing we can do
        sys.path.insert(0, str(generated))
        for src in sorted((root / "modules").glob("*/python/src")):
            sys.path.insert(0, str(src))


_ensure_importable()

from vfhe.arith import Ring
from vfhe.fhe import CKKS_Scheme

TOL = 0.05  # CKKS is approximate; results match plaintext to a few decimals


def _check(label: str, got: list[complex], want: list[complex]) -> bool:
    err = max(abs(g - w) for g, w in zip(got, want))
    ok = err < TOL
    print(f"  {label:<26} max|error| = {err:.2e}  [{'ok' if ok else 'FAIL'}]")
    return ok


def main() -> int:
    N = 256
    slots = N // 2
    print(f"vfhe CKKS — ring degree N={N} ({slots} complex slots), scale 2^49\n")

    scheme = CKKS_Scheme(
        Ring(N, 300, split_degree=1), scaling_factor=2**49, special_primes=0
    )
    key = scheme.key_gen_sparse(N // 8, 3.2)
    s = key.poly[0]
    scheme.rlk = scheme.gen_ksk(key, [-(s * s)])  # relinearization key for ct * ct

    # Two plaintext vectors; only the first few slots are non-zero for a legible trace.
    a = [
        complex(1.5, 0.5),
        complex(-2.0, 1.0),
        complex(0.5, -0.5),
        complex(3.0, 0.0),
    ] + [0j] * (slots - 4)
    b = [
        complex(0.5, -0.5),
        complex(1.0, 0.0),
        complex(-1.5, 0.5),
        complex(2.0, 1.0),
    ] + [0j] * (slots - 4)

    ca = scheme.encrypt(scheme.encode(a), key)
    cb = scheme.encrypt(scheme.encode(b), key)
    print("  a[:4] =", [f"{z:.1f}" for z in a[:4]])
    print("  b[:4] =", [f"{z:.1f}" for z in b[:4]], "\n")

    ok = True
    ok &= _check("encrypt / decrypt a", scheme.decode(scheme.decrypt(ca, key)), a)

    c_add = ca + cb
    ok &= _check(
        "a + b",
        scheme.decode(scheme.decrypt(c_add, key)),
        [x + y for x, y in zip(a, b)],
    )

    c_mul = ca * cb  # discrete convolution + relinearization + rescale
    ok &= _check(
        "a * b",
        scheme.decode(scheme.decrypt(c_mul, key), scaling_factor=c_mul.delta),
        [x * y for x, y in zip(a, b)],
    )

    print("\n" + ("OK — homomorphic results match plaintext." if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
