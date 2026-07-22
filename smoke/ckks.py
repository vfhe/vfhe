#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Smoke test: end-to-end CKKS on encrypted vectors, verified against plaintext.

Standalone and self-verifying: needs an importable vfhe, exits non-zero on
mismatch.
"""

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
    print(f"vfhe CKKS: ring degree N={N} ({slots} complex slots), scale 2^49\n")

    scheme = CKKS_Scheme(
        Ring(N, 300, split_degree=1), scaling_factor=2**49, special_primes=0
    )
    key = scheme.key_gen_sparse(N // 8, 3.2)
    secret = key.poly[0]
    # Relinearization key for ciphertext * ciphertext.
    scheme.rlk = scheme.gen_ksk(key, [-(secret * secret)])

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

    ct_a = scheme.encrypt(scheme.encode(a), key)
    ct_b = scheme.encrypt(scheme.encode(b), key)
    print("  a[:4] =", [f"{z:.1f}" for z in a[:4]])
    print("  b[:4] =", [f"{z:.1f}" for z in b[:4]], "\n")

    ok = True
    ok &= _check("encrypt / decrypt a", scheme.decode(scheme.decrypt(ct_a, key)), a)

    ct_sum = ct_a + ct_b
    ok &= _check(
        "a + b",
        scheme.decode(scheme.decrypt(ct_sum, key)),
        [x + y for x, y in zip(a, b)],
    )

    ct_prod = ct_a * ct_b  # discrete convolution + relinearization + rescale
    ok &= _check(
        "a * b",
        scheme.decode(scheme.decrypt(ct_prod, key), scaling_factor=ct_prod.delta),
        [x * y for x, y in zip(a, b)],
    )

    print("\n" + ("OK: homomorphic results match plaintext." if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
