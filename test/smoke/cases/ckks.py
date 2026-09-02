# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Smoke test: end-to-end CKKS on encrypted vectors, verified against plaintext."""

from _report import check, exit_status
from vfhe.arith import Ring
from vfhe.fhe import CKKS_Scheme

TOL = 0.05  # CKKS is approximate


def _close(label: str, got: list[complex], want: list[complex]) -> bool:
    err = max(abs(g - w) for g, w in zip(got, want, strict=True))
    return check(f"{label}  max|error| = {err:.2e}", err < TOL)


def main() -> int:
    N = 256
    slots = N // 2
    print(f"vfhe CKKS: ring degree N={N} ({slots} complex slots), scale 2^49\n")

    scheme = CKKS_Scheme(
        Ring(N, 300, split_degree=1), scaling_factor=2**49, special_primes=0
    )
    key = scheme.key_gen_sparse(N // 8, 3.2)
    secret = key.poly[0]
    # Needed for ciphertext * ciphertext.
    scheme.rlk = scheme.gen_rlk(key, [-(secret * secret)])

    def plain(ct, scaling_factor: float | None = None) -> list[complex]:
        return scheme.decode(scheme.decrypt(ct, key), scaling_factor)

    # Only the first four slots are non-zero, for a legible trace.
    a = [1.5 + 0.5j, -2 + 1j, 0.5 - 0.5j, 3 + 0j] + [0j] * (slots - 4)
    b = [0.5 - 0.5j, 1 + 0j, -1.5 + 0.5j, 2 + 1j] + [0j] * (slots - 4)
    pairs = list(zip(a, b, strict=True))

    ct_a = scheme.encrypt(scheme.encode(a), key)
    ct_b = scheme.encrypt(scheme.encode(b), key)
    print("  a[:4] =", [f"{z:.1f}" for z in a[:4]])
    print("  b[:4] =", [f"{z:.1f}" for z in b[:4]], "\n")

    ok = _close("encrypt / decrypt a", plain(ct_a), a)
    ok &= _close("a + b", plain(ct_a + ct_b), [x + y for x, y in pairs])

    ct = ct_a * ct_b  # tensor product + relinearization + rescale
    ok &= _close("a * b", plain(ct, scaling_factor=ct.delta), [x * y for x, y in pairs])

    return exit_status(ok, "homomorphic results match plaintext.")


if __name__ == "__main__":
    raise SystemExit(main())
