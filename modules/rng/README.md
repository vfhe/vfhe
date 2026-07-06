# rng

Cryptographic randomness and sampling: random bytes, Gaussian noise, and
sparse ternary vectors. The byte-generation backend is chosen at compile
time: AES-NI counter mode on x86 SIMD builds, BLAKE3 otherwise; seeding uses
RDRAND on x86 or `/dev/urandom` elsewhere.

Internal C only — no Python API. It stays a separate module (rather than a
corner of `arith`) because it is the security-sensitive entropy boundary and
because future modules (`mlwe`, `fhe`) will consume it directly, without
going through the arithmetic engine.

Depends on `base` and the vendored BLAKE3.
