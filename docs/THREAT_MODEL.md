<!-- SPDX-License-Identifier: Apache-2.0 -->

# Threat model

Scope: the VFHE library as shipped. This is a living document, revised for new
features and breaking changes. It complements the reporting scope in
[SECURITY.md](../SECURITY.md).

VFHE is pre-release and unaudited; this model states what the project reasons
about, not a guarantee.

## Actors and trust boundaries

| Actor | Trust | Can influence |
|---|---|---|
| Library caller | trusted with its own data | all public API inputs: ring parameters, plaintexts, ciphertexts, keys, serialized protobuf |
| Attacker supplying data to a caller | untrusted | any bytes that reach the API as ciphertexts or serialized messages |
| Host RNG | trusted | key and noise generation (BLAKE3-seeded PRNG / AES-CTR) |
| Build/supply chain | trusted, verified | the compiled `_vfhe_native` extension and vendored BLAKE3 |
| `vfhe.misc.dynamic_extensions` caller | trusted (compiles C) | runtime-compiled native code |

The primary trust boundary is the **Python -> C (cffi) boundary**: every public
Python call crosses into unmanaged C, where memory-safety errors become
exploitable rather than exceptions.

## Assets

- Secret keys and plaintexts (confidentiality).
- Correctness of homomorphic results (integrity of computation).
- Host process memory (no corruption from library inputs).

## Threats and mitigations

- **Memory-safety bugs in the C kernels reachable from inputs.** Highest-impact
  class: an out-of-bounds or use-after-free driven by attacker-influenced
  ciphertext/parameter data. Mitigations: libFuzzer harnesses with ASan/UBSan
  on changed code per pull request and one hour nightly; the C test suite runs
  under sanitizers; CodeQL on every change. Gap: fuzz coverage currently spans
  the NTT surface, not all kernels.
- **Cryptographic incorrectness.** A broken NTT/FFT, bad parameter derivation,
  or insufficient noise silently breaks security. Mitigations: characterization
  tests over the public API, end-to-end CKKS validation against plaintext, and
  the SIMD/portable engines cross-checked. The `ntt_new_proc` non-termination
  fix is an example of this class.
- **Weak or predictable randomness.** Keys and noise depend on the C PRNG.
  Mitigation: hardware-seeded BLAKE3/AES-CTR; the deterministic seed exists only
  for tests and is never the default.
- **Engine-mode mismatch.** A process mixing portable and CPU-tuned kernels
  crashed; `dynamic_extensions` now inherits the loaded engine's mode.
- **Malicious runtime compilation.** `dynamic_extensions` compiles caller-
  provided C into the process by design; it is as trusted as the calling code
  and must never be fed untrusted C. Documented, not sandboxed.
- **Supply-chain tampering.** Mitigations: SHA-pinned actions, digest-pinned
  build container, vendored deps pinned to commits, SBOM and Sigstore provenance
  per release, Trusted Publishing (no stored tokens).

## Explicitly out of scope

- **Timing and other side channels.** VFHE is not constant-time; do not use it
  where an adversary observes timing, cache, or power.
- **Production use.** Unaudited pre-release software.
