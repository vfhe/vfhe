<!-- SPDX-FileCopyrightText: 2026 Alin-Petru Roșu -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Security policy

## Reporting a vulnerability

**Please do not open a public issue for a security problem.** Report it
privately by email to <security@vfhe.ai>.

> **Before you invest time.** vFHE is unaudited pre-release software, not
> yet ready for production use, and it changes freely between `0.x`
> releases. In the period before the first major release, while findings
> are welcome and much appreciated, they are worth more — to you and to
> us — once there is a stable release worth attacking. Until then we run no formal
> disclosure programme: GitHub's private reporting form is switched off, and
> the only coordination available on these matters is over email.

Please include what we need to reproduce it: the affected version or
commit, build configuration (which engine, compiler and
version), parameters, and, if you have it, code that triggers the issue.

<!--
We will acknowledge your report as soon as possible and keep you updated as we
investigate. Disclosure will be coordinated with you. Fixed vulnerabilities are
published as [GitHub Advisories](https://github.com/vfhe/vfhe/security/advisories),
naming the affected versions and the remediation; you will be credited unless
you ask us not to.
-->

## Supported versions

At **major version zero** (`0.x`), every release is a pre-release and may
contain breaking changes. Only the **latest release** on
[PyPI](https://pypi.org/project/vfhe/) receives bug fixes and security
updates; upgrading is the remediation path for any fixed issue. This
policy will be revised at the first stable (`1.0`) release.

## Threat model

Scope: the vFHE library as shipped. A living document, revised for new
features and breaking changes, and the definition of what is in and out of
scope for a report. vFHE is pre-release and unaudited; the model states
what the project reasons about, not a guarantee.

### Actors and trust boundaries

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

### Assets

- Secret keys and plaintexts (confidentiality).
- Correctness of homomorphic results (integrity of computation).
- Host process memory (no corruption from library inputs).

### Threats and mitigations

- **Memory-safety bugs in the C kernels reachable from inputs.** Highest-impact
  class: an out-of-bounds or use-after-free driven by attacker-influenced
  ciphertext/parameter data. Mitigations: libFuzzer harnesses with ASan/UBSan
  (changed code per pull request, the full set nightly); the C test suite
  runs under sanitizers; CodeQL on every change — where each runs is in the
  [development guide](https://github.com/vfhe/vfhe/blob/main/DEVELOPMENT.md).
  Gap: fuzz coverage currently spans the NTT surface, not all kernels.
- **Cryptographic incorrectness.** A broken NTT/FFT, bad parameter derivation,
  or insufficient noise silently breaks security. Mitigations: characterization
  tests over the public API, end-to-end CKKS validation against plaintext, and
  every engine cross-checked against the portable one. The `ntt_new_proc` non-termination
  fix is an example of this class.
- **Weak or predictable randomness.** Keys and noise depend on the C PRNG.
  Mitigation: hardware-seeded BLAKE3/AES-CTR; the deterministic seed exists only
  for tests and is never the default.
- **Engine mismatch.** Mixing two engines' kernels in one
  process corrupts state. Structurally prevented: the engine is fixed when
  the binary is built, and runtime-compiled modules take theirs from the
  loaded binary (`vfhe_engine_active()`).
- **Malicious runtime compilation.** `dynamic_extensions` compiles caller-
  provided C into the process by design; it is as trusted as the calling code
  and must never be fed untrusted C. Documented, not sandboxed.
- **Supply-chain tampering.** The threat is a gap between the audited
  source and the shipped artifact. The pinning, provenance, and publishing
  machinery answering it is documented in
  [Secrets and credentials](#secrets-and-credentials).

### Explicitly out of scope

- **Timing and other side channels.** vFHE is not constant-time; do not use it
  where an adversary observes timing, cache, or power.
- **Production use.** Unaudited pre-release software; a finding that
  amounts to "it is unaudited" is not a report — the README already says
  so.

## Secrets and credentials

The project stores no long-lived publishing or code-access secrets in version
control or CI; the single stored CI secret is a Zulip API key that posts
build-failure notifications to the team chat, held by a workflow that never
executes repository code. Publishing to PyPI uses
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (short-lived
OIDC tokens minted per run), and release provenance is signed with
Sigstore's ephemeral keys; there is nothing to rotate. GitHub Actions runs with
a read-only default token, elevated per job only where required. A
`detect-private-key` pre-commit hook and GitHub push protection block
accidental credential commits. Any credential that must exist (e.g. a future
external service token) is a GitHub Actions secret or environment secret,
readable only by maintainers, never hard-coded.

## Dependency and static-analysis policy

Every change is automatically evaluated before it can merge:

- **Dependencies (SCA):** `dependency-review` blocks any pull request that adds
  a dependency with a known vulnerability; Dependabot proposes fixes for
  existing ones. **Threshold:** no dependency with a known high- or
  critical-severity vulnerability, and no dependency under a license
  incompatible with Apache-2.0, may be merged or released. Lower-severity
  findings are triaged and tracked. A release is blocked while any unresolved
  high/critical dependency finding applies to the shipped code.
- **Code (SAST):** static analysis runs on every pull request and the default
  branch, one leg per engine so no kernel is preprocessed away.
  **Threshold:** no alert of high severity or above may be merged
  unresolved; each is fixed or, if a true false positive, dismissed with a
  written justification recorded in the alert.

Findings on dependency code paths that vFHE does not execute are recorded as
non-exploitable in the finding itself — dismissed with a written justification —
rather than treated as release blockers.
