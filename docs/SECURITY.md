# Security policy

For anyone reporting a vulnerability in vFHE, and for readers assessing
what the project defends against.

## Reporting a vulnerability

**Please do not open a public issue for a security problem.** Report it
privately by email to <security@vfhe.ai>.

> **Before you invest time.** vFHE is unaudited pre-release software, not
> yet ready for production use, and it changes freely between `0.x`
> releases. Findings are welcome and appreciated, but they carry more
> value — for you and for us — against a stable release. Until the first
> stable release the project runs no formal disclosure programme: GitHub's
> private reporting form is switched off, and coordination happens over
> email.

Please include what we need to reproduce it: the affected version or
commit, build configuration (which engine, compiler and version),
parameters, and, if you have it, code that triggers the issue.

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

This model defines what is in and out of scope for a report.
It is a living document, revised for new features and breaking changes.
It states what the project reasons about, not a guarantee.

### Actors and trust boundaries

| Actor | Trust | Can influence |
|---|---|---|
| Library caller | trusted with its own data | all public API inputs: ring parameters, plaintexts, ciphertexts, keys, serialized data |
| Attacker supplying data to a caller | untrusted | any bytes that reach the API as ciphertexts or serialized messages |
| Host RNG | trusted | key and noise generation |
| Build/supply chain | trusted, verified | the compiled extension and vendored software |
| Caller of dynamic extensions | trusted (compiles C) | runtime-compiled native code |

The primary trust boundary is the **Python -> C (cffi) boundary**: every
public Python call crosses into C, where nothing checks bounds or lifetimes
at runtime — a memory error corrupts the process. Different architectures are
supported by specific **engines**, some adding instruction-level
optimization; an install carries every engine its architecture can run and
loads the optimal one at import.

### Assets

- Secret keys and plaintexts (confidentiality).
- Correctness of homomorphic results (integrity of computation).
- Host process memory (no corruption from library inputs).

### Threats and mitigations

- **Memory-safety bugs in the C kernels.** Attacker-shaped data —
  ciphertexts, parameters, serialized messages — reaches C code, where a bug
  corrupts memory instead of raising an error. Mitigation: fuzzing, the test
  suites under sanitizers (ASan, UBSan), and static analysis, on every
  engine; no static-analysis alert of high severity or above may be merged
  unresolved, and a true false positive is dismissed with a written
  justification recorded in the alert.
- **Implementation incorrectness.** A miscomputed transform, wrong modular
  arithmetic, bad parameter derivation, or insufficient noise breaks
  security silently. Mitigation: testing, with coverage measured on every
  change — results are compared against plaintext computation, the portable
  engine, and algebraic identities.
- **Weak or predictable randomness.** Predictable key material breaks
  everything built on it. Mitigation: seeds come from the CPU or the
  operating system and are expanded with a cryptographic generator; only
  test code can make the stream deterministic.
- **Supply-chain tampering.** The shipped artifact could differ from the
  audited source. Mitigation: signed provenance and pinned dependencies —
  see [Supply-chain security](#supply-chain-security).

### Explicitly out of scope

- **The cryptographic soundness of the implemented schemes.** vFHE implements
  published schemes and assumes their security claims. Report a scheme or
  parameter regime shown insecure by cryptanalysis to the scheme's authors.
- **C given to runtime compilation.** vFHE compiles caller-provided C into
  the process on request via dynamic extensions, which runs with
  the caller's full trust.
- **Timing and other side channels.** vFHE is not constant-time; do not use it
  where an adversary observes timing, cache, or power.
- **Production use.** Unaudited pre-release software; a finding that
  amounts to "it is unaudited" is not a report — the README already says
  so.

## Supply-chain security

The threat is a gap between the audited source and the shipped artifact.

- **Provenance.** Release artifacts carry signed build provenance from
  ephemeral keys, and each release verifies what the index actually serves
  against the signing workflow. The README's
  [Verifying a release](https://github.com/vfhe/vfhe/blob/main/README.md#verifying-a-release)
  shows the check any user can run.
- **Publishing.** PyPI uploads use
  [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) —
  short-lived OIDC tokens minted per run — so there is no publishing
  credential to steal or rotate.
- **Dependencies.** Everything entering the build is pinned: submodules to
  commits, CI actions to hashes, the CI toolchain to exact versions.
- **Known vulnerabilities.** No dependency with a known high- or
  critical-severity vulnerability, and none under a license incompatible
  with Apache-2.0, may be merged or released.
- **Secrets.** The project stores no long-lived publishing or code-access
  secrets in version control or CI. Secret scanning covers the whole
  history, and push protection rejects an accidental credential before the
  commit lands. Any credential that must exist is a repository secret, known
  only by maintainers, never hard-coded.
