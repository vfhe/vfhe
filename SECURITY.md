<!-- SPDX-License-Identifier: Apache-2.0 -->

# Security policy

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Report it privately through either channel:

- [GitHub private vulnerability reporting](https://github.com/vfhe/vfhe/security/advisories/new)
  (Security > Report a vulnerability), which is preferred: it keeps the
  discussion, the fix, and the advisory in one place.
- Email <maintainers@vfhe.ai>.

Please include what you need us to reproduce it: affected version or commit,
build configuration (portable or AVX-512, compiler and version), parameters, and
a proof of concept if you have one.

We will acknowledge your report within 7 days and keep you updated as we
investigate. Disclosure is coordinated with you: we ask for the customary 60
days to release a fix before details become public. Fixed vulnerabilities are
published as [GitHub security advisories](https://github.com/vfhe/vfhe/security/advisories),
naming the affected versions and the remediation; you will be credited unless
you ask us not to.

## Supported versions

VFHE is **pre-release** software (`0.x`). Only the latest release on
[PyPI](https://pypi.org/project/vfhe/) receives fixes; there are no maintained
back-branches.

A threat model and attack-surface analysis is maintained in
[docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).

## Scope

VFHE implements fully homomorphic encryption and related proof systems. The
following are in scope and worth reporting:

- Cryptographic errors: incorrect arithmetic, a broken NTT or FFT transform, bad
  parameter derivation, insufficient noise, faulty randomness.
- Memory safety in the C kernels: out-of-bounds access, use-after-free, integer
  overflow reachable from library inputs.
- Anything that lets an attacker recover a key or plaintext.

Two limitations are known and documented rather than reported:

- **This library is not yet audited and is not for production use.** The README
  states this, and a finding that amounts to "it is unaudited" is not a report.
- **Timing side channels are not currently in scope.** VFHE is not hardened for
  constant-time execution. Reports are still welcome; treat such a finding as a
  design gap rather than a vulnerability with a patch pending.

## Secrets and credentials

The project stores no long-lived secrets in version control or CI. Publishing
to PyPI uses [Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
(short-lived OIDC tokens minted per run), and release provenance is signed with
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
- **Code (SAST):** CodeQL analyses every pull request and the default branch.
  **Threshold:** no CodeQL alert of high severity or above may be merged
  unresolved; each is fixed or, if a true false positive, dismissed with a
  written justification recorded in the alert.

Findings on dependency code paths that VFHE does not execute are documented as
non-exploitable in the project's [VEX feed](docs/vex/) rather than treated as
release blockers.
