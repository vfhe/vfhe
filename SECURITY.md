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

We will acknowledge your report and keep you updated as we investigate. Once a
fix is available we will credit you in the advisory unless you ask us not to.

## Supported versions

VFHE is **pre-release** software (`0.x`). Only the latest release on
[PyPI](https://pypi.org/project/vfhe/) receives fixes; there are no maintained
back-branches.

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
