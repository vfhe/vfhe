<!-- SPDX-License-Identifier: Apache-2.0 -->

# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until 1.0.0, minor
versions may contain breaking changes.

## [Unreleased]

## [0.0.2] - 2026-07-22

### Added

- `AUTHORS.md`, `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and this
  changelog.
- Pre-commit hooks (ruff, clang-format, file hygiene, workflow validation) and
  Dependabot updates for actions and Python dependencies.
- Coverage reporting for both Python and C: job summary and pull request
  comment, informational only.
- ClusterFuzzLite: batch fuzzing nightly and per-pull-request fuzzing of
  changed code (build under `.clusterfuzzlite/`), with `make fuzz-local` for
  short local runs.
- Python 3.14 support: tested in CI and declared in the classifiers.
- Releases attach a CycloneDX SBOM and Sigstore build provenance to the GitHub
  Release (verifiable with `gh attestation verify`) and publish to PyPI via
  Trusted Publishing.

### Changed

- Build tooling consolidated under `packaging/` (previously split across
  `native/`, `stubs/`, and part of `scripts/`); `scripts/` now holds only
  runnable developer tools.
- The development guide moved from the README to `docs/DEVELOPMENT.md`; the
  README now targets users and renders cleanly on PyPI.
- CI reworked end to end: parallel required checks behind a single `CI OK`
  gate, SHA-pinned actions, hardened permissions, and an sdist
  install-and-smoke check in a clean environment.

### Fixed

- `ntt_new_proc` could loop forever searching for a primitive root of unity with
  certain prime and ring-size combinations; the search is now deterministic and
  always terminates.

## [0.0.1] - 2026-07-08

### Added

- Distribution as sdist; the package builds against the host CPU at
  install time.
- Initial pre-release on PyPI: RNS polynomial arithmetic with incomplete NTTs
  (`arith`), LWE / Module-LWE and MGSW (`mlwe`), CKKS with CGGI16 and GP25
  bootstrapping (`fhe`), layered GKR circuits (`circuit`), and an AVX-512 or
  portable native engine.

[Unreleased]: https://github.com/vfhe/vfhe/compare/0.0.2...HEAD
[0.0.2]: https://github.com/vfhe/vfhe/compare/0.0.1...0.0.2
[0.0.1]: https://github.com/vfhe/vfhe/releases/tag/0.0.1
