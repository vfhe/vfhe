<!-- SPDX-FileCopyrightText: 2026 Alin-Petru Roșu -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until 1.0.0, minor
versions may contain breaking changes.

## [Unreleased]

### Added

- Add engine selection at import. An install carries one extension per
  instruction-set level its architecture supports (`portable`, `avx512ifma`,
  `neon`) and loads the best one this CPU can run, asking a separate probe
  extension so choosing an engine never loads one. `VFHE_ENGINE=<name>` pins
  the choice and refuses a CPU that cannot run it, and
  `vfhe_engine_active()` reports it.
- Add wheels for manylinux x86_64/aarch64 and macOS arm64, published
  alongside the sdist with the same provenance attestation, so most installs
  need no compiler.
- Add arm64 Linux to the tested platforms.
- Add `python -m vfhe.info`, which prints the version, the selected engine,
  anything faster this CPU could have run, and the platform — the whole
  environment a bug report needs.
- Add PEP 561 typing markers to every subpackage, so a type checker resolves
  `vfhe.*` against an install.
- Add the metadata a scanner and a redistributor need to every wheel:
  - CycloneDX SBOM fragments at `.dist-info/sboms/`, the location PEP 770
    standardises — the vendored BLAKE3 sources, plus pedigree entries for code
    adapted from MOSFHET, Intel HEXL, and a modular-inverse routine whose
    upstream licence is unstated. A metadata scan cannot see C compiled into
    an extension, and install tools copy that directory, so a scanner reading
    an installed wheel can.
  - The Apache-2.0 licence text and `NOTICE` at `.dist-info/licenses/`, so a
    redistributor receives the attribution the licence requires for the C
    vFHE ships.

### Changed

- **BREAKING**: Split `vfhe.misc` into `vfhe.engine` (the native handle) and
  `vfhe.dynamic_extensions` (runtime C compilation), and move randomness into
  a new internal `crypto` module. Import `from vfhe.engine import ffi, lib`
  in place of `vfhe.misc.libvfhe`, and `vfhe.dynamic_extensions` in place of
  `vfhe.misc.dynamic_extensions`.
- Change `vfhe.dynamic_extensions` to compile only your files and link them
  against the shipped `libvfhe_<engine>.a`, rather than rebuilding the whole
  library from source. A runtime compile drops from the full kernel set to
  your snippet, and an install no longer carries C sources or build
  machinery.
- Change the build backend to meson (`meson-python`). `setup.py` and
  `MANIFEST.in` are gone, and an sdist vendors the BLAKE3 sources with a
  frozen version, so building one needs neither git nor submodules.
- Verify a release's provenance after publishing as well as before: every
  file the index serves is downloaded and checked against the one workflow
  allowed to have signed it.
- Change the security contact to <security@vfhe.ai> and the conduct contact
  to <conduct@vfhe.ai>.

### Removed

- **BREAKING**: Remove the `VFHE_PORTABLE` and `VFHE_TUNED` build knobs.
  Every build now carries every engine its architecture supports and picks
  one at import.
- Remove the CycloneDX SBOM asset from releases. It described one CI runner's
  dependency resolution, which no user reproduces; the vendored-C record it
  uniquely held now ships inside every wheel.

### Fixed

- Fix `vfhe.dynamic_extensions`:
  - It failed from an installed package, because the C sources it recompiles
    were never installed and a clean Python 3.12 or later lacks the
    setuptools cffi compiles through. Installs now ship the build inputs as
    `vfhe/_source`.
  - User-supplied compiler and linker flags replaced the build plan's instead
    of extending them, reinitialization failures degraded to warnings,
    repeated compiles accumulated `sys.path` entries, and `XDG_CACHE_HOME`
    was ignored.
- Fix an async multilinear-polynomial evaluation being collected mid-flight,
  because the event loop holds tasks only weakly and the spawned task was
  never referenced.
- Fix the generated protobuf bindings:
  - `import vfhe.circuit` raised `VersionError` on protobuf 6.x, which the
    declared floor allowed, because the bindings are generated against 7.35.1
    and protobuf refuses a runtime older than its gencode. The floor now
    states what they need (`protobuf>=7.35.1,<8`).
  - They installed as implicit namespace packages, so another distribution
    shipping a `_vfhe_proto` directory would have merged into the same
    namespace instead of conflicting.

## [0.0.2] - 2026-07-22

### Added

- Add `AUTHORS.md`, `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  and this changelog.
- Add Python 3.14 support, tested in CI and declared in the classifiers.
- Add coverage reporting for Python and C, informational only.
- Add nightly and per-pull-request fuzzing (ClusterFuzzLite).
- Attach a CycloneDX SBOM and Sigstore build provenance to each GitHub
  Release (verifiable with `gh attestation verify`), and publish to PyPI via
  Trusted Publishing.

### Changed

- Move the development guide from the README to `docs/DEVELOPMENT.md`. The
  README now targets users and renders cleanly on PyPI.
- Rework CI end to end: parallel required checks behind one gate, SHA-pinned
  actions, hardened permissions, and an sdist install-and-smoke check.
- Raise the minimum `cffi` to 2.1 at runtime and `setuptools-scm` to 10.2 at
  build time.

### Fixed

- Fix `ntt_new_proc` looping forever searching for a primitive root of unity
  with certain prime and ring-size combinations. The search is now
  deterministic and always terminates.
- Fix a module compiled at runtime auto-tuning independently of the loaded
  engine, so a portable process could load AVX-512 kernels and crash. A
  custom build now inherits the loaded engine's mode.

## [0.0.1] - 2026-07-08

### Added

- Publish the initial pre-release on PyPI: RNS polynomial arithmetic with
  incomplete NTTs (`arith`), LWE / Module-LWE and MGSW (`mlwe`), CKKS with
  CGGI16 and GP25 bootstrapping (`fhe`), layered GKR circuits (`circuit`),
  and an AVX-512 or portable native engine.
- Distribute as an sdist, which builds against the host CPU at install time.

[Unreleased]: https://github.com/vfhe/vfhe/compare/0.0.2...HEAD
[0.0.2]: https://github.com/vfhe/vfhe/compare/0.0.1...0.0.2
[0.0.1]: https://github.com/vfhe/vfhe/releases/tag/0.0.1
