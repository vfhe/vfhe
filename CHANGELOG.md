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

- Commits must carry the Developer Certificate of Origin sign-off
  (`git commit -s`), enforced on every pull request by CI's `DCO` check and
  locally by a commit-msg pre-commit hook.
- `.github/CODEOWNERS` routes reviews: the crypto authors own the C kernels,
  the infrastructure maintainer owns packaging and CI.
- `GOVERNANCE.md`: roles, how decisions are made, and the two-maintainer rule
  that keeps access to every sensitive resource survivable.
- `AUTHORS.md` records the project's history and funding origin, and states
  that a steering committee will be established upon the first release.
- A CI secret scan (gitleaks, sha256-pinned) over the full git history, on
  every pull request.
- Sanitizer legs: the C tests run under ASan+UBSan for the portable engine on
  every pull request, and for the `avx512ifma` engine nightly on real AVX-512
  IFMA hardware.
- The workflow security linter zizmor: in CI lint, the dev group, and the
  pre-commit hooks (on the workflow and action files a commit touches).
- `ci-constraints-requirements.txt` pins the entire CI Python
  toolchain (applied via `PIP_CONSTRAINT`), so an upstream release cannot
  change the gate under a commit; Dependabot keeps it fresh, and also now
  watches the git submodules.
- Engine selection at import: an install carries one extension per engine its
  architecture can build (meson compiles the tree once per engine — today
  `portable` and `avx512ifma`), and the pure-Python `_vfhe_native` picker
  imports the best one this CPU can run. `vfhe_cpu_supports("<capability>")`
  answers that in C, so it stays right per architecture, from its own ~50 KB
  `_vfhe_cpu` extension — choosing an engine never loads one, and exactly one
  engine is imported. `VFHE_ENGINE=<name>` pins the choice (and refuses on a
  CPU that cannot run that engine), `vfhe_engine_active()` reports it, and
  runtime extensions link the active engine's archive. No symbol renaming, no
  trampolines, no generated dispatch glue: the build system does the
  multiplying. The old `VFHE_PORTABLE`/`VFHE_TUNED` build knobs are gone.
- A nightly Hardware Test workflow runs the complete `avx512ifma` suites natively on
  the project's AVX-512 IFMA machine — the standing cross-check of the
  emulator against real silicon. It never gates a merge and never executes
  pull-request code.
- Nightly corpus pruning alongside ClusterFuzzLite batch fuzzing (a
  ClusterFuzzLite requirement), keeping the managed fuzz corpus minimal. Pull
  requests fuzz what the diff touched for five minutes (`code-change` mode).
  The fuzz build targets whichever engine its machine executes — a fuzzer runs
  what it builds — so `batch-fuzzing.yml` covers the portable kernels and
  `hardware-batch-fuzzing.yml`, on the project's own machine, covers
  `avx512ifma`, which nothing fuzzed before. The sessions are ordered rather
  than parallel, a single managed corpus having one writer, and a dispatch input
  starts one of them without the other.
- The hardware lane also smoke-tests an install: it is the only place the engine
  picker selects the SIMD extension and runs it, every other install landing on
  a CPU that falls back to portable.
- The Intel SDE pin (version, sha256, mirrors)
  moved to `.sde.json` at the repo root; CI keys the SDE cache on that
  file's hash. Fuzzing is CI-only: the local fuzz runner was removed
  (reproduce CI findings via the OSS-Fuzz helper, documented in the
  development guide).
- Wheels (cibuildwheel): manylinux x86_64/aarch64 and macOS arm64, carrying
  every engine their architecture has, published alongside the sdist with
  the same provenance attestation.
  Each wheel runs the fast test suite from its own install on its build
  runner — the artifact is tested at birth, in build environments the CI
  matrix never sees; macOS x86_64 stays on the sdist for now.

### Changed

- Runtime extensions (`vfhe.misc.dynamic_extensions`) compile only the
  user's files and link them against a shipped static library
  (`vfhe/_source/lib/libvfhe_<engine>.a`, with `engine-<engine>.json`
  recording the matching flags) instead of rebuilding the whole library from
  source. Runtime
  compiles drop from the full kernel set to the user's snippet, and the
  installed package stops carrying C sources and build machinery.
- The native build is Meson (`meson-python` backend): each module lists
  its C in `modules/*/meson.build`, the root `meson.build` assembles each
  engine's extension and archive with per-file incremental rebuilds, and
  engine facts still come from `tools/_engines.py` — meson queries it,
  so the two cannot drift. `setup.py` and `MANIFEST.in` are gone; sdists
  are cut by `meson dist` from committed state, vendoring BLAKE3 and a
  frozen version.
- All repo tooling lives in one folder, `tools/`, with one command per make
  target (so the menu and the tree agree), the parts they share as `_noun`
  files beside them, and the commands *meson* runs in `tools/meson/`. It is
  small: three parts (`_common`, `_engines`, `_sde`) plus those commands,
  replacing the former `packaging/discovery.py` + `scripts/` split. One
  rule: a verb file is a command you run, a `_noun` file is a part you
  import.
- `python -m vfhe.info` prints an install's version, the engine it selected,
  anything faster the CPU could have run, and the platform — one command for a
  bug report, which the report form now asks for instead of four fields and a
  dropdown of engine names that would go stale as engines are added.
- The pre-commit hooks are the single definition of every static check, and
  `make lint` — the command CI's lint job runs — executes them over the whole
  tree, so a hook, a local lint and CI cannot disagree. Beyond the formatters
  they add `actionlint`, `shellcheck`, `codespell`, `markdownlint`, an SPDX
  REUSE licensing check, and JSON-schema validation of the workflow, Dependabot,
  issue-form and citation files — 22 checks where CI previously enforced 5. The
  formatter hooks check instead of rewriting (a commit no longer edits your
  files behind you); `make format` is what fixes. Lint runs on documentation
  changes too, so a docs-only pull request is no longer merged unchecked.
- The project is [REUSE](https://reuse.software) 3.3 compliant: all 192 files
  state their copyright and licence, the licence texts live in `LICENSES/`, and
  `reuse lint` is a hook, so a new file cannot arrive without saying who owns it
  and under what terms. Files that cannot carry a comment — `.gitmodules`,
  `.python-version`, JSON data, `py.typed` — are annotated in `REUSE.toml`.
- `make` is the front end of the whole lifecycle, one target per stage
  (`build`, `test`, `sdist`, `install`, `smoke`, `sbom`, `release-notes`), with meson and `tools/` behind it — so every step CI takes
  runs locally by the same command, and the workflows keep only their glue:
  checkout, setup, artifact upload, and the few lines that speak GitHub's own
  protocol. `.github/scripts/` is gone. `tools/` is grouped by subject
  (`test/`, `coverage/`, `sbom/`, `release/`, `meson/`) with the shared parts
  at its top level.
- The dependency groups split into `dev` and `release` (cyclonedx, twine,
  build), with `dev` including `release`: a release job installs only what it
  needs, and a developer installs one group to do everything.
- Engines are a registry (`tools/_engines.py`): one entry per instruction-set
  level, naming the architecture it builds on, the CPU capability it needs,
  its ISA flags, its extra sources, and how to emulate it. Everything derives
  from that list — the archives, extensions, `engine-<name>.json`, the
  picker's table, meson's per-engine test suites — so adding
  an engine is an entry plus its kernels, with no build, test, or CI rewiring.
- Every x86_64 build now compiles the SIMD kernels (a host builds every engine
  its architecture allows), so the separate cross-compile check of those
  kernels — and the CI job that existed to run it — are gone: any x86_64 leg
  catches a broken SIMD kernel.
- The fuzzing container builds its kernels with meson too (`-Dfuzz=true`
  configures the instrumented archives and nothing Python, so no Python
  headers or protobuf toolchain are needed there), leaving its build script
  to do only what needs `$LIB_FUZZING_ENGINE`: compile and link the
  harnesses. Fuzzed code can no longer drift from built code.
- The C unit tests are meson tests: `meson test` runs them in parallel with
  timeouts, sanitizers come from meson's `-Db_sanitize`, and the emulated
  emulated leg is `meson test --wrapper`, so the hand-rolled C harness (and
  its own sanitizer and wrapper plumbing) is gone.
- The SIMD engine is now executed, not just compile-checked, and testing is
  one matrix with one entry point: `tools/test/run.py <engine> <suites>`,
  where suites are `c`, `fast`, and `complete`. `make test` covers every
  engine this CPU runs; `make test ENGINE=<name>` insists on one —
  natively where the CPU has its ISA, otherwise under the emulator the
  registry names for it (for `avx512ifma`, the Intel Software Development
  Emulator on x86_64 Linux). CI exercises every engine on every pull request
  and fills every suite x engine cell across the gate: the fast suite on pull
  requests, the complete suite on pushes to `main`.
- The `avx512ifma` engine gates every merge under Intel SDE — deterministic and
  hosted, so no infrastructure of ours can block development. Pull requests
  gate on the fast suite, pushes to `main` on the complete one.
- Coverage now measures every engine — one parallel leg per engine, each
  building its own instrumented binary — and unions
  them, so the SIMD kernels are counted rather than compiled out of the
  measurement: on every pull request, the portable engine on the complete
  suite, `avx512ifma` on the fast suite under Intel SDE (hosted and
  deterministic, so no report depends on infrastructure of ours), and a
  pre-wired arm64 leg that starts measuring the moment arm64 kernels
  land. Informational: coverage gates nothing.
- The project name is styled **vFHE** in prose; identifiers, the package, and
  build output stay `vfhe`.
- Security reports go to <security@vfhe.ai> and conduct reports to
  <conduct@vfhe.ai>; GitHub's private reporting form
  is off until the first stable release, and `SECURITY.md` says what to expect
  from a pre-release, unaudited library in the meantime.
- CI builds the engine it tests explicitly (`VFHE_ENGINE=portable`, so a
  capable runner cannot silently switch engines) and on one more platform: arm64 Linux
  (`ubuntu-24.04-arm`). The pre-wired arm64 test rail no-ops until an
  arm64 engine lands (`tools/_engines.py`), then runs natively on both arm64
  runners.
- CI runs in standard stage families — prepare (classification), checks,
  test, package — parallel rather than barriered, so code changes
  get every signal at once and docs-only pull requests run only the
  classifier and the secret scan.
- Static analysis runs one leg per engine (`code-scanning.yml`, CodeQL
  today), because an analyser only sees the ISA variant its build compiled:
  the AVX-512 kernels are analysed instead of preprocessed away. The analyser
  is named nowhere else, so it can be replaced without touching a caller.
- Docs-only pull requests skip the build/test pyramid: an in-repo `changes`
  job classifies the diff, and the required `CI OK` check accepts exactly those
  skips — any other skip, a cancelled run, or a job that vanished from the
  workflow fails it.
- Deadlocked maintainer decisions escalate to the VERIFHE project lead
  (`GOVERNANCE.md`); the status quo stands pending the ruling.
- One naming convention across `.github/`: kebab-case filenames (Python
  keeps snake_case) matching noun display names (Hardware Test, Static Analysis, Scorecard Analysis, Zulip
  Alert), verb names for composite actions (`setup-dev`, `setup-sde`), terse
  verb-phrase step names.
- Scheduled workflows run at off-hour minutes. Zulip alerting is a shared
  `workflow_call` job appended to every workflow with non-PR triggers — no
  `workflow_run` fan-in, no display-name coupling — and ignores pull-request
  runs (red checks are already visible on the PR).
- The issue templates ask only what routes a report: the feature form is a
  problem statement plus an optional sketch, and the bug form's engine
  question lists the engine names.
- The ruff rule set widened from the former default (E4/E7/E9, F, I) to add
  bugbear, comprehensions, perf, pie, ruff-specific, bandit, simplify, and
  pyupgrade groups; the codebase now conforms (parallel-array `zip`s are
  `strict=True`, `assert False` branches raise `NotImplementedError`).

- Installs are PEP 561-typed (`py.typed` in every subpackage), the sdist no
  longer sweeps in git/CI-only files, and `NOTICE` records the OpenFHE
  (BSD 2-Clause) adaptation and the modular-inverse routine whose upstream
  licensing is under review.
- Pushes to `main` now record what a pull request cannot afford: `avx512ifma`
  natively on the project's own machine (the cross-check of Intel SDE) and the C
  suites under ASan+UBSan, in `hardware-tests.yml` — one job of ordered steps,
  so a failure in one shape does not hide the rest, even though the single
  runner executes them in turn. It replaces the nightly hardware lane, and runs per
  commit. Every
  main commit also builds the entire release artifact set (sdist, the wheel
  matrix, the SBOM) at the version setuptools-scm derives from git, which nothing publishes, so
  "main is releasable" is checked rather than assumed.
- A release proves its provenance instead of only generating it: the bundle is
  verified offline before the upload, and afterwards every file the index serves
  is downloaded and `gh attestation verify` runs over each, against the one
  workflow allowed to have signed it. Neither release target will publish a
  commit `main` did not prove green.
- CI is two workflows with one job list: `ci-presubmit.yml` gates pull
  requests at fast depth, `ci-postsubmit.yml` records every `main` commit at
  complete depth and alerts when it goes red. Packaging rides along in both —
  the sdist is built and its install smoke-tested per commit, and `main` also
  builds the release's whole wheel matrix as a dev artifact nothing publishes —
  so a packaging break surfaces where it was introduced, not on release day.
- The release is two dispatch workflows, each owning its target's form,
  validation, credentials, and publish job (Release TestPyPI with a required
  version input; Release PyPI run from the tag, no inputs); both call the
  shared artifact pipeline, and assert only what a tag can get wrong — the
  commit is contained in `origin/main`, and the version has the shape its
  index accepts. The sdist is built by the gate-pinned backend
  without a pip cache, and dependency review enforces the
  Apache-2.0-compatible license allow-list `SECURITY.md` promises.
- The release SBOM records the vendored BLAKE3 — version and pinned
  submodule commit, read from the tree, nothing hand-maintained — alongside
  the Python dependency closure: a Python-environment scan cannot see C
  compiled into the extension, so scanners fed the SBOM would never have
  flagged a BLAKE3 advisory.

### Fixed

- A missing-submodule clone now fails fast with the remediation command on
  the dev path too (`make build`, C tests), not just `pip install`; C
  compiler warnings are no longer swallowed on successful test builds.
- `vfhe.misc.dynamic_extensions`: user-supplied extra compiler/linker flags
  extended rather than replaced the build plan's; reinitialization failures
  propagate instead of degrading to warnings; repeated compiles no longer
  accumulate `sys.path` entries; the cache honours `XDG_CACHE_HOME`; the
  build helper is loaded under a private name instead of a generic
  `import discovery`.
- Static analysis could silently pick up AVX-512 IFMA on a hosted runner,
  tracing that preprocessor variant and the portable one never; it now runs a
  leg per engine, each building that engine's kernels alone
  (`make build KERNELS=<engine>`), so every variant is traced and no finding is
  attributed to the wrong one.
- The coverage summary would have dropped the SIMD engine's row: the renderer
  filtered rows against a hardcoded leg list. It now renders whatever legs
  reported, so a new engine needs no renderer change.
- The required `CI OK` check could be satisfied by cancelling a run (a skipped
  aggregation job passes branch protection); it now runs under `always()` and
  fails unless every gated job succeeded, counting a skip as a failure except
  where the docs-only classification caused it.
- The coverage exclusion for the runtime-recompilation helpers sat under the
  wrong `coverage.py` section and was silently ignored.
- An async multilinear-polynomial evaluation could be garbage-collected
  mid-flight: the event loop holds tasks only weakly and the spawned task was
  never referenced. Background tasks are now kept alive until done.
- `vfhe.misc.dynamic_extensions` did not work from an installed package: the
  C sources it recompiles were never installed, and clean Python >= 3.12
  environments lack the setuptools cffi compiles through. Installs now ship
  the build inputs as `vfhe/_source`, setuptools is a runtime dependency, and
  a smoke test exercises the installed-package path in CI.
- The arith C tests fed plain `malloc` buffers to SIMD kernels that require
  64-byte alignment (`safe_aligned_malloc`); crashed the SIMD engine, found
  by its first emulated run.

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
- Minimum requirements raised: `cffi` 2.1 (runtime) and `setuptools-scm` 10.2
  (build).

### Fixed

- `ntt_new_proc` could loop forever searching for a primitive root of unity with
  certain prime and ring-size combinations; the search is now deterministic and
  always terminates.
- A module compiled at runtime by `vfhe.misc.dynamic_extensions` auto-tuned
  independently of the loaded engine, so a portable process could load AVX-512
  kernels and crash; the custom build now inherits the loaded engine's mode.

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
