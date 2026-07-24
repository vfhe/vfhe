<!-- SPDX-License-Identifier: Apache-2.0 -->

# Developing VFHE

The contributor guide: repository layout, build system, testing, coverage,
and CI. Contribution expectations are in
[CONTRIBUTING.md](https://github.com/vfhe/vfhe/blob/main/CONTRIBUTING.md).

## Repository layout

```
modules/<mod>/                     one folder per module
  python/src/vfhe/<mod>/           Python package source (physical namespace layout)
    __init__.py                    thin: re-exports the module's public API
    <impl>.py                      implementation; talks to C via `from vfhe.misc.libvfhe import ffi, lib`
  python/cdef/<mod>.cdef           Python-facing ABI: hand-written cffi decls (opaque handles)
  python/test/                     pytest suite (public-API characterization tests)
  c/include/                       public C headers (umbrella <mod>.h)
  c/src/                           pure C kernels
  c/test/                          C unit tests (plain assert/main() or Unity)
  c/fuzz/                          optional libFuzzer harnesses (fuzzed in CI)
  proto/                           optional protobuf schema for this module

packaging/build_ffi.py             cffi build: all c/src kernels -> one ext `_vfhe_native`
packaging/discovery.py             shared source/include/arch discovery (build + tests)
packaging/generate_protos.py       protoc: all proto schemas -> `_vfhe_proto` package
packaging/typings/                 typing boundary for the cffi ext (Any)
scripts/_common.py                 shared script plumbing: repo root, discovery import, logging, OS differences
scripts/run_c_tests.py             compiles + runs the C tests (VFHE_SANITIZE=...)
scripts/run_c_fuzz_tests_local.py  builds + briefly fuzzes the c/fuzz harnesses locally (make fuzz-local)
scripts/check_simd_build.py        compile-checks the x86 AVX-512 engine paths
scripts/run_tuned_tests.py         tests this arch's optimised engine (x86-64: native IFMA or Intel SDE)
scripts/check_install.py           installs the sdist into a scratch venv, proves it compiles
scripts/run_smoke_tests.py         runs smoke/*.py with a given interpreter
smoke/ckks.py                      smoke test: self-verifying end-to-end CKKS computation
.github/workflows/                 CI, coverage, fuzzing, release, and security workflows
.clusterfuzzlite/                  ClusterFuzzLite build (Dockerfile, build.sh, build_c_fuzz_tests_ci.py)
docs/DEVELOPMENT.md                this guide
external/unity/                    Unity C test framework (git submodule, test-only)
external/blake3/                   BLAKE3 hash (git submodule; c/ compiled into the ext)
setup.py / pyproject.toml          package assembly + build config
pyrightconfig.json                 editor import paths (Pyright / Pylance)
.generated/                        all build artifacts (gitignored)
```

### The native boundary

Every module's `c/src` is compiled into **one** LTO'd extension,
`_vfhe_native`, so kernels inline across module boundaries. Each Python-facing
module hand-writes `python/cdef/<mod>.cdef` declaring the C ABI it calls;
handles are passed as opaque `void *`, with a few structs cdef'd so Python can
read fields. The Python wrappers reach C through `from vfhe.misc.libvfhe import
ffi, lib`.

### Generated files

Everything generated goes into `.generated/`, which is gitignored in full:

| Artifact | Produced by | Purpose |
|---|---|---|
| `_vfhe_native.*.so` | `packaging/build_ffi.py` (cffi) | the compiled C kernels, one LTO'd extension |
| `_vfhe_proto` | `packaging/generate_protos.py` (protoc) | protobuf bindings |

---

## Prerequisites (development)

- Python **3.10+**
- `pip` **25.1+** (for `pip install --group`, PEP 735 dependency groups)
- A C compiler (clang/gcc; Xcode Command Line Tools on macOS)
- `git` (Unity and BLAKE3 are vendored as submodules; clone with `--recursive`)

---

## 1. One-time setup

```bash
git clone --recursive https://github.com/vfhe/vfhe vfhe && cd vfhe
# already cloned without --recursive? fetch the submodules (Unity + BLAKE3):
git submodule update --init --recursive

python3 -m venv .venv && source .venv/bin/activate
make deps                                        # dev dependencies + git hooks
```

`make deps` installs the **`dev` dependency group from `pyproject.toml`** (PEP
735, **not** the package) and the `pre-commit` hooks: ruff and clang-format run
on the files you touch, so CI never fails you for formatting alone. The hooks
call the versions pinned in that same group (via `language: system`), so a hook
and CI cannot disagree, but they need the venv active. Run them over everything
with `pre-commit run --all-files`.

> **Submodules are required to build.** BLAKE3's C sources are compiled into the
> native extension, so a non-recursive clone will fail fast with a clear message.
> Released sdists bundle BLAKE3's sources, so `pip install vfhe` needs no submodules.

> `.venv/` is not managed by the tooling; activate it for every session. All commands below
> assume an activated environment (or pass `make PYTHON=/path/to/python`).

---

## 2. Development loop

```bash
make build      # (re)generate proto bindings + compile the C kernels into .generated/
make test       # build, then run the C and Python (pytest) suites
make format     # format all Python + C (see "Formatting" below)
```

### Editor setup

Point your editor at the venv interpreter (the one `make deps` installed into).
Import paths are configured IDE-agnostically in `pyrightconfig.json` (read by
Pylance, the `pyright` CLI, and any Pyright-based LSP): `vfhe.*` resolves from
`modules/*/python/src`, and the generated `_vfhe_native` / `_vfhe_proto` from
`.generated/`. Run `make build` once so `.generated/` exists, then reload the
editor.

### Testing

`make test` runs the complete suite; `make test-fast` runs the fast subset:

- **Python**: `pytest` has two modes. The default is **fast** (quick tests).
  Heavy end-to-end FHE bootstraps are marked `@pytest.mark.complete` and run only
  with **`pytest --complete`**. CI runs the complete suite, which includes the
  fast subset.
- **C**: `scripts/run_c_tests.py` compiles every `modules/*/c/test/*.c` against
  the portable engine and runs it (a non-zero exit fails the suite). Tests are
  Unity suites; the harness links Unity only for files that include it, so a
  plain `assert`/`main()` program works too.
- **Smoke**: `smoke/` holds standalone, self-verifying programs; today
  `smoke/ckks.py` (encrypt two vectors, add and multiply under encryption,
  decrypt, and check the results against plaintext).
  `make smoke` (and the CI/release sdist jobs) composes two scripts:
  `scripts/check_install.py` recreates a scratch venv at `.cache/install/venv`
  and pip-installs the sdist (proves it compiles), then
  `scripts/run_smoke_tests.py --python <venv python>` runs every `smoke/*.py`
  against it (proves it computes). Your environment is never touched;
  `make clean` removes the venv. A new smoke test is just a new file in
  `smoke/`. The CKKS one also runs against the source build as a pytest
  (`modules/fhe/python/test/test_smoke.py`).

**Coverage.** `coverage.yml` measures *both* languages in one run, from a
gcov-instrumented build (`VFHE_COVERAGE=1 make build`) exercised by the
**complete** suite, so the FHE-bootstrap C kernels are counted. Python is
measured by [coverage.py](https://coverage.readthedocs.io) (via `pytest --cov`)
and C by `gcovr` reading the resulting `.gcda`. Both emit JSON, which
`.github/render_coverage.py` fills into `.github/coverage-comment.md`; the
result goes to the run's job summary, to one sticky pull-request comment
(updated in place, not stacked), and to the `coverage` artifact, which carries
one human-readable (HTML, per-line annotated source) and one machine-readable
(JSON) report per language.

**Nothing blocks a merge on coverage.** It is a number a reviewer weighs, not a
rule. GitHub supports thresholds as branch ruleset rules (minimum coverage, and
maximum drop against the default branch) if that changes; a repository setting,
not a code change.

When code genuinely cannot be tested (unreachable defensive branches, OOM paths,
platform-specific fallbacks), **exclude it explicitly rather than lowering the
floor**: the exclusion is reviewable, a lower floor is not:

```python
if impossible_state:  # pragma: no cover
    raise AssertionError("unreachable")
```

```c
if (rc != 0) /* GCOVR_EXCL_LINE */
    abort();
/* GCOVR_EXCL_START ... GCOVR_EXCL_STOP  for a whole region */
```

Coverage settings live in `pyproject.toml` under `[tool.coverage.*]`, so a local
`pytest --cov=vfhe` measures exactly what CI measures.

Extra, deeper checks:

```bash
VFHE_SANITIZE=address,undefined python scripts/run_c_tests.py   # ASan + UBSan
make fuzz-local                                                 # fuzz each c/fuzz harness for 60s (needs a clang with libFuzzer)
make smoke                                                      # sdist -> scratch venv install -> smoke tests
python scripts/check_simd_build.py                              # compile-check the SIMD engine paths (--require to fail, not skip)
make test-tuned                                                 # C tests + fast suite on this arch's optimised engine (x86-64: IFMA or SDE)
```

**The optimised engines.** The library has one engine per architecture rail:
the portable baseline (every platform), the x86 AVX-512 tuned engine, and, in
the future, an arm-tuned one. Each rail is tested deterministically: the
`test-c`/`test-python` matrix pins the baseline (`VFHE_PORTABLE=1`, so a
runner that happens to have IFMA cannot silently switch engines), and
`make test-tuned` (`scripts/run_tuned_tests.py`) runs *this* architecture's
optimised engine — on any other architecture it errors. On x86-64 it builds
with explicit ISA flags (`VFHE_TUNED=1`) and runs natively when the CPU has
real AVX-512 IFMA; otherwise it runs under the
[Intel Software Development Emulator](https://www.intel.com/content/www/us/en/download/684897/intel-software-development-emulator.html)
posing as an Ice Lake CPU (`sde64 -icl`), downloaded (sha256-pinned) into
`.cache/sde/` on first use — **Linux only**, since SDE instruments real x86
processes and cannot itself run under Rosetta or QEMU. `--suite
c|fast|complete` picks the depth; emulation costs 10-50x, so budget minutes
for `c`, tens of minutes for `fast`, hours for `complete`. CI runs the x86
rail on every PR under SDE (`--emulate`, one deterministic path across the
mixed runner fleet): `test-sde` for the `c` tier, `test-sde-python` for
`fast` and `complete`. On arm64 no tuned kernels exist yet: the pre-wired
`test-tuned-arm` jobs (`--if-supported`) no-op until
`discovery.TUNED_ARCHES` declares arm64, then run the suites natively on
`macos-latest`/`ubuntu-24.04-arm` — NEON is baseline arm64, no emulator.

libFuzzer harnesses live in `modules/<mod>/c/fuzz/`. Two ways to run them:
`make fuzz-local` (`scripts/run_c_fuzz_tests_local.py`) builds each with local
clang and fuzzes it briefly; CI runs ClusterFuzzLite, whose build lives
entirely in `.clusterfuzzlite/` (changed code for 5 min in ci.yml, the full
set for 1h nightly in `cflite-batch.yml`). Local fuzzing needs a clang that
ships the libFuzzer runtime; Apple's does not, so on macOS `brew install llvm`
and set `CC` to its clang.

---

## 3. Code & library generation

Nothing generated is committed; regenerate locally as needed:

| Run | When |
|---|---|
| `make proto` | after editing a `.proto` |
| `make build` | after editing C, or to refresh everything (`proto` + native) |

`setup.py` reruns proto generation automatically during a build, so a clean
checkout (or an sdist install) needs nothing pre-generated.

---

## 4. Formatting & commits

**`make format`** formats everything in place: `ruff` for Python (format plus
auto-fixable lint, imports included) and `clang-format` for C (style in `.clang-format`). Run it before
committing.

**Commit messages** follow the 50/72 convention (subject <= 50 chars, body wrapped
at 72) as a *recommendation, not enforced*. Opt into the editor guide once:

```bash
git config commit.template .gitmessage
```

---

## 5. Building the distribution

VFHE releases **as an sdist only**, with no wheels, so every install compiles from
source and tunes to the target CPU:

```bash
make sdist        # -> dist/vfhe-<version>.tar.gz
```

The sdist bundles the `vfhe.*` packages, all C/cdef sources, `packaging/build_ffi.py`,
and the vendored BLAKE3 C sources (so no submodule is needed to build it). Proto
bindings are regenerated during the build. Publishing is handled by the release
workflow in section 6.

### Dependencies

How dependencies are selected, obtained, and tracked:

- **Selection**: the runtime surface is deliberately small (`cffi`,
  `protobuf`, `mpmath`); anything else must earn its place in review. Version
  floors state the oldest supported release and are only raised when a newer
  feature is required.
- **Obtaining**: Python dependencies are declared in `pyproject.toml`
  (runtime, build, and PEP 735 dev/sbom groups) and resolved from PyPI by pip
  at build time. Native third-party code (BLAKE3, Unity) is vendored as
  git submodules pinned to exact commits; provenance is recorded in `NOTICE`.
- **Tracking**: Dependabot watches all ecosystems weekly (GitHub Actions by
  commit SHA, pip, the fuzzing container by image digest);
  `dependency-review.yml` blocks pull requests that introduce known-vulnerable
  dependencies, and each release ships a CycloneDX SBOM of the resolved
  runtime environment.

---

## 6. Continuous integration & releases

Workflows live in `.github/workflows/` (POSIX only: Linux and macOS):

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | push / PR | the merge gate, eight parallel tracks: lint (ruff, pyright, clang-format); the SIMD compile check (which no other job compiles) with both toolchains, gcc on Linux and Apple clang cross-compiling on macOS; the C tests then the fast and complete Python suites on the tuned AVX-512 engine under Intel SDE (Linux; SDE has no macOS build); the pre-wired arm-tuned jobs (no-ops until `discovery.TUNED_ARCHES` declares arm64); ClusterFuzzLite fuzzing of the changed code (5 min); sdist build + clean-venv install + smoke tests; and C tests followed by the complete Python suite, baseline engine pinned, on Linux (x86-64 and arm64) and macOS x py3.10-3.14 |
| `coverage.yml` | push to `main` / PR / manual | C and Python coverage from one gcov-instrumented run; job summary + PR comment; informational, blocks nothing |
| `cflite-batch.yml` | nightly / manual | ClusterFuzzLite batch fuzzing of the `c/fuzz/*` harnesses (1h, managed corpus) |
| `codeql.yml`, `scorecard.yml`, `dependency-review.yml` | push / PR / schedule | static analysis and supply-chain hygiene (incl. OpenSSF Scorecard) |
| `notify-zulip.yml` | any workflow fails | posts the failed run's link to the project Zulip (`verifhe-ci` > `failures`) |
| `release.yml` | see below | build the sdist and SBOM, then publish |

**Releases** are manual, one workflow, two targets (Actions > Release > Run
workflow):

- **TestPyPI**: pick `target: testpypi` and give an RC `version` (e.g.
  `0.0.2rc1`). Run it from any branch.
- **PyPI**: push a bare-semver tag (e.g. `0.1.0`, *no* leading `v`), then run
  the workflow **from that tag** with `target: pypi`. The version comes from the
  tag, and a GitHub Release with the sdist, SBOM, and build provenance is
  created alongside.

Both use **Trusted Publishing** (OIDC, no API tokens) with PEP 740 attestations.
One-time setup before the first release: create the `vfhe` project + a Trusted
Publisher for this repo/workflow on **both** TestPyPI and PyPI, and add GitHub
Environments named `testpypi` and `pypi` (protect `pypi` with required reviewers
so the manual publish still needs an approval).

---

## Adding a new module

A module can be **Python-facing** (has `python/` + a cdef exposing C to Python)
or **internal C-only** (no `python/`; used by other modules' kernels).

1. (Python, optional) create `modules/<mod>/python/src/vfhe/<mod>/__init__.py`
   (+ impl `.py`). Omit for internal C-only modules.
2. (C, optional) add `c/src/*.c` + `c/include/*.h`. If the module exposes any C
   to Python, hand-write `python/cdef/<mod>.cdef` declaring that ABI (opaque
   handles + the functions Python calls). Internal C-only modules have **no**
   cdef; they contribute compiled code but no Python symbols.
3. (C tests) add `c/test/test_<mod>.c`, a plain `assert`/`main()` program
   (non-zero exit = failure) or a Unity suite.
4. (Proto, optional) add `proto/vfhe/<mod>/<name>/v1/<name>.proto` with `package
   VFHE.<mod>.<name>.v1` (buf convention: the file's path must equal its package
   path). Each `proto/` dir has a `buf.yaml` as the buf module root. Bindings are
   generated to `_vfhe_proto.vfhe.<mod>.<name>.v1.<name>_pb2`.

`setup.py`, `packaging/build_ffi.py`, `packaging/generate_protos.py`, and `scripts/run_c_tests.py`
all auto-discover modules (recursively). Python-facing modules list their import
paths in `pyrightconfig.json` (editor); tests resolve via `modules/conftest.py`,
which globs the module src dirs; no per-module edits.

### Architecture-specific sources & SIMD

`build_ffi.py` picks the engine by build context:

- **Source builds** (the default: pip compiling the sdist on the target
  machine) **auto-tune to that CPU**: on x86 with AVX-512 IFMA they drop
  `PORTABLE_BUILD` to light up the arith IFMA engine + AES-NI in `misc`, add
  BLAKE3's SIMD kernels, and pass `-march=native`; otherwise they build the
  portable engine. IFMA is detected by asking the compiler what `-march=native`
  enables (`__AVX512IFMA__`), so the gate matches exactly what the kernels are
  guarded on.
- **Forced portable** (`VFHE_PORTABLE=1`) compiles the baseline (`PORTABLE_BUILD`,
  scalar paths, BLAKE3 portable core) and skips detection, for building in one
  place to run on another (Docker/build farms). `VFHE_COVERAGE=1` also implies it.
- **Forced tuned** (`VFHE_TUNED=1`) compiles the AVX-512 engine with explicit
  ISA flags instead of `-march=native`, so it builds on x86-64 hosts without
  IFMA. The result runs only on IFMA hardware or under an emulator
  (`make test-tuned`); it is a test knob, not an install option.

`c/src` is scanned **recursively** and compiles both C and hand-written assembly
(`.S`), so vendored trees drop straight in; sources under `c/src/arch/<name>/`
build **only** on a matching machine (`x86_64` aliases: `x86_64/amd64/x86-64/x64`;
`arm64` aliases: `arm64/aarch64`). The engine selects SIMD via compile-time
macros in the sources (not `arch/` dirs); the `arch/` mechanism remains for any
future hand-written per-arch kernels.

Because a source build is `-march=native`, it targets **this** machine; never
copy one to a different CPU. Set `VFHE_PORTABLE=1` if you must build somewhere
other than where it will run.

---
