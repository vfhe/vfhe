<!-- SPDX-FileCopyrightText: 2026 Alin-Petru Roșu -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Developing vFHE

The contributor guide: repository layout, build system, testing, coverage,
and CI. Contribution expectations are in
[CONTRIBUTING.md](https://github.com/vfhe/vfhe/blob/main/CONTRIBUTING.md).

## Repository layout

```text
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

meson.build                        the native build: every module's C -> one extension + archive PER ENGINE, and every C test
modules/*/meson.build              each module's contribution: its sources, headers, cdefs, C tests, installs
modules/misc/python/src/_vfhe_native.py   the import-time engine picker (asks `_vfhe_cpu`, honours VFHE_ENGINE)
modules/misc/c/src/cpu_probe.c     the CPU probes; also built alone as the tiny `_vfhe_cpu` extension
tools/                             one command per make target, grouped by subject, plus the shared parts
tools/_common.py                   part: repo paths, logging, the host's engine capability
tools/_engines.py                  part: the engine registry — one entry per ISA level, as data
tools/test/run.py                  picks the engines this host can honestly test, then runs meson's suites
tools/test/run_smoke.py            smoke/*.py against the interpreter it runs under, each from a temp dir
tools/test/_sde.py                 part: Intel SDE download, verify, cache, wrap (pin: .sde.json)
tools/coverage/merge.py            unions the per-engine legs into one report
tools/coverage/render.py           fills summary.md.in — the table make prints and CI comments
tools/install.py                   a scratch venv holding one distribution — what `smoke` and `sbom` both read
tools/sbom/amend.py                adds the vendored C (BLAKE3) that a Python-environment scan cannot see
tools/release/extract_notes.py     the changelog section a release carries, printed before you tag
tools/meson/                       commands ONLY meson runs: engine_info, generate_ffi_c, generate_proto_bindings, print_version, prepare_sdist
tools/typings/                     typing boundary for the cffi ext (Any)
smoke/ckks.py                      smoke test: self-verifying end-to-end CKKS computation
smoke/info.py                      smoke test: `python -m vfhe.info` answers from an install
.github/workflows/                 CI, the extended suites, fuzzing, release, and security workflows
.clusterfuzzlite/                  fuzzing container: build.sh runs meson (-Dfuzz=true) for the kernels, then links the harnesses
DEVELOPMENT.md                     this guide
external/unity/                    Unity C test framework (git submodule, test-only)
external/blake3/                   BLAKE3 hash (git submodule; c/ compiled into the ext)
pyproject.toml / Makefile          package metadata + the one command menu
build/                             meson's tree: everything compiled (gitignored)
```

### The native boundary

Every module's `c/src` is compiled into **one** LTO'd extension,
`_vfhe_native`, so kernels inline across module boundaries. Each Python-facing
module hand-writes `python/cdef/<mod>.cdef` declaring the C ABI it calls;
handles are passed as opaque `void *`, with a few structs cdef'd so Python can
read fields. The Python wrappers reach C through `from vfhe.misc.libvfhe import
ffi, lib`.

### Generated files

One folder, gitignored: **`build/`** is meson's tree. Everything it produces
lives there — compiled extensions, static archives, and generated code — and the
dev loop, the tests, and pyright all read it straight from there.

| Artifact | Where | Produced by |
|---|---|---|
| `_vfhe_native_<engine>.*.so` | `build/` | meson: the compiled kernels, one LTO'd extension per engine |
| `_vfhe_cpu.*.so` | `build/` | meson: CPU probes only, so picking an engine never loads one |
| `libvfhe_<engine>.a` + `engine-<engine>.json` | `build/` | meson: each engine as a static archive, for runtime extensions to link |
| `_vfhe_proto` | `build/` | protoc: the protobuf bindings |

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
735, **not** the package) and the `pre-commit` hooks, which are where every
formatter and validator is *defined*: a commit checks the files it touches, and
`make lint` — the same command CI's lint job runs — checks the whole tree, so
neither can drift from the other. They check rather than rewrite, since a hook
that edits a file mid-commit hides what you are about to push; `make format` is
the one that fixes. The formatters come from that same dependency group (via
`language: system`), so they need the venv active.

> **Submodules are required to build.** BLAKE3's C sources are compiled into the
> native extension, so a non-recursive clone will fail fast with a clear message.
> Released sdists bundle BLAKE3's sources, so `pip install vfhe` needs no submodules.

> `.venv/` is not managed by the tooling; activate it for every session. All commands below
> assume an activated environment (or pass `make PYTHON=/path/to/python`).

---

## 2. Development loop

```bash
make build      # compile the C kernels and bindings into build/
make test       # build, then run the C and Python (pytest) suites
make format     # format all Python + C (see "Formatting" below)
```

### Editor setup

Point your editor at the venv interpreter (the one `make deps` installed into).
Import paths are configured IDE-agnostically in `[tool.pyright]` in `pyproject.toml` (read by
Pylance, the `pyright` CLI, and any Pyright-based LSP): `vfhe.*` resolves from
`modules/*/python/src`, and the extensions and protobuf bindings from `build/`.
Run `make build` once, then reload the editor.

### Testing

vFHE is a *kernel-owning* library, not a binding over someone else's crypto:
the C kernels are the product, so they are tested at their own level, in C
(that harness also carries the sanitizer legs, the fuzz builds, and the cheap
emulated leg), while the wrappers and schemes are tested in Python through
the real extension, and the installed package by the smoke suite. One layer
per owner — the pyca model of Python-only tests applies to projects whose C
is tested upstream, which is not this repo.

The test matrix has two orthogonal axes, and every entry point names both:

- **Suites** say *what* is tested: the **C unit tests** (`modules/*/c/test`)
  exercise the kernels directly; the **Python suite** exercises the wrappers
  and the engine through the extension (**fast** is the default subset,
  **complete** adds the heavy `@pytest.mark.complete` FHE bootstraps); the
  **smoke suite** (`smoke/*.py`, via `make smoke`) tests the *installed
  package* — sdist, clean-venv install, runtime recompilation — which is how
  the dynamic-extensions pip bug escaped before this suite existed.
- **Engines** say *which implementation* runs underneath: one per
  instruction-set level with kernels — **portable** (the baseline every
  platform gets) and **avx512ifma** today, more as kernels land
  (`tools/_engines.py` is the list). A run names the engines it skips here and
  why, so the answer that matters — what this machine can actually run — comes
  from running it.

**Test sources are engine-invariant.** One C API, one set of test files, N
implementations behind the ISA macros: the same C tests and the same pytest
suite run against every engine, so the engine is a *build parameter*, never a
test parameter. A test that genuinely applies to a single engine asks which
one is loaded (`vfhe_engine_active()` in C, `active_engine()` in Python)
rather than living in a separate suite. Comprehensive testing means
filling every suite x engine cell that exists for the platform — which is
exactly what CI's job list is.

`tools/test/run.py` is the one entry point, taking an engine and suites:

```bash
make test                          # C + complete suites, every engine this CPU runs
make test SUITES=c,fast            # the fast depth instead, same engines
make test ENGINE=avx512ifma        # one engine alone (emulated if this CPU lacks its ISA)
make install                       # DIST (default: a fresh sdist) into a scratch venv
make smoke                         # then smoke/*.py against that install
```

`all` (the default) runs every engine this CPU executes natively and skips
the rest, printing how to run them instead of surprising you with an
emulator download and a 10-50x-slower run; `make test ENGINE=<name>`
is that explicit step, and where the CPU lacks the ISA it starts the
emulator the registry names for that engine — for `avx512ifma`, Intel SDE
(sha256-pinned, `.cache/sde/`, x86_64 Linux) posing as an Ice Lake CPU.
Every suite underneath is **meson's**: each `modules/*/c/test/*.c` becomes
a test binary per engine, and each Python depth a pytest run per engine, so
one runner gives parallelism, timeouts, and per-cell selection (each test
joins the suites `<engine>`, `<depth>`, and `<engine>-<depth>`, since
meson's `--suite` flags union). Sanitizers come from `-Db_sanitize`
(`make build VFHE_SANITIZE=address,undefined`) and the emulated leg is
`meson test --wrapper`. `tools/test/run.py` is left with the one judgement
meson cannot make: which engines this machine can actually execute.

**A docs-only pull request is one whose every changed path is a `*.md`**, and
what keeps that safe is an invariant: **no file that any tool reads may be named
`*.md`.** A markdown file that feeds a program is a template, not documentation,
so it carries `.in` — hence `tools/coverage/summary.md.in`, the table CI
comments. Break that invariant and a behaviour change can ride in labelled as
documentation, skipping lint, analysis, and every test. Two knowing exceptions
sit at the root: `README.md` is the package's long description, so a rendering
break in it surfaces post-merge in `test-sdist`; `CHANGELOG.md` feeds
`make release-notes`, which fails loudly at release if a version has no
section.

**No `${{ }}` reaches a `run:` block.** Every expression — even a `matrix` value
that is a literal three lines up — travels through `env:` and is used quoted, so
nothing can be read as shell. `make lint` runs `zizmor --persona=pedantic`, which
enforces this rather than leaving "is this context attacker-controlled?" to be
judged afresh at each call site. Nothing in `.github/` carries a suppression,
and a finding counts as a defect until proven otherwise: the last one standing
was hiding a real gap, a `workflow_dispatch` trigger with no concurrency group.

**Workflow comments are one line each**, the one in the SPDX block included —
and that one says why the file exists, not what it does: a workflow should be
skimmable, so anything that needs a paragraph belongs in this file and the
workflow points at it. Displayed names and comments are prose
(**Python**, **C**); lowercase is reserved for verbatim tokens — job ids,
matrix keys, engine names, language identifiers, commands.

**Anything unordered is alphabetical**: the keys of `with`, `env`, `inputs`,
`outputs`, `permissions` and `matrix`, the entries of a matrix row, and the
lists where order carries nothing (`needs`, the `os` axis). Jobs are not
unordered — they read as the pipeline runs, grouped by the stage banners — and
neither are steps or `runs-on` labels.

In CI the same cells appear as jobs, under one doctrine: **required checks
run only on infrastructure anyone can reproduce** — hosted runners or a
deterministic emulator. The `test-c` and `test-python` actions are one leg each,
and the pipelines hold the matrices: portable is the rail on the three hosted
platforms; `avx512ifma` gates that engine under Intel SDE, always — the fast
suite on pull requests, the complete suite on pushes to `main`, where silicon
runs the same engine natively. The emulated leg stays on `main` even so: the
machine can be offline, and a rail `main` cannot see is a rail `main` does not
have. And the sdist is installed and smoke-tested the way a user installs it. An engine the registry does not declare has no rail: adding one means
adding its matrix rows, which is the same edit as declaring it.

**Coverage.** Every engine is measured, because measuring the portable engine
alone would leave the SIMD kernels uncounted — and nothing is executed twice to
get it. Each suite is measured by the leg that owns it: one C leg and one
Python leg per engine pass `coverage: 1` and run instrumented
(`VFHE_COVERAGE=1`) instead of the release shape (`-O3`, LTO) their siblings run,
which is what a wheel carries. Both halves come out of both legs — gcov reports
whatever executed the instrumented kernels, so the Python suite covers 59.4% of
C lines through the extension and the C tests add the ones no API reaches, 61.5%
together. A **leg** is therefore one measured run, `coverage-<engine>-<depth>`,
and `tools/coverage/merge.py` unions them — `coverage combine` and
`gcovr --add-tracefile` — per engine for its row and across engines for the
total, so a line counts as covered when *any* leg covered it. Depth follows the
caller, so a pull request's numbers are `fast`-depth and sit below main's
`complete`-depth ones — compare like with like. Python is measured by
[coverage.py](https://coverage.readthedocs.io) (via `pytest --cov`) and C by
`gcovr`; `tools/coverage/render.py` renders a row per engine plus the
combined figure, and every report ships both human-readable (HTML, per-line
annotated source) and machine-readable (JSON) forms. A measured run ends in
those same two tools locally, over whatever legs `.coverage/legs` holds, so the
table you get is the table a reviewer reads — measuring a second engine adds
its row.

**Coverage gates nothing.** It is a number a reviewer weighs, informational
by design.

Untestable code is excluded explicitly so reports stay honest — the
markers and the testing policy they belong to are in
[CONTRIBUTING.md](https://github.com/vfhe/vfhe/blob/main/CONTRIBUTING.md#about-coverage)
(`# pragma: no cover`, `GCOVR_EXCL_LINE`, and `GCOVR_EXCL_START/STOP` for
a region).

Coverage settings live in `pyproject.toml` under `[tool.coverage.*]`, so a local
`pytest --cov=vfhe` measures exactly what CI measures.

Extra, deeper checks:

```bash
make test SUITES=c VFHE_SANITIZE=address,undefined        # ASan + UBSan (meson's -Db_sanitize)
make test ENGINE=portable SUITES=c,fast VFHE_COVERAGE=1  # the same run, measured, ending in the table
make test ENGINE=avx512ifma                              # complete suite on one engine (native AVX-512 IFMA, else SDE)
make smoke DIST=vfhe==1.2.3                              # a published release instead of a local build
```

**The engines.** One engine per instruction-set level the kernels have a
branch for, each analysed and tested on a host of its architecture. The
ones marked *to do* are the work still ahead — this table is that list, and
each needs its matrix rows added alongside its registry entry.

| Engine | Hosts | State | What it buys |
|---|---|---|---|
| `portable` | x86_64, arm64, macOS | shipping | the scalar baseline every CPU runs |
| `avx512ifma` | x86_64 | shipping | 52-bit integer multiply-add (`madd52`), the widest path we have |
| `avx512f` | x86_64 | to do | AVX-512 F/DQ/VL without IFMA — most kernels need no `madd52` |
| `avxifma` | x86_64 | to do | the same IFMA at 256 bits (VEX), for CPUs without AVX-512 |
| `avx2` | x86_64 | to do | 256-bit integer SIMD, the widest baseline on old x86_64 |
| `neon` | arm64, macOS | to do | arm64's baseline SIMD (no 64-bit multiply, no IFMA) |

`portable` (every platform) and `avx512ifma` (x86_64) ship today. A
build carries every engine its architecture can compile; the picker takes
the best this CPU can run, and `VFHE_ENGINE=<name>` pins one. Each is tested
deterministically: the `test-c`/`test-python` matrix pins the baseline
(`VFHE_ENGINE=portable`, so a runner that happens to have IFMA cannot
silently switch engines), and `make test ENGINE=<name>`
(`tools/test/run.py <name> c,complete`) runs exactly that engine — on a
host that cannot build it, it says so. `avx512ifma` runs
natively when the CPU has real AVX-512 IFMA; otherwise under the
[Intel Software Development Emulator](https://www.intel.com/content/www/us/en/download/684897/intel-software-development-emulator.html)
posing as an Ice Lake CPU (`sde64 -icl`), downloaded (pinned in `.sde.json`
at the repo root: version, sha256, mirrors) into
`.cache/sde/` on first use — **Linux only**, since SDE instruments real x86
processes and cannot itself run under Rosetta or QEMU. The suites argument
(`c,fast,complete`) picks the depth. Emulation costs 10-50x per instruction,
but the suites are small enough that it barely shows: `complete` under SDE runs
in about ten minutes today. That number is the one thing here certain to age —
it grows with every suite added, so measure before assuming. CI runs
that engine under SDE (`--emulate`, one deterministic path across the mixed
runner fleet) in the `avx512ifma` job: the `fast` suite on pull requests, the
`complete` suite on pushes to `main`. On arm64 no SIMD engine
exists yet, so arm64 runs the portable engine only. When one lands, its rows
run natively on `macos-26`/`ubuntu-24.04-arm` — NEON is baseline arm64, no
emulator — and `make test IF_SUPPORTED=1` is there for a rail that must
tolerate a host which cannot build the engine yet.

Fuzzing follows the same rule, for the same reason a fuzzer must execute what it
builds: `batch-fuzzing.yml` holds one session per machine — hosted for portable,
the project's own for `avx512ifma`, an arm64 runner for `neon` when it exists —
and a dispatch input starts one without the others.

**Sanitizers follow the engine, not the platform.** ASan and UBSan instrument
the code that runs, and which C runs is the engine's choice, so each engine is
sanitized on a host that executes it *natively*: `portable` anywhere, `neon` on
the arm64 runners once it exists, `avx512ifma` only on the project's machine.
Never under an emulator: we sanitize the release build (`buildtype=release`
plus `-Db_sanitize`, so ~2-3x, not the 20x the folklore quotes for `-O0`), but
even 2-3x on top of SDE's 10-50x is not a test run. Sanitized legs run the
**fast** suite, hardware included — a sanitizer answers "does this path corrupt
memory", and the fast suite already walks every kernel and every public API;
sweeping more parameter sets is a numerical question, which the native complete
run answers on the same machine. `portable` is sanitized on both hosted
platforms — Linux instruments the interpreter too, macOS the C only — and the
gate runs the same two legs the push does, because the fuzzers already build
with ASan there, so the marginal cost is a leg behind a five-minute fuzz job.
`avx512ifma` is sanitized on silicon or not at all. A new engine adds its
sanitized leg wherever its ISA is native.

**Adding an engine** is one entry plus its kernels — no rewiring.
`tools/_engines.py` is the single switch: an entry names the architecture it
builds on, the capability its CPU must have (`vfhe_cpu_supports()`), the
explicit ISA flags (they define the macros the kernels guard on), any
arch-specific vendored sources, and how to emulate it (`None` when the ISA
is architecture-baseline, like NEON on arm64). Everything else derives:
meson grows that engine's archive, extension, `engine-<name>.json` and test
suites, and the picker's table grows a row. Two CI
edits remain deliberate rather than automatic: a leg in
`code-scanning.yml`'s matrix (an analyser must be told which host compiles
which variant) and, where the engine needs emulation, its own test rail. Arch-specific kernels live in
`arch/<arch>/` directories under `c/src/` (foreign-arch dirs are excluded
from the build) or behind the ISA macros in shared files.

**Analysis runs per engine too.** Compiled-out code is invisible to every
build-time tool, so executing the engines is not enough: CodeQL builds every engine on each platform, so every preprocessor variant is
analyzed; the C tests run under ASan+UBSan on hardware that executes the engine
natively, which today means `avx512ifma` on the self-hosted machine —
instrumentation under an emulator is impractically slow, and a portable-only
sanitizer leg would add little, since every engine compiles the same scalar C
apart from the `PORTABLE_BUILD` fallbacks. Fuzzing follows the same rule: a
fuzzer runs what it builds, so `.clusterfuzzlite/build.sh` targets whichever
engine the machine executing it supports — the portable one on hosted runners,
`avx512ifma` on the project's own machine, which fuzzes those kernels nightly in
its own batch session.

libFuzzer harnesses live in `modules/<mod>/c/fuzz/`; the build lives
entirely in `.clusterfuzzlite/`. Fuzzing runs only in CI, in two shapes: five
minutes against what the diff touched on every pull request (`code-change`
mode, in `ci-presubmit.yml`, where a crash the change introduced is worth more
than depth), and nightly sessions of an hour on the managed corpus
(`batch-fuzzing.yml`), whose value is CPU-time and continuity — which no
laptop and no merge gate can contribute. To reproduce or debug a CI finding locally,
use ClusterFuzzLite's own containers via the OSS-Fuzz helper (needs
docker):

```bash
git clone --depth 1 https://github.com/google/oss-fuzz .cache/oss-fuzz
python .cache/oss-fuzz/infra/helper.py build_fuzzers --external . --sanitizer address
python .cache/oss-fuzz/infra/helper.py run_fuzzer --external . <target> [reproducer-file]
```

---

## 3. Code & library generation

Nothing generated is committed; regenerate locally as needed:

| Run | When |
|---|---|
| `make build` | after editing C, or to refresh everything (`proto` + native) |

`setup.py` reruns proto generation automatically during a build, so a clean
checkout (or an sdist install) needs nothing pre-generated.

---

## 4. Formatting & commits

**`make format`** formats everything in place: `ruff` for Python (format plus
auto-fixable lint, imports included) and `clang-format` for C (style in
`.clang-format`). Run it before committing — the hooks report what it would
change, they do not change it for you.

Beyond the formatters, the hooks check what no compiler will: workflow
semantics (`actionlint` — job references, contexts, and the shell inside `run:`
blocks), shell scripts (`shellcheck`), spelling (`codespell`), markdown
structure (`markdownlint`, configured in `.markdownlint.yaml`), and the JSON
schemas of the workflow, Dependabot, issue-form and citation files.

**Licensing is machine-checked.** The project is
[REUSE](https://reuse.software) 3.3 compliant: every file states its copyright
and licence, the texts live in `LICENSES/`, and `reuse lint` runs as a hook. A
file says it in its own header; the few that cannot — formats without comments,
and the ones git and pyenv read verbatim — are annotated in `REUSE.toml`
instead.

**Commits must be signed off** (`git commit -s`): CI enforces the
[DCO](https://developercertificate.org) on every pull-request commit; see
[CONTRIBUTING.md](https://github.com/vfhe/vfhe/blob/main/CONTRIBUTING.md#sign-off-dco)
for what the sign-off asserts.

**Commit messages** follow the 50/72 convention (subject <= 50 chars, body wrapped
at 72) as a *recommendation, not enforced*. Opt into the editor guide once:

```bash
git config commit.template .gitmessage
```

---

## 5. Building the distribution

vFHE releases as wheels (Linux x86_64/arm64, macOS arm64) plus the sdist;
every install — prebuilt or built from the sdist — carries both engines and
picks at import. Sdists are cut by
`meson dist` from the **last commit** — commit before `make sdist` or
`make smoke`, or the archive will not match (or build from) your working
tree:

```bash
make sdist        # -> dist/vfhe-<version>.tar.gz
```

Wheels are never built by hand: the release builds the matrix with cibuildwheel
(`[tool.cibuildwheel]` in `pyproject.toml`), which cross-builds every wheel vFHE
ships and runs the fast suite inside each one.

The sdist bundles the git-tracked tree plus the vendored BLAKE3 C sources
and a frozen `.version` (so neither submodule nor git is needed to build
it). Proto
bindings are regenerated during the build. Publishing is handled by the release
workflow in section 6.

### Dependencies

How dependencies are selected, obtained, and tracked:

- **Selection**: the runtime surface is deliberately small (`cffi`,
  `protobuf`, `mpmath`); anything else must earn its place in review. Version
  floors state the oldest supported release and are only raised when a newer
  feature is required.
- **Obtaining**: Python dependencies are declared in `pyproject.toml`
  (runtime, build, and PEP 735 `dev`/`release` groups — `dev` includes
  `release`, so one install can do everything CI does) and resolved from PyPI by pip
  at build time. Native third-party code (BLAKE3, Unity) is vendored as
  git submodules pinned to exact commits; provenance is recorded in `NOTICE`.
- **Tracking**: Dependabot watches all ecosystems weekly (GitHub Actions by
  commit SHA — workflows and the composite actions alike — pip, the BLAKE3
  submodule, the fuzzing container by image digest). Two pins sit outside every
  ecosystem, because they are tarballs fetched by URL rather than packages:
  **gitleaks** (`.github/gitleaks.json`, read by the workflow that downloads it)
  and **Intel SDE** (`.sde.json`, read by `tools/test/_sde.py` and the CI action
  alike — hence the root, since a tool you run locally needs it). Nothing bumps them, and their digests prove integrity, not
  freshness — a stale secret scanner simply knows fewer patterns. Check them
  when touching either file;
  `ci-presubmit.yml`'s dependency review blocks pull requests that introduce known-vulnerable
  dependencies, and each release ships a CycloneDX SBOM of the resolved
  runtime environment plus the vendored native components
  (`tools/sbom/amend.py` — a Python-environment scan cannot see C
  compiled into the extension).

---

## 6. Continuous integration & releases

**Who tests what, where** — one direction, no cycles:

1. **Source** is tested by CI on every pull request (fast suites) and every
   push to `main` (complete suites), static analysis included. The pull request
   ends in one required check, `CI OK`, which asserts every gated job succeeded:
   it counts a skip as a failure unless the change classification caused it, and
   it runs under `always()`, so a cancelled or vanished job fails the gate rather
   than satisfying it. On `main` there is no aggregate — every job reports its own
   check run, since nothing gates a push.
2. **A release never re-tests source.** It asserts what a tag can get wrong —
   that its commit is contained in `origin/main`, and that the version has the
   shape its index accepts — and then that `main` actually proved that commit.
   What makes `main` tested is branch protection: a commit gets there through a
   green pull request, and its own push run records the complete status.
3. **Artifacts are tested at birth, inside the release**: `twine check` on
   the sdist, and every wheel runs the fast suite from its own install —
   covering the build environments CI's matrix never sees. Then once more
   after publishing, installed from the index a user installs from.
4. **Every main commit is proven packageable**: `ci-postsubmit.yml` builds the
   sdist, smoke-tests the install, and builds the release's whole wheel matrix as
   a dev artifact (setuptools-scm's git-derived version, e.g. `0.0.3.dev22+g036c735`, which nothing publishes) — each wheel running
   its own fast suite. A packaging break surfaces on the commit that caused it
   rather than on release day, and the only thing a release then adds is the act
   of publishing.
5. **Coverage** is informational (PRs only); **silicon** runs on every push to
   `main` that touches something other than documentation.

**A workflow is extracted for what it answers; a shape is an action.** Reading
the tree, producing the artifact and running the *installed* one are three
questions with three answers of their own — `static-checks.yml`,
`distribution-build.yml`, `smoke-tests.yml` — and each is worth a file because more than
one pipeline asks it and the answer is the same every time. **Running the
suites is not one of them.** The suites vary along four axes at once — engine,
platform, interpreter, and shape (depth, sanitizers, instrumentation) — and a
called workflow can only take those as inputs and rebuild the matrix inside
itself, which is how a pull request's seven legs and main's twenty came to be
one file pretending they were the same shape. So the unit is a step, not a
workflow: `test-c` and `test-python` are composite actions that run exactly one
leg, and **the pipeline holds the matrix** — presubmit spells out the legs a
gate can afford, postsubmit the grid it cannot, each visible where it is
decided. The one split by *runner* is `hardware-tests.yml`, and its reason is
contention rather than shape: that machine takes one job at a time, so anything
sharing a workflow with it either waits for it or makes it wait. Alone, it can
be push-only while `ci-postsubmit.yml` stays dispatchable — nobody can start a
run that blocks silicon, and fork code still never reaches it.

A job belongs to `ci-postsubmit.yml` for one of three reasons, and the file is
short because nothing else qualifies: it is **deeper** than a pull request can
afford (the complete suites, the sanitized legs the gate skips); it **only counts
on the default branch** (CodeQL's baseline is what a pull request's analysis is
diffed against, and Scorecard scores main); or it is the **last line for a commit
that never saw a pull request** (`lint`, `secret-scan`). Three presubmit jobs
cannot follow: dependency review and the coverage comment need a pull request to
exist, and `code-change` fuzzing needs a diff — nightly batch fuzzing covers main
instead. `main` has no gate, because a gate blocks a merge that already happened;
its mirror image is the alert, since nobody is watching a push.

A run covers the tip of a push, not each commit in it: rebase merge pushes a
pull request's commits together, so `main`'s record is one run on the last of
them. Bisecting can therefore stop at a commit CI never built — squash merge is
what would make commit and status one-to-one, at the cost of the linear history
rebase gives.

No workflow *inherits* another's status: no `workflow_run` fan-ins, no shared
state, nothing that passes because something else was green. The release is the
one workflow that reads a status it did not produce, and it reads it as a
precondition it fails on, never as work it skips. Rebase merge is what makes
step 2 sound — it mints new SHAs on `main`, so a pull request's status never
belongs to the commit that ships, and `main`'s own push run is the record (and
this is why the merge method must stay rebase).

Workflows come in three kinds, and the triggers say which:

- **Entry points** start from an event and compose the rest — `ci-presubmit`
  (pull request), `ci-postsubmit` and `hardware-tests` (push to main),
  `release-pypi` and
  `release-testpypi` (dispatch), `batch-fuzzing` and `scorecard-analysis`
  (schedule), `published-smoke` (nightly, or dispatched after a release).
  Nothing calls them.
- **Stages** are `workflow_call` and do one thing: `static-checks`,
  `code-scanning`, `distribution-build`, `smoke-tests`, `release-artifacts`,
  `alert-channel`. A stage adds `workflow_dispatch` only when running it alone
  is meaningful — a distribution build — and not otherwise, which is why
  `static-checks` and `smoke-tests` are call-only. `code-scanning` is the one
  hybrid: a stage that also runs weekly, because its analyser gains queries
  whether or not the code changed. The trigger block says which kind a file is;
  nothing in the filename does.
- **Composite actions** are shared steps inside a job: `setup-dev`, `setup-sde`,
  `scan-secrets`, `classify-changes`, `check-workflow-status`, `test-c`,
  `test-python`, `merge-coverage`, `verify-provenance`. A step that holds
  *logic* belongs in one whether or not it has a second caller: a workflow is
  meant to be skimmable, and shell that decides something is not. One caller is not enough to earn
  one — the indirection has to save something: a pinned version that would
  otherwise live in two files, or one leg of a matrix the caller owns. **An
  action owns its environment**: `test-c`, `test-python` and `merge-coverage`
  each run `setup-dev` themselves, and the test ones install the emulator when
  their `emulate` input asks for it, so a leg is one step and a caller owes it
  only a checkout — which is the one thing an action cannot do for itself, since
  a local `uses: ./…` is read from the workspace. Repeats are cheap by
  construction — `setup-dev` marks `$RUNNER_TEMP` (job-scoped, so it cannot go
  stale) and the second call through skips `make deps` — while a leg that
  silently inherited a half-built environment would cost a green run that proved
  nothing.

The self-hosted machine is a *resource*, not a kind: whichever workflow needs it
names its label — `hardware-tests` for the suites, `batch-fuzzing` for a
fuzzing session — and the runner executing one job at a time is what serialises
them. It gets no workflow of its own, or every activity on it would have to be
squeezed behind one trigger.

Workflows live in `.github/workflows/` (POSIX only: Linux and macOS):

| Workflow | Trigger | What it does |
|---|---|---|
| `ci-presubmit.yml` | PR | the merge gate, in stage families (parallel, not barriers): **prepare** the docs-only classification; **checks** `static-checks.yml` (the history scan and `make lint`), code scanning, DCO sign-off and dependency review; **test** the C legs, then the Python ones at `fast` depth, plus `portable` under ASan+UBSan on both hosted platforms, then the coverage comment their measured legs uploaded; **fuzz** five minutes on the changed code (portable engine); **package** `distribution-build.yml` for the sdist, then `smoke-tests.yml` installing it the way a user would. **gate** `ci-ok`, the one check branch protection requires — coverage is not among its dependencies, because it gates nothing. A docs-only pull request runs prepare plus the checks that are not about code |
| `ci-postsubmit.yml` | push to main / manual | the same jobs at their complete depth over the whole grid, nothing classified or skipped, the sanitized legs again (a gate's status belongs to a SHA rebase discards), and `release-artifacts.yml` for the whole artifact set — sdist, wheels, SBOM — under a dev version nothing publishes. No DCO (a pull request gate), no dependency review (the action needs a pull request), and no coverage comment (no pull request to post on) |
| `hardware-tests.yml` | push to main | everything the project's own machine can do and a hosted runner cannot, as one job of ordered steps: `avx512ifma` natively in the shipped shape (which is also what tests the SDE), that engine under ASan+UBSan at a speed emulation cannot reach, and `make smoke`, the only place an *installed* package selects the SIMD extension and runs it rather than merely shipping it. Its own workflow because the runner takes one job at a time: sharing a file with anything else would mean waiting for silicon or making silicon wait, and it has no dispatch so nobody can queue in front of a push. The one place `main` classifies anything: `paths-ignore: **.md`, because a machine with no second job to spare should not spend hours on a document |
| `published-smoke.yml` | nightly / manual | what an index actually serves, re-proven on a clock: `smoke-tests.yml` across the hosted platforms and `hardware-smoke-tests.yml` on the project's machine. Nightly because `pip` resolves a different dependency closure each run — the one input that changes while a published wheel does not — and because no release event can start a workflow: `GITHUB_TOKEN` suppresses the ones a workflow emits. Dispatch it with a version straight after a release |
| `hardware-smoke-tests.yml` | called by `published-smoke.yml` / manual | installs what an index serves on the project's own machine and runs `smoke/*.py` unpinned, so the picker selecting `avx512ifma` is itself under test — the only thing this machine can answer that a hosted leg cannot, which is why it holds nothing else. The one dispatch allowed to queue on that runner, because it takes minutes rather than hours |
| `batch-fuzzing.yml` | nightly | ClusterFuzzLite batch sessions over the `c/fuzz/*` harnesses, each followed by a corpus prune. One leg per engine, on a machine that executes it natively — portable on a hosted runner, `avx512ifma` on the project's own — with `max-parallel: 1`, because the managed corpus takes one writer. No dispatch: fuzzing is worth continuity rather than a session someone starts, and a crash is reproduced locally from the corpus the run leaves |
| `alert-channel.yml` | called by other workflows | the shared Zulip message. A workflow alerts when its failure would otherwise go unseen — a schedule or a push to main — but not when it was dispatched, since someone is already watching that; a release alerts either way, because a failed release is the team's problem. Only failures: a successful release is already on the repository's webhooks. Holds the single (post-only) CI secret |
| `static-checks.yml` | called by both CI workflows | the tree-reading checks that always run: the gitleaks history scan and `make lint`. Identical for a pull request and for main, so it is one file, and it takes no inputs — what a caller may want to skip, the caller calls itself |
| `code-scanning.yml` | weekly / called by both CI workflows / manual | one analysis leg per engine, because an analyser only sees the ISA variant a build compiled; the analyser is named nowhere else, so it can be replaced without touching a caller |
| `scorecard-analysis.yml` | push / branch protection change / weekly | OpenSSF Scorecard, the supply-chain posture score |
| `distribution-build.yml` | called by both CI workflows and the release | assembles the distributions: the sdist always (`twine check --strict`), the wheel matrix when the caller names a version to stamp — a release, or main's dev artifact (cibuildwheel, each wheel running the fast suite from its own install). Nothing here publishes or installs |
| `smoke-tests.yml` | called by both CI workflows and the releases | `smoke/*.py` against an *installed* vfhe across platform x interpreter — the sdist this run built when the caller names nothing, or the requirement it names (a release, from the index it just reached). One target either way: `make smoke DIST=…` |
| `release-artifacts.yml` | called by both release workflows | the shared artifact pipeline: `distribution-build.yml` with wheels, then the CycloneDX SBOM of an install |
| `release-testpypi.yml`, `release-pypi.yml` | manual | each target's form, validation, and publish job; both call `release-artifacts.yml` (sdist + wheels + SBOM), then `smoke-tests.yml` against the index they just published to. PyPI additionally verifies the signed bundle offline before uploading — the last point at which stopping is free — and afterwards downloads every file the index serves and runs `gh attestation verify` over each, so the provenance is proven rather than merely generated. Both refuse to publish a commit `main` did not prove: `validate` asserts the shape of the request (a tag, bare semver, an ancestor of `main`) and then `check-workflow-status` asserts the status `CI Postsubmit` recorded for that exact SHA — `Hardware Tests` too when it ran, which a changelog-only release commit skips. Policy: pre-releases (`aN`/`bN`/`rcN`/`.devN`) exist only on TestPyPI, as a dispatch input, never a git tag — PyPI's validate accepts bare `X.Y.Z`, so users can never see one and the tag list stays one-to-one with real releases |

### The self-hosted AVX-512 IFMA runner

Real silicon lives in one workflow, `hardware-tests.yml`, on a runner
labelled `self-hosted, Linux, X64, avx512ifma`. It runs that engine's complete
suite natively — the cross-check of Intel SDE against physical hardware — its
sanitizer pass at a speed emulation cannot reach, and `make smoke` *unpinned*,
the one place an *installed* package selects the SIMD extension and runs it
rather than merely shipping it — hosted smoke legs pin `VFHE_ENGINE=portable`,
since the picker asks the CPU and that fleet is mixed. Ordered steps rather than separate jobs: the runner takes one job at a time anyway, and steps pay for
the checkout and the dev environment once, with `!cancelled()` on the later ones
so a failing shape does not hide the rest. It only runs on pushes to `main`, so
it **never executes pull-request code**; a red or endlessly-queued run here is the outage alert, and
it blocks no merge, since the merge already happened. The
A job that needs this runner while it is offline **queues rather than fails**,
and a queued job triggers no alert: the symptom is a run that never finishes,
not a red one. Job timeouts do not apply to queue time. The
runner needs no secrets: only `git`, a C compiler, and network access for
the dev dependencies. Register it ephemeral (`config.sh --ephemeral`) and
repo-scoped, so each job starts on a clean machine. Note: GitHub disables
every scheduled workflow after 60 days without repository activity.

**Releases** are manual: one dispatch workflow per target, each owning its form,
validation, publish credentials, and publish job; both call the shared
`release-artifacts.yml` (sdist and wheels in parallel, then the SBOM of an
install). Neither will publish a commit `main` did not prove: `validate` reads
the status `CI Postsubmit` recorded for that exact SHA, and `Hardware Tests` too
when it ran, which a documentation-only release commit skips:

- **Release TestPyPI**: give an RC `version` (e.g. `0.0.2rc1`). Dispatch it
  from `main`.
- **Release PyPI**: push a bare-semver tag (e.g. `0.1.0`, *no* leading `v`),
  then run the workflow **from that tag**. The version comes from the tag,
  and a GitHub Release with the sdist, wheels, SBOM, and build provenance is
  created alongside.

After a release, dispatch **Published Smoke** with the version (and the index,
for TestPyPI); it also runs nightly against whatever PyPI serves. It answers what the release deliberately cannot: that the
*published wheel* picks the SIMD extension and runs it, rather than merely
shipping it — hosted smoke legs pin `VFHE_ENGINE=portable`, so this is the only
place selection is proven against an artifact an index served. Locally it is
`make smoke DIST=vfhe==<version>`.

It is dispatched rather than chained because a release must not wait on a
machine that can be offline, and a workflow cannot start another with
`GITHUB_TOKEN` — a fire-and-forget dispatch would need a long-lived token on the
publish path and would hide the result besides.

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
   vfhe.<mod>.<name>.v1` (buf convention: the file's path must equal its package
   path). Each `proto/` dir has a `buf.yaml` as the buf module root. Bindings are
   generated to `_vfhe_proto.vfhe.<mod>.<name>.v1.<name>_pb2`.

Each module's `meson.build` lists its C sources and C tests explicitly (a
new file is one line there); the proto bindings auto-discover modules
recursively. Python-facing modules list their import
paths in `[tool.pyright]` (editor); tests resolve via `modules/conftest.py`,
which globs the module src dirs.

### Architecture-specific sources & SIMD

Every build is **multi-engine**. Meson compiles the tree once per engine
the host's architecture allows — `_vfhe_native_portable`,
`_vfhe_native_avx512ifma`, one module per registry entry — and the
pure-Python `_vfhe_native` shim picks one at import: `VFHE_ENGINE=<name>` if
pinned, else the best whose capability `vfhe_cpu_supports()` confirms. That
probe lives in C (so the answer is right per architecture) inside its own
~50 KB `_vfhe_cpu` extension, so asking the question never loads an engine
and **exactly one** engine is ever imported. Kernels that are never
imported cannot SIGILL, so one install moves freely between machines of its
architecture. The portable engine compiles the scalar paths
(`PORTABLE_BUILD`); every other engine compiles with its explicit ISA flags
(`tools/_engines.py` — meson queries it through `engine_info.py`, so the two
cannot drift), so it builds on any host of its architecture even where it
cannot run.
`VFHE_COVERAGE=1` instruments the build with gcov.
`vfhe.misc.dynamic_extensions` compiles only the *user's* files at runtime
and links them against the shipped `libvfhe_<engine>.a` of the *active*
engine; `engine-<engine>.json` beside it records the flags a matching
compile needs (the public headers change types under them). An installed
package therefore carries headers, cdefs, and the archives — no C sources, no
build machinery.

Sources are **listed** in each module's `meson.build` (meson does not glob,
by design: a forgotten file is a loud link error, never a silent omission),
and both C and hand-written assembly (`.S`) compile. The engines select SIMD
via compile-time macros in the sources; for the tools that compile outside
meson — the fuzz harnesses — `tools/_engines.py` still scans `c/src`
recursively and skips other architectures' `c/src/arch/<name>/`
(`x86_64` aliases: `x86_64/amd64/x86-64/x64`; `arm64`: `arm64/aarch64`).

---
