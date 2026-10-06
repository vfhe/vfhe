# Developing vFHE

For contributors changing the library itself. It covers the build, the test
matrix, coverage, and packaging. All library code lives in self-contained
modules under `modules/`, and each module's README maps its own contents.

Contribution policy is in
[CONTRIBUTING.md](https://github.com/vfhe/vfhe/blob/main/docs/CONTRIBUTING.md);
CI and releasing are in
[WORKFLOWS.md](https://github.com/vfhe/vfhe/blob/main/docs/WORKFLOWS.md).

## Prerequisites

- **Python 3.10 or later**
- **pip 25.1 or later**, which added `pip install --group` for PEP 735
  dependency groups
- **A C compiler** — clang or gcc; on macOS,
  [Xcode Command Line Tools](https://developer.apple.com/xcode/resources/)
  (`xcode-select --install`)
- **git**, to fetch the two submodules the build compiles

## Getting started

### Step 1: Clone with submodules

```bash
git clone --recursive https://github.com/vfhe/vfhe vfhe && cd vfhe
```

Confirm submodules arrived:

```bash
git submodule status
```

```text
 f3149ec5bb… external/blake3/blake3 (1.8.7)
 b6763fbd9c… external/unity/unity (v2.7.0)
```

An already-cloned tree fetches them with
`git submodule update --init --recursive`. Both are required to build, because
BLAKE3's C compiles into the native extension and Unity runs the C tests. A
non-recursive clone fails the configure step with a message naming the fix.

### Step 2: Create and activate a virtual environment

```bash
python3 -m venv .venv && source .venv/bin/activate
```

Nothing in the tooling manages `.venv/`, so activate it in every new shell.
Every command below otherwise needs `make PYTHON=/path/to/python`.

### Step 3: Install the toolchain and the git hooks

```bash
make dev-env
```

```text
pre-commit installed at .git/hooks/pre-commit
pre-commit installed at .git/hooks/commit-msg
```

This installs the `dev` dependency group from `pyproject.toml` — the group,
not the package — and the `pre-commit` hooks.

### Step 4: Build the native code

```bash
make build
```

`build/` then holds one extension per engine this architecture can run, plus
the CPU probe:

```bash
ls build/*.so
```

```text
build/_vfhe_native_neon.cpython-314-darwin.so
build/_vfhe_native_portable.cpython-314-darwin.so
```

Nothing generated is committed, because meson regenerates the bindings on
every build. A clean checkout, or an sdist install, needs nothing
pre-generated.

### Step 5: Verify the setup

```bash
make test SUITES=c,fast
```

The run ends with meson's tally, and every count outside `Ok` reads zero:

```text
Ok:                10
Fail:              0
```

Engines this CPU cannot run are left unselected, so a `Skip` would come from
a test's own skip.

### Next steps

- Run the whole matrix with `make test`, which defaults to the complete depth.
- Read [Testing](#testing) for how a test's name selects it and how
  `EMULATE=1` reaches Intel SDE.
- Point your editor at `.venv`'s interpreter. `pyrightconfig.json` configures
  import paths for any Pyright-based tool, so `vfhe.*` resolves from
  `modules/*/python/src` and the extensions from `build/`. Build once, then
  reload the editor.

## How the build works

Every build is **multi-engine**, producing one extension and archive per
engine the host's architecture allows. Every module's `c/src` compiles into
that one LTO'd extension per engine, `_vfhe_native_<engine>`, so kernels
inline across module boundaries, and the hand-written cdefs pass opaque
`void *` handles plus a few structs so Python can read fields.

### The engines

One engine exists per instruction-set level the kernels have a branch for.
The *to do* rows are the work ahead, and this table is that list.

| Engine | Hosts | State | What it buys |
|---|---|---|---|
| `portable` | x86_64, arm64, macOS | shipping | the scalar baseline every CPU runs |
| `avx512ifma` | x86_64 | shipping | 52-bit integer multiply-add (`madd52`), the widest path we have |
| `avx512f` | x86_64 | to do | AVX-512 F/DQ/VL without IFMA — most kernels need no `madd52` |
| `avxifma` | x86_64 | to do | the same IFMA at 256 bits (VEX), for CPUs without AVX-512 |
| `avx2` | x86_64 | to do | 256-bit integer SIMD, the widest baseline on old x86_64 |
| `neon` | arm64, macOS | shipping | BLAKE3's NEON hashing today. vFHE's own kernels still take the portable path here, because they guard on `__AVX512IFMA__`; a `#if defined(__ARM_NEON)` branch slots in with no registry change |

Meson pins every test to its engine (`VFHE_ENGINE`), so a machine that
happens to have IFMA cannot silently switch engines. Testing `avx512ifma`
without that hardware needs
[Intel SDE](https://www.intel.com/content/www/us/en/download/684897/intel-software-development-emulator.html),
which `tools/test/sde/fetch.sh` fetches into `.cache/sde/` on first use, at the
version `tools/test/sde/.env` pins.
SDE instruments real x86 processes, so it runs on **Linux x86_64 only**.

### Selecting an engine

At import, `vfhe.util.bindings` loads the engine `vfhe.util._engine` selects: the
fastest installed one that `vfhe.util._cpu` confirms this CPU runs. The probe is
pure Python and reads the CPU's flags, because loading an engine the CPU cannot
run executes illegal instructions. `VFHE_ENGINE=<name>` pins an engine, and a pin
skips the probe, so an emulator runs an engine whose instructions the host CPU
lacks. `bindings` is the only loader and the import system runs it once, so a
process holds one engine, and an install moves freely between machines of its
architecture.

### Architecture-specific sources

The portable engine compiles the scalar paths (`PORTABLE_BUILD`), and every
other engine compiles with its explicit ISA flags, which define the macros
the kernels guard on, so it builds on any host of its architecture even where
it cannot run. `vfhe.util.kernels` compiles the caller's files *and* vfhe's own into a single
engine with link-time optimisation, so their loops inline vfhe's primitives and
the process still holds exactly one copy of the library. The generated
`_vfhe_engines` table records each engine's flags and the exact sources meson
compiled for it, so a rebuild cannot drift from the shipped one. An installed
package therefore carries that table plus one tarball,
`vfhe/_source/kernels-src.tar.gz`, which holds the C sources, the public headers and the
cdefs together. They are not installed a second time beside it: `vfhe.util.kernels`
unpacks the tarball and compiles from there, so an unpacked copy would be read
by nothing.

Sources are **listed**, never globbed, so a forgotten file becomes a loud
link error rather than a silent omission. Both C and hand-written assembly
(`.S`) compile.

The build directory is named after what identifies the build -- the vfhe
version, the engine, the compiler and its version, the Python version, and the
paths of the caller's sources and declarations -- so the same inputs land in the
same place and a second call is a cache hit, and two projects never share a
directory. File *contents* stay out of the name: ninja owns staleness inside the
directory, reading the compiler's own depfiles, so editing one file of sixty
recompiles one file of sixty in the same place.

Building needs a C compiler, meson and ninja, the `kernels` extra
(`pip install vfhe[kernels]`); loading a prebuilt engine needs neither. The
engine built carries the prebuilt engine's module name and its directory goes
first on `sys.path`, so `vfhe.util.bindings` imports it in place of the prebuilt
one. CPython keeps an extension loaded for the life of the process, so `ExtensionBuilder.build()`
runs before anything imports vfhe's C and raises when called later.
`python -m vfhe.util.kernels` builds the same artefact and exits; its directory
on `PYTHONPATH` removes the ordering rule and lets an image ship without a
compiler. The same works from Python when generating the C needs vfhe loaded:
`compile()` writes the engine without touching the process, and a child process
with its directory on `PYTHONPATH` runs the kernel.

## Testing

The matrix has two orthogonal axes.

- **Suites** say *what* runs — **c** (`modules/*/c/test`), **fast** (the
  default Python subset), **complete** (adds the heavy
  `@pytest.mark.complete` computations), and the **smoke** cases
  (`test/smoke`) against an installed package.
- **Engines** say *which implementation* runs underneath — **portable**
  everywhere, **avx512ifma** on x86_64, **neon** on arm64, and more as
  kernels land.

**Test sources are engine-invariant.** One C API and one set of test files
cover every implementation behind the ISA macros, so the engine is a build
parameter rather than a test parameter. A test that applies to one engine
asks which one is loaded. An arithmetic *implementation* or *backend* is the
opposite: chosen per object at runtime, so it is a legitimate `parametrize`
argument (see `modules/arith/python/test/test_spec.py`).

```bash
make test                          # C + complete suites, on every built engine
make test SUITES=c,fast            # the fast depth instead, same engines
make test ENGINE=avx512ifma        # one engine alone
make test ENGINE=avx512ifma EMULATE=1  # the same under Intel SDE, on Linux x86_64
make smoke                         # test/smoke against DIST (default: a fresh sdist)
make smoke SMOKE_CASES=info        # one case, in that same sandbox venv
```

`ENGINE` defaults to `all`: `tools/test/unit/run.sh` reads the built engines off
meson's test list, keeps those this CPU runs, and selects their tests. Naming
one engine selects its tests alone, and stops when this build lacks it or this
CPU cannot run it. A test is
named `<engine>-c-<stem>` or `<engine>-py-<depth>`, and that name is the
selection: a glob that matches nothing exits 1, where `--suite` would exit 0
and report success having run nothing. `build/meson-logs/testlog.txt` holds
the details of anything under `Fail`.

`EMULATE=1` runs `tools/test/sde/fetch.sh`, which fetches Intel SDE into
`.cache/sde` on first use, reconfigures so meson finds the launcher there, and
selects `meson test --setup <engine>_emulated`, whose `exe_wrapper` launches
each test under SDE with the engine's flags. Configure itself downloads
nothing, and without the launcher the setup does not exist.

A smoke case asserts what only an installed distribution can answer: the
version is real, every declared engine shipped, a runtime-compiled extension
links. tox gives it a venv holding only the distribution and runs it from a
temp directory, so the source tree is unreachable. A helper both a smoke case
and an in-tree suite want is a sign the smoke case is testing the tree.

```bash
make test SUITES=c VFHE_SANITIZE=address,undefined           # ASan + UBSan (meson's -Db_sanitize)
make test ENGINE=portable SUITES=c,fast VFHE_COVERAGE=true   # the same run, measured
make smoke REQUIREMENT=vfhe==1.2.3                           # a published release instead of a local build
make test SUITES=fast PYTEST_ADDOPTS="-k ntt"                # narrow the pytest selection
```

### Coverage

`VFHE_COVERAGE=true` swaps the release shape (`-O3`, LTO) for gcov
instrumentation and leaves the reports in `build/meson-logs/`. Codecov
combines CI's uploads into the figure on the pull request, and coverage gates
nothing. `[tool.coverage.*]` in `pyproject.toml` makes a local
`pytest --cov=vfhe` measure what CI measures, and untestable code is excluded
with the markers in
[CONTRIBUTING.md](https://github.com/vfhe/vfhe/blob/main/docs/CONTRIBUTING.md#about-coverage).

### Sanitizers

Sanitize an engine on a host that runs it natively, never under an emulator,
whose slowdown multiplies the sanitizer's. The **fast** suite suffices,
because it already walks every kernel and public API.

### Static analysis and fuzzing

CodeQL analyses every engine it can build, because compiled-out code is
invisible to a build-time tool. neon is not one: the CLI has no linux/arm64
build, and on macOS it traces the build as x86_64, so an arm64-gated engine
never configures.

libFuzzer fuzzers live in `modules/<mod>/c/test/fuzz/`, built by meson from
the same flags and includes as the archive they link, so a fuzzer cannot drift
from the kernels it exercises. Fuzzing runs only in CI. Reproduce a finding
locally with the OSS-Fuzz helper, which needs docker:

```bash
git clone --depth 1 https://github.com/google/oss-fuzz .cache/oss-fuzz
python .cache/oss-fuzz/infra/helper.py build_fuzzers --external . --sanitizer address
python .cache/oss-fuzz/infra/helper.py run_fuzzer --external . <target> [reproducer-file]
```

## Documentation

`make docs` builds the site into `build/docs/html` from `docs/site/`, a Sphinx
project in reStructuredText: a stub per guide beside it, which stays Markdown
for GitHub; the Python API from the docstrings (autodoc); the C API from the
Doxygen blocks in each module's `c/include` (Doxygen's XML, rendered by
Breathe). It needs `doxygen` on `PATH` besides the `docs` dependency group. CI builds it on
every pull request with warnings as errors, so an undocumented declaration or a
malformed docstring fails the gate, and publishes it from `main` to
<https://vfhe.github.io/vfhe/>. The API pages list `util` and `info`; a module
joins them once its headers and docstrings pass the generators.

The generators read these conventions. A C declaration carries a `/** ... */`
block of `@` commands: `@brief`, `@param[in]`, `@return`, `@warning`, `@note`.
A Python docstring opens with a verb, names code in `` `backticks` ``, which
Sphinx cross-references, and literals in ``` ``double backticks`` ```. A
package's `__init__` docstring states the exports, what a user must know to use
them safely, and one example.

## Formatting and licensing

The hooks report problems without rewriting anything, so run `make format`
before committing. They also check what no compiler will, and
`.pre-commit-config.yaml` lists all of them.

Licensing is machine-checked. The project follows
[REUSE](https://reuse.software), so every file states its copyright and
licence in its own header, or in `REUSE.toml` where the format carries no
comments.

Commit style and the enforced DCO sign-off are in
[CONTRIBUTING.md](https://github.com/vfhe/vfhe/blob/main/docs/CONTRIBUTING.md#commits).

## Extending

Both a new module and a new engine cost one entry in the build plus their
own code; nothing else needs rewiring.

### Adding a module

A module carries only the parts it needs. `circuit` is Python with no C,
`polycom` has no C tests, and `arith` has everything.

1. **Python package.** `modules/<mod>/python/src/vfhe/<mod>/__init__.py` plus
   its implementation modules, and the pytest suite in `python/test/`. A
   module holding several implementations of one interface gives each its own
   subpackage, `impl/<name>/` (`arith`).
2. **C sources.** `c/src/*.c` and `c/include/*.h`, grouped in subdirectories
   where a module has many (`arith`); `<file>_rns.c` holds the part of
   `<file>.c` that needs one representation. A module exposing C to
   Python also writes `python/cdef/<name>.h`, the declarations Python may
   call, in cffi's cdef dialect. Kernels worth testing below the Python
   surface add a Unity suite in `c/test/unit/`, and a libFuzzer fuzzer in
   `c/test/fuzz/`.
3. **Protobuf schema.** Not under the module: every wire format lives in the
   repository's `proto/` tree, as `proto/vfhe/<mod>/<name>/v1/<name>.proto`
   with `package vfhe.<mod>.<name>.v1`, following buf's convention that a
   file's path equals its package path. A schema is a contract between a
   producer and a consumer, often in different modules and in the PoC across a
   network, so it is reviewed as one surface: a single `proto/buf.yaml` lints
   it and checks it for breaking changes, one `-I` root lets a schema import
   another module's, and `proto/meson.build` generates the lot into
   `_vfhe_proto.vfhe.<mod>.<name>.v1.<name>_pb2`.

Whichever surfaces the module adds need tests, per
[CONTRIBUTING.md](https://github.com/vfhe/vfhe/blob/main/docs/CONTRIBUTING.md#about-coverage)'s
testing policy.

Each module's `meson.build` appends what it contributes to the lists the
root `meson.build` declares — `vfhe_c_sources`, `vfhe_c_include_dirs`,
`vfhe_c_public_headers`, `vfhe_c_unit_tests`, `vfhe_c_fuzz_tests`,
`vfhe_python_cdefs` — and installs what it owns. A new module costs its
directory plus one `subdir()` line in the root; a new file costs one line in
the module's own list. Misspell a list and meson stops, naming the file and
the correct name, because the lists are plain variables.

The root reads the lists to build one extension per engine, and
`test/meson.build` to build the static libraries, the test programs and the
fuzzers. Python-facing modules also list their import path in
`pyrightconfig.json` and in `test/meson.build`'s `pythonpath`.

### Adding an engine

An engine costs one entry in the root `meson.build`'s `engines` dict, its
name mapped to the C flags that define the macros its kernels guard on, plus
those kernels and its CI rows. Kernels live behind the ISA macros in shared
files, or in `arch/<arch>/` under `c/src/`.

## Building the distribution

vFHE releases as wheels (Linux x86_64/arm64, macOS arm64) plus the sdist.
Every install carries each engine its architecture can run and picks one at
import.

```bash
make sdist        # -> dist/vfhe-<version>.tar.gz
```

`meson dist` cuts the sdist from the **last commit**, so commit before
`make sdist` or `make smoke` or the archive will not match your working tree.
The archive bundles the git-tracked tree plus the vendored BLAKE3 C sources,
and the version is the static one in `pyproject.toml`, so building it needs
neither submodule nor git.

Wheels are never built by hand. The release builds the whole matrix with
cibuildwheel (`[tool.cibuildwheel]` in `pyproject.toml`). Publishing is
[WORKFLOWS.md](https://github.com/vfhe/vfhe/blob/main/docs/WORKFLOWS.md)'s
Releasing section.

### Dependencies

- **Selection.** The runtime surface is deliberately small (`cffi`,
  `protobuf`, `mpmath`), and anything else must earn its place in review.
  Version floors state the oldest supported release, raised only when a newer
  feature is required.
- **Obtaining.** Python dependencies are declared in `pyproject.toml`
  (runtime, build, and PEP 735 `dev`/`release` groups, where `dev` includes
  `release`) and resolved from PyPI by pip at build time. The native
  third-party submodules are pinned to exact commits, with provenance in
  `NOTICE`.
- **Tracking.** Dependabot watches GitHub Actions by commit SHA, pip, and
  both Dockerfiles by image digest. Outside it, the **submodules** are bumped
  by hand to a tag, because Dependabot's submodule updater only moves a pin
  to the tracked branch's newest commit and preferring tags is an upstream
  request rather than an option. The **pre-commit hook revisions** are bumped
  by hand too, with `pre-commit autoupdate`. The **Intel SDE** download
  (`tools/test/sde/.env`) is a URL-fetched binary rather than a package, so
  its pin sits beside the code reading it, nothing bumps it, and its digest
  proves integrity rather than freshness — check it when touching that file.
  Every wheel carries a CycloneDX description of the C compiled into it at
  `.dist-info/sboms/`, the location PEP 770 standardises; Python dependencies
  need no such record, because `Requires-Dist` metadata already states them.
