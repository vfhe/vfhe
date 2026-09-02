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
 b6763fbd9c… external/unity (v2.7.0)
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
build/_vfhe_cpu.cpython-314-darwin.so
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

A `Skip` count is normal — it names engines this CPU cannot execute.

### Next steps

- Run the whole matrix with `make test`, which defaults to the complete depth.
- Read
  [`test/README.md`](https://github.com/vfhe/vfhe/blob/main/test/README.md)
  for how a test's name selects it.
- Point your editor at `.venv`'s interpreter. `[tool.pyright]` in
  `pyproject.toml` configures import paths for any Pyright-based tool, so
  `vfhe.*` resolves from `modules/*/python/src` and the extensions from
  `build/`. Build once, then reload the editor.

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
which `tools/sde/fetch.sh` pins and fetches into `.cache/sde/` on first use.
SDE instruments real x86 processes, so it runs on **Linux x86_64 only**.

### Selecting an engine

At import, `vfhe.engine` picks the best engine `vfhe_cpu_supports()` confirms
(or the `VFHE_ENGINE=<name>` pin), asking the separate `_vfhe_cpu` extension
rather than trying a candidate, because loading an engine this CPU cannot run
executes illegal instructions. Exactly one engine is ever imported, so an
install moves freely between machines of its architecture.

### Architecture-specific sources

The portable engine compiles the scalar paths (`PORTABLE_BUILD`), and every
other engine compiles with its explicit ISA flags, which define the macros
the kernels guard on, so it builds on any host of its architecture even where
it cannot run. `vfhe.dynamic_extensions` compiles only the *user's* files at
runtime, linking them against the shipped `libvfhe_<engine>.a` of the active
engine; `engine-<engine>.json` beside it records the flags a matching compile
needs, because the public headers change types under them. An installed
package therefore carries headers, cdefs, and archives, with no C sources and
no build machinery.

Sources are **listed**, never globbed, so a forgotten file becomes a loud
link error rather than a silent omission. Both C and hand-written assembly
(`.S`) compile.

## Testing

The matrix has two orthogonal axes.

- **Suites** say *what* runs — **c** (`modules/*/c/test`), **fast** (the
  default Python subset), **complete** (adds the heavy
  `@pytest.mark.complete` computations), and the **smoke** cases
  (`test/smoke/cases`) against an installed package.
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
make test ENGINE=avx512ifma        # one engine alone (emulated if this CPU lacks its ISA)
make smoke                         # test/smoke/cases against DIST (default: a fresh sdist)
make smoke SMOKE_CASES=info        # one case, in that same sandbox venv
```

`ENGINE` defaults to `all`, running every engine the build produced, and
naming one narrows the run to it.
[`test/README.md`](https://github.com/vfhe/vfhe/blob/main/test/README.md)
states the mechanism — how a test's name selects it, how an engine this CPU
cannot execute skips rather than fails, and how `EMULATE=1` reaches for Intel
SDE.

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

libFuzzer harnesses live in `modules/<mod>/c/fuzz/`, built by meson from the
same flags and includes as the archive they link, so a harness cannot drift
from the kernels it exercises. Fuzzing runs only in CI. Reproduce a finding
locally with the OSS-Fuzz helper, which needs docker:

```bash
git clone --depth 1 https://github.com/google/oss-fuzz .cache/oss-fuzz
python .cache/oss-fuzz/infra/helper.py build_fuzzers --external . --sanitizer address
python .cache/oss-fuzz/infra/helper.py run_fuzzer --external . <target> [reproducer-file]
```

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

A module carries only the parts it needs. `crypto` is C kernels alone,
`circuit` is Python and a protobuf schema with no C, and `arith` has
everything.

1. **Python package.** `modules/<mod>/python/src/vfhe/<mod>/__init__.py` plus
   its implementation modules, and the pytest suite in `python/test/`. A
   module holding several implementations of one interface gives each its own
   subpackage, `impl/<name>/` (`arith`).
2. **C sources.** `c/src/*.c` and `c/include/*.h`, grouped in subdirectories
   where a module has many (`arith`); `<file>_rns.c` holds the part of
   `<file>.c` that needs one representation. A module exposing C to
   Python also needs a hand-written `python/cdef/<mod>.cdef` declaring that
   ABI. Kernels worth testing below the Python surface add
   `c/test/test_<mod>.c`, either a plain `assert`/`main()` program whose
   non-zero exit means failure, or a Unity suite.
3. **Protobuf schema.** `proto/vfhe/<mod>/<name>/v1/<name>.proto` with
   `package vfhe.<mod>.<name>.v1`, following buf's convention that a file's
   path equals its package path. Each `proto/` directory carries a `buf.yaml`
   as its buf module root, and bindings generate to
   `_vfhe_proto.vfhe.<mod>.<name>.v1.<name>_pb2`.

Whichever surfaces the module adds need tests, per
[CONTRIBUTING.md](https://github.com/vfhe/vfhe/blob/main/docs/CONTRIBUTING.md#about-coverage)'s
testing policy.

Each module's `meson.build` declares what it contributes — sources, headers,
cdefs, C tests, protos — and the root `meson.build`'s `parts` list names the
module's path, so a new module costs its directory plus that one line, and a
new file costs one line in the module's own list. Python-facing modules also
list their import path in `[tool.pyright]`, and tests resolve through
`modules/conftest.py`, which globs the module source directories.

### Adding an engine

An engine costs one entry in `meson.build`'s engine list plus its kernels.
The entry names the architecture, the capability, the ISA flags (which define
the macros the kernels guard on), any arch-specific vendored sources, and how
to emulate it. Its CI rows are added alongside. Kernels live behind the ISA
macros in shared files, or in `arch/<arch>/` under `c/src/`.

## Building the distribution

vFHE releases as wheels (Linux x86_64/arm64, macOS arm64) plus the sdist.
Every install carries each engine its architecture can run and picks one at
import.

```bash
make sdist        # -> dist/vfhe-<version>.tar.gz
```

`meson dist` cuts the sdist from the **last commit**, so commit before
`make sdist` or `make smoke` or the archive will not match your working tree.
The archive bundles the git-tracked tree plus the vendored BLAKE3 C sources
and a frozen `.version`, so building it needs neither submodule nor git.

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
  request rather than an option. The **Intel SDE** download
  (`tools/sde/fetch.sh`) is a URL-fetched binary rather than a package, so
  its pin sits beside the code reading it, nothing bumps it, and its digest
  proves integrity rather than freshness — check it when touching that file.
  Every wheel carries a CycloneDX description of the C compiled into it at
  `.dist-info/sboms/`, the location PEP 770 standardises; Python dependencies
  need no such record, because `Requires-Dist` metadata already states them.
