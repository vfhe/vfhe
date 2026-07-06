# vfhe

Verifiable Fully Homomorphic Encryption library.

---

## Repository structure

```
modules/<mod>/                     one folder per module (arith, base, rng, circuit, …)
  python/src/vfhe/<mod>/           Python package source (physical namespace layout)
    __init__.py                    THIN: re-exports the module's public API
    <impl>.py                      implementation; `from _vfhe_native import ffi, lib`
  python/cdef/<mod>.cdef           Python-facing ABI: hand-written cffi decls (opaque handles)
  python/test/                     pytest suite
  c/include/                       public C headers (umbrella <mod>.h; large modules
                                   namespace per-layer headers under include/<mod>/)
  c/src/                           pure C kernels, one subdirectory per concern
  c/test                           Unity C tests (targeted per-layer suites)
  c/fuzz                           OPTIONAL libFuzzer harnesses (nightly CI)
  proto                            OPTIONAL protobuf schema for this module

native/build_ffi.py                cffi build: ALL c/src kernels -> one ext `_vfhe_native`
native/discovery.py                shared source/include/arch discovery (build + tests)
scripts/gen_proto.py               protoc: all proto schemas -> `_vfhe_proto` package
scripts/run_c_tests.py             compiles + runs the Unity C tests (VFHE_SANITIZE=…)
scripts/build_fuzzers.py           builds/runs the c/fuzz libFuzzer harnesses
scripts/check_simd_build.py        compile-checks the x86 AVX-512 engine paths
scripts/smoke_import.py            post-install import smoke test (CI + cibuildwheel)
.github/workflows/                 CI, nightly, release, and security workflows
external/unity/                    Unity C test framework (git submodule, test-only)
external/blake3/                   BLAKE3 hash (git submodule; c/ compiled into the ext)
setup.py / pyproject.toml          package assembly + build config
pyrightconfig.json                 editor import paths (Pyright / Pylance)
stubs/_vfhe_native.pyi             typing boundary for the cffi ext (Any)
.generated/                        ALL build artifacts (gitignored — see below)
```

### Generated files

Everything generated goes into `.generated/`, which is gitignored in full:

| Artifact | Produced by | Purpose |
|---|---|---|
| `_vfhe_native.*.so` | `native/build_ffi.py` (cffi) | the compiled C kernels, one LTO'd extension |
| `_vfhe_proto` | `scripts/gen_proto.py` (protoc) | protobuf bindings |

---

## Prerequisites

- Python **3.9+**
- `pip` **25.1+** (for `pip install --group`, PEP 735 dependency groups)
- A C compiler (clang/gcc — Xcode Command Line Tools on macOS)
- `git` (Unity and BLAKE3 are vendored as submodules — clone with `--recursive`)

---

## 1. One-time setup

```bash
git clone --recursive https://github.com/vfhe/vfhe vfhe && cd vfhe
# already cloned without --recursive? fetch the submodules (Unity + BLAKE3):
git submodule update --init --recursive

python3 -m venv .venv && source .venv/bin/activate
make deps                                        # install dev dependencies (see below)
```

> **Submodules are required to build.** BLAKE3's C sources are compiled into the
> native extension, so a non-recursive clone will fail fast with a clear message.
> Released sdists bundle BLAKE3's sources, so `pip install vfhe` needs no submodules.

`make deps` runs `pip install --group dev`, installing the **`dev` dependency
group declared in `pyproject.toml`** (PEP 735) — **not** the package.

> `.venv/` is yours to manage; activate it for every session. All commands below
> assume an activated environment (or pass `make PYTHON=/path/to/python`).

---

## 2. Development loop

```bash
make build      # (re)generate proto bindings + compile the C kernels into .generated/
make test       # build, then run the C (Unity) and Python (pytest) suites
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

`make test` runs both suites: the per-layer Unity C tests
(`scripts/run_c_tests.py`, discovered under each `c/test/`) and the Python
pytest suites. Extra, deeper checks:

```bash
VFHE_SANITIZE=address,undefined python scripts/run_c_tests.py   # ASan + UBSan
python scripts/build_fuzzers.py --run --time 60                 # fuzz c/fuzz/* (needs clang)
PYTHONPATH=scripts python scripts/check_simd_build.py           # x86 AVX-512 compile-check
```

Fuzz harnesses live in `modules/<mod>/c/fuzz/` (built with
`-fsanitize=fuzzer,address,undefined` and a deterministic RNG so crashes
reproduce); libFuzzer needs a Linux/clang toolchain, so they run in the nightly
CI rather than on macOS. The sanitizer and fuzz jobs are wired into the
workflows below.

---

## 3. Code & Library Generation

Nothing generated is committed — regenerate locally as needed:

| Run | When |
|---|---|
| `make proto` | after editing a `.proto` |
| `make build` | after editing C, or to refresh everything (`proto` + native) |

`setup.py` reruns proto generation automatically during a wheel build, so a
clean checkout needs nothing pre-generated.

---

## 4. Formatting & commits

**`make format`** formats everything in place — `ruff` for Python (format +
import sorting) and `clang-format` for C (style in `.clang-format`). Run it before
committing.

**Commit messages** follow the 50/72 convention (subject ≤ 50 chars, body wrapped
at 72) as a *recommendation, not enforced*. Opt into the editor guide once:

```bash
git config commit.template .gitmessage
```

---

## 5. Building the wheel

```bash
make wheel        # -> dist/vfhe-<version>-<pytag>-<platform>.whl
```

This builds a wheel for the **current platform / interpreter** via the standard
PEP 517 frontend (`python -m build`). It contains: the `vfhe.*` packages, the
compiled `_vfhe_native` extension, and the freshly generated `_vfhe_proto`
bindings. For the full cross-platform wheel matrix and publishing, see the
release workflow in §6.

---

## 6. Continuous integration & releases

Workflows live in `.github/workflows/` (POSIX only — Linux + macOS):

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | push / PR | lint (ruff + pyright + clang-format), tests on Linux+macOS × py3.9–3.13, x86 AVX-512 compile-check, sdist/wheel smoke |
| `nightly.yml` | nightly / manual | C tests under ASan + UBSan, libFuzzer on `c/fuzz/*` |
| `codeql.yml`, `scorecard.yml`, `dependency-review.yml` | push / PR / schedule | static analysis + supply-chain hygiene |
| `release.yml` | see below | build wheels + sdist + SBOM, then publish |

**Releases** build once and publish two ways:

- **Push a bare-semver tag** (e.g. `0.1.0` — *no* leading `v`) → auto-publish to
  **TestPyPI** + cut a GitHub Release.
- **Run `release.yml` manually** (Actions → Run workflow) → publish to **real
  PyPI**.

Both use **Trusted Publishing** (OIDC, no API tokens) with PEP 740 attestations.
One-time setup before the first release: create the `vfhe` project + a Trusted
Publisher for this repo/workflow on **both** TestPyPI and PyPI, and add GitHub
Environments named `testpypi` and `pypi` (protect `pypi` with required reviewers
so the manual publish still needs an approval).

---

## Adding a new module

A module can be **Python-facing** (has `python/` + a cdef exposing C to Python)
or **internal C-only** (no `python/`; used by other modules' kernels — like
`base` and `rng`).

1. (Python, optional) create `modules/<mod>/python/src/vfhe/<mod>/__init__.py`
   (+ impl `.py`). Omit for internal C-only modules.
2. (C, optional) add `c/src/*.c` + `c/include/*.h`. If the module exposes any C
   to Python, hand-write `python/cdef/<mod>.cdef` declaring that ABI (opaque
   handles + the functions Python calls). Internal C-only modules (`base`, `rng`)
   have **no** cdef — they contribute compiled code but no Python symbols.
3. (C tests) add `c/test/test_<mod>.c` (Unity).
4. (Proto, optional) add `proto/vfhe/<mod>/<name>/v1/<name>.proto` with `package
   vfhe.<mod>.<name>.v1` (buf convention: the file's path must equal its package
   path). Each `proto/` dir has a `buf.yaml` as the buf module root. Bindings are
   generated to `_vfhe_proto.vfhe.<mod>.<name>.v1.<name>_pb2`.

`setup.py`, `native/build_ffi.py`, `scripts/gen_proto.py`, and `scripts/run_c_tests.py`
all auto-discover modules (recursively). Python-facing modules list their import
paths in `pyrightconfig.json` (editor); tests resolve via `modules/conftest.py`,
which globs the module src dirs — no per-module edits.

### Architecture-specific sources & SIMD

The default build is **portable** and runs on any x86-64 or arm64 CPU:
`build_ffi.py` compiles with `PORTABLE_BUILD` (the engine's scalar paths) and
links BLAKE3's portable core. `c/src` is scanned **recursively** and compiles
both C and hand-written assembly (`.S`), so vendored trees drop straight in;
sources under `c/src/arch/<name>/` build **only** on a matching machine (`x86_64`
aliases: `x86_64/amd64/x86-64/x64`; `arm64` aliases: `arm64/aarch64`).

Fast paths are **opt-in** via `VFHE_SIMD=1` (x86-64 only): it enables the arith
AVX-512 IFMA engine, AES-NI in `rng`, and BLAKE3's SIMD kernels. This yields an
**AVX-512-only** binary — do not ship it as the default wheel. The engine selects
SIMD via compile-time macros in the sources (not `arch/` dirs); the `arch/`
mechanism remains for any future hand-written per-arch kernels.

Do **not** use `-march=native` in a distributed wheel — it targets the *build*
machine's CPU and will crash elsewhere.
