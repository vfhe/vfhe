# vfhe

Verifiable Fully Homomorphic Encryption library — C compute kernels exposed to
Python through a single CFFI extension.

---

## Modules

Each module is a self-contained folder under `modules/`. A module is
**Python-facing** (ships a `python/` package + a `cdef` exposing its C to
Python) or **internal C-only** (contributes compiled kernels used by other
modules, no Python symbols).

| Module | Kind | What it provides |
|---|---|---|
| `arith` | Python-facing | RNS polynomial arithmetic over `Z_q[X]/(X^N+1)`: incomplete NTT, `Ring`/`Polynomial`, CKKS complex encoding, multiprecision, number theory |
| `misc` | Python-facing | The native handle (`ffi`/`lib`/`libvfhe`) plus the internal C utilities: BLAKE3-seeded PRNG, AES-CTR RNG, aligned allocation, mod-switching helpers |
| `mlwe` | Python-facing | LWE / Module-LWE and MGSW: key generation, encryption, key-switching, automorphisms, external product |
| `fhe` | Python-facing | Schemes on top of `mlwe`: CKKS (encode/encrypt/rescale/rotate/multiply), CGGI16 functional bootstrap, GP25 sparse-amortized bootstrap |
| `piop` | Python-facing | Multilinear extensions (dense/sparse/`ML_Polynomial`) and the IOP prover/verifier scaffolding |
| `circuit` | Python-facing | Layered GKR arithmetic circuits (protobuf wire format) and their polynomial export to `arith` |
| `compiler`, `polycom`, `snark`, `vfhe` | placeholder | reserved for the compiler frontend, polynomial commitments, the SNARK layer, and the top-level assembly |

```
modules/<mod>/                     one folder per module
  python/src/vfhe/<mod>/           Python package source (physical namespace layout)
    __init__.py                    THIN: re-exports the module's public API
    <impl>.py                      implementation; talks to C via `from vfhe.misc.libvfhe import ffi, lib`
  python/cdef/<mod>.cdef           Python-facing ABI: hand-written cffi decls (opaque handles)
  python/test/                     pytest suite (public-API characterization tests)
  c/include/                       public C headers (umbrella <mod>.h)
  c/src/                           pure C kernels
  c/test/                          C unit tests (plain assert/main() or Unity)
  c/fuzz/                          OPTIONAL libFuzzer harnesses (nightly CI)
  proto/                           OPTIONAL protobuf schema for this module

native/build_ffi.py                cffi build: ALL c/src kernels -> one ext `_vfhe_native`
native/discovery.py                shared source/include/arch discovery (build + tests)
scripts/gen_proto.py               protoc: all proto schemas -> `_vfhe_proto` package
scripts/run_c_tests.py             compiles + runs the C tests (VFHE_SANITIZE=…)
scripts/build_fuzzers.py           builds/runs any c/fuzz libFuzzer harnesses
scripts/check_simd_build.py        compile-checks the x86 AVX-512 engine paths
scripts/smoke.py                   end-to-end CKKS smoke / integration test (CI, run vs the sdist)
.github/workflows/                 CI, nightly, release, and security workflows
external/unity/                    Unity C test framework (git submodule, test-only)
external/blake3/                   BLAKE3 hash (git submodule; c/ compiled into the ext)
setup.py / pyproject.toml          package assembly + build config
pyrightconfig.json                 editor import paths (Pyright / Pylance)
stubs/_vfhe_native.pyi             typing boundary for the cffi ext (Any)
.generated/                        ALL build artifacts (gitignored)
```

### The native boundary

Every module's `c/src` is compiled into **one** LTO'd extension,
`_vfhe_native`, so kernels inline across module boundaries. Each Python-facing
module hand-writes `python/cdef/<mod>.cdef` declaring the C ABI it calls —
handles are passed as opaque `void *`, with a few structs cdef'd so Python can
read fields. The Python wrappers reach C through `from vfhe.misc.libvfhe import
ffi, lib`.

### Generated files

Everything generated goes into `.generated/`, which is gitignored in full:

| Artifact | Produced by | Purpose |
|---|---|---|
| `_vfhe_native.*.so` | `native/build_ffi.py` (cffi) | the compiled C kernels, one LTO'd extension |
| `_vfhe_proto` | `scripts/gen_proto.py` (protoc) | protobuf bindings |

---

## Installing

```bash
pip install vfhe
```

vfhe ships **as an sdist only — no pre-built wheels** — so every install compiles
from source and **tunes to your CPU**. On an x86 machine with AVX-512 IFMA this
enables `-march=native`, the SIMD kernels, AES-NI, and BLAKE3 SIMD; on any other
CPU it builds the portable engine (and says so). The result targets **this**
machine. You need a C compiler (clang/gcc); the sdist bundles the BLAKE3 sources,
so no submodules are required.

> **Choosing the compiler.** The build uses your interpreter's compiler by
> default (no configuration needed). To pick a specific one — e.g. if you have
> several installed — set the standard `CC` environment variable; it steers both
> the CPU detection and the compile, in lockstep:
>
> ```bash
> CC=gcc-14 pip install vfhe
> ```

> **Forcing a portable build.** If you build in one place and run in another
> (Docker image built on a big CI box, run on a smaller node), set
> `VFHE_PORTABLE=1` to skip CPU detection and build the portable engine that runs
> on any CPU — `-march=native` would otherwise bake in the *build* host's
> features and crash elsewhere. When the portable engine ends up on a CPU that
> *does* support AVX-512 IFMA, vfhe prints a one-time hint to rebuild tuned
> (silence it with the usual `warnings` filters).

---

## Prerequisites (development)

- Python **3.10+**
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
make deps                                        # install dev dependencies
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

- **Python** — `pytest` has two modes. The default is **fast** (quick tests).
  Heavy end-to-end FHE bootstraps are marked `@pytest.mark.complete` and run only
  with **`pytest --complete`**. CI runs the fast suite first as a gate, then the
  complete tests only if it passes.
- **C** — `scripts/run_c_tests.py` compiles every `modules/*/c/test/*.c` against
  the portable engine and runs it (a non-zero exit fails the suite). Tests are
  Unity suites; the harness links Unity only for files that include it, so a
  plain `assert`/`main()` program works too.
- **Smoke / integration** — `scripts/smoke.py` is an end-to-end CKKS run
  (encrypt two vectors, add and multiply under encryption, decrypt, and check
  the results against plaintext). It runs standalone (`make smoke`), as a pytest
  (`modules/fhe/python/test/test_smoke.py`), and in CI against the sdist install
  — the single check that the whole stack computes, not just imports.

**Coverage.** The CI `coverage` job measures *both* languages in one run — no
external service — from a gcov-instrumented build (`VFHE_COVERAGE=1 make build`)
exercised by the **complete** suite (so the FHE-bootstrap C kernels are counted):
Python via `coverage.py` (`pytest --complete --cov=vfhe`) and C via `gcovr` on
the resulting `.gcda`. `-O0` + gcov makes it slow (a few minutes) — deliberately,
to gate merges. It renders a summary on the run page, uploads the reports as
artifacts, and **fails if either drops below a floor**. The floors are
intentionally low for now and set by the `COV_MIN_PY` / `COV_MIN_C_LINE` /
`COV_MIN_C_BRANCH` workflow env vars (override per-repo via Actions *Variables* —
no code edit).

Extra, deeper checks:

```bash
VFHE_SANITIZE=address,undefined python scripts/run_c_tests.py   # ASan + UBSan
python scripts/build_fuzzers.py --run --time 60                 # fuzz c/fuzz/* (needs clang)
PYTHONPATH=scripts python scripts/check_simd_build.py           # x86 AVX-512 compile-check
```

Optional libFuzzer harnesses live in `modules/<mod>/c/fuzz/` (built with
`-fsanitize=fuzzer,address,undefined`); libFuzzer needs a Linux/clang
toolchain, so they run in nightly CI rather than on macOS.

---

## 3. Code & library generation

Nothing generated is committed — regenerate locally as needed:

| Run | When |
|---|---|
| `make proto` | after editing a `.proto` |
| `make build` | after editing C, or to refresh everything (`proto` + native) |

`setup.py` reruns proto generation automatically during a build, so a clean
checkout (or an sdist install) needs nothing pre-generated.

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

## 5. Building the distribution

vfhe releases **as an sdist only** — no wheels — so every install compiles from
source and tunes to the target CPU:

```bash
make sdist        # -> dist/vfhe-<version>.tar.gz
```

The sdist bundles the `vfhe.*` packages, all C/cdef sources, `native/build_ffi.py`,
and the vendored BLAKE3 C sources (so no submodule is needed to build it). Proto
bindings are regenerated during the build. Publishing is handled by the release
workflow in §6.

---

## 6. Continuous integration & releases

Workflows live in `.github/workflows/` (POSIX only — Linux + macOS):

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | push / PR | lint (ruff + pyright + clang-format); fast tests on Linux+macOS × py3.10–3.13 (gate) → complete suite; C + Python coverage (gated, gcov + coverage.py); x86 AVX-512 compile-check; sdist build + install-from-source smoke |
| `nightly.yml` | nightly / manual | C tests under ASan + UBSan, libFuzzer on any `c/fuzz/*` |
| `codeql.yml`, `scorecard.yml`, `dependency-review.yml` | push / PR / schedule | static analysis + supply-chain hygiene (incl. OpenSSF Scorecard) |
| `release.yml` | see below | build the sdist + SBOM, then publish |

**Releases** publish the **sdist** two ways:

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
or **internal C-only** (no `python/`; used by other modules' kernels).

1. (Python, optional) create `modules/<mod>/python/src/vfhe/<mod>/__init__.py`
   (+ impl `.py`). Omit for internal C-only modules.
2. (C, optional) add `c/src/*.c` + `c/include/*.h`. If the module exposes any C
   to Python, hand-write `python/cdef/<mod>.cdef` declaring that ABI (opaque
   handles + the functions Python calls). Internal C-only modules have **no**
   cdef — they contribute compiled code but no Python symbols.
3. (C tests) add `c/test/test_<mod>.c` — a plain `assert`/`main()` program
   (non-zero exit = failure) or a Unity suite.
4. (Proto, optional) add `proto/vfhe/<mod>/<name>/v1/<name>.proto` with `package
   vfhe.<mod>.<name>.v1` (buf convention: the file's path must equal its package
   path). Each `proto/` dir has a `buf.yaml` as the buf module root. Bindings are
   generated to `_vfhe_proto.vfhe.<mod>.<name>.v1.<name>_pb2`.

`setup.py`, `native/build_ffi.py`, `scripts/gen_proto.py`, and `scripts/run_c_tests.py`
all auto-discover modules (recursively). Python-facing modules list their import
paths in `pyrightconfig.json` (editor); tests resolve via `modules/conftest.py`,
which globs the module src dirs — no per-module edits.

### Architecture-specific sources & SIMD

`build_ffi.py` picks the engine by build context:

- **Source builds** (the default — pip compiling the sdist on the target
  machine) **auto-tune to that CPU**: on x86 with AVX-512 IFMA they drop
  `PORTABLE_BUILD` to light up the arith IFMA engine + AES-NI in `misc`, add
  BLAKE3's SIMD kernels, and pass `-march=native`; otherwise they build the
  portable engine. IFMA is detected by asking the compiler what `-march=native`
  enables (`__AVX512IFMA__`), so the gate matches exactly what the kernels are
  guarded on.
- **Forced portable** (`VFHE_PORTABLE=1`) compiles the baseline (`PORTABLE_BUILD`,
  scalar paths, BLAKE3 portable core) and skips detection — for building in one
  place to run on another (Docker/build farms). `VFHE_COVERAGE=1` also implies it.

`c/src` is scanned **recursively** and compiles both C and hand-written assembly
(`.S`), so vendored trees drop straight in; sources under `c/src/arch/<name>/`
build **only** on a matching machine (`x86_64` aliases: `x86_64/amd64/x86-64/x64`;
`arm64` aliases: `arm64/aarch64`). The engine selects SIMD via compile-time
macros in the sources (not `arch/` dirs); the `arch/` mechanism remains for any
future hand-written per-arch kernels.

Because a source build is `-march=native`, it targets **this** machine — never
copy one to a different CPU; set `VFHE_PORTABLE=1` if you must build somewhere
other than where it will run.
