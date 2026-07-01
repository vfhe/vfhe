# vfhe

Verifiable Fully Homomorphic Encryption library.

---

## Repository structure

```
modules/<mod>/                     one folder per module (arith, crypto, circuit, …)
  python/src/vfhe/<mod>/           Python package source (physical namespace layout)
    __init__.py                    THIN: re-exports the module's public API
    <impl>.py                      implementation; `from _vfhe_native import ffi, lib`
  python/test.                     pytest suite
  c/include/<mod>.h                kernel header
  c/src/<mod>.c                    pure C kernel (no Python headers)
  c/cdef/<mod>.cdef                OPTIONAL explicit cffi decls (if header isn't cffi-clean)
  c/test                           Unity C tests
  proto                            OPTIONAL protobuf schema for this module

native/build_ffi.py                cffi build: ALL c/src kernels -> one ext `_vfhe_native`
scripts/gen_proto.py               protoc: all proto schemas -> `_vfhe_proto` package
scripts/run_c_tests.py             compiles + runs the Unity C tests
external/unity/                    Unity C test framework (git submodule)
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
- `git` (the Unity test framework is a submodule)

---

## 1. One-time setup

```bash
git clone https://github.com/vfhe/vfhe vfhe && cd vfhe
git submodule update --init external/unity      # fetch the Unity C test framework

python3 -m venv .venv && source .venv/bin/activate
make deps                                        # install dev dependencies (see below)
```

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

This is the current end goal. The wheel is built for the **current platform /
interpreter** via the standard PEP 517 frontend (`python -m build`). It contains:
the `vfhe.*` packages, the compiled `_vfhe_native` extension, and the freshly
generated `_vfhe_proto` bindings.

---

## Adding a new module

1. Create `modules/<mod>/python/src/vfhe/<mod>/__init__.py` (+ impl `.py`).
2. (C) add `c/src/<mod>.c` + `c/include/<mod>.h`; add a `c/cdef/<mod>.cdef` only
   if the header isn't cffi-clean (has macros / `static inline` / `__attribute__`).
3. (C tests) add `c/test/test_<mod>.c` (Unity).
4. (Proto) add `proto/vfhe/<mod>/<name>/v1/<name>.proto` with `package
   vfhe.<mod>.<name>.v1` (buf convention: the file's path must equal its package
   path). Each `proto/` dir has a `buf.yaml` as the buf module root. Bindings are
   generated to `_vfhe_proto.vfhe.<mod>.<name>.v1.<name>_pb2`.

`setup.py`, `native/build_ffi.py`, `scripts/gen_proto.py`, and `scripts/run_c_tests.py`
all auto-discover modules (recursively). The import paths for every template
module are already listed in `pyrightconfig.json` (editor); tests resolve via
`modules/conftest.py`, which globs the module src dirs — no per-module edits.

### Assembly & architecture-specific sources

`c/src` is scanned **recursively** and compiles both C and hand-written assembly
(`.S`), so vendored/third-party trees drop straight in. Architecture-specific
sources go under `c/src/arch/<name>/` and are compiled **only** on a matching
machine (`x86_64` aliases: `x86_64/amd64/x86-64/x64`; `arm64` aliases:
`arm64/aarch64`). Anything outside an `arch/` directory always builds — keep the
portable baseline there and use runtime CPU dispatch for SIMD.
Do **not** use `-march=native` in a wheel — it targets the *build* machine's CPU
and will crash elsewhere. See `modules/arith/c/src/arch/*/mul64.S` for a live
example (`vfhe.arith.mul64`).
