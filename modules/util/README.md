<!-- SPDX-FileCopyrightText: 2026 The vFHE Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# vfhe.util

The runtime beneath every vfhe module: which engine runs, how Python reaches C,
and the C substrate every kernel calls.

- `python/src/vfhe/util/bindings/`: loads the engine and exports its `ffi` and
  `lib`; every other Python module reaches C through
  `from vfhe.util.bindings import ffi, lib`. The import system runs it once, so
  a process holds one engine.
- `python/src/vfhe/util/kernels/`: compiles the caller's C together with vfhe's
  own into one engine, LTO on, so the compiler inlines across the boundary. The
  engine it builds is the one the process then loads, so `ExtensionBuilder.build()` runs before
  `bindings` is imported. `import vfhe.util` leaves it alone, so only a caller
  who builds pays for cffi and a compiler.
- `python/src/vfhe/util/_engine/`: the engines this build knows, as an enum.
  `select()` returns the pinned one (`VFHE_ENGINE`) or the fastest installed one
  this CPU runs; `active()` returns the one loaded.
- `python/src/vfhe/util/_cpu/`: facts about this machine, in pure Python.
  `is_arm`, `is_x86_64` and `has_flag` say what instructions run here, a
  correctness gate whose wrong answer is a crash, so `has_flag` answers `None`
  when it cannot judge. `last_level_cache_bytes` says how big the last level of
  cache is, a tuning hint whose wrong answer only costs speed, so `None` keeps a
  working set whole.
- `python/src/vfhe/info/`: `python -m vfhe.info`, a JSON report of the install
  for bug reports.
- `python/cdef/parallel.h`: what Python may call of the C below, the thread
  limit.
- `c/include/`, `c/src/`: allocation that exits on failure (`alloc`), the thread
  limit and the parallel loop (`parallel`), and the `VFHE_HAVE_*` macros for
  x86-64 vector extensions (`x86_64.h`). Randomness lives in `crypto`, index and
  modulus helpers in `arith`.

`util` depends on nothing above it: its headers include only libc. Anything
needing an `arith` type belongs in `arith`.

## Environment

| variable | read by | meaning |
|---|---|---|
| `VFHE_ENGINE` | `_engine`, on selection | the engine to load, by name; honoured without asking the CPU, so an emulator can run one the host lacks |
| `VFHE_NUM_THREADS` | `parallel.c`, on first use | the thread limit when a positive integer; otherwise 1 |
| `VFHE_BUILD_DIR` | `kernels` | the build cache; default `$XDG_CACHE_HOME/vfhe`, else `~/.cache/vfhe` |
| `XDG_CACHE_HOME` | `kernels` | where the default build cache goes |
| `CC` | `kernels` | the compiler; default the one Python was built with, else `cc` |
| `PYTHONPATH` | Python | a directory `python -m vfhe.util.kernels` printed, first, so `bindings` imports that engine |
