<!-- SPDX-FileCopyrightText: 2026 Antonio Guimarães -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# vfhe.util

The native boundary, and the substrate every kernel calls. Two Python packages
live here, because the engine handle and the runtime toolchain answer different
questions and almost nobody needs the second.

- `python/src/vfhe/engine/`: the handle to the compiled extension. It re-exports
  `ffi` / `lib` from the loaded `_vfhe_native_<engine>` and a `libvfhe` singleton; every other Python
  module reaches C through `from vfhe.engine import ffi, lib`.
- `python/src/vfhe/dynamic_extensions/`: compiles and reloads user C against the
  installed library. A top-level package rather than a subpackage, so importing
  `vfhe` never pays for cffi and a compiler probe.
- `python/src/vfhe/engine/_native.py`: the import-time picker. It asks the CPU probe
  what this machine can run and imports exactly one engine extension.
- `c/src/cpu_probe.c`: that probe, in its own translation unit with no engine
  dependencies, so asking the question never loads an engine. Built twice: as
  the tiny `_vfhe_cpu` extension, and as the `vfhe-cpu` command the test runner
  uses.
- `c/src/`: what the kernels share — allocation that aborts rather than
  returning NULL (`alloc.c`), index and modulus helpers (`util.c`), and the
  engine's own name (`build_info.c`). Randomness lives in `crypto`.

`util` depends on nothing above it: `util.h` includes only libc and
`vfhe_cpu.h`. Anything needing an `arith` type belongs in `arith` — that is why
`new_ntt_list` and `compute_RNS_Qhat_array` live there, and the LWE sampler in
`mlwe`.

Its cdefs are deliberately small. `engine.cdef` promises one symbol, and hooks
that only a test should call are not among them: a test declares those itself.
