# SPDX-FileCopyrightText: 2026 The vfhe authors
# SPDX-License-Identifier: Apache-2.0
"""Holds the runtime beneath every vfhe module: which engine runs and how Python reaches C.

- `bindings` loads the engine and exports `ffi` and `lib`; every other module
  reaches C through them.
- `kernels` compiles your C into an engine of your own.
- `io` serializes vfhe objects.
- `parallel` sets how many threads vfhe uses.
- `memory_pool` controls the pool that keeps the buffers vfhe releases.
- `trace` records the ring operations vfhe runs, the execution tape.
- `benchmark` is the pytest plugin the benchmarks run under.
- `gpu` loads the CUDA extension, when the install has one.
- `info` prints a report of the install, for bug reports.

Importing `vfhe.util` imports none of them; import the one you use:

>>> from vfhe.util import bindings
"""
