<!-- SPDX-FileCopyrightText: 2026 The vFHE Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# vfhe.vfhe

Top-level assembly that ties the modules into vFHE (the package root): what
needs every module at once.

## Layout

- `python/src/vfhe/benchmark/`: `python -m vfhe.benchmark`, which times every
  module's `bench` package with pytest-benchmark on the installed library.

## Dependencies

Every other module.
