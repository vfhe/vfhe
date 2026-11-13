<!-- SPDX-FileCopyrightText: 2026 The vfhe authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# vfhe.vfhe

Top-level assembly that ties the modules into vfhe (the package root): what
needs every module at once.

## Layout

- `python/src/vfhe/vfhe/`: what an arithmetization reads from the tape.
  - `computation`: the computation a trace recorded, read into its steps
    in vfhe's own terms (multiplications, each a tensor product, a key
    switch and its rescales; tensor products and key switches outside a
    multiplication; rescales with the value each consumed, through the lift
    of a rational rescale; plaintext-linear sums) with the recorded values,
    each step with its layer and number of primes.
  - `witness`: the witness of a computation under an arithmetization: the
    named values with their roles and bounds, the relations between them,
    `check`, the ideal decider, and the range classes of the bounded values
    with their signed coefficients.
  - `cccfgs26`: the arithmetization of [CCCFGS26, App. A.5], the paper's
    names and relations.
- `python/src/vfhe/benchmark/`: `python -m vfhe.benchmark`, which times every
  module's `bench` package with pytest-benchmark on the installed library.

## Dependencies

Every other module.
