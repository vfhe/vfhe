<!-- SPDX-FileCopyrightText: 2026 The vFHE Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# vfhe.vfhe

Top-level assembly that ties the modules into vFHE (the package root).

- `computation`: the computation a trace recorded, read into its steps
  in vfhe's own terms (a multiplication's tensor product, key switch and
  rescales) with the recorded values.
- `witness`: the witness of a computation under an arithmetization: the
  named values with their roles and bounds, the relations between them,
  and `check`, the ideal decider.
- `cccfgs26`: the arithmetization of [CCCFGS26, App. A.5], the paper's
  names and relations.
