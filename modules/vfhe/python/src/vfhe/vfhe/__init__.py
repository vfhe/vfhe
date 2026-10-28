# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Ties the modules into vFHE: what an arithmetization reads from the
execution tape.

- `computation` reads a trace into its steps in vfhe's own terms.
- `witness` applies an arithmetization to it: the named values, their
  roles and bounds, and the relations between them.
- `cccfgs26` is the arithmetization of [CCCFGS26, App. A.5].

Importing `vfhe.vfhe` imports none of them; import the one you use:

>>> from vfhe.vfhe import cccfgs26, witness
"""
