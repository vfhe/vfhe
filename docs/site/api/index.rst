.. SPDX-FileCopyrightText: 2026 The vFHE Authors
.. SPDX-License-Identifier: Apache-2.0

API reference
=============

.. include:: ../../../README.md
   :parser: myst_parser.sphinx_
   :start-after: ## Modules
   :end-before: ## Development

Every module's Python package is documented here from its docstrings, its
public API being what ``vfhe.<module>`` exports. The C headers of ``crypto``
and ``util`` are documented from their Doxygen blocks; the other modules'
headers join as their blocks reach that standard.

.. toctree::
   :maxdepth: 1

   arith/index
   circuit/index
   crypto/index
   fhe/index
   mlwe/index
   piop/index
   polycom/index
   util/index
   vfhe/index
