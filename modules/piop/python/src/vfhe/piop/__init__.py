# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The interactive-oracle-proof framework: an `IOP` runs a `Protocol` on a `Statement` of a `Relation` between a `Prover` and a `Verifier`, the messages on a `Transcript`; `FS_Verifier` is the Fiat-Shamir transform and `Proof` its output.

A statement names oracles (`TableOracle`, `CommittedOracle`, `ImplicitOracle`,
`VirtualOracle`), never tables: the prover holds the tables in its
``witnesses``. The relations are in `vfhe.piop.relations` and the protocols in
`vfhe.piop.protocols`; `IOP.register` binds a protocol to the relation it
reduces. The polynomials are `vfhe.arith`'s `MLE` and `SparseMLE`. Under
development: the API changes between releases.
"""

from .composer import Composed, compose
from .digest import element_digest
from .evaluation import Relation_Eval, with_start
from .fs import FS_Verifier
from .iop import IOP
from .oracle import CommittedOracle, ImplicitOracle, Oracle, OracleKind, TableOracle
from .party import Party, Prover, Verifier
from .protocol import Protocol, Rejection, exceptional_set_size
from .relation import Relation, Statement, statement_oracles
from .transcript import Proof, Transcript, Value, Variable
from .virtual import VirtualOracle

__all__ = [
    "IOP",
    "CommittedOracle",
    "Composed",
    "FS_Verifier",
    "ImplicitOracle",
    "Oracle",
    "OracleKind",
    "Party",
    "Proof",
    "Protocol",
    "Prover",
    "Rejection",
    "Relation",
    "Relation_Eval",
    "Statement",
    "TableOracle",
    "Transcript",
    "Value",
    "Variable",
    "Verifier",
    "VirtualOracle",
    "compose",
    "element_digest",
    "exceptional_set_size",
    "statement_oracles",
    "with_start",
]
