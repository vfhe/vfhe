# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The relation toolbox (piop.md section 4)."""

from vfhe.piop.evaluation import Relation_Eval

from .circuit import (
    Relation_Circuit,
    Relation_CircuitSparseIndex,
    Relation_Equal,
    Relation_Layer,
    wiring_predicates,
)
from .hypercube import Relation_Sum, Relation_SumProd, Relation_Zero
from .lookup import Relation_Lookup
from .range import Relation_Range
from .sparse import FamilyShape, Relation_SpEval, dense_relation
from .sumdef import Relation_SumDef

__all__ = [
    "FamilyShape",
    "Relation_Circuit",
    "Relation_CircuitSparseIndex",
    "Relation_Equal",
    "Relation_Eval",
    "Relation_Layer",
    "Relation_Lookup",
    "Relation_Range",
    "Relation_SpEval",
    "Relation_Sum",
    "Relation_SumDef",
    "Relation_SumProd",
    "Relation_Zero",
    "dense_relation",
    "wiring_predicates",
]
