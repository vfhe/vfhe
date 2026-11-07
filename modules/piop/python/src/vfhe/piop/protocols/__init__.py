# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The protocols (piop.md section 5): the reductions registered against the
relations of the toolbox. The sumcheck protocols are a subpackage,
`sumcheck`, with their round-message kernels (`sumcheck.rounds`) and the
prover's forms of a virtual oracle (`sumcheck.prover_view`)."""

from .gkr import GKR, GKR_Start, OutputTest
from .logup import LogUpGKR
from .lookup import Lookup
from .range import RangeDecomposition, ResidueRangeDecomposition
from .sparse import dense_layering
from .sumcheck import Sumcheck, SumcheckProd
from .virtual import SumDefStart, VirtualEval

__all__ = [
    "GKR",
    "GKR_Start",
    "LogUpGKR",
    "Lookup",
    "OutputTest",
    "RangeDecomposition",
    "ResidueRangeDecomposition",
    "SumDefStart",
    "Sumcheck",
    "SumcheckProd",
    "VirtualEval",
    "dense_layering",
]
