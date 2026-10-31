# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
# vfhe.polycom public API re-exports.
from .basefold import (
    Basefold,
    BasefoldCommitment,
    BasefoldEval,
    BasefoldMessages,
    BasefoldOpening,
    pair_digest,
)
from .batchfold import BatchFoldEval
from .code import (
    DEFAULT_SEED,
    DISTANCE_BOUNDS,
    INSTANTIATIONS,
    FoldableRS,
    batch_inverse,
    bit_reverse,
    bit_reverse_permutation,
    foldable_relative_distance,
)
from .field_code import ROOT_FIELDS, TWIST_FIELDS, FieldFoldableRS
from .parameters import BasefoldParameters, basefold_parameters
from .queries import query_positions

__all__ = [
    "DEFAULT_SEED",
    "DISTANCE_BOUNDS",
    "INSTANTIATIONS",
    "ROOT_FIELDS",
    "TWIST_FIELDS",
    "Basefold",
    "BasefoldCommitment",
    "BasefoldEval",
    "BasefoldMessages",
    "BasefoldOpening",
    "BasefoldParameters",
    "BatchFoldEval",
    "FieldFoldableRS",
    "FoldableRS",
    "basefold_parameters",
    "batch_inverse",
    "bit_reverse",
    "bit_reverse_permutation",
    "foldable_relative_distance",
    "pair_digest",
    "query_positions",
]
