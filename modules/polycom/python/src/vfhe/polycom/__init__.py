# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
# vfhe.polycom public API re-exports.
from .basefold import (
    Basefold,
    BasefoldCommitment,
    BasefoldEval,
    BasefoldOpening,
    pair_digest,
)
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
from .field_code import FieldFoldableRS
from .queries import query_positions

__all__ = [
    "DEFAULT_SEED",
    "DISTANCE_BOUNDS",
    "INSTANTIATIONS",
    "Basefold",
    "BasefoldCommitment",
    "BasefoldEval",
    "BasefoldOpening",
    "FieldFoldableRS",
    "FoldableRS",
    "batch_inverse",
    "bit_reverse",
    "bit_reverse_permutation",
    "foldable_relative_distance",
    "pair_digest",
    "query_positions",
]
