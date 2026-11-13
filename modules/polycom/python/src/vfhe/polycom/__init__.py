# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Polynomial commitments for the oracles of `vfhe.piop`: `Basefold` over a foldable Reed-Solomon code, `FoldableRS` over the ring or `FieldFoldableRS` over a field, with the evaluation protocols `BasefoldEval` and `BatchFoldEval`.

``scheme.commit(f)`` returns a `BasefoldCommitment`, the root a statement
carries, and a `BasefoldOpening`, the prover's data. `BasefoldEval`, registered
for `Relation_Eval`, proves an evaluation claim by a sumcheck interleaved with
folds of the codeword, then answers queries at `query_positions`. Fiat-Shamir
comes from ``IOP(fiat_shamir=True)``. Under development: the API changes
between releases.
"""

from .basefold import (
    Basefold,
    BasefoldCommitment,
    BasefoldEval,
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
    "BasefoldOpening",
    "BatchFoldEval",
    "FieldFoldableRS",
    "FoldableRS",
    "batch_inverse",
    "bit_reverse",
    "bit_reverse_permutation",
    "foldable_relative_distance",
    "pair_digest",
    "query_positions",
]
