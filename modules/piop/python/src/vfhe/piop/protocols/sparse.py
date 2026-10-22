# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The protocol of a dense-index relation (`relations.sparse`): a two-layer
composition (`composer.py`), the inner relation over restrictions of a fresh
master, then the realization's statement headed at that master."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vfhe.piop.relation import Statement


def dense_layering(root: Statement) -> list[list[Statement]]:
    """The protocol of a dense-index relation as a layered reduction: `L_0`
    the inner relation over the root's instance fields, its sparse index
    fields holding restrictions of a fresh master P, headed at the root's
    head when the root is started (the Composer then starts it with the
    root's claims); `L_1` the realization over `(P, P_dense, shape)`, headed
    at P through its index. Rule 1: P heads `L_1`; rule 2: `L_0` reaches P
    through its index, `L_1` mentions only P_dense beside its head. No
    message, no challenge, soundness error 0. Register
    `compose(dense_layering)` against the dense relation and its `Start`.
    """
    inner_cls = root.relation.inner
    shape = root.index["shape"]
    P = shape.implicit()
    sparse = inner_cls.sparse_index_fields
    rest = {n: root.index[n] for n in inner_cls.index_fields if n not in sparse}
    inner = inner_cls(index={**rest, **shape.place(P)})
    realization = root.index["speval"](
        index={"sparse": P, "dense": root.index["dense"], "shape": shape}
    )
    fields = {n: root.fields[n] for n in inner_cls.fields if n in root.fields}
    return [
        [root.reduce_to(inner, head=root.head, **fields)],
        [root.reduce_to(realization)],
    ]
