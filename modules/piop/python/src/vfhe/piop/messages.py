# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Oracle messages: how a table the prover sends as an oracle travels.

In the ideal model an oracle message is the table itself: the prover writes
its entries to the transcript and both parties name it through a plain
`TableOracle`. A commitment scheme replaces that message by a commitment
and the evaluation claims on it by its evaluation protocol; a scheme
supplies that variant by overriding `send` and `receive`. The protocols that
send tables take one of these objects and are otherwise the same in both
models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vfhe.arith import MLE, FieldVector
from vfhe.arith.mle import vector_table
from vfhe.crypto import hash_bytes

from .oracle import Oracle, TableOracle
from .protocol import Rejection

if TYPE_CHECKING:
    from .party import Prover, Verifier


def entries_message(table: MLE):
    """Return the message that carries the entries of `table`: its
    `FieldVector` for a field table, which a transcript digests with one
    hash, and a tuple of its entries otherwise. `entries_table` reads it
    back."""
    return table.table if vector_table(table) else tuple(table.table)


def entries_table(message, variables: list, field: Any) -> MLE | None:
    """Read a table over `field` with `variables` from a message that
    `entries_message` wrote. Returns None when the message is not one:
    a vector over another field, or a count of entries other than
    ``2^len(variables)``."""
    count = 1 << len(variables)
    if isinstance(message, FieldVector):
        if message.field != field or len(message) != count:
            return None
        return MLE(field=field, variables=variables, evaluations=message)
    if not isinstance(message, tuple | list) or len(message) != count:
        return None
    return MLE(field=field, variables=variables, evaluations=list(message))


class OracleMessages:
    """The ideal model: the table's entries are the message.

    `send` and `receive` are the two halves of one message under one
    transcript label. Each party gets its own oracle: the prover's names its
    table, the verifier's a table built from the entries it read, over the
    variables it passes (`MLE_Variable`s compare by identity, so each party
    uses its own).
    """

    def digest(self) -> bytes:
        """What a transcript absorbs for the way tables travel: the ideal
        model has no parameters."""
        return hash_bytes(b"oracle-messages")

    def send(self, prover: Prover, label: str, table: MLE) -> Oracle:
        """Write `table` under `label` and return the prover's oracle to it."""
        prover.iop.transcript.write(label, entries_message(table))
        return TableOracle(table)

    async def receive(
        self, verifier: Verifier, label: str, variables: list, field: Any
    ) -> Oracle:
        """Read the table sent under `label`, a table over `field` with
        `variables`, and return the verifier's oracle to it."""
        table = entries_table(
            await verifier.iop.transcript.read(label), variables, field
        )
        if table is None:
            raise Rejection(
                f"{label}: expected {1 << len(variables)} entries over the field"
            )
        return TableOracle(table)
