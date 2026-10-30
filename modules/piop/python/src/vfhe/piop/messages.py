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

from vfhe.arith import MLE

from .oracle import Oracle, TableOracle
from .protocol import Rejection

if TYPE_CHECKING:
    from .party import Prover, Verifier


class OracleMessages:
    """The ideal model: the table's entries are the message.

    `send` and `receive` are the two halves of one message under one
    transcript label. Each party gets its own oracle: the prover's names its
    table, the verifier's a table built from the entries it read, over the
    variables it passes (`MLE_Variable`s compare by identity, so each party
    uses its own).
    """

    def send(self, prover: Prover, label: str, table: MLE) -> Oracle:
        """Write `table` under `label` and return the prover's oracle to it."""
        prover.iop.transcript.write(label, tuple(table.table))
        return TableOracle(table)

    async def receive(
        self, verifier: Verifier, label: str, variables: list, field: Any
    ) -> Oracle:
        """Read the table sent under `label`, a table over `field` with
        `variables`, and return the verifier's oracle to it."""
        entries = await verifier.iop.transcript.read(label)
        if len(entries) != 1 << len(variables):
            raise Rejection(
                f"{label}: expected {1 << len(variables)} entries, got {len(entries)}"
            )
        return TableOracle(
            MLE(field=field, variables=variables, evaluations=list(entries))
        )
