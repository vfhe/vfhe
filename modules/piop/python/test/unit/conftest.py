# SPDX-FileCopyrightText: 2026 The vfhe authors
# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the piop tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from vfhe.arith import FieldVector

if TYPE_CHECKING:
    from collections.abc import Callable


def _bump_first_entry(value, one):
    """Return `value`, a table message, with `one` added to its first entry:
    a changed copy of a `FieldVector`, or a new tuple."""
    if isinstance(value, FieldVector):
        out = value.copy()
        out[0] = out[0] + one
        return out
    return (value[0] + one, *value[1:])


@pytest.fixture
def bump_first_entry() -> Callable:
    """The tamper that adds `one` to the first entry of a table message:
    ``bump_first_entry(value, one)``."""
    return _bump_first_entry
