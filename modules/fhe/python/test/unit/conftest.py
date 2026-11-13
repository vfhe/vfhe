# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The seeded source the bootstrap tests build their schemes over."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from vfhe.crypto.random import Random

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def source() -> Callable[[int], Random]:
    """A `Random` from an integer seed, so a bootstrap test cannot flake.

    These tests assert exact decryption, which fails with a small probability
    over the keys and noise; a scheme built over a seeded source draws the
    same ones every run (seeds chosen to decrypt cleanly).
    """
    return lambda seed: Random(seed.to_bytes(8, "little"))
