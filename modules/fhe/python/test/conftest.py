# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Seeding hooks for the bootstrap tests."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import pytest
from vfhe.crypto import entropy

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@pytest.fixture
def deterministic_prng() -> Iterator[Callable[[int], None]]:
    """Pin the C PRNG to a fixed seed, so a bootstrap test cannot flake.

    These tests assert exact decryption over keys and noise drawn from the
    hardware-seeded PRNG, which carries a small but nonzero failure
    probability. The seed is dropped afterwards, so every other test keeps
    using hardware entropy.
    """
    with contextlib.ExitStack() as stack:

        def pin(seed: int) -> None:
            stack.enter_context(entropy.deterministic(seed))

        yield pin
