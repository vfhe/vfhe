# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Process-global state an implementation keeps, and how to empty it.

Some implementations cache native objects for the lifetime of the process --
shared bases, plans, precomputed tables. A test wanting the caches empty so it
does not inherit another test's should not have to know which implementation
holds them or in which module it lives, so an implementation registers a
handler and callers ask for the effect. Registering is idempotent per function,
so importing a module twice does not run its handler twice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

_RESETS: list[Callable[[], None]] = []


def register_reset(fn: Callable[[], None]) -> Callable[[], None]:
    """Register `fn` to empty one implementation's caches.

    The object holding them stays; only its contents go, so references taken
    before the reset stay valid.
    """
    if fn not in _RESETS:
        _RESETS.append(fn)
    return fn


def reset() -> None:
    """Empty every registered implementation's caches."""
    for handler in _RESETS:
        handler()
