# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Process-global state an implementation keeps, and how to disturb it.

Some implementations cache native objects for the lifetime of the process --
shared bases, plans, precomputed tables. Two things outside arith need to act
on that state and should not have to know which implementation holds it or in
which module it lives:

- a test that wants the caches empty so it does not inherit another test's,
- the dynamic-extension reloader, after it swaps the native library under a
  running process, since anything holding the retired ``lib`` must be rebuilt.

An implementation registers a handler for each; callers ask for the effect.
Registering is idempotent per function, so importing a module twice does not
run its handler twice.
"""

from __future__ import annotations

from collections.abc import Callable

_RESETS: list[Callable[[], None]] = []
_REBINDS: list[Callable[[], None]] = []


def _add(handlers: list[Callable[[], None]], fn: Callable[[], None]):
    if fn not in handlers:
        handlers.append(fn)
    return fn


def register_reset(fn: Callable[[], None]) -> Callable[[], None]:
    """Register `fn` to empty one implementation's caches.

    The object holding them stays; only its contents go, so references taken
    before the reset stay valid.
    """
    return _add(_RESETS, fn)


def register_rebind(fn: Callable[[], None]) -> Callable[[], None]:
    """Register `fn` to rebuild one implementation's state against the native
    library that is loaded *now*.

    Unlike a reset, this may replace the object itself: state that pins ``lib``
    as an instance attribute cannot be patched in place. Reach such state
    through its accessor rather than binding it at import.
    """
    return _add(_REBINDS, fn)


def reset() -> None:
    """Empty every registered implementation's caches."""
    for handler in _RESETS:
        handler()


def rebind() -> None:
    """Rebuild every registered implementation's state after a library swap."""
    for handler in _REBINDS:
        handler()
