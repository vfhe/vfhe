# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Seeding hooks for the bootstrap tests.

The library exports PRNG seeding only a test should call. Declaring it in a
.cdef would advertise it in every installed engine's public ABI, so this binds
it in cffi's ABI mode from a declaration owned here.
"""

from __future__ import annotations

import importlib.util
from functools import cache
from typing import Protocol, cast

import pytest

SEEDING_CDEF = """
void vfhe_prng_set_deterministic_seed(uint64_t seed);
void vfhe_prng_clear_deterministic_seed(void);
"""


class Seeding(Protocol):
    """The same two symbols as SEEDING_CDEF, for the type checker."""

    def vfhe_prng_set_deterministic_seed(self, seed: int) -> None: ...
    def vfhe_prng_clear_deterministic_seed(self) -> None: ...


@cache
def _seeding() -> Seeding:
    from cffi import FFI
    from vfhe.engine import active_engine

    spec = importlib.util.find_spec(f"_vfhe_native_{active_engine()}")
    if spec is None or spec.origin is None:
        raise RuntimeError(f"no extension found for engine {active_engine()!r}")
    ffi = FFI()
    ffi.cdef(SEEDING_CDEF)
    return cast("Seeding", ffi.dlopen(spec.origin))


@pytest.fixture
def deterministic_prng():
    """Pin the C PRNG to a fixed seed, so a bootstrap test cannot flake.

    These tests assert exact decryption over keys and noise drawn from the
    hardware-seeded PRNG, which carries a small but nonzero failure
    probability. The seed is dropped afterwards, so every other test keeps
    using hardware entropy.
    """
    lib = _seeding()
    lib.vfhe_prng_clear_deterministic_seed()
    yield lib.vfhe_prng_set_deterministic_seed
    lib.vfhe_prng_clear_deterministic_seed()
