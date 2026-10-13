# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Gives every test under modules/ empty arithmetic caches."""


def pytest_runtest_setup() -> None:
    """Empties arith's process-global caches before each test.

    Tests share one process and arith caches native objects for its lifetime,
    so an earlier test's registrations would otherwise reach a later one.
    """
    from vfhe.arith import reset_state  # at call time, so collection loads no engine

    reset_state()
