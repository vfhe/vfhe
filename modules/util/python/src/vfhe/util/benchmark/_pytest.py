# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The pytest plugin for the files under a ``bench`` directory: every public function of theirs is a benchmark, run with arith's caches emptied and marked `complete`."""

from __future__ import annotations

import gc
import inspect
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Iterator


def _is_benchmark(path: pathlib.Path) -> bool:
    return "bench" in path.parts


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "complete: a benchmark, or a heavy test; the fast suites leave it out",
    )


def pytest_pycollect_makeitem(
    collector: pytest.Module | pytest.Class, name: str, obj: object
) -> list[pytest.Item] | None:
    """Collects every public function a benchmark file defines, whatever its name."""
    if not (_is_benchmark(collector.path) and inspect.isfunction(obj)):
        return None
    if name.startswith("_") or obj.__module__ != collector.obj.__name__:
        return None
    return list(collector._genfunctions(name, obj))  # noqa: SLF001 - the collection pytest-asyncio and friends use


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Marks every benchmark `complete`."""
    for item in items:
        if _is_benchmark(item.path):
            item.add_marker("complete")


@pytest.fixture(autouse=True)
def benchmark_reset(request: pytest.FixtureRequest) -> Iterator[None]:
    """Empties arith's caches before a benchmark; other tests untouched."""
    if not _is_benchmark(request.path):
        yield
        return
    from vfhe.arith import reset_state

    reset_state()
    gc.collect()
    yield
