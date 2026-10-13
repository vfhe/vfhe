# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Fixtures the util suites share."""

import pathlib

import pytest


@pytest.fixture
def cache(tmp_path, monkeypatch) -> pathlib.Path:
    """A build cache of the test's own, set as ``VFHE_BUILD_DIR``."""
    monkeypatch.setenv("VFHE_BUILD_DIR", str(tmp_path))
    return tmp_path.resolve()
