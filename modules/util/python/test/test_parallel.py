# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The library-wide thread limit."""

import os
import subprocess
import sys

import pytest
from vfhe.util.bindings import lib


@pytest.fixture(autouse=True)
def restore_the_default():
    yield
    lib.vfhe_set_num_threads(0)


def test_the_limit_is_set_and_restored() -> None:
    default = lib.vfhe_num_threads()
    assert default >= 1
    lib.vfhe_set_num_threads(3)
    assert lib.vfhe_num_threads() == 3
    lib.vfhe_set_num_threads(0)
    assert lib.vfhe_num_threads() == default


def test_the_environment_sets_the_default() -> None:
    # The child picks its own engine: under an emulator it runs on the bare CPU.
    env = {
        **{k: v for k, v in os.environ.items() if k != "VFHE_ENGINE"},
        "VFHE_NUM_THREADS": "2",
    }
    out = subprocess.run(
        [
            sys.executable,
            "-W",
            "ignore",
            "-c",
            "from vfhe.util.bindings import lib; print(lib.vfhe_num_threads())",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "2"
