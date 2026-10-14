# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The library-wide thread limit, and the local one."""

import os
import subprocess
import sys

import pytest
from vfhe.util import parallel
from vfhe.util.bindings import lib


@pytest.fixture(autouse=True)
def restore_the_default():
    yield
    parallel.set_num_threads()


def test_the_limit_is_set_and_restored() -> None:
    default = parallel.num_threads()
    assert default >= 1
    parallel.set_num_threads(3)
    assert parallel.num_threads() == 3
    assert lib.vfhe_num_threads() == 3
    parallel.set_num_threads()
    assert parallel.num_threads() == default
    with pytest.raises(ValueError, match="at least 1"):
        parallel.set_num_threads(0)


def test_a_local_number_of_threads_holds_inside_and_is_restored() -> None:
    assert lib.vfhe_local_num_threads() == 0
    with parallel.local_num_threads(3):
        assert lib.vfhe_local_num_threads() == 3
        with parallel.local_num_threads(5):  # a nested one only lowers
            assert lib.vfhe_local_num_threads() == 3
        with parallel.local_num_threads(2):
            assert lib.vfhe_local_num_threads() == 2
        assert lib.vfhe_local_num_threads() == 3
    assert lib.vfhe_local_num_threads() == 0
    with parallel.local_num_threads(0):
        assert lib.vfhe_local_num_threads() == 0
    with pytest.raises(ValueError, match="at least 0"), parallel.local_num_threads(-1):
        pass


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
            "from vfhe.util import parallel; print(parallel.num_threads())",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "2"
