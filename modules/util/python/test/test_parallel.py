# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The library-wide thread limit."""

import os
import subprocess
import sys

import pytest
import vfhe.engine as engine
from vfhe.dynamic_extensions._reload import reinit_thread_limit


@pytest.fixture(autouse=True)
def restore_the_default():
    yield
    engine.set_num_threads()


def test_the_limit_is_set_and_restored():
    default = engine.num_threads()
    assert default >= 1
    engine.set_num_threads(3)
    assert engine.num_threads() == 3
    engine.set_num_threads()
    assert engine.num_threads() == default
    with pytest.raises(ValueError, match="at least 1"):
        engine.set_num_threads(0)


def test_the_environment_sets_the_default():
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
            "import vfhe.engine as e; print(e.num_threads())",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "2"


def test_a_reloaded_library_gets_the_limit():
    # A module loaded by vfhe.dynamic_extensions holds its own copy of the
    # limit; the reinitializer is what carries it over. Here it is handed the
    # library already loaded, reset behind its back.
    engine.set_num_threads(3)
    engine.lib.vfhe_set_num_threads(0)
    reinit_thread_limit(engine.ffi, engine.lib)
    assert engine.num_threads() == 3
