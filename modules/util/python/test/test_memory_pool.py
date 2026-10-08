# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The memory pool's controls."""

import os
import subprocess
import sys

import pytest
import vfhe.engine as engine
from vfhe.dynamic_extensions._reload import reinit_memory_pool_settings
from vfhe.engine import memory_pool


@pytest.fixture(autouse=True)
def restore_the_default():
    yield
    memory_pool.set_capacity()
    memory_pool.set_wipe_on_release()


def test_the_capacity_is_set_and_restored():
    default = memory_pool.capacity()
    memory_pool.set_capacity(1 << 20)
    assert memory_pool.capacity() == 1 << 20
    assert memory_pool.statistics().retention_limit_bytes == 1 << 20
    memory_pool.set_capacity("auto")
    assert memory_pool.capacity() == "auto"
    memory_pool.set_capacity(0)
    assert memory_pool.capacity() == 0
    memory_pool.set_capacity()
    assert memory_pool.capacity() == default


@pytest.mark.parametrize("bad", [-1, "half", 2**64, 1.5])
def test_a_bad_capacity_is_refused(bad):
    with pytest.raises(ValueError, match="number of bytes"):
        memory_pool.set_capacity(bad)


def test_release_all_empties_the_pool():
    memory_pool.release_all()
    assert memory_pool.statistics().retained_bytes == 0


@pytest.mark.parametrize(
    ("variable", "value", "query", "expected"),
    [
        ("VFHE_MEMPOOL_CAPACITY", "12345", "capacity", "12345"),
        ("VFHE_MEMPOOL_CAPACITY", None, "capacity", "auto"),
        ("VFHE_MEMPOOL_WIPE_ON_RELEASE", "1", "wipe_on_release", "True"),
        ("VFHE_MEMPOOL_WIPE_ON_RELEASE", None, "wipe_on_release", "False"),
    ],
)
def test_the_environment_sets_the_default(variable, value, query, expected):
    # The child picks its own engine: under an emulator it runs on the bare CPU.
    env = {k: v for k, v in os.environ.items() if k not in ("VFHE_ENGINE", variable)}
    if value is not None:
        env[variable] = value
    out = subprocess.run(  # noqa: S603 - the command line is this test's own constants
        [
            sys.executable,
            "-W",
            "ignore",
            "-c",
            f"from vfhe.engine import memory_pool; print(memory_pool.{query}())",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == expected


def test_a_reloaded_library_gets_the_capacity():
    # A module loaded by vfhe.dynamic_extensions has its own pool; the
    # reinitializer is what carries the setting over. Here it is handed the
    # library already loaded, reset behind its back.
    memory_pool.set_capacity(12345)
    memory_pool.set_wipe_on_release(True)
    engine.lib.mempool_set_capacity(0)
    engine.lib.mempool_set_wipe_on_release(0)
    reinit_memory_pool_settings(engine.ffi, engine.lib)
    assert memory_pool.capacity() == 12345
    assert memory_pool.wipe_on_release()


def test_wiping_is_switched_and_restored():
    default = memory_pool.wipe_on_release()
    memory_pool.set_wipe_on_release(True)
    assert memory_pool.wipe_on_release()
    memory_pool.set_wipe_on_release(False)
    assert not memory_pool.wipe_on_release()
    memory_pool.set_wipe_on_release()
    assert memory_pool.wipe_on_release() == default
