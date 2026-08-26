# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"Dev test wiring (pytest loads this before collecting any test under modules/)."

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True  # keep .pyc out of the source tree

try:
    # Imported for its side effect: proving the built extension is importable.
    import _vfhe_native  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT / "build"))  # extensions and archives
    for src in sorted(ROOT.glob("modules/*/python/src")):
        sys.path.insert(0, str(src))


# --- fast / complete test modes -------------------------------------------
# Default is the FAST suite (quick, small parameters). `--complete` additionally
# runs the heavy, thorough tests marked `@pytest.mark.complete` (end-to-end FHE
# bootstraps). This mirrors the original --fast / --complete split.
def pytest_addoption(parser):
    parser.addoption(
        "--complete",
        action="store_true",
        default=False,
        help="run the complete test suite (adds the heavy @complete tests); "
        "default runs only the fast subset",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--complete"):
        return
    skip_complete = pytest.mark.skip(reason="fast run; pass --complete to include")
    for item in items:
        if "complete" in item.keywords:
            item.add_marker(skip_complete)


@pytest.fixture
def deterministic_prng():
    """Pin the C PRNG to a fixed seed for reproducible probabilistic tests.

    The heavy FHE-bootstrap tests assert exact decryption over keys/noise drawn
    from the hardware-seeded C PRNG, which carries a small but nonzero
    decryption-failure probability -> intermittent CI failures. A test that
    takes this fixture calls it with a seed to make its run reproducible; the
    seed is discarded afterwards so every other test keeps using hardware
    entropy. Yields the seeding callable.
    """
    from vfhe.misc.libvfhe import lib

    lib.vfhe_prng_clear_deterministic_seed()  # start from a known (hardware) state
    yield lib.vfhe_prng_set_deterministic_seed
    lib.vfhe_prng_clear_deterministic_seed()


@pytest.fixture(autouse=True)
def _reset_arith_state():  # pyright: ignore[reportUnusedFunction]
    """Give every test empty arithmetic caches.

    An implementation's caches are process-global. RNS keys its prime pool on
    (N, split_degree), so unrelated ring families sharing those parameters
    accumulate primes in one growing pool, and a ring's primes occupy
    contiguous indices from 0 only when it is the first registered for its
    key. A few low-level paths (LWE extraction, packing key-switch) rely on
    that contiguity. The original suites ran each test file as its own
    process; emptying the caches before every test reproduces that isolation
    so cross-file ordering cannot leak state.
    """
    from vfhe.arith import reset_state

    reset_state()
    yield
