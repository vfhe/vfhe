# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Times every module's benchmarks and prints the table: ``python -m vfhe.benchmark [PYTEST OPTIONS]``.

Runs pytest-benchmark, installed as ``vfhe[benchmark]``, over each module's
installed ``bench`` directory on the engine this process loads;
``VFHE_ENGINE`` picks it and ``VFHE_NUM_THREADS`` the thread limit. Options
pass through to pytest: ``-k ntt`` selects by name, ``--benchmark-save=NAME``
keeps the run and ``--benchmark-compare`` sets a run against the last one
kept.
"""

import sys
from importlib.resources import files
from pathlib import Path

import pytest

MODULES = ["arith", "circuit", "crypto", "fhe", "mlwe", "piop", "polycom"]

benches = [Path(str(files(f"vfhe.{m}"))) / "bench" for m in MODULES]
# The root is where ``vfhe`` is installed and ``vfhe`` is a namespace package,
# so pytest names a module ``vfhe.<mod>.bench.<topic>`` and finds the packages
# above it already imported, importing no second copy of them.
sys.exit(
    pytest.main(
        [
            *map(str, benches),
            "--rootdir",
            str(benches[0].parents[2]),
            "--import-mode=importlib",
            "-o",
            "python_files=bench/*.py",
            "-o",
            "consider_namespace_packages=true",
            "--benchmark-enable",
            "--benchmark-only",
            "--benchmark-disable-gc",
            "-p",
            "vfhe.util.benchmark",
            "-p",
            "no:cacheprovider",
            *sys.argv[1:],
        ]
    )
)
