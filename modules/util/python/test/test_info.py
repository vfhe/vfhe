# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""`python -m vfhe.info` is what a bug report pastes, so it reports the engine
truthfully and exits 0 on every install, including one that cannot load an
engine."""

import json
import os
import subprocess
import sys

from vfhe.util._engine import Engine


def _info(**env: str) -> tuple[dict, str]:
    done = subprocess.run(
        [sys.executable, "-m", "vfhe.info"],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, **env},
    )
    return json.loads(done.stdout), done.stderr


def test_the_report_holds_every_fact() -> None:
    facts, _ = _info()
    assert list(facts) == [
        "version",
        "engine",
        "engine_file",
        "runs",
        "python",
        "platform",
    ]


def test_the_report_names_the_selected_engine() -> None:
    facts, _ = _info()
    assert facts["engine"] == Engine.select().value


def test_runs_matches_what_each_engine_reports() -> None:
    facts, _ = _info()
    assert facts["runs"] == {engine.value: engine.runnable() for engine in Engine}


def test_an_engine_that_cannot_load_is_reported() -> None:
    """The install that cannot load one is the install this command is run on."""
    facts, err = _info(VFHE_ENGINE="nonsense")
    assert facts["engine"] is None
    assert facts["engine_file"] is None
    assert "unknown VFHE_ENGINE" in err
