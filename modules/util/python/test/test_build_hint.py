# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The hint fires when a faster engine than the active one could run here, and
not when the active engine is already the best available."""

import warnings

import pytest
from vfhe.engine import warn_if_faster_engine_available as hint


def test_hint_fires_when_a_faster_engine_could_run():
    with pytest.warns(RuntimeWarning, match="avx512ifma"):
        hint("portable", ["avx512ifma", "portable"])


@pytest.mark.parametrize(
    "engine,choices",
    [
        ("avx512ifma", ["avx512ifma", "portable"]),  # already the best
        ("portable", ["portable"]),  # nothing else runs here
        ("portable", []),  # nothing reported at all
    ],
)
def test_hint_silent_otherwise(engine, choices):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would raise
        hint(engine, choices)
