# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Names the parameter sets benchmarks share, so their results line up across modules.

A benchmark lists the names it runs at; ``harness.Shape`` hands it the
parameters. A parameter set used by one benchmark alone belongs in that
benchmark's file, as a ``Shape``.
"""

from typing import Any

__all__ = ["SHAPES"]

SHAPES: dict[str, dict[str, Any]] = {
    # CKKS chains: ring degree 2^log_n; prime_bits are the chain's prime
    # sizes from level 0 up; special_bits the primes key switching adds;
    # scale_bits the encoding scale.
    "ckks-n12-3p": {
        "log_n": 12,
        "prime_bits": [60, 40, 40],
        "special_bits": [60],
        "scale_bits": 40,
    },
    "ckks-n13-7p": {
        "log_n": 13,
        "prime_bits": [60] + [40] * 6,
        "special_bits": [60],
        "scale_bits": 40,
    },
    "ckks-n14-7p": {
        "log_n": 14,
        "prime_bits": [60] + [40] * 6,
        "special_bits": [60],
        "scale_bits": 40,
    },
}
