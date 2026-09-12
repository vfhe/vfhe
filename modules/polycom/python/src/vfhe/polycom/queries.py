# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Spot-check positions for a folding commitment scheme.

Every scheme whose verifier checks a chain of folds needs the same sampler:
expand a published bit-string challenge into `rep` positions of the top
oracle, rejecting a candidate whose walk *ends* where an earlier one's does.
The rejection is the part worth having written once -- it is a soundness
argument rather than a property of any one code, and dropping it costs
repetitions silently (see `query_positions`).

What differs between schemes is only the geometry of the walk: how many
positions the top oracle has, and how many index bits one walk consumes. Both
are arguments here, so a scheme that folds several levels per published
oracle uses the same function as one that folds one.
"""

from __future__ import annotations

from vfhe.crypto import DIGEST_LEN, hash_bytes


def query_positions(seed: bytes, span: int, shift: int, rep: int) -> tuple[int, ...]:
    """`rep` distinct spot-check positions in ``[0, span)``, derived from
    ``seed``.

    :param seed: The published bit-string challenge. The positions are a
        deterministic function of it, so prover and verifier derive the same
        ones -- the prover to answer them, the verifier to check.
    :param span: How many positions the top oracle has. A power of two, so
        that masking a uniform word is unbiased and no rejection is needed
        for the range itself.
    :param shift: How many low index bits one walk consumes, i.e. how far a
        position must be shifted right to give the position its walk ends at.
        ``0`` makes every position its own endpoint and the rejection below a
        plain distinctness check.
    :param rep: How many positions to draw.

    The seed is expanded with BLAKE3 in counter mode
    (``BLAKE3(seed || counter)``, eight candidates per digest), and a
    candidate is **rejected on its projection** ``candidate >> shift``: two
    queries that meet at the end of their walks run the same final check
    twice, so sampling with replacement there buys fewer effective
    repetitions than ``rep`` claims while reporting ``rep``. That is a
    silent loss of soundness, which is why it is rejected rather than
    tolerated.

    Distinct endpoints imply distinct positions all the way up, since a
    position's projection at any height extends its endpoint. Termination
    needs at least ``rep`` endpoints to exist, which is checked.
    """
    if span < 1 or span & (span - 1):
        raise ValueError(f"span must be a power of two, got {span}")
    if shift < 0 or shift >= span.bit_length():
        raise ValueError(f"shift {shift} does not fit a span of {span}")
    if rep < 1:
        raise ValueError(f"rep must be at least 1, got {rep}")
    endpoints = span >> shift
    if rep > endpoints:
        raise ValueError(
            f"rep = {rep} positions but the walks end in only {endpoints}: "
            "they are rejection-sampled to end apart, so this cannot terminate"
        )

    positions: list[int] = []
    seen: set[int] = set()
    counter = 0
    while len(positions) < rep:
        digest = hash_bytes(seed + counter.to_bytes(8, "little"))
        counter += 1
        for off in range(0, DIGEST_LEN, 8):
            candidate = int.from_bytes(digest[off : off + 8], "little") & (span - 1)
            end = candidate >> shift
            if end in seen:
                continue
            seen.add(end)
            positions.append(candidate)
            if len(positions) == rep:
                break
    return tuple(positions)
