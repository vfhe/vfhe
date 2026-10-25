# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The Fiat-Shamir verifier [FS86], in the BCS chain form [BCS16].

`FS_Verifier` is a `Verifier` that derives its randomness instead of
sampling it: both draw hooks (`_draw_challenge`, `_draw_bits`) hash the
transcript's chained state — seeded with the root statement's digest, so
every challenge binds both the claim and every message written before it
(omitting the statement is the "weak Fiat-Shamir" bug [DMWG23]). Nothing
else changes: same transcript, same compute-if-absent contract, same
protocols — the IOP machinery stays FS-unaware, and a run is fully
deterministic (same statement, same protocols => byte-identical transcript).

Domain challenges need a deterministic bytes -> exceptional-element map:
the domain's own seeded sampler, `domain.random_exceptional(seed=...)`,
which every `vfhe.arith` ring and field provides.
"""

from __future__ import annotations

from vfhe.crypto import hash_bytes

from .party import Verifier


def expand_bytes(seed: bytes, nbytes: int) -> bytes:
    """`nbytes` of BLAKE3 output in counter mode over `seed`."""
    out = b""
    counter = 0
    while len(out) < nbytes:
        out += hash_bytes(seed + counter.to_bytes(8, "little"))
        counter += 1
    return out[:nbytes]


class FS_Verifier(Verifier):
    """Derives both kinds of verifier randomness from the transcript chain.

    Each value is `H(state | tag | label)`: the chain state over everything
    written so far (`Transcript.state()`, seeded with the statement digest
    by `IOP.run`), a tag separating the two samplers, and the challenge's
    own label (so back-to-back challenges with no message in between still
    differ). The derived value is then written to the transcript like any
    challenge, extending the chain for the next one — exactly the BCS
    sigma-chain [BCS16].
    """

    def _derive(self, tag: bytes, label: str) -> bytes:
        transcript = self.iop.transcript
        return hash_bytes(transcript.state() + tag + label.encode())

    def _draw_challenge(self, label: str, domain=None):
        seed = self._derive(b"challenge", label)
        domain = self.iop.domain if domain is None else domain
        if domain is None:
            raise RuntimeError("this IOP has no domain")
        return domain.random_exceptional(seed=seed)

    def _draw_bits(self, label: str, nbytes: int) -> bytes:
        return expand_bytes(self._derive(b"bits", label), nbytes)
