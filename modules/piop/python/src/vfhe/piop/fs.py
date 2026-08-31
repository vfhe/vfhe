# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
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

Domain challenges need a deterministic bytes -> exceptional-element map.
Over a `Ring` this module derives one itself (`ring_exceptional_from_seed`);
a domain may instead provide its own `exceptional_from_seed(seed)` method,
which takes precedence (the future arith-side hook, and the test-stub one).
"""

from __future__ import annotations

from vfhe.arith import Polynomial, Ring

from .merkle import hash_bytes
from .piop import Verifier


def expand_bytes(seed: bytes, nbytes: int) -> bytes:
    """`nbytes` of BLAKE3 output in counter mode over `seed`."""
    out = b""
    counter = 0
    while len(out) < nbytes:
        out += hash_bytes(seed + counter.to_bytes(8, "little"))
        counter += 1
    return out[:nbytes]


def ring_exceptional_from_seed(ring: Ring, seed: bytes) -> Polynomial:
    """A deterministic exceptional element of `ring` derived from `seed`.

    The element is a constant chunk: `split_degree` integer coefficients,
    each uniform below min(primes) (128 bits reduced per coefficient, so the
    modulo bias is ~2^-80). These chunks form an exceptional set of the
    documented size |A| = min(p)^split_degree: a nonzero difference has some
    coefficient in (-min p, min p) \\ {0}, hence is nonzero mod every prime,
    and a nonzero polynomial of degree < split_degree is coprime to each
    irreducible degree-split_degree factor of X^N + 1, so the difference is
    invertible. (A seeded arith-side `sample_exceptional` can replace this;
    `sample_exceptional`'s own set — a broadcast NTT slot — is sampled
    differently but has the same size and role.)
    """
    bound = min(ring.primes)
    stream = expand_bytes(seed, 16 * ring.split_degree)
    coefficients = [
        int.from_bytes(stream[16 * i : 16 * (i + 1)], "little") % bound
        for i in range(ring.split_degree)
    ]
    return Polynomial(ring).from_array(coefficients)


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
        assert self.iop is not None
        transcript = self.iop.transcript
        return hash_bytes(transcript.state() + tag + label.encode())

    def _draw_challenge(self, label: str):
        seed = self._derive(b"challenge", label)
        assert self.iop is not None
        domain = self.iop.domain
        derive = getattr(domain, "exceptional_from_seed", None)
        if callable(derive):
            return derive(seed)
        if isinstance(domain, Ring):
            return ring_exceptional_from_seed(domain, seed)
        raise TypeError(
            f"no deterministic exceptional sampler for domain "
            f"{type(domain).__name__}; give it an exceptional_from_seed method"
        )

    def _draw_bits(self, label: str, nbytes: int) -> bytes:
        return expand_bytes(self._derive(b"bits", label), nbytes)
