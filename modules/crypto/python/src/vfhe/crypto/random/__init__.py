# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The randomness: `Random`, the source a scheme's samplers draw from, and `seeded_below_many`, values another party recomputes.

``Random()`` is the entropy stream, bytes no one can predict; ``Random(seed)``
is a stream the seed reproduces, for a reproducible scheme. `seeded_below_many`
is the other kind of determinism: value `i` is a pure function of
``(context, seed)`` and `i`, so a verifier recomputes a prover's draws from a
transcript.

>>> from vfhe.crypto.random import Random, seeded_below_many
>>> len(Random().bytes(16))
16
>>> seeded_below_many(2, 1000, b"example", b"seed") == seeded_below_many(2, 1000, b"example", b"seed")
True
"""

from ._random import Random
from ._seeded import seeded_below_many

__all__ = ["Random", "seeded_below_many"]
