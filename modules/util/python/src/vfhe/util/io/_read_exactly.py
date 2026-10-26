# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""What a decoder needs of a stream: `read_exactly`, a read that completes or fails."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import IO


def read_exactly(stream: IO[bytes], n: int) -> bytes:
    """Returns the next ``n`` bytes of ``stream``, in as many reads as the stream needs. Raises `EOFError` when the stream ends first."""
    out = bytearray()
    while len(out) < n:
        chunk = stream.read(n - len(out))
        if not chunk:
            raise EOFError("truncated vfhe.util.io stream")
        out += chunk
    return bytes(out)
