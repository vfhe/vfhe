# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The type classes: `Encoder`, `Decoder`, and `Codec`, both at once."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from typing import IO

T = TypeVar("T")


class Encoder(ABC, Generic[T]):
    """Writes a ``T`` to a binary stream.

    An encoder writes the object's own bytes and nothing around them, so what
    one encoder writes, another may follow on the same stream.
    """

    @abstractmethod
    def encode(self, obj: T, stream: IO[bytes], /) -> None:
        """Writes ``obj`` to ``stream``. Raises `TypeError` for an object the codec does not cover and `ValueError` for one it cannot write."""


class Decoder(ABC, Generic[T]):
    """Reads a ``T`` from a binary stream.

    A decoder consumes exactly the bytes its encoder wrote, so decoders chain
    on one stream in the order their encoders ran.
    """

    @abstractmethod
    def decode(self, stream: IO[bytes], /) -> T:
        """Returns the object at the head of ``stream``. Raises `EOFError` when the stream ends inside the object and `ValueError` on bytes it cannot read."""

    def decode_strict(self, stream: IO[bytes], /) -> T:
        """`decode`, then checks that the stream has ended: for a whole file or buffer. Raises `ValueError` when bytes follow the object. On a socket the check waits for the peer to close, so use `decode` there."""
        obj = self.decode(stream)
        if stream.read(1):
            raise ValueError("trailing bytes after the decoded object")
        return obj


class Codec(Encoder[T], Decoder[T]):
    """An `Encoder` and a `Decoder` of one ``T``: what a type's codec implements."""
