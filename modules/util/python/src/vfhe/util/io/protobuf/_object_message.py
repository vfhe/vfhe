# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""`ObjectMessageCodec`, object <-> message: a codec whose object is one protobuf message, in the stream through `MessageBytesCodec`."""

from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Generic, TypeVar

from google.protobuf.message import Message
from vfhe.util.io import Codec

from ._message_bytes import MessageBytesCodec

if TYPE_CHECKING:
    from typing import IO

T = TypeVar("T")
M = TypeVar("M", bound=Message)


class ObjectMessageCodec(Codec[T], Generic[T, M]):
    """Object <-> message: a codec whose object is one protobuf ``message``, in the stream through `MessageBytesCodec`.

    A subclass names its ``message`` and writes `to_message` and
    `from_message`; `encode` and `decode` are those two through the bytes
    codec.
    An object holding more than one message's worth, such as a key and its
    samples, gets a `Codec` of its own, which writes its header through
    `MessageBytesCodec` and its parts through their codecs.
    """

    #: The message class `decode` parses: a subclass sets it to its ``M``.
    message: type[M]

    @abstractmethod
    def to_message(self, obj: T, /) -> M:
        """Returns the message holding ``obj``. Raises `TypeError` for an object of another type and `ValueError` for one that cannot be written."""

    @abstractmethod
    def from_message(self, message: M, /) -> T:
        """Returns the object ``message`` holds. Raises `ValueError` for a message this library cannot read."""

    @cached_property
    def _bytes(self) -> MessageBytesCodec[M]:
        return MessageBytesCodec(self.message)

    def encode(self, obj: T, stream: IO[bytes], /) -> None:
        self._bytes.encode(self.to_message(obj), stream)

    def decode(self, stream: IO[bytes], /) -> T:
        return self.from_message(self._bytes.decode(stream))
