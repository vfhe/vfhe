# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""`MessageBytesCodec`, message <-> bytes: a protobuf message in a stream, its bytes behind their size as a varint."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from google.protobuf.message import DecodeError, Message
from vfhe.util.io import Codec, read_exactly

if TYPE_CHECKING:
    from typing import IO

M = TypeVar("M", bound=Message)

#: Protobuf's largest message: a longer claim in a length prefix is malformed.
MAX_SIZE = 2**31 - 1


class MessageBytesCodec(Codec[M]):
    """Message <-> bytes: the codec of a protobuf ``message`` as itself, its serialized bytes behind their size as a varint, protobuf's own framing.

    This is protobuf's ``serialize_length_prefixed`` and ``parse_length_prefixed``
    written for streams: protobuf's parser reads the message in one ``read``
    and calls a shorter answer corruption, which a socket or a pipe gives on a
    sound stream. `decode` reads in as many reads as the stream needs; it
    raises `EOFError` at the stream's end or when the message is cut, and
    `ValueError` on malformed bytes or a size past `MAX_SIZE`, before reading
    a byte of it. `encode` writes the message in two writes.
    """

    def __init__(self, message: type[M]) -> None:
        self.message = message

    def encode(self, obj: M, stream: IO[bytes], /) -> None:
        data = obj.SerializeToString()
        stream.write(self._prefix(len(data)))
        stream.write(data)

    def decode(self, stream: IO[bytes], /) -> M:
        size = self._length(stream)
        if size > MAX_SIZE:
            raise ValueError(
                f"malformed vfhe.util.io stream: a message of {size} bytes"
            )
        data = read_exactly(stream, size)
        message = self.message()
        try:
            message.ParseFromString(data)
        except DecodeError as e:
            raise ValueError(f"malformed vfhe.util.io stream: {e}") from e
        return message

    @staticmethod
    def _prefix(size: int) -> bytes:
        """``size`` as a varint: seven bits a byte, low bits first, the high bit set while more follow."""
        out = bytearray()
        while size > 0x7F:
            out.append(size & 0x7F | 0x80)
            size >>= 7
        out.append(size)
        return bytes(out)

    @staticmethod
    def _length(stream: IO[bytes]) -> int:
        """The varint at the head of ``stream``. Raises `EOFError` when the stream has ended, `ValueError` on a varint past 64 bits."""
        byte = stream.read(1)
        if not byte:
            raise EOFError("truncated vfhe.util.io stream")
        size, shift = 0, 0
        while True:
            size |= (byte[0] & 0x7F) << shift
            if not byte[0] & 0x80:
                return size
            shift += 7
            if shift > 63:
                raise ValueError("malformed vfhe.util.io stream: a length past 64 bits")
            byte = read_exactly(stream, 1)
