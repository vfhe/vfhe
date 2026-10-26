# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Codecs of plain values and the combinators that build one codec from others: an optional, a list, a tuple, a dict.

Counts and ints are 8 little-endian bytes and floats IEEE 754 doubles, so a
stream reads the same on every machine. Every `decode` raises `EOFError` when
the stream ends inside the value.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from ._codec import Codec
from ._read_exactly import read_exactly

if TYPE_CHECKING:
    from typing import IO

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

_COUNT = struct.Struct("<Q")
_INT = struct.Struct("<q")
_FLOAT = struct.Struct("<d")


class IntCodec(Codec[int]):
    """An int as 8 little-endian bytes, two's complement. `encode` raises `struct.error` for a value outside 64 bits."""

    def encode(self, obj: int, stream: IO[bytes], /) -> None:
        stream.write(_INT.pack(obj))

    def decode(self, stream: IO[bytes], /) -> int:
        return _INT.unpack(read_exactly(stream, _INT.size))[0]


class FloatCodec(Codec[float]):
    """A float as an IEEE 754 double, little-endian."""

    def encode(self, obj: float, stream: IO[bytes], /) -> None:
        stream.write(_FLOAT.pack(obj))

    def decode(self, stream: IO[bytes], /) -> float:
        return _FLOAT.unpack(read_exactly(stream, _FLOAT.size))[0]


class BoolCodec(Codec[bool]):
    """A bool as one byte, ``0`` or ``1``; any other byte reads as ``True``."""

    def encode(self, obj: bool, stream: IO[bytes], /) -> None:
        stream.write(b"\x01" if obj else b"\x00")

    def decode(self, stream: IO[bytes], /) -> bool:
        return read_exactly(stream, 1) != b"\x00"


class BytesCodec(Codec[bytes]):
    """Bytes behind their count."""

    def encode(self, obj: bytes, stream: IO[bytes], /) -> None:
        stream.write(_COUNT.pack(len(obj)))
        stream.write(obj)

    def decode(self, stream: IO[bytes], /) -> bytes:
        return read_exactly(stream, _COUNT.unpack(read_exactly(stream, _COUNT.size))[0])


class StrCodec(Codec[str]):
    """A str as UTF-8 behind its byte count. `decode` raises `UnicodeDecodeError` on invalid UTF-8."""

    _bytes = BytesCodec()

    def encode(self, obj: str, stream: IO[bytes], /) -> None:
        self._bytes.encode(obj.encode(), stream)

    def decode(self, stream: IO[bytes], /) -> str:
        return self._bytes.decode(stream).decode()


class OptionalCodec(Codec[Any], Generic[T]):
    """``None`` or a ``T``: a flag byte, then the value through ``item`` when there is one."""

    def __init__(self, item: Codec[T]) -> None:
        self.item = item

    def encode(self, obj: T | None, stream: IO[bytes], /) -> None:
        if obj is None:
            stream.write(b"\x00")
        else:
            stream.write(b"\x01")
            self.item.encode(obj, stream)

    def decode(self, stream: IO[bytes], /) -> T | None:
        return self.item.decode(stream) if read_exactly(stream, 1) != b"\x00" else None


class ListCodec(Codec[list], Generic[T]):
    """A list of ``T``: its count, then each item through ``item``."""

    def __init__(self, item: Codec[T]) -> None:
        self.item = item

    def encode(self, obj: list[T], stream: IO[bytes], /) -> None:
        stream.write(_COUNT.pack(len(obj)))
        for x in obj:
            self.item.encode(x, stream)

    def decode(self, stream: IO[bytes], /) -> list[T]:
        n = _COUNT.unpack(read_exactly(stream, _COUNT.size))[0]
        return [self.item.decode(stream) for _ in range(n)]


class TupleCodec(Codec[tuple]):
    """A tuple: each position through its own codec, in order, with no count. `encode` raises `ValueError` for a tuple of another length."""

    def __init__(self, *items: Codec[Any]) -> None:
        self.items = items

    def encode(self, obj: tuple, stream: IO[bytes], /) -> None:
        if len(obj) != len(self.items):
            raise ValueError(f"{len(obj)} values for {len(self.items)} codecs")
        for codec, x in zip(self.items, obj, strict=True):
            codec.encode(x, stream)

    def decode(self, stream: IO[bytes], /) -> tuple:
        return tuple(codec.decode(stream) for codec in self.items)


class DictCodec(Codec[dict], Generic[K, V]):
    """A dict: its count, then each pair in the dict's order, the key through ``key`` and the value through ``value``."""

    def __init__(self, key: Codec[K], value: Codec[V]) -> None:
        self.key = key
        self.value = value

    def encode(self, obj: dict[K, V], stream: IO[bytes], /) -> None:
        stream.write(_COUNT.pack(len(obj)))
        for k, v in obj.items():
            self.key.encode(k, stream)
            self.value.encode(v, stream)

    def decode(self, stream: IO[bytes], /) -> dict[K, V]:
        n = _COUNT.unpack(read_exactly(stream, _COUNT.size))[0]
        return {self.key.decode(stream): self.value.decode(stream) for _ in range(n)}
