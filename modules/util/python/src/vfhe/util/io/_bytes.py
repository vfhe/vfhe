# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""`Sink` and `Payload`: the bytes of a stream, counted and checksummed as they pass."""

from __future__ import annotations

from typing import IO, Any


class Sink:
    """The output stream, counting bytes and feeding the record checksum."""

    def __init__(self, f: IO[bytes]) -> None:
        self._f = f
        self.pos = 0
        self.hash: Any = None

    def write(self, data: Any) -> None:
        view = memoryview(data).cast("B")
        n = len(view)
        while view:
            written = self._f.write(view)
            if written is None or written >= len(view):
                break
            view = view[written:]
        if self.hash is not None:
            self.hash.update(memoryview(data).cast("B"))
        self.pos += n


class Payload:
    """One record's payload, read in order and exactly once."""

    def __init__(self, source: _Source, size: int) -> None:
        self._source = source
        self.remaining = size

    def read(self, n: int) -> bytes:
        buf = bytearray(n)
        self.readinto(buf)
        return bytes(buf)

    def readinto(self, buf: Any) -> None:
        view = memoryview(buf).cast("B")
        if len(view) > self.remaining:
            raise ValueError("a codec read past its record's payload")
        self._source.readinto(view)
        self.remaining -= len(view)


class _Source:
    def __init__(self, f: IO[bytes]) -> None:
        self._f = f
        self.pos = 0
        self.hash: Any = None

    def readinto(self, view: memoryview) -> None:
        got = 0
        n = len(view)
        reader = getattr(self._f, "readinto", None)
        while got < n:
            if reader is not None:
                k = reader(view[got:])
            else:
                chunk = self._f.read(n - got)
                k = len(chunk)
                view[got : got + k] = chunk
            if not k:
                raise EOFError("truncated vfhe.util.io stream")
            got += k
        if self.hash is not None:
            self.hash.update(view)
        self.pos += n

    def read(self, n: int) -> bytes:
        buf = bytearray(n)
        self.readinto(memoryview(buf))
        return bytes(buf)
