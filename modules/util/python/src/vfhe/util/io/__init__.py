# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Serialization of vfhe objects: `Serializer`, and the codec interface.

A stream holds one value: a vfhe object, ``None``, a JSON scalar, or a list,
tuple or dict (``str`` or ``int`` keys) of values, nested freely. Objects that
several others share, such as rings and schemes, are written once.

>>> s = Serializer("compact")
>>> with open("keys.vfhe", "wb") as f:
...     s.dump({"rlk": rlk, "rot": rotation_keys}, f)
>>> with open("keys.vfhe", "rb") as f:
...     keys = s.load(f, schemes=scheme)

Format v1, integers little-endian:

- Header, 16 bytes: ``b"VFHEIO"``, u8 major, u8 minor, u32 flags
  (`FLAG_CHECKSUM`, `FLAG_SECRET`; bits 8-15 the checksum algorithm), u32
  reserved.
- Records: u8 kind, 3 reserved bytes, u32 tag length, u64 meta length, u64
  payload length; the tag (UTF-8); the meta (JSON); if there is a payload,
  zero padding to a multiple of `ALIGN` bytes from the stream's start and
  the payload; if checksummed, a 32-byte checksum of tag, meta and payload.
- A list or dict record is followed by its items. A definition record comes
  before its first use, which refers to it by the ``_id`` in its meta.
- An end record closes the stream.
"""

from __future__ import annotations

import io
from typing import IO, TYPE_CHECKING, Any

from ._bytes import Payload, Sink
from ._codec import Codec, Encoded, codec_for, codec_for_tag, register
from ._format import (
    ALIGN,
    CHECKSUM_BLAKE2B,
    CHECKSUM_BLAKE3,
    FLAG_CHECKSUM,
    FLAG_SECRET,
    register_checksum,
)
from ._options import PROFILES, Options, options_for
from ._stream import ReadContext, StreamReader, StreamWriter, WriteContext

if TYPE_CHECKING:
    from collections.abc import Iterable


def _seq(objs: Any) -> list[Any]:
    if objs is None:
        return []
    if isinstance(objs, (list, tuple)):
        return list(objs)
    return [objs]


class Serializer:
    """Writes and reads vfhe objects, with a choice of what to optimize.

    ``profile`` picks a preset from `PROFILES` (``"default"``, ``"compact"``,
    ``"fast"``), and each keyword overrides one of its `Options`.

    Secret keys are never written by `dump`; `dump_secret` is the only way,
    so that a key cannot end up in a file by being inside something else.
    """

    def __init__(
        self,
        profile: str | None = None,
        *,
        seeded: bool | None = None,
        packing: str | None = None,
        domain: str | None = None,
        checksum: bool | None = None,
        validate: bool | None = None,
    ) -> None:
        self.options = options_for(
            profile,
            {
                "seeded": seeded,
                "packing": packing,
                "domain": domain,
                "checksum": checksum,
                "validate": validate,
            },
        )

    def _dump(self, obj: Any, f: IO[bytes], secret: bool) -> None:
        writer = StreamWriter(f, self.options, secret)
        writer.header()
        writer.put(obj)
        writer.end()

    def dump(self, obj: Any, f: IO[bytes]) -> None:
        """Write ``obj`` to the binary file-like ``f``, streaming."""
        self._dump(obj, f, secret=False)

    def dumps(self, obj: Any) -> bytes:
        buf = io.BytesIO()
        self.dump(obj, buf)
        return buf.getvalue()

    def dump_secret(self, obj: Any, f: IO[bytes]) -> None:
        """`dump`, allowing secret keys anywhere in ``obj``."""
        self._dump(obj, f, secret=True)

    def dumps_secret(self, obj: Any) -> bytes:
        buf = io.BytesIO()
        self.dump_secret(obj, buf)
        return buf.getvalue()

    def load(
        self,
        f: IO[bytes],
        *,
        schemes: Any = None,
        rings: Iterable[Any] | None = None,
    ) -> Any:
        """Read one value from ``f``.

        Each scheme in the stream binds to the one of ``schemes`` (a scheme or
        a sequence) with the same primes and parameters, and its rings to that
        scheme's ring objects; one that matches none is rebuilt from the
        record, with a warning when ``schemes`` was given. Rings outside any
        scheme bind to ``rings``, or are built. Only `Options.validate`
        applies to reading: everything else is read from the stream.
        """
        reader = StreamReader(f, self.options, _seq(schemes), _seq(rings))
        value = reader.get()
        reader.end()
        return value

    def loads(self, data: bytes, **kwargs: Any) -> Any:
        return self.load(io.BytesIO(data), **kwargs)


__all__ = [
    "ALIGN",
    "CHECKSUM_BLAKE2B",
    "CHECKSUM_BLAKE3",
    "FLAG_CHECKSUM",
    "FLAG_SECRET",
    "PROFILES",
    "Codec",
    "Encoded",
    "Options",
    "Payload",
    "ReadContext",
    "Serializer",
    "Sink",
    "WriteContext",
    "codec_for",
    "codec_for_tag",
    "register",
    "register_checksum",
]
