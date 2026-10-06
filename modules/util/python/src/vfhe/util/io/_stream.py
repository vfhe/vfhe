# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""`StreamWriter` and `StreamReader`, and the `WriteContext` and `ReadContext` handed to codecs."""

from __future__ import annotations

import importlib
import json
import warnings
from typing import IO, TYPE_CHECKING, Any

from ._bytes import Payload, Sink, _Source
from ._codec import codec_for, codec_for_tag
from ._format import (
    _CHECKSUM_PROVIDERS,
    _CHECKSUM_SHIFT,
    _CHECKSUMS,
    _DIGEST,
    _HEADER,
    _RECORD,
    _REF_TAG,
    ALIGN,
    FLAG_CHECKSUM,
    FLAG_SECRET,
    KIND_DEFINITION,
    KIND_DICT,
    KIND_END,
    KIND_LIST,
    KIND_NONE,
    KIND_OBJECT,
    KIND_VALUE,
    MAGIC,
    VERSION,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable

    from ._codec import Encoded
    from ._options import Options


def _checksum(algorithm: int) -> Callable[[], Any]:
    if algorithm not in _CHECKSUMS and algorithm in _CHECKSUM_PROVIDERS:
        importlib.import_module(_CHECKSUM_PROVIDERS[algorithm])
    if algorithm not in _CHECKSUMS:
        raise ValueError(f"unknown checksum algorithm {algorithm}")
    return _CHECKSUMS[algorithm]


class WriteContext:
    """Handed to `Codec.encode`: the options, and definitions by reference."""

    def __init__(self, writer: StreamWriter, options: Options) -> None:
        self._writer = writer
        self.options = options
        self._ids: dict[int, int] = {}
        # Keeps every defined object alive for the dump, so id() stays unique.
        self._defined: list[Any] = []

    def ref(self, obj: Any) -> int:
        """The id of definition ``obj``, writing its record on first use."""
        key = id(obj)
        if key in self._ids:
            return self._ids[key]
        codec = codec_for(obj)
        if not codec.definition:
            raise TypeError(f"{codec.tag} is not a definition codec")
        enc = codec.encode(obj, self)
        if enc.children:
            raise ValueError("a definition cannot have children")
        n = len(self._defined)
        self._ids[key] = n
        self._defined.append(obj)
        self._writer.record(KIND_DEFINITION, codec.tag, {**enc.meta, "_id": n}, enc)
        return n


class ReadContext:
    """Handed to `Codec.decode`: options, definitions, binding, a cache."""

    def __init__(
        self, options: Options, schemes: Iterable[Any], rings: Iterable[Any]
    ) -> None:
        self.options = options
        self.defs: dict[int, Any] = {}
        #: Scratch space for codecs, per load (e.g. rings built so far).
        self.cache: dict[Any, Any] = {}
        self._bound: dict[tuple[str, Hashable], Any] = {}
        schemes = list(schemes)
        self._wanted = bool(schemes)
        for obj in [*schemes, *rings]:
            self._bind(obj)

    def _bind(self, obj: Any) -> None:
        codec = codec_for(obj)
        ident = codec.identity(obj)
        if ident is not None:
            self._bound.setdefault((codec.tag, ident), obj)
        for sub in codec.bindings(obj):
            self._bind(sub)

    def bound(self, tag: str, identity: Hashable) -> Any | None:
        """The object the caller asked to bind to with this identity, if any."""
        return self._bound.get((tag, identity))

    def unbound(self, what: str) -> None:
        """Report a definition that matched nothing passed to ``load``."""
        if self._wanted:
            warnings.warn(
                f"{what} in the stream matches none of the schemes passed to "
                "load(); a new one was built from the record",
                stacklevel=4,
            )

    def deref(self, n: int) -> Any:
        return self.defs[n]


class StreamWriter:
    def __init__(self, f: IO[bytes], options: Options, secret: bool) -> None:
        self.sink = Sink(f)
        self.options = options
        self.secret = secret
        self.ctx = WriteContext(self, options)
        self.algorithm = max(_CHECKSUMS)
        self._hasher = _CHECKSUMS[self.algorithm]

    def header(self) -> None:
        flags = (
            FLAG_CHECKSUM | self.algorithm << _CHECKSUM_SHIFT
            if self.options.checksum
            else 0
        ) | (FLAG_SECRET if self.secret else 0)
        self.sink.write(_HEADER.pack(MAGIC, VERSION[0], VERSION[1], flags, 0))

    def record(
        self, kind: int, tag: str, meta: dict[str, Any], enc: Encoded | None
    ) -> None:
        tag_b = tag.encode()
        meta_b = json.dumps(meta, separators=(",", ":")).encode() if meta else b""
        size = enc.size if enc is not None else 0
        sink = self.sink
        sink.write(_RECORD.pack(kind, len(tag_b), len(meta_b), size))
        if self.options.checksum:
            sink.hash = self._hasher()
        sink.write(tag_b)
        sink.write(meta_b)
        if size:
            hashing, sink.hash = sink.hash, None
            sink.write(bytes(-sink.pos % ALIGN))
            sink.hash = hashing
            start = sink.pos
            if enc is None or enc.write is None:
                raise ValueError(f"{tag} announced a payload and gave no writer")
            enc.write(sink)
            if sink.pos - start != size:
                raise RuntimeError(
                    f"{tag} wrote {sink.pos - start} payload bytes, announced {size}"
                )
        if self.options.checksum:
            digest = sink.hash.digest()
            sink.hash = None
            sink.write(digest)

    def put(self, obj: Any) -> None:
        if obj is None:
            self.record(KIND_NONE, "", {}, None)
        elif isinstance(obj, (bool, int, float, str)):
            self.record(KIND_VALUE, "", {"v": obj}, None)
        elif isinstance(obj, (list, tuple)):
            self.record(
                KIND_LIST, "", {"n": len(obj), "tuple": isinstance(obj, tuple)}, None
            )
            for item in obj:
                self.put(item)
        elif isinstance(obj, dict):
            keys = list(obj)
            if not all(
                isinstance(k, (str, int)) and not isinstance(k, bool) for k in keys
            ):
                raise TypeError("dict keys must be str or int")
            self.record(KIND_DICT, "", {"keys": keys}, None)
            for k in keys:
                self.put(obj[k])
        else:
            codec = codec_for(obj)
            if codec.secret and not self.secret:
                raise TypeError(
                    f"{type(obj).__qualname__} is secret: write it with dump_secret"
                )
            if codec.definition:
                self.record(KIND_OBJECT, _REF_TAG, {"ref": self.ctx.ref(obj)}, None)
                return
            enc = codec.encode(obj, self.ctx)
            meta = enc.meta
            if enc.children:
                if enc.size:
                    raise ValueError(f"{codec.tag}: a record has a payload or children")
                meta = {**meta, "_n": len(enc.children)}
            self.record(KIND_OBJECT, codec.tag, meta, enc)
            for child in enc.children:
                self.put(child)

    def end(self) -> None:
        self.record(KIND_END, "", {}, None)


class StreamReader:
    def __init__(
        self,
        f: IO[bytes],
        options: Options,
        schemes: Iterable[Any],
        rings: Iterable[Any],
    ) -> None:
        self.source = _Source(f)
        magic, major, _minor, flags, _ = _HEADER.unpack(self.source.read(_HEADER.size))
        if magic != MAGIC:
            raise ValueError("not a vfhe.util.io stream")
        if major != VERSION[0]:
            raise ValueError(
                f"vfhe.util.io format {major}.x, this reader knows {VERSION[0]}.x"
            )
        self.checksum = bool(flags & FLAG_CHECKSUM)
        if self.checksum:
            self._hasher = _checksum(flags >> _CHECKSUM_SHIFT & 0xFF)
        self.ctx = ReadContext(options, schemes, rings)

    def _record(self) -> tuple[int, str, dict[str, Any], int]:
        source = self.source
        kind, tag_len, meta_len, size = _RECORD.unpack(source.read(_RECORD.size))
        if self.checksum:
            source.hash = self._hasher()
        tag = source.read(tag_len).decode()
        meta = json.loads(source.read(meta_len)) if meta_len else {}
        if size:
            hashing, source.hash = source.hash, None
            source.read(-source.pos % ALIGN)
            source.hash = hashing
        return kind, tag, meta, size

    def _finish(self, payload: Payload | None, tag: str) -> None:
        if payload is not None and payload.remaining:
            raise ValueError(f"{tag} left {payload.remaining} payload bytes unread")
        if self.checksum:
            digest = self.source.hash.digest()
            self.source.hash = None
            if self.source.read(_DIGEST) != digest:
                raise ValueError(f"checksum mismatch in a {tag or 'container'} record")

    def get(self) -> Any:
        while True:
            kind, tag, meta, size = self._record()
            if kind == KIND_DEFINITION:
                payload = Payload(self.source, size)
                obj = codec_for_tag(tag).decode(meta, payload, [], self.ctx)
                self._finish(payload, tag)
                self.ctx.defs[meta["_id"]] = obj
                continue
            if kind == KIND_OBJECT and tag != _REF_TAG:
                codec = codec_for_tag(tag)
                n = meta.pop("_n", 0)
                if n:
                    self._finish(None, tag)
                    children = [self.get() for _ in range(n)]
                    return codec.decode(
                        meta, Payload(self.source, 0), children, self.ctx
                    )
                payload = Payload(self.source, size)
                obj = codec.decode(meta, payload, [], self.ctx)
                self._finish(payload, tag)
                return obj
            self._finish(None, tag)
            if kind == KIND_OBJECT:
                return self.ctx.deref(meta["ref"])
            if kind == KIND_NONE:
                return None
            if kind == KIND_VALUE:
                return meta["v"]
            if kind == KIND_LIST:
                items = [self.get() for _ in range(meta["n"])]
                return tuple(items) if meta["tuple"] else items
            if kind == KIND_DICT:
                return {k: self.get() for k in meta["keys"]}
            if kind == KIND_END:
                raise ValueError("vfhe.util.io stream ended before its value")
            raise ValueError(f"unknown record kind {kind}")

    def end(self) -> None:
        kind, tag, _, _ = self._record()
        self._finish(None, tag)
        if kind != KIND_END:
            raise ValueError("vfhe.util.io stream holds more than one value")
