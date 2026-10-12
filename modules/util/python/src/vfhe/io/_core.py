# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The container format: records, the codec registry, and the two contexts.

The format knows no vfhe types: a type becomes serializable when the module
that owns it registers a `Codec`. See ``vfhe.io`` for the layout.
"""

from __future__ import annotations

import importlib
import json
import struct
import sys
import warnings
from dataclasses import dataclass, replace
from typing import IO, TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Sequence

if sys.byteorder != "little":
    raise ImportError("vfhe.io writes little-endian words as they are in memory")

MAGIC = b"VFHEIO"
VERSION = (1, 0)
#: Records end with a 32-byte checksum of their tag, meta and payload.
FLAG_CHECKSUM = 1 << 0
#: The stream was written by ``dump_secret``: it may hold secret keys.
FLAG_SECRET = 1 << 1
#: Flag bits 8-15: which checksum the records carry, when they carry one.
_CHECKSUM_SHIFT = 8

#: The checksum algorithm, by the id the header records: BLAKE3, from
#: ``vfhe.crypto``.
CHECKSUM_BLAKE3 = 2


def _checksum(algorithm: int) -> Callable[[], Any]:
    """A factory of hashers with ``update`` and a 32-byte ``digest``."""
    if algorithm != CHECKSUM_BLAKE3:
        raise ValueError(f"unknown checksum algorithm {algorithm}")
    return importlib.import_module("vfhe.crypto").Blake3Stream


KIND_END = 0
KIND_OBJECT = 1
KIND_DEFINITION = 2
KIND_LIST = 3
KIND_DICT = 4
KIND_NONE = 5
KIND_VALUE = 6

#: Payloads start at a multiple of this many bytes from the stream's start.
ALIGN = 64
_HEADER = struct.Struct("<6sBBII")
_RECORD = struct.Struct("<B3xIQQ")
_DIGEST = 32
#: The tag of a reference to a definition dumped as a value.
_REF_TAG = "io.ref"

PACKINGS = ("word", "tight")
DOMAINS = ("held", "mul", "canonical")


@dataclass(frozen=True)
class Options:
    """What a `Serializer` optimizes for. See `PROFILES` for the presets.

    ``seeded``
        Write a fresh sample's mask as its seed, halving the size of fresh
        ciphertexts and keys; the mask is expanded again on load.
    ``packing``
        ``"word"``: each residue as a 4- or 8-byte word, copied as is.
        ``"tight"``: each residue in exactly the bits its prime needs.
    ``domain``
        The domain rows are written in: ``"held"`` (as held, no conversion),
        ``"mul"`` or ``"canonical"``. The size is the same; ``canonical``
        does not depend on the NTT convention but costs a transform on load.
    ``checksum``
        A checksum per record, verified on load. It detects corruption, not
        tampering.
    ``validate``
        On load, check that every residue is below its prime.
    """

    seeded: bool = True
    packing: str = "word"
    domain: str = "held"
    checksum: bool = True
    validate: bool = True

    def __post_init__(self) -> None:
        if self.packing not in PACKINGS:
            raise ValueError(f"packing must be one of {PACKINGS}, got {self.packing!r}")
        if self.domain not in DOMAINS:
            raise ValueError(f"domain must be one of {DOMAINS}, got {self.domain!r}")


#: ``compact`` minimizes size; ``fast`` minimizes load time (no expansion,
#: unpacking, checksum or range check).
PROFILES: dict[str, Options] = {
    "default": Options(),
    "compact": Options(packing="tight"),
    "fast": Options(seeded=False, checksum=False, validate=False),
}


@dataclass
class Encoded:
    """A codec's encoding of an object.

    ``meta`` is JSON-serializable. A record has either a payload of exactly
    ``size`` bytes, written by ``write(sink)``, or ``children``: values written
    as records of their own and handed back to `Codec.decode`.
    """

    meta: dict[str, Any]
    size: int = 0
    write: Callable[[Sink], None] | None = None
    children: Sequence[Any] = ()


class Codec:
    """How one type becomes a record and back.

    ``tag`` is ``"<package>.<name>"``, where ``vfhe.<package>`` registers the
    codec; a reader meeting an unknown tag imports that package. ``types`` are
    the classes it writes, subclasses included unless they have their own.
    A ``secret`` codec is only written by ``dump_secret``.

    A ``definition`` codec writes objects that others share: each is written
    once, before its first use, and referred to by the id `WriteContext.ref`
    returns. ``identity`` is what `ReadContext.bound` matches against the
    objects passed to ``load``; ``bindings`` lists the definitions an object
    contains, which are bound along with it.
    """

    tag: ClassVar[str] = ""
    types: ClassVar[tuple[type, ...]] = ()
    secret: ClassVar[bool] = False
    definition: ClassVar[bool] = False

    # Positional-only, so implementations may rename their parameters.

    def encode(self, obj: Any, ctx: WriteContext, /) -> Encoded:
        raise NotImplementedError

    def decode(
        self,
        meta: dict[str, Any],
        payload: Payload,
        children: list[Any],
        ctx: ReadContext,
        /,
    ) -> Any:
        """The object back. ``payload`` is empty for a record with children."""
        raise NotImplementedError

    def identity(self, _obj: Any, /) -> Hashable | None:
        return None

    def bindings(self, _obj: Any, /) -> Iterable[Any]:
        return ()


_BY_TAG: dict[str, Codec] = {}
_BY_TYPE: dict[type, Codec] = {}


def register(codec: Codec) -> Codec:
    """Make ``codec`` the one for its tag and types. Returns it."""
    if not codec.tag or "." not in codec.tag:
        raise ValueError(f"a codec tag is '<package>.<name>', got {codec.tag!r}")
    _BY_TAG[codec.tag] = codec
    for t in codec.types:
        _BY_TYPE[t] = codec
    return codec


def codec_for(obj: Any) -> Codec:
    for t in type(obj).__mro__:
        codec = _BY_TYPE.get(t)
        if codec is not None:
            return codec
    raise TypeError(
        f"no vfhe.io codec for {type(obj).__module__}.{type(obj).__qualname__}"
    )


def codec_for_tag(tag: str) -> Codec:
    codec = _BY_TAG.get(tag)
    if codec is None:
        # The package named by the tag registers its codecs on import.
        importlib.import_module("vfhe." + tag.split(".", 1)[0])
        codec = _BY_TAG.get(tag)
    if codec is None:
        raise ValueError(f"unknown record type {tag!r}")
    return codec


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
                raise EOFError("truncated vfhe.io stream")
            got += k
        if self.hash is not None:
            self.hash.update(view)
        self.pos += n

    def read(self, n: int) -> bytes:
        buf = bytearray(n)
        self.readinto(memoryview(buf))
        return bytes(buf)


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
        self.algorithm = CHECKSUM_BLAKE3
        self._hasher = _checksum(self.algorithm)

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
            raise ValueError("not a vfhe.io stream")
        if major != VERSION[0]:
            raise ValueError(
                f"vfhe.io format {major}.x, this reader knows {VERSION[0]}.x"
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
                raise ValueError("vfhe.io stream ended before its value")
            raise ValueError(f"unknown record kind {kind}")

    def end(self) -> None:
        kind, tag, _, _ = self._record()
        self._finish(None, tag)
        if kind != KIND_END:
            raise ValueError("vfhe.io stream holds more than one value")


def options_for(profile: str | None, overrides: dict[str, Any]) -> Options:
    if profile is None:
        profile = "default"
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {sorted(PROFILES)}, got {profile!r}")
    return replace(
        PROFILES[profile], **{k: v for k, v in overrides.items() if v is not None}
    )
