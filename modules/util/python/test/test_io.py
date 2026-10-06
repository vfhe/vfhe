# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The vfhe.util.io container: framing, containers, definitions, checksums and the
secret guard, through codecs defined here (the real ones live with their
types and are tested there)."""

import io
import json
import struct

import pytest
from vfhe.util.io import ALIGN, Codec, Encoded, Serializer, register


class Blob:
    def __init__(self, data: bytes, shape=None):
        self.data = data
        self.shape = shape


class Shape:
    def __init__(self, name: str):
        self.name = name


class Secret:
    def __init__(self, value: int):
        self.value = value


class Pair:
    def __init__(self, left, right):
        self.left, self.right = left, right


class ShapeCodec(Codec):
    tag = "testio.shape"
    types = (Shape,)
    definition = True

    def identity(self, obj, /):
        return obj.name

    def encode(self, obj, _ctx, /):
        return Encoded({"name": obj.name})

    def decode(self, meta, _payload, _children, ctx, /):
        return ctx.bound(self.tag, meta["name"]) or Shape(meta["name"])


class BlobCodec(Codec):
    tag = "testio.blob"
    types = (Blob,)

    def encode(self, obj, ctx, /):
        meta = {"shape": None if obj.shape is None else ctx.ref(obj.shape)}
        return Encoded(meta, len(obj.data), lambda sink: sink.write(obj.data))

    def decode(self, meta, payload, _children, ctx, /):
        shape = None if meta["shape"] is None else ctx.deref(meta["shape"])
        return Blob(payload.read(payload.remaining), shape)


class SecretCodec(Codec):
    tag = "testio.secret"
    types = (Secret,)
    secret = True

    def encode(self, obj, _ctx, /):
        return Encoded({"v": obj.value})

    def decode(self, meta, _payload, _children, _ctx, /):
        return Secret(meta["v"])


class PairCodec(Codec):
    tag = "testio.pair"
    types = (Pair,)

    def encode(self, obj, _ctx, /):
        return Encoded({}, children=[obj.left, obj.right])

    def decode(self, _meta, _payload, children, _ctx, /):
        return Pair(*children)


for _codec in (ShapeCodec(), BlobCodec(), SecretCodec(), PairCodec()):
    register(_codec)


def _records(data: bytes):
    """(kind, tag, meta, payload offset, payload size) of each record."""
    flags = struct.unpack_from("<I", data, 8)[0]
    pos, out = 16, []
    while True:
        kind, tag_len, meta_len, size = struct.unpack_from("<B3xIQQ", data, pos)
        pos += 24
        tag = data[pos : pos + tag_len].decode()
        pos += tag_len
        meta = json.loads(data[pos : pos + meta_len]) if meta_len else {}
        pos += meta_len
        if size:
            pos += -pos % ALIGN
        out.append((kind, tag, meta, pos, size))
        pos += size + (32 if flags & 1 else 0)
        if kind == 0:
            assert pos == len(data)
            return out


@pytest.mark.parametrize("profile", ["default", "compact", "fast"])
def test_values_and_containers_round_trip(profile):
    s = Serializer(profile)
    shape = Shape("square")
    value = {
        "a": [1, 2.5, "x", None, True],
        3: (Blob(b"abc", shape), Blob(b"", shape), Blob(bytes(range(200)))),
        "nested": [[], {}, ()],
        "pair": Pair(Blob(b"L", shape), [Blob(b"R")]),
    }
    back = s.loads(s.dumps(value))
    assert back["a"] == [1, 2.5, "x", None, True]
    assert [b.data for b in back[3]] == [b"abc", b"", bytes(range(200))]
    assert isinstance(back[3], tuple)
    assert back["nested"] == [[], {}, ()]
    # One definition, shared by every object that used it.
    assert back[3][0].shape is back[3][1].shape is back["pair"].left.shape
    assert back["pair"].right[0].data == b"R"


def test_payloads_are_aligned_and_definitions_written_once():
    shape = Shape("s")
    data = Serializer().dumps([Blob(b"x" * 5, shape), Blob(b"y" * 70, shape), shape])
    records = _records(data)
    assert sum(1 for r in records if r[0] == 2) == 1
    for _, _, _, offset, size in records:
        if size:
            assert offset % ALIGN == 0


def test_binding_returns_the_callers_object():
    shape = Shape("mine")
    back = Serializer().loads(Serializer().dumps(Blob(b"z", shape)), rings=[shape])
    assert back.shape is shape


def test_secrets_need_dump_secret():
    s = Serializer()
    with pytest.raises(TypeError, match="dump_secret"):
        s.dumps([Secret(7)])
    assert s.loads(s.dumps_secret([Secret(7)]))[0].value == 7


def test_checksum_catches_a_flipped_byte():
    s = Serializer()
    data = bytearray(s.dumps(Blob(b"payload bytes")))
    _, _, _, offset, _ = next(r for r in _records(bytes(data)) if r[1] == "testio.blob")
    data[offset] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        s.loads(bytes(data))
    # Without checksums the flip goes through: that is what they are for.
    fast = Serializer("fast")
    raw = bytearray(fast.dumps(Blob(b"payload bytes")))
    _, _, _, offset, _ = next(r for r in _records(bytes(raw)) if r[1] == "testio.blob")
    raw[offset] ^= 1
    assert fast.loads(bytes(raw)).data != b"payload bytes"


def test_malformed_streams_are_refused():
    s = Serializer()
    data = s.dumps([Blob(b"abc")])
    with pytest.raises(EOFError):
        s.loads(data[:-40])
    with pytest.raises(ValueError, match=r"not a vfhe\.util\.io stream"):
        s.loads(b"NOTVFHE" + data[7:])
    with pytest.raises(ValueError, match="more than one value"):
        # A second value where the end record should be.
        first = data[: -(24 + 32)]
        s.loads(first + data[16:])
    with pytest.raises(TypeError, match=r"no vfhe\.util\.io codec"):
        s.dumps(object())


def test_options_are_checked():
    with pytest.raises(ValueError, match="packing"):
        Serializer(packing="loose")
    with pytest.raises(ValueError, match="profile"):
        Serializer("smallest")
    assert Serializer("fast", checksum=True).options.checksum


def test_streams_through_file_objects(tmp_path):
    path = tmp_path / "x.vfhe"
    s = Serializer()
    with open(path, "wb") as f:
        s.dump([Blob(b"q" * 1000)], f)
    with open(path, "rb") as f:
        assert s.load(f)[0].data == b"q" * 1000
    # A reader without readinto works too.

    class ReadOnly:
        def __init__(self, data):
            self._b = io.BytesIO(data)

        def read(self, n):
            return self._b.read(min(n, 7))

    assert s.load(ReadOnly(path.read_bytes()))[0].data  # pyright: ignore[reportArgumentType] == b"q" * 1000


def test_the_checksum_algorithm_is_recorded(monkeypatch):
    from vfhe.util.io import _stream

    def algorithm(data):
        return struct.unpack_from("<I", data, 8)[0] >> 8 & 0xFF

    with monkeypatch.context() as m:
        m.setattr(_stream, "_CHECKSUMS", {1: _stream._CHECKSUMS[1]})
        blake2 = Serializer().dumps(Blob(b"x"))
    assert algorithm(blake2) == 1
    import vfhe.crypto  # noqa: F401  # pyright: ignore[reportUnusedImport]

    blake3 = Serializer().dumps(Blob(b"x"))
    assert algorithm(blake3) == 2
    for data in (blake2, blake3):
        assert Serializer().loads(data).data == b"x"
