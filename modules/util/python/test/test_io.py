# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""vfhe.util.io: the plain-value codecs and combinators, a protobuf message codec, a
hand-written codec, streaming through files, and truncated streams (the vfhe
codecs live with their types and are tested there)."""

import io

import pytest
from google.protobuf.wrappers_pb2 import BytesValue
from vfhe.util.io import (
    BoolCodec,
    BytesCodec,
    Codec,
    DictCodec,
    FloatCodec,
    IntCodec,
    ListCodec,
    OptionalCodec,
    StrCodec,
    TupleCodec,
)
from vfhe.util.io.protobuf import ObjectMessageCodec


class Blob:
    def __init__(self, data: bytes):
        self.data = data


class Secret:
    def __init__(self, value: int):
        self.value = value


class BlobCodec(ObjectMessageCodec[Blob, BytesValue]):
    message = BytesValue

    def to_message(self, obj, /):
        return BytesValue(value=obj.data)

    def from_message(self, message, /):
        return Blob(message.value)


class SecretCodec(Codec[Secret]):
    """A codec from the base API alone: bytes out, bytes in."""

    def encode(self, obj, stream, /):
        stream.write(obj.value.to_bytes(8, "little", signed=True))

    def decode(self, stream, /):
        return Secret(int.from_bytes(stream.read(8), "little", signed=True))


def dumps(codec, obj) -> bytes:
    buf = io.BytesIO()
    codec.encode(obj, buf)
    return buf.getvalue()


def loads(codec, data):
    return codec.decode(io.BytesIO(data))


def test_plain_values_round_trip():
    codec = TupleCodec(
        IntCodec(),
        FloatCodec(),
        BoolCodec(),
        StrCodec(),
        BytesCodec(),
        OptionalCodec(IntCodec()),
        ListCodec(StrCodec()),
        DictCodec(StrCodec(), IntCodec()),
    )
    value = (-5, 2.5, True, "x", b"\x00\xff", None, ["a", "b"], {"k": 1, "j": 2})
    assert loads(codec, dumps(codec, value)) == value
    with pytest.raises(ValueError, match="values for"):
        dumps(TupleCodec(IntCodec()), (1, 2))


def test_codecs_compose():
    codec = ListCodec(OptionalCodec(BlobCodec()))
    blobs = [Blob(b"abc"), None, Blob(b""), Blob(bytes(range(200)))]
    back = loads(codec, dumps(codec, blobs))
    assert [b and b.data for b in back] == [b"abc", None, b"", bytes(range(200))]


def test_a_hand_written_codec():
    codec = ListCodec(SecretCodec())
    assert [s.value for s in loads(codec, dumps(codec, [Secret(7), Secret(-1)]))] == [
        7,
        -1,
    ]


def test_streams_through_files(tmp_path):
    codec = ListCodec(BlobCodec())
    path = tmp_path / "x.vfhe"
    with open(path, "wb") as f:
        codec.encode([Blob(b"q" * 1000)] * 3, f)
    with open(path, "rb") as f:
        assert [b.data for b in codec.decode(f)] == [b"q" * 1000] * 3
        assert f.read() == b""  # consumed exactly


def test_decode_strict_refuses_trailing_bytes():
    codec = ListCodec(BlobCodec())
    data = dumps(codec, [Blob(b"abc")])
    assert [b.data for b in codec.decode_strict(io.BytesIO(data))] == [b"abc"]
    with pytest.raises(ValueError, match="trailing"):
        codec.decode_strict(io.BytesIO(data + b"\x00"))


def test_truncated_and_malformed_streams_are_refused():
    codec = ListCodec(BlobCodec())
    data = dumps(codec, [Blob(b"abc")])
    with pytest.raises(EOFError):
        loads(codec, data[:8])  # the count, then nothing
    with pytest.raises(EOFError):
        loads(IntCodec(), b"\x01")
    with pytest.raises(EOFError):
        loads(codec, data[:-2])  # a message cut short
    with pytest.raises(ValueError, match="malformed"):
        loads(BlobCodec(), b"\x05\xff\xff\xff\xff\xff")
    with pytest.raises(ValueError, match="malformed"):
        loads(BlobCodec(), b"\xff" * 10)  # a length past 64 bits
    with pytest.raises(ValueError, match="malformed"):
        loads(BlobCodec(), b"\x80\x80\x80\x80\x08" + b"x")  # a claim of 2 GiB


class _Trickle(io.RawIOBase):
    """A stream handing out one byte per read, as a socket may."""

    def __init__(self, data: bytes) -> None:
        self._data = io.BytesIO(data)

    def readable(self) -> bool:
        return True

    def readinto(self, buf) -> int:
        chunk = self._data.read(1)
        buf[: len(chunk)] = chunk
        return len(chunk)


def test_short_reads_are_completed():
    codec = TupleCodec(ListCodec(BlobCodec()), StrCodec(), IntCodec())
    value = ([Blob(b"q" * 300), Blob(b"")], "abc", -1)
    got = codec.decode(_Trickle(dumps(codec, value)))  # pyright: ignore[reportArgumentType]
    assert ([b.data for b in got[0]], *got[1:]) == ([b"q" * 300, b""], "abc", -1)
