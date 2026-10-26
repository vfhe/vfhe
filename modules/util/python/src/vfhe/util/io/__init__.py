# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Serialization of vfhe objects: the type classes `Encoder`, `Decoder` and `Codec`, and the codecs of plain values.

A codec writes an object to a binary stream and reads one back, consuming
exactly the bytes it wrote, so codecs chain on one stream and compose: a key's
codec writes its header, then every sample through the sample codec it was
built with, and a key of any size streams one sample at a time. The caller
names the whole structure once, as scodec does. A codec of an object that
others refer to, such as a ring or a scheme, is built with the caller's own
objects and hands them back where the stream's parameters match; built with
none, it rebuilds them from the stream.

>>> from vfhe.mlwe.codec import KeySwitchKeyCodec, MLWESchemeCodec, SampleCodec
>>> keys = KeySwitchKeyCodec(SampleCodec(MLWESchemeCodec(scheme), packing="tight"))
>>> with open("rlk.vfhe", "wb") as f:
...     keys.encode(rlk, f)
>>> with open("rlk.vfhe", "rb") as f:
...     rlk = keys.decode_strict(f)

A stream is a binary file, a buffer or a buffered socket. A decoder raises
`EOFError` when the stream ends inside an object and `ValueError` on bytes it
cannot read; `Decoder.decode_strict` also refuses bytes after the object.
`vfhe.util.io.protobuf` is the format the vfhe codecs use: a protobuf message
per object, length-prefixed in the stream.
"""

from ._codec import Codec, Decoder, Encoder
from ._read_exactly import read_exactly
from ._values import (
    BoolCodec,
    BytesCodec,
    DictCodec,
    FloatCodec,
    IntCodec,
    ListCodec,
    OptionalCodec,
    StrCodec,
    TupleCodec,
)

__all__ = [
    "BoolCodec",
    "BytesCodec",
    "Codec",
    "Decoder",
    "DictCodec",
    "Encoder",
    "FloatCodec",
    "IntCodec",
    "ListCodec",
    "OptionalCodec",
    "StrCodec",
    "TupleCodec",
    "read_exactly",
]
