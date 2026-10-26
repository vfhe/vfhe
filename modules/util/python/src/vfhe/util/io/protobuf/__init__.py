# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Protobuf in a stream, object <-> message <-> bytes: `ObjectMessageCodec` maps an object to one message of a ``vfhe.<module>.io.v1`` schema under ``proto/``, and `MessageBytesCodec` puts a message in the stream, length-prefixed."""

from ._message_bytes import MessageBytesCodec
from ._object_message import ObjectMessageCodec

__all__ = ["MessageBytesCodec", "ObjectMessageCodec"]
