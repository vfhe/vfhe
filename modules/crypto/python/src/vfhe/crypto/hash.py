# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Incremental BLAKE3 over the native hasher."""

from __future__ import annotations

from vfhe.util.bindings import ffi, lib


class Blake3Stream:
    """BLAKE3 over a stream, with the `hashlib` interface: `update`, `digest`."""

    def __init__(self) -> None:
        self._state = ffi.new("uint8_t[]", lib.hash_stream_size())
        lib.hash_stream_init(self._state)

    def update(self, data) -> None:
        view = memoryview(data).cast("B")
        lib.hash_stream_update(self._state, ffi.from_buffer(view), len(view))

    def digest(self) -> bytes:
        out = ffi.new("uint8_t[32]")
        lib.hash_stream_digest(self._state, out)
        return bytes(out)
