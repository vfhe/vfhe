# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""`CKKSCiphertextCodec`: a sample and its scaling factor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from _vfhe_proto.vfhe.fhe.io.v1 import io_pb2 as pb
from vfhe.fhe.ckks import CKKS_Ciphertext
from vfhe.mlwe.codec import SampleCodec
from vfhe.util.io.protobuf import MessageBytesCodec

from .scheme import CKKSSchemeCodec

if TYPE_CHECKING:
    from typing import IO


class CKKSCiphertextCodec(SampleCodec):
    """A CKKS ciphertext: a sample and its actual scaling factor; ``schemes`` is a `CKKSSchemeCodec`."""

    cls = CKKS_Ciphertext

    body = MessageBytesCodec(pb.CKKSCiphertext)

    def __init__(
        self,
        schemes: CKKSSchemeCodec | None = None,
        seeded: bool = True,
        packing: str = "word",
        domain: str = "held",
    ) -> None:
        super().__init__(schemes or CKKSSchemeCodec(), seeded, packing, domain)

    def write(self, c: Any, scheme: Any, stream: IO[bytes]) -> None:
        if c.scheme != scheme:
            raise ValueError("a ciphertext of another scheme than the one written")
        self.body.encode(
            pb.CKKSCiphertext(sample=self.to_message(c), delta=c.delta), stream
        )

    def read(self, scheme: Any, stream: IO[bytes]) -> CKKS_Ciphertext:
        m = self.body.decode(stream)
        c = self.from_message(scheme, m.sample)
        c.delta = m.delta
        return c
