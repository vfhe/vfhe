# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""`CKKSSchemeCodec` and `BFVSchemeCodec`: the MLWE scheme message with the kind's own parameters."""

from __future__ import annotations

from _vfhe_proto.vfhe.fhe.io.v1 import io_pb2 as pb
from vfhe.fhe.bfv import BFV_Scheme
from vfhe.fhe.ckks import CKKS_Scheme
from vfhe.mlwe.codec import SchemeCodec


class CKKSSchemeCodec(SchemeCodec[CKKS_Scheme, pb.CKKSScheme]):
    message = pb.CKKSScheme
    kind = CKKS_Scheme

    def to_message(self, scheme: CKKS_Scheme, /) -> pb.CKKSScheme:
        return pb.CKKSScheme(
            scheme=self.scheme_message(scheme), scaling_factor=scheme.scaling_factor
        )

    def from_message(self, m: pb.CKKSScheme, /) -> CKKS_Scheme:
        return self.resolve(
            m.scheme,
            (m.scaling_factor,),
            lambda args: CKKS_Scheme(scaling_factor=m.scaling_factor, **args),
        )


class BFVSchemeCodec(SchemeCodec[BFV_Scheme, pb.BFVScheme]):
    message = pb.BFVScheme
    kind = BFV_Scheme

    def to_message(self, scheme: BFV_Scheme, /) -> pb.BFVScheme:
        # The plaintext modulus is the first primes of level 0, in its order.
        return pb.BFVScheme(
            scheme=self.scheme_message(scheme),
            plaintext_primes=scheme.plaintext_ring.ell,
        )

    def from_message(self, m: pb.BFVScheme, /) -> BFV_Scheme:
        return self.resolve(
            m.scheme,
            (m.plaintext_primes,),
            lambda args: BFV_Scheme(plaintext_primes=m.plaintext_primes, **args),
        )
