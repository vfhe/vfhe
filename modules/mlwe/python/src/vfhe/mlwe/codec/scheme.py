# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The scheme codecs: `SchemeCodec`, what its kinds share, and the MLWE and MGSW kinds."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, TypeVar

from _vfhe_proto.vfhe.mlwe.io.v1 import io_pb2 as pb
from google.protobuf.message import Message
from vfhe.arith.codec import RingCodec
from vfhe.mlwe.mgsw import MGSW_Scheme, MGSWParameters
from vfhe.mlwe.mlwe import (
    GadgetParams,
    MLWE_Scheme,
    SchemeParameters,
)
from vfhe.util.io import Codec
from vfhe.util.io.protobuf import ObjectMessageCodec

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import IO


T = TypeVar("T")
S = TypeVar("S", bound=MLWE_Scheme)


M = TypeVar("M", bound=Message)


def _gadget(params: GadgetParams) -> pb.Gadget:
    gadget = pb.Gadget(primes_per_digit=params.primes_per_digit)
    if params.log_base is not None:
        gadget.log_base = params.log_base
    if params.digits is not None:
        gadget.digits = params.digits
    return gadget


def _gadget_params(gadget: pb.Gadget) -> GadgetParams:
    return GadgetParams(
        gadget.log_base if gadget.HasField("log_base") else None,
        gadget.digits if gadget.HasField("digits") else None,
        gadget.primes_per_digit,
    )


class SchemeCodec(ObjectMessageCodec[S, M]):
    """What the scheme codecs share: a scheme is one message holding its chain of rings.

    A scheme read back is one of ``known`` where the parameters match; else it
    is built from the stream, once, with a warning when ``known`` was given.
    `rings` is the ring codec the schemes' rings go through, which the sample
    codecs share. A subclass names its ``kind`` and ``message`` and writes
    `to_message` around `scheme_message`, `from_message` around `resolve`.
    Thread-unsafe, as `RingCodec` is: each thread decodes through its own.
    """

    kind: type[S]

    def __init__(self, *known: S) -> None:
        self.schemes: dict[SchemeParameters, S] = {s.parameters: s for s in known}
        self.rings = RingCodec(
            *[r for s in known for r in (*s.rings, *s.special_rings)]
        )
        self._warn = bool(known)

    def scheme_message(self, scheme: MLWE_Scheme) -> pb.Scheme:
        """The `Scheme` message of ``scheme``: what every kind shares. Raises `TypeError` for a scheme of another kind, which would come back without its own parameters."""
        if type(scheme) is not self.kind:
            raise TypeError(f"a {type(scheme).__qualname__} needs its own codec")
        return pb.Scheme(
            rings=[self.rings.to_message(r) for r in scheme.rings],
            special_rings=[self.rings.to_message(r) for r in scheme.special_rings],
            special_primes=scheme.special_primes,
            rank=scheme.r,
            balanced=scheme.balanced,
        )

    def resolve(
        self, message: pb.Scheme, extra: tuple, build: Callable[[dict[str, Any]], S]
    ) -> S:
        """The known scheme with these parameters, else ``build`` of the constructor arguments, remembered."""
        args = {
            "rings": [self.rings.from_message(m) for m in message.rings],
            "special_rings": [
                self.rings.from_message(m) for m in message.special_rings
            ],
            "special_primes": message.special_primes,
            "module_rank": message.rank,
            "balanced": message.balanced,
        }
        key = SchemeParameters(
            self.kind,
            tuple(r.parameters for r in args["rings"]),
            tuple(r.parameters for r in args["special_rings"]),
            args["module_rank"],
            extra,
        )
        scheme = self.schemes.get(key)
        if scheme is None:
            if self._warn:
                warnings.warn(
                    f"a {self.kind.__name__} in the stream matches none of the known "
                    "schemes; a new one was built from the stream",
                    stacklevel=4,
                )
            scheme = self.schemes[key] = build(args)
        return scheme


class MLWESchemeCodec(SchemeCodec[MLWE_Scheme, pb.Scheme]):
    """A plain MLWE scheme."""

    message = pb.Scheme
    kind = MLWE_Scheme

    def to_message(self, scheme: MLWE_Scheme, /) -> pb.Scheme:
        return self.scheme_message(scheme)

    def from_message(self, message: pb.Scheme, /) -> MLWE_Scheme:
        return self.resolve(message, (), lambda args: MLWE_Scheme(**args))


class MGSWSchemeCodec(ObjectMessageCodec[MGSW_Scheme, pb.MGSWScheme]):
    """An MGSW scheme: its MLWE scheme, through ``inner``, and its gadget. ``known`` as for `SchemeCodec`, with the same warning."""

    message = pb.MGSWScheme

    def __init__(
        self, *known: MGSW_Scheme, inner: MLWESchemeCodec | None = None
    ) -> None:
        self.schemes: dict[MGSWParameters, MGSW_Scheme] = {
            s.parameters: s for s in known
        }
        self.inner = inner or MLWESchemeCodec(*[s.mlwe_scheme for s in known])
        self._warn = bool(known)

    def to_message(self, scheme: MGSW_Scheme) -> pb.MGSWScheme:
        return pb.MGSWScheme(
            mlwe_scheme=self.inner.scheme_message(scheme.mlwe_scheme),
            ell=scheme.ell,
            gadget=_gadget(scheme.gadget_params),
            balanced=scheme.balanced,
        )

    def from_message(self, message: pb.MGSWScheme) -> MGSW_Scheme:
        inner = self.inner.from_message(message.mlwe_scheme)
        params = _gadget_params(message.gadget)
        key = MGSWParameters(inner.parameters, message.ell, params)
        scheme = self.schemes.get(key)
        if scheme is None:
            if self._warn:
                warnings.warn(
                    "an MGSW_Scheme in the stream matches none of the known schemes; "
                    "a new one was built from the stream",
                    stacklevel=4,
                )
            scheme = self.schemes[key] = MGSW_Scheme(
                inner, ell=message.ell, gadget_params=params, balanced=message.balanced
            )
        return scheme


class BundleCodec(Codec[T]):
    """A bundle of keys of one ``scheme``, such as a dict of rotation keys: the scheme once, through ``schemes``, then the bundle through the codec ``build`` makes of a scheme.

    ``build`` composes the bundle's shape from codecs `bound` to the scheme it
    is given, such as ``DictCodec(IntCodec(), ListCodec(keys.bound(scheme)))``.
    `encode` builds it for ``scheme``; `decode` for the scheme it reads, which
    is ``scheme`` where the parameters match.
    """

    def __init__(
        self, schemes: SchemeCodec, scheme: Any, build: Callable[[Any], Codec[T]]
    ) -> None:
        self.schemes = schemes
        self.scheme = scheme
        self.build = build

    def encode(self, obj: T, stream: IO[bytes], /) -> None:
        self.schemes.encode(self.scheme, stream)
        self.build(self.scheme).encode(obj, stream)

    def decode(self, stream: IO[bytes], /) -> T:
        return self.build(self.schemes.decode(stream)).decode(stream)
