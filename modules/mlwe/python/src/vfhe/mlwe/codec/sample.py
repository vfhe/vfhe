# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Samples and what is made of them: `SampleCodec`, `KeySwitchKeyCodec` and `MGSWCodec`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from _vfhe_proto.vfhe.mlwe.io.v1 import io_pb2 as pb
from vfhe.arith import repr as Repr
from vfhe.arith.codec.rows import (
    DOMAINS,
    RowReader,
    RowWriter,
    held_domain,
    rows_options,
    target_domain,
)
from vfhe.mlwe.mgsw import MGSW
from vfhe.mlwe.mlwe import (
    MLWE,
    MLWE_Set,
)
from vfhe.util.bindings import lib
from vfhe.util.io import Codec
from vfhe.util.io.protobuf import MessageBytesCodec

from .scheme import (
    MGSWSchemeCodec,
    MLWESchemeCodec,
    SchemeCodec,
    _gadget,
    _gadget_params,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import IO


SEED_BYTES = 32


def seed_still_holds(c: MLWE) -> bytes | None:
    """``c.seed`` if ``c``'s mask still matches it, else None (and clears it).

    Checks the first and last prime's row of each mask component, which any
    operation rewriting the mask changes.
    """
    seed = c.seed
    if seed is None or c.is_extended or c.repr not in DOMAINS:
        return None
    canonical = c.repr == Repr.coeff
    rows = {min(c.ring.prime_indices), max(c.ring.prime_indices)}
    for j in range(c.r):
        for i in rows:
            if not lib.polynomial_RNS_matches_seeded(
                c.obj_a_i(j), canonical, i, seed, len(seed), j
            ):
                c.seed = None
                return None
    return seed


class SampleCodec(Codec[MLWE]):
    """One MLWE sample: its scheme through ``schemes``, then its rows.

    ``seeded`` writes a fresh sample's mask as its seed, halving the size of
    fresh ciphertexts and keys; ``packing`` and ``domain`` are as
    `vfhe.arith.codec.rows_options` says. A subclass wraps the `Sample`
    message in its own and adds its ciphertext type's state.
    """

    cls: type[MLWE] = MLWE

    body: MessageBytesCodec[Any] = MessageBytesCodec(pb.Sample)

    def __init__(
        self,
        schemes: SchemeCodec | None = None,
        seeded: bool = True,
        packing: str = "word",
        domain: str = "held",
    ) -> None:
        self.schemes = schemes or MLWESchemeCodec()
        self.seeded = seeded
        self.packing, self.domain = rows_options(packing, domain)

    def encode(self, obj: MLWE, stream: IO[bytes], /) -> None:
        self.schemes.encode(obj.scheme, stream)
        self.write(obj, stream)

    def decode(self, stream: IO[bytes], /) -> MLWE:
        return self.read(self.schemes.decode(stream), stream)

    def write(self, c: MLWE, stream: IO[bytes]) -> None:
        """Writes ``c`` after its scheme, which a key's codec writes once for all its samples."""
        self.body.encode(self.to_message(c), stream)

    def read(self, scheme: Any, stream: IO[bytes]) -> Any:
        """The sample of ``scheme`` at the head of ``stream``."""
        return self.from_message(scheme, self.body.decode(stream))

    def to_message(self, c: MLWE) -> pb.Sample:
        if type(c) is not self.cls:
            # A subclass would come back as `cls`, without its own state.
            raise TypeError(f"a {type(c).__qualname__} needs its own codec")
        if c.repr not in DOMAINS:
            raise ValueError("cannot write a sample that holds no value")
        seed = seed_still_holds(c) if self.seeded else None
        domain = target_domain(c.repr, self.domain)
        view = c
        if c.repr != domain:
            view = c.copy()
            if domain == Repr.ntt:
                view.to_NTT()
            else:
                view.to_coeff()
        rows = RowWriter(c.ring, self.packing)
        message = pb.Sample(
            ring=self.schemes.rings.to_message(c.ring),
            lvl=c.lvl,
            rank=c.r,
            domain=DOMAINS[domain],
            packing=self.packing,
            extended=c.is_extended,
            b=rows.rows(view.obj_b()),
        )
        if seed:
            message.seed = seed
            message.seed_expansion = pb.SEED_EXPANSION_AES128_CTR
        else:
            message.a.extend(rows.rows(view.obj_a_i(j)) for j in range(c.r))
        return message

    def from_message(self, scheme: Any, message: pb.Sample) -> Any:
        ring = self.schemes.rings.from_message(message.ring)
        domain = held_domain(message.domain)
        if domain == Repr.ntt or message.seed:
            self.schemes.rings.require_same_transform(ring)
        rows = RowReader(ring, message.packing)
        c = self.cls(scheme, lvl=message.lvl, ring=ring, rank=message.rank)
        c.is_extended = message.extended
        rows.read(message.b, c.obj_b())
        if message.seed:
            if message.seed_expansion != pb.SEED_EXPANSION_AES128_CTR:
                raise ValueError(
                    f"unknown seed expansion {message.seed_expansion}: written by a newer vfhe?"
                )
            for j in range(message.rank):
                a_j = c.obj_a_i(j)
                lib.polynomial_RNS_expand_seeded(a_j, message.seed, SEED_BYTES, j)
                if domain == Repr.coeff:
                    lib.polynomial_RNS_to_RNSc(a_j, a_j)
            c.seed = message.seed
        else:
            if len(message.a) != message.rank:
                raise ValueError(
                    f"{len(message.a)} mask components for rank {message.rank}: corrupt stream"
                )
            for j, a in enumerate(message.a):
                rows.read(a, c.obj_a_i(j))
        c.repr = domain
        return c


def _shared_scheme(samples: Iterable[MLWE]) -> Any:
    """The one scheme of ``samples``. Raises `ValueError` when they differ."""
    schemes = {c.scheme for c in samples}
    if len(schemes) > 1:
        raise ValueError("the samples of one key share one scheme")
    return next(iter(schemes), None)


class KeySwitchKeyCodec(Codec[MLWE_Set]):
    """An `MLWE_Set`: a header, the samples' scheme once, then the samples through ``samples``; or its nested sets."""

    header = MessageBytesCodec(pb.KeySwitchKey)

    def __init__(self, samples: SampleCodec | None = None) -> None:
        self.samples = samples or SampleCodec()

    def encode(self, obj: MLWE_Set, stream: IO[bytes], /) -> None:
        header = pb.KeySwitchKey(
            gadget=_gadget(obj.gadget_params), balanced=obj.balanced
        )
        if obj.dim > 2:
            sets = vars(obj)["_children"]
            header.sets = len(sets)
            self.header.encode(header, stream)
            for s in sets:
                self.encode(s, stream)
            return
        header.lengths.extend(-1 if c is None else len(c) for c in obj.mlwe)
        self.header.encode(header, stream)
        samples = [c for component in obj.mlwe for c in component or ()]
        if samples:
            self.samples.schemes.encode(_shared_scheme(samples), stream)
        for c in samples:
            self.samples.write(c, stream)

    def decode(self, stream: IO[bytes], /) -> MLWE_Set:
        header = self.header.decode(stream)
        if header.sets:
            return MLWE_Set.flatten_array(
                [self.decode(stream) for _ in range(header.sets)]
            )
        scheme = (
            self.samples.schemes.decode(stream)
            if any(n > 0 for n in header.lengths)
            else None
        )
        components = [
            None if n < 0 else [self.samples.read(scheme, stream) for _ in range(n)]
            for n in header.lengths
        ]
        return MLWE_Set(components, _gadget_params(header.gadget), header.balanced)


class MGSWCodec(Codec[MGSW]):
    """An MGSW ciphertext: its scheme through ``schemes`` and a count, then its samples through ``samples``, of the scheme's MLWE scheme."""

    header = MessageBytesCodec(pb.MGSW)

    def __init__(
        self, schemes: MGSWSchemeCodec | None = None, samples: SampleCodec | None = None
    ) -> None:
        self.schemes = schemes or MGSWSchemeCodec()
        self.samples = samples or SampleCodec(self.schemes.inner)

    def encode(self, obj: MGSW, stream: IO[bytes], /) -> None:
        if _shared_scheme(obj.obj) not in (None, obj.scheme.mlwe_scheme):
            raise ValueError("the samples of an MGSW ciphertext are of its MLWE scheme")
        self.header.encode(
            pb.MGSW(scheme=self.schemes.to_message(obj.scheme), samples=len(obj.obj)),
            stream,
        )
        for c in obj.obj:
            self.samples.write(c, stream)

    def decode(self, stream: IO[bytes], /) -> MGSW:
        header = self.header.decode(stream)
        scheme = self.schemes.from_message(header.scheme)
        samples = [
            self.samples.read(scheme.mlwe_scheme, stream) for _ in range(header.samples)
        ]
        return MGSW(scheme, obj=samples)
