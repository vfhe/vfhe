# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""`Codec`, `Encoded`, and the registry of codecs by tag and by type."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Sequence

    from ._bytes import Payload, Sink
    from ._stream import ReadContext, WriteContext


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
        f"no vfhe.util.io codec for {type(obj).__module__}.{type(obj).__qualname__}"
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
