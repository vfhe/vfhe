# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""`vfhe.io` codecs for MLWE schemes, samples, key-switch keys, MGSW and LWE.

Samples sharing scheme, ring, level and rank (a key-switch key, an MGSW
ciphertext) go in one record. A sample is ``b`` and then ``a``, encoded as
`vfhe.arith.impl.rns.io` encodes polynomials; a fresh sample whose mask still
matches its seed is written as the 32-byte seed and ``b``.
"""

from __future__ import annotations

import array
from typing import TYPE_CHECKING, Any

from vfhe.arith import RNSRing
from vfhe.arith import repr as Repr
from vfhe.arith.impl.rns.io import (
    DOMAIN_NAMES,
    DOMAINS_BY_NAME,
    RowReader,
    RowWriter,
    require_same_transform,
    ring_key,
    target_domain,
)
from vfhe.engine import ffi, lib
from vfhe.io import (
    Codec,
    Encoded,
    Payload,
    ReadContext,
    Sink,
    WriteContext,
    codec_for,
    register,
)

from .lwe import LWE, LWE_Key, _native_limbs
from .mgsw import MGSW, MGSW_Scheme
from .mlwe import MLWE, MLWE_Key, MLWE_Scheme, MLWE_Set

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

SEED_BYTES = 32


# --- schemes ---------------------------------------------------------------


def scheme_identity(tag: str, scheme: MLWE_Scheme, *extra: Hashable) -> tuple:
    """What a scheme record is matched against a caller's scheme by."""
    return (
        tag,
        tuple(ring_key(r) for r in scheme.rings),
        tuple(ring_key(r) for r in scheme.special_rings),
        scheme.r,
        *extra,
    )


def scheme_meta(scheme: MLWE_Scheme, ctx: WriteContext) -> dict[str, Any]:
    return {
        "rings": [ctx.ref(r) for r in scheme.rings],
        "special_rings": [ctx.ref(r) for r in scheme.special_rings],
        "special_primes": scheme.special_primes,
        "rank": scheme.r,
    }


def scheme_arguments(meta: dict[str, Any], ctx: ReadContext) -> dict[str, Any]:
    """The constructor keywords every `MLWE_Scheme` takes, from its record."""
    return {
        "rings": [ctx.deref(i) for i in meta["rings"]],
        "special_rings": [ctx.deref(i) for i in meta["special_rings"]],
        "special_primes": meta["special_primes"],
        "module_rank": meta["rank"],
    }


class SchemeCodec(Codec):
    """A scheme as its chain of rings, plus a subclass's ``parameters``.

    The identity is the chain and those parameters, so a record is matched
    against a caller's scheme without building one.
    """

    tag = "mlwe.scheme"
    types = (MLWE_Scheme,)
    definition = True
    #: The meta keys `parameters` writes.
    parameter_keys: tuple[str, ...] = ()

    def parameters(self, _obj: Any, /) -> dict[str, Any]:
        return {}

    def build(self, args: dict[str, Any], _meta: dict[str, Any], /) -> Any:
        return MLWE_Scheme(**args)

    def _parameter_key(self, given: dict[str, Any]) -> tuple:
        return tuple(sorted(given.items()))

    def identity(self, obj: Any, /) -> tuple:
        return scheme_identity(
            self.tag, obj, *self._parameter_key(self.parameters(obj))
        )

    def bindings(self, obj: Any, /) -> list[RNSRing]:
        return [*obj.rings, *obj.special_rings]

    def encode(self, obj: Any, ctx: WriteContext, /) -> Encoded:
        if type(obj) not in self.types:
            # A subclass would come back as this class, without its own state.
            raise TypeError(f"no vfhe.io codec for {type(obj).__qualname__}")
        return Encoded({**scheme_meta(obj, ctx), **self.parameters(obj)})

    def decode(
        self, meta: dict, _payload: Payload, _children: list, ctx: ReadContext, /
    ):
        args = scheme_arguments(meta, ctx)
        given = {k: meta[k] for k in self.parameter_keys}
        ident = (
            self.tag,
            tuple(ring_key(r) for r in args["rings"]),
            tuple(ring_key(r) for r in args["special_rings"]),
            args["module_rank"],
            *self._parameter_key(given),
        )
        found = ctx.bound(self.tag, ident)
        if found is not None:
            return found
        ctx.unbound(f"a {self.types[0].__name__}")
        return self.build(args, meta)


class MGSWSchemeCodec(Codec):
    tag = "mlwe.mgsw_scheme"
    types = (MGSW_Scheme,)
    definition = True

    def identity(self, obj: MGSW_Scheme, /) -> tuple:
        inner = codec_for(obj.mlwe_scheme).identity(obj.mlwe_scheme)
        return (inner, obj.ell, obj.radix_log_base)

    def bindings(self, obj: MGSW_Scheme, /) -> list[Any]:
        return [obj.mlwe_scheme]

    def encode(self, obj: MGSW_Scheme, ctx: WriteContext, /) -> Encoded:
        return Encoded(
            {
                "mlwe_scheme": ctx.ref(obj.mlwe_scheme),
                "ell": obj.ell,
                "radix_log_base": obj.radix_log_base,
            }
        )

    def decode(
        self, meta: dict, _payload: Payload, _children: list, ctx: ReadContext, /
    ):
        inner = ctx.deref(meta["mlwe_scheme"])
        ident = (codec_for(inner).identity(inner), meta["ell"], meta["radix_log_base"])
        found = ctx.bound(self.tag, ident)
        if found is not None:
            return found
        return MGSW_Scheme(
            inner, ell=meta["ell"], radix_log_base=meta["radix_log_base"]
        )


# --- samples ---------------------------------------------------------------


def seed_still_holds(c: MLWE) -> bytes | None:
    """``c.seed`` if ``c``'s mask still matches it, else None (and clears it).

    Checks the first and last prime's row of each mask component, which any
    operation rewriting the mask changes.
    """
    seed = c.seed
    if seed is None or c.is_extended or c.repr not in DOMAIN_NAMES:
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


class SampleBatch:
    """Samples of one scheme, ring, level and rank, as one payload."""

    def __init__(self, samples: Sequence[MLWE], ctx: WriteContext) -> None:
        if not samples:
            raise ValueError("an empty batch of samples")
        first = samples[0]
        for c in samples:
            if (c.scheme, c.ring, c.lvl, c.r) != (
                first.scheme,
                first.ring,
                first.lvl,
                first.r,
            ):
                raise ValueError(
                    "the samples of one record must share scheme, ring, level, rank"
                )
            if c.repr not in DOMAIN_NAMES:
                raise ValueError("cannot write a sample that holds no value")
        opts = ctx.options
        self.samples = samples
        self.rows = RowWriter(first.ring, opts.packing == "tight")
        self.seeds = [seed_still_holds(c) if opts.seeded else None for c in samples]
        self.domains = [target_domain(c.repr, opts.domain) for c in samples]
        self.meta = {
            "scheme": ctx.ref(first.scheme),
            "ring": ctx.ref(first.ring),
            "lvl": first.lvl,
            "rank": first.r,
            "tight": self.rows.tight,
            "domains": "".join(DOMAIN_NAMES[d][0] for d in self.domains),
            "seeded": "".join("1" if s else "0" for s in self.seeds),
            "extended": "".join("1" if c.is_extended else "0" for c in samples),
        }
        polys = sum(1 if s else 1 + first.r for s in self.seeds)
        self.size = polys * self.rows.size + SEED_BYTES * sum(
            1 for s in self.seeds if s
        )

    def write(self, sink: Sink) -> None:
        for c, seed, domain in zip(self.samples, self.seeds, self.domains, strict=True):
            view = c
            if c.repr != domain:
                view = c.copy()
                if domain == Repr.ntt:
                    view.to_NTT()
                else:
                    view.to_coeff()
            if seed:
                sink.write(seed)
            self.rows.write(sink, view.obj_b())
            if not seed:
                for j in range(c.r):
                    self.rows.write(sink, view.obj_a_i(j))

    @staticmethod
    def read(
        meta: dict, payload: Payload, ctx: ReadContext, cls: type[MLWE]
    ) -> list[MLWE]:
        scheme = ctx.deref(meta["scheme"])
        ring = ctx.deref(meta["ring"])
        rank = meta["rank"]
        rows = RowReader(ring, meta["tight"], ctx.options.validate)
        if "n" in meta["domains"] or "1" in meta["seeded"]:
            require_same_transform(ring, ctx)
        out = []
        for d, s, e in zip(
            meta["domains"], meta["seeded"], meta["extended"], strict=True
        ):
            domain = DOMAINS_BY_NAME["ntt" if d == "n" else "coeff"]
            c = cls(scheme, lvl=meta["lvl"], ring=ring, rank=rank)
            c.is_extended = e == "1"
            if s == "1":
                seed = payload.read(SEED_BYTES)
                rows.read(payload, c.obj_b())
                for j in range(rank):
                    a_j = c.obj_a_i(j)
                    lib.polynomial_RNS_expand_seeded(a_j, seed, SEED_BYTES, j)
                    if domain == Repr.coeff:
                        lib.polynomial_RNS_to_RNSc(a_j, a_j)
                c.seed = seed
            else:
                rows.read(payload, c.obj_b())
                for j in range(rank):
                    rows.read(payload, c.obj_a_i(j))
            c.repr = domain
            out.append(c)
        return out


class SampleCodec(Codec):
    """One MLWE sample. Subclasses carry their ciphertext type's metadata."""

    tag = "mlwe.sample"
    types = (MLWE,)
    cls: type[MLWE] = MLWE

    def parameters(self, _obj: Any, /) -> dict[str, Any]:
        return {}

    def restore(self, _obj: Any, _meta: dict[str, Any], /) -> None:
        pass

    def encode(self, obj: MLWE, ctx: WriteContext, /) -> Encoded:
        if type(obj) not in self.types:
            raise TypeError(f"no vfhe.io codec for {type(obj).__qualname__}")
        batch = SampleBatch([obj], ctx)
        return Encoded({**batch.meta, **self.parameters(obj)}, batch.size, batch.write)

    def decode(
        self, meta: dict, payload: Payload, _children: list, ctx: ReadContext, /
    ):
        (obj,) = SampleBatch.read(meta, payload, ctx, self.cls)
        self.restore(obj, meta)
        return obj


class KeySwitchKeyCodec(Codec):
    """An `MLWE_Set`: its gadget samples in one record, or nested sets."""

    tag = "mlwe.key_switch_key"
    types = (MLWE_Set,)

    def encode(self, obj: MLWE_Set, ctx: WriteContext, /) -> Encoded:
        if obj.dim > 2:
            return Encoded({"log_base": obj.log_base}, children=vars(obj)["_children"])
        lengths = [None if comp is None else len(comp) for comp in obj.mlwe]
        samples = [c for comp in obj.mlwe if comp is not None for c in comp]
        batch = SampleBatch(samples, ctx)
        meta = {**batch.meta, "log_base": obj.log_base, "lengths": lengths}
        return Encoded(meta, batch.size, batch.write)

    def decode(self, meta: dict, payload: Payload, children: list, ctx: ReadContext, /):
        if children:
            return MLWE_Set.flatten_array(children)
        samples = iter(SampleBatch.read(meta, payload, ctx, MLWE))
        components = [
            None if n is None else [next(samples) for _ in range(n)]
            for n in meta["lengths"]
        ]
        return MLWE_Set(components, meta["log_base"] or None)


class MGSWCodec(Codec):
    tag = "mlwe.mgsw"
    types = (MGSW,)

    def encode(self, obj: MGSW, ctx: WriteContext, /) -> Encoded:
        batch = SampleBatch(obj.obj, ctx)
        meta = {**batch.meta, "mgsw_scheme": ctx.ref(obj.scheme)}
        return Encoded(meta, batch.size, batch.write)

    def decode(
        self, meta: dict, payload: Payload, _children: list, ctx: ReadContext, /
    ):
        samples = SampleBatch.read(meta, payload, ctx, MLWE)
        return MGSW(ctx.deref(meta["mgsw_scheme"]), obj=samples)


# --- LWE ---------------------------------------------------------------------


def _limb_order(ring: RNSRing) -> list[int]:
    """`LWE`'s limb positions (by base index) sorted by prime value."""
    limbs = _native_limbs(ring)
    return sorted(range(len(limbs)), key=lambda j: ring.primes[limbs[j]])


class LWECodec(Codec):
    """An LWE sample: per limb in ascending prime, ``a`` then ``b`` as 8-byte
    little-endian words."""

    tag = "mlwe.lwe"
    types = (LWE,)

    def encode(self, obj: LWE, ctx: WriteContext, /) -> Encoded:
        n, order = obj.n, _limb_order(obj.ring)
        struct = ffi.cast("LWE", obj.obj)

        def write(sink: Sink) -> None:
            for j in order:
                sink.write(ffi.buffer(struct.a[j], 8 * n))
                sink.write(ffi.buffer(struct.b + j, 8))

        return Encoded(
            {"ring": ctx.ref(obj.ring), "n": n}, 8 * (n + 1) * len(order), write
        )

    def decode(
        self, meta: dict, payload: Payload, _children: list, ctx: ReadContext, /
    ):
        ring, n = ctx.deref(meta["ring"]), meta["n"]
        out = LWE(ring, n=n)
        struct = ffi.cast("LWE", out.obj)
        limbs = _native_limbs(ring)
        for j in _limb_order(ring):
            payload.readinto(ffi.buffer(struct.a[j], 8 * n))
            payload.readinto(ffi.buffer(struct.b + j, 8))
            if ctx.options.validate:
                q = ring.primes[limbs[j]]
                if max(ffi.unpack(struct.a[j], n)) >= q or struct.b[j] >= q:
                    raise ValueError(
                        f"a residue modulo {q} is not below it: corrupt stream"
                    )
        return out


# --- secret keys -------------------------------------------------------------


def _signed_code(width: int) -> str:
    """The `array` type code of a signed integer of ``width`` bytes."""
    return next(c for c in "bhilq" if array.array(c).itemsize == width)


def _pack_signed(values: Sequence[int]) -> tuple[int, bytes]:
    """The smallest width in 1, 2, 4, 8 bytes holding every value, and the
    values packed at that width."""
    lo, hi = min(values), max(values)
    for width in (1, 2, 4, 8):
        if -(1 << (8 * width - 1)) <= lo and hi < (1 << (8 * width - 1)):
            return width, array.array(_signed_code(width), values).tobytes()
    raise ValueError("a secret key coefficient does not fit in 64 bits")


def _unpack_signed(width: int, data: bytes) -> list[int]:
    return array.array(_signed_code(width), data).tolist()


class MLWEKeyCodec(Codec):
    """A secret key, as its small signed coefficients."""

    tag = "mlwe.key"
    types = (MLWE_Key,)
    secret = True

    def encode(self, obj: MLWE_Key, ctx: WriteContext, /) -> Encoded:
        flat = [x for comp in obj.key for x in comp]
        width, data = _pack_signed(flat)
        meta = {
            "scheme": ctx.ref(obj.scheme),
            "ring": ctx.ref(obj.ring),
            "sigma": obj.sigma_err,
            "width": width,
            "lengths": [len(comp) for comp in obj.key],
        }
        return Encoded(meta, len(data), lambda sink: sink.write(data))

    def decode(
        self, meta: dict, payload: Payload, _children: list, ctx: ReadContext, /
    ):
        lengths, width = meta["lengths"], meta["width"]
        flat = _unpack_signed(width, payload.read(width * sum(lengths)))
        key, at = [], 0
        for n in lengths:
            key.append(flat[at : at + n])
            at += n
        return MLWE_Key(
            key, meta["sigma"], ctx.deref(meta["scheme"]), ring=ctx.deref(meta["ring"])
        )


class LWEKeyCodec(Codec):
    tag = "mlwe.lwe_key"
    types = (LWE_Key,)
    secret = True

    def encode(self, obj: LWE_Key, ctx: WriteContext, /) -> Encoded:
        width, data = _pack_signed(obj.get_s())
        meta = {
            "ring": ctx.ref(obj.ring),
            "n": obj.n,
            "sigma": obj.err_sigma,
            "width": width,
        }
        return Encoded(meta, len(data), lambda sink: sink.write(data))

    def decode(
        self, meta: dict, payload: Payload, _children: list, ctx: ReadContext, /
    ):
        s = _unpack_signed(meta["width"], payload.read(meta["width"] * meta["n"]))
        return LWE_Key(
            ctx.deref(meta["ring"]), key=s, n=meta["n"], err_sigma=meta["sigma"]
        )


register(SchemeCodec())
register(MGSWSchemeCodec())
register(SampleCodec())
register(KeySwitchKeyCodec())
register(MGSWCodec())
register(LWECodec())
register(MLWEKeyCodec())
register(LWEKeyCodec())
