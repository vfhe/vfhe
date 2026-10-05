# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""`vfhe.io` codecs for the schemes, ciphertexts and keys of this package.

The schemes are `vfhe.mlwe.io` scheme records plus their own parameters;
a BFV ciphertext is a plain MLWE sample of a BFV scheme. Key containers
(`SAB_Key`, `CGGI16_Key`) and `CKKS_LinearTransform` are records whose
children are their samples and plaintexts.
"""

from __future__ import annotations

from typing import Any

from vfhe.io import Codec, Encoded, Payload, ReadContext, WriteContext, register
from vfhe.mlwe.io import SampleCodec, SchemeCodec

from .bfv import BFV_Scheme
from .cggi16 import CGGI16_Key
from .ckks import CKKS_Ciphertext, CKKS_Scheme
from .gp25 import SAB_Key
from .linear_transform import CKKS_LinearTransform


class CKKSSchemeCodec(SchemeCodec):
    tag = "fhe.ckks_scheme"
    types = (CKKS_Scheme,)
    parameter_keys = ("scaling_factor",)

    def parameters(self, obj: CKKS_Scheme, /) -> dict[str, Any]:
        return {"scaling_factor": obj.scaling_factor}

    def build(self, args: dict[str, Any], meta: dict[str, Any], /) -> CKKS_Scheme:
        return CKKS_Scheme(scaling_factor=meta["scaling_factor"], **args)


class BFVSchemeCodec(SchemeCodec):
    tag = "fhe.bfv_scheme"
    types = (BFV_Scheme,)
    parameter_keys = ("plaintext_primes",)

    def parameters(self, obj: BFV_Scheme, /) -> dict[str, Any]:
        # The plaintext modulus is the first primes of level 0, in its order.
        return {"plaintext_primes": obj.plaintext_ring.ell}

    def build(self, args: dict[str, Any], meta: dict[str, Any], /) -> BFV_Scheme:
        return BFV_Scheme(plaintext_primes=meta["plaintext_primes"], **args)


class CKKSCiphertextCodec(SampleCodec):
    """A CKKS ciphertext: a sample and its actual scaling factor."""

    tag = "fhe.ckks_ciphertext"
    types = (CKKS_Ciphertext,)
    cls = CKKS_Ciphertext

    def parameters(self, obj: CKKS_Ciphertext, /) -> dict[str, Any]:
        return {"delta": obj.delta}

    def restore(self, obj: CKKS_Ciphertext, meta: dict[str, Any], /) -> None:
        obj.delta = meta["delta"]


class SABKeyCodec(Codec):
    tag = "fhe.gp25_sab_key"
    types = (SAB_Key,)
    _fields = ("s", "s_sign", "packing_key", "hw_reducing_key", "trace_repack_key")

    def encode(self, obj: SAB_Key, _ctx: WriteContext, /) -> Encoded:
        meta = {"h": obj.h, "b_prec": obj.b_prec, "r_prec": obj.r_prec}
        return Encoded(meta, children=[getattr(obj, f) for f in self._fields])

    def decode(
        self, meta: dict, _payload: Payload, children: list, _ctx: ReadContext, /
    ):
        out = SAB_Key()
        for name, value in zip(self._fields, children, strict=True):
            setattr(out, name, value)
        out.h, out.b_prec, out.r_prec = meta["h"], meta["b_prec"], meta["r_prec"]
        return out


class CGGI16KeyCodec(Codec):
    tag = "fhe.cggi16_key"
    types = (CGGI16_Key,)

    def encode(self, obj: CGGI16_Key, _ctx: WriteContext, /) -> Encoded:
        meta = {
            "b_prec": obj.b_prec,
            "n": obj.n,
            "unfolding": obj.unfolding,
        }
        return Encoded(meta, children=[obj.bk])

    def decode(
        self, meta: dict, _payload: Payload, children: list, _ctx: ReadContext, /
    ):
        out = CGGI16_Key()
        (out.bk,) = children
        out.b_prec = meta["b_prec"]
        # A record without `n` and `unfolding` holds one key per coefficient.
        out.n = meta.get("n", len(out.bk))
        out.unfolding = meta.get("unfolding", 1)
        return out


class LinearTransformCodec(Codec):
    """The encoded diagonals, so the encoding need not be redone."""

    tag = "fhe.ckks_linear_transform"
    types = (CKKS_LinearTransform,)

    def encode(self, obj: CKKS_LinearTransform, ctx: WriteContext, /) -> Encoded:
        layout = [[j, i] for j, row in obj.plaintexts.items() for i in row]
        meta = {
            "scheme": ctx.ref(obj.scheme),
            "n": obj.n,
            "lvl": obj.lvl,
            "baby_steps": obj.baby_steps,
            "scale": obj.scale,
            "layout": layout,
        }
        return Encoded(meta, children=[obj.plaintexts[j][i] for j, i in layout])

    def decode(
        self, meta: dict, _payload: Payload, children: list, ctx: ReadContext, /
    ):
        out = CKKS_LinearTransform.__new__(CKKS_LinearTransform)
        out.scheme = ctx.deref(meta["scheme"])
        out.n, out.lvl = meta["n"], meta["lvl"]
        out.baby_steps, out.scale = meta["baby_steps"], meta["scale"]
        out.plaintexts = {}
        for (j, i), pt in zip(meta["layout"], children, strict=True):
            pt.to_NTT()
            out.plaintexts.setdefault(j, {})[i] = pt
        return out


register(CKKSSchemeCodec())
register(BFVSchemeCodec())
register(CKKSCiphertextCodec())
register(SABKeyCodec())
register(CGGI16KeyCodec())
register(LinearTransformCodec())
