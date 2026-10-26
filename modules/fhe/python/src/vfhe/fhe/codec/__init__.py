# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""`vfhe.util.io` codecs for the schemes, ciphertexts and keys of this package.

The schemes wrap `vfhe.mlwe.codec`'s scheme message with their own parameters; a
BFV ciphertext is a plain MLWE sample of a BFV scheme. The key containers
`SAB_Key` and `CGGI16_Key` and `CKKS_LinearTransform` are a header followed
by their parts, through the codecs they were built with.
"""

from .ciphertext import CKKSCiphertextCodec
from .key import CGGI16KeyCodec, SABKeyCodec
from .linear_transform import LinearTransformCodec
from .scheme import BFVSchemeCodec, CKKSSchemeCodec

__all__ = [
    "BFVSchemeCodec",
    "CGGI16KeyCodec",
    "CKKSCiphertextCodec",
    "CKKSSchemeCodec",
    "LinearTransformCodec",
    "SABKeyCodec",
]
