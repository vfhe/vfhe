# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""`vfhe.util.io` codecs for MLWE schemes, samples, key-switch keys, MGSW and LWE.

A sample's stream is its scheme, through the scheme codec the sample codec was
built with, then its rows, encoded as `vfhe.arith.codec` encodes
polynomials; a fresh sample whose mask still matches its seed is written as
the 32-byte seed and ``b``. A key-switch key or an MGSW ciphertext is a header,
the scheme its samples share, once, then the samples, so a key of any size
streams one sample at a time.

A codec built with the caller's schemes hands them back where a stream's
parameters match, so a loaded ciphertext refers to the scheme holding the
keys; a scheme codec built with none rebuilds the scheme from the stream. A
`BundleCodec` writes one scheme for a whole bundle of keys.
"""

from .key import LWEKeyCodec, MLWEKeyCodec
from .lwe import LWECodec
from .sample import KeySwitchKeyCodec, MGSWCodec, SampleCodec, seed_still_holds
from .scheme import BundleCodec, MGSWSchemeCodec, MLWESchemeCodec, SchemeCodec

__all__ = [
    "BundleCodec",
    "KeySwitchKeyCodec",
    "LWECodec",
    "LWEKeyCodec",
    "MGSWCodec",
    "MGSWSchemeCodec",
    "MLWEKeyCodec",
    "MLWESchemeCodec",
    "SampleCodec",
    "SchemeCodec",
    "seed_still_holds",
]
