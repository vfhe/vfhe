# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""`Options`, what a `Serializer` optimizes for, and the `PROFILES` presets."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

PACKINGS = ("word", "tight")
DOMAINS = ("held", "mul", "canonical")


@dataclass(frozen=True)
class Options:
    """What a `Serializer` optimizes for. See `PROFILES` for the presets.

    ``seeded``
        Write a fresh sample's mask as its seed, halving the size of fresh
        ciphertexts and keys; the mask is expanded again on load.
    ``packing``
        ``"word"``: each residue as a 4- or 8-byte word, copied as is.
        ``"tight"``: each residue in exactly the bits its prime needs.
    ``domain``
        The domain rows are written in: ``"held"`` (as held, no conversion),
        ``"mul"`` or ``"canonical"``. The size is the same; ``canonical``
        does not depend on the NTT convention but costs a transform on load.
    ``checksum``
        A checksum per record, verified on load. It detects corruption, not
        tampering.
    ``validate``
        On load, check that every residue is below its prime.
    """

    seeded: bool = True
    packing: str = "word"
    domain: str = "held"
    checksum: bool = True
    validate: bool = True

    def __post_init__(self) -> None:
        if self.packing not in PACKINGS:
            raise ValueError(f"packing must be one of {PACKINGS}, got {self.packing!r}")
        if self.domain not in DOMAINS:
            raise ValueError(f"domain must be one of {DOMAINS}, got {self.domain!r}")


#: ``compact`` minimizes size; ``fast`` minimizes load time (no expansion,
#: unpacking, checksum or range check).
PROFILES: dict[str, Options] = {
    "default": Options(),
    "compact": Options(packing="tight"),
    "fast": Options(seeded=False, checksum=False, validate=False),
}


def options_for(profile: str | None, overrides: dict[str, Any]) -> Options:
    if profile is None:
        profile = "default"
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {sorted(PROFILES)}, got {profile!r}")
    return replace(
        PROFILES[profile], **{k: v for k, v in overrides.items() if v is not None}
    )
