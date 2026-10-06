# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Exports `is_arm`, `is_x86_64`, `has_flag` and `last_level_cache_bytes`, which
report this machine's instruction set and cache.

>>> from vfhe.util import _cpu
>>> _cpu.has_flag("avx512ifma")
True
>>> _cpu.last_level_cache_bytes()
33554432
"""

from .cache import last_level_cache_bytes
from .cpu import has_flag, is_arm, is_x86_64

__all__ = ["has_flag", "is_arm", "is_x86_64", "last_level_cache_bytes"]
