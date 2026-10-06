# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Exports `Engine`, which lists the engines known to this build.

>>> from vfhe.util._engine import Engine
>>> Engine.select()
<Engine.AVX512IFMA: 'avx512ifma'>
>>> [e.value for e in Engine if e.installed()]
['avx512ifma', 'portable']
"""

from .engine import Engine

__all__ = ["Engine"]
