# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
# The engine picker's surface: the chosen engine's cffi handles (typed Any —
# the real types live on the Python wrappers), which engine that is, and
# which others could run here.
from typing import Any

ffi: Any
lib: Any
active: str
runnable: list[str]
