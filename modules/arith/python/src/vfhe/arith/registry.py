# SPDX-License-Identifier: Apache-2.0
"""Process-wide registry of native ring handles.

The registry owns identity and routing -- nothing else. It caches one native
``ring_t`` per ``(N, split_degree)`` pair, tracks which primes occupy which
pool slots inside each handle, and caches base-conversion plans derived from
them. Rings that draw on the same prime pool therefore share twiddle tables
and reduction contexts instead of rebuilding them.
"""

from __future__ import annotations

import weakref

from _vfhe_native import ffi, lib


class RingRegistry:
    """Singleton cache of native ring handles and base-conversion plans.

    ``RingRegistry()`` always returns the one shared instance. Handles are
    keyed by ``(N, split_degree)``; the C pool inside a handle is append-only,
    so prime indices handed out earlier stay valid as new primes register.

    The C-owned base-conversion plans are released at interpreter teardown via
    a :func:`weakref.finalize` tied to the instance. Ring handles themselves
    are deliberately leaked to the OS: polynomials may outlive the registry
    during teardown and freeing the rings under them would crash.
    """

    _instance: RingRegistry | None = None

    def __new__(cls) -> RingRegistry:
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._init_state()
            cls._instance = instance
        return cls._instance

    def _init_state(self) -> None:
        # Keys below are (N, split_degree).
        self.handles: dict = {}  # -> native ring_t
        self.primes: dict = {}  # -> list of primes, in pool order
        self.prime_to_index: dict = {}  # -> {prime: pool slot}
        # (N, split_degree, in_mask, out_mask, primes) -> baseconv_plan_t
        self.conversion_plans: dict = {}
        self._finalizer = weakref.finalize(self, self._release, self.conversion_plans)

    @staticmethod
    def _release(conversion_plans: dict) -> None:
        """Free the C-owned base-conversion plans. Runs once, at teardown."""
        for plan in conversion_plans.values():
            if plan:
                lib.baseconv_plan_free(plan)
        conversion_plans.clear()

    def register_ring_primes(self, primes, N, split_degree) -> list[int]:
        """Ensure a handle exists for ``(N, split_degree)`` and covers ``primes``.

        Extends the pool with any not-yet-seen primes and returns each prime's
        pool slot within the handle (the bit positions used in limb masks).
        """
        key = (N, split_degree)
        if key not in self.handles:
            self.primes[key] = list(primes)
            self.prime_to_index[key] = {p: i for i, p in enumerate(primes)}
            arr = ffi.new("uint64_t[]", list(primes))
            self.handles[key] = lib.ring_new(arr, split_degree, N, len(primes))
        else:
            prime_map = self.prime_to_index[key]
            new_primes = [p for p in primes if p not in prime_map]
            if new_primes:
                start_idx = len(self.primes[key])
                arr = ffi.new("uint64_t[]", new_primes)
                lib.ring_extend(self.handles[key], arr, len(new_primes))
                for i, p in enumerate(new_primes):
                    self.primes[key].append(p)
                    prime_map[p] = start_idx + i
        return [self.prime_to_index[key][p] for p in primes]

    def handle(self, N, split_degree):
        """The native ring handle for ``(N, split_degree)`` (must exist)."""
        return self.handles[(N, split_degree)]

    def pool_size(self, N, split_degree) -> int:
        """Number of primes currently in the ``(N, split_degree)`` pool."""
        return len(self.primes[(N, split_degree)])

    def get_conversion_plan(self, N, split_degree, in_mask, out_mask):
        """The (cached) base-conversion plan for ``in_mask -> out_mask``."""
        primes = tuple(self.primes[(N, split_degree)])
        key = (N, split_degree, in_mask, out_mask, primes)
        if key not in self.conversion_plans:
            handle = self.handles[(N, split_degree)]
            self.conversion_plans[key] = lib.baseconv_plan_new(
                handle, in_mask, out_mask
            )
        return self.conversion_plans[key]
