# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import atexit

from vfhe.misc.libvfhe import ffi, lib


class RNS_Base_Registry:
    """The process's native ``RNS_Base`` objects, one per ``(N, split_degree)``.

    A base is shared by every `Ring` with that key, and it is append-only: a
    ring introducing a prime extends the existing base in place rather than
    getting its own. `register_ring_primes` is what does the extending, and it
    returns the ring's prime *indices* into the shared pool -- which are stable,
    unlike the pool's length (see `Ring.rns_rows`).
    """

    def __init__(self):
        self.bases = {}  # (N, split_degree) -> RNS_Base pointer
        self.primes = {}  # (N, split_degree) -> list of primes
        self.prime_to_index = {}  # (N, split_degree) -> {prime: index}
        self.lib = lib

        # Cache for base conversion parameters: (N, split_degree, in_mask, out_mask) -> params_pointer
        self.conversion_params_cache = {}

    def register_ring_primes(self, primes, N, split_degree):
        key = (N, split_degree)
        if key not in self.bases:
            self.primes[key] = list(primes)
            self.prime_to_index[key] = {p: i for i, p in enumerate(primes)}
            self.bases[key] = self.lib.new_rns_base(
                ffi.new("uint64_t[]", list(primes)), split_degree, N, len(primes)
            )
        else:
            prime_map = self.prime_to_index[key]
            new_primes = [p for p in primes if p not in prime_map]

            if new_primes:
                start_idx = len(self.primes[key])
                self.lib.rns_base_extend_with_primes(
                    self.bases[key],
                    ffi.new("uint64_t[]", new_primes),
                    len(new_primes),
                )
                for i, p in enumerate(new_primes):
                    self.primes[key].append(p)
                    prime_map[p] = start_idx + i

        return [self.prime_to_index[key][p] for p in primes]

    def get_conversion_params(self, N, split_degree, in_mask, out_mask):
        primes_tuple = tuple(self.primes[(N, split_degree)])
        key = (N, split_degree, in_mask, out_mask, primes_tuple)
        if key not in self.conversion_params_cache:
            base = self.bases[(N, split_degree)]
            params = self.lib.init_base_conversion_params(base, in_mask, out_mask)
            self.conversion_params_cache[key] = params
        return self.conversion_params_cache[key]

    def cleanup(self):
        for params in self.conversion_params_cache.values():
            if params:
                self.lib.free_base_conversion_params(params)
        self.conversion_params_cache.clear()


rns_base_registry = RNS_Base_Registry()
atexit.register(rns_base_registry.cleanup)


def registry() -> RNS_Base_Registry:
    """The registry in force right now.

    Call this instead of importing `rns_base_registry` directly. A registry
    holds `lib` as an *instance* attribute, so a dynamic-extension reload
    cannot patch it the way it patches module-level `lib` / `ffi` -- it
    replaces the whole instance instead (`_reload.reinit_rns_base`). A name
    bound at import time therefore keeps pointing at the retired registry, and
    would go on building RNS bases in the unloaded library.
    """
    return rns_base_registry
