from __future__ import annotations

import atexit

from vfhe.misc.libvfhe import ffi, lib


class NTT_Processor:
    def __init__(self):
        self.incNTTs = {}  # (N, split_degree) -> incNTT pointer
        self.primes = {}  # (N, split_degree) -> list of primes
        self.prime_to_index = {}  # (N, split_degree) -> {prime: index}
        self.lib = lib

        # Cache for base conversion parameters: (N, split_degree, in_mask, out_mask) -> params_pointer
        self.conversion_params_cache = {}

    def register_ring_primes(self, primes, N, split_degree):
        key = (N, split_degree)
        if key not in self.incNTTs:
            self.primes[key] = list(primes)
            self.prime_to_index[key] = {p: i for i, p in enumerate(primes)}
            self.incNTTs[key] = self.lib.new_incomplete_ntt_list(
                ffi.new("uint64_t[]", list(primes)), split_degree, N, len(primes)
            )
        else:
            prime_map = self.prime_to_index[key]
            new_primes = []
            for p in primes:
                if p not in prime_map:
                    new_primes.append(p)

            if new_primes:
                start_idx = len(self.primes[key])
                self.lib.incNTT_extend_with_primes(
                    self.incNTTs[key],
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
            incNTT = self.incNTTs[(N, split_degree)]
            params = self.lib.init_base_conversion_params(incNTT, in_mask, out_mask)
            self.conversion_params_cache[key] = params
        return self.conversion_params_cache[key]

    def cleanup(self):
        for params in self.conversion_params_cache.values():
            if params:
                self.lib.free_base_conversion_params(params)
        self.conversion_params_cache.clear()


NTT_processor_instance = NTT_Processor()
atexit.register(NTT_processor_instance.cleanup)
