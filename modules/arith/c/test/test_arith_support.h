/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_arith_support.h
 * @brief Shared helpers for the arith test suites (deterministic RNG, ring
 *        construction, and a negacyclic schoolbook oracle).
 *
 * Everything here is @c static @c inline so each test translation unit gets a
 * private copy -- the runner (scripts/run_c_tests.py) compiles every test file
 * independently against the full kernel set.
 */
#ifndef VFHE_TEST_ARITH_SUPPORT_H
#define VFHE_TEST_ARITH_SUPPORT_H

#include <stdint.h>

#include "arith.h"

/** Deterministic 64-bit LCG (Knuth constants); seeded once per TU. */
static inline uint64_t lcg(void)
{
    static uint64_t state = 0x123456789ABCDEFULL;
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return state;
}

/** Fresh 2-limb test ring whose primes support the split's transform size. */
static inline ring_t make_ring(uint64_t N, uint64_t split_degree, uint64_t *primes_out)
{
    primes_out[0] = nt_next_ntt_prime(1ULL << 30, N / split_degree, false);
    primes_out[1] = nt_next_ntt_prime(primes_out[0], N / split_degree, false);
    return ring_new(primes_out, split_degree, N, 2);
}

/** Negacyclic schoolbook product over Z_q[X]/(X^N + 1). */
static inline void negacyclic_schoolbook(uint64_t *out, const uint64_t *a, const uint64_t *b,
                                         uint64_t N, uint64_t q)
{
    for (uint64_t k = 0; k < N; k++)
        out[k] = 0;
    for (uint64_t i = 0; i < N; i++)
    {
        for (uint64_t j = 0; j < N; j++)
        {
            uint64_t prod = (uint64_t)((unsigned __int128)a[i] * b[j] % q);
            uint64_t k = (i + j) % N;
            out[k] = (i + j >= N) ? (out[k] + q - prod) % q : (out[k] + prod) % q;
        }
    }
}

#endif /* VFHE_TEST_ARITH_SUPPORT_H */
