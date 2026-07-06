// SPDX-License-Identifier: Apache-2.0
/**
 * @file arith/nt.h
 * @brief Number theory on plain 64-bit integers.
 *
 * Pure functions with no context and no state: primality testing, modular
 * exponentiation/inversion, and root-of-unity / NTT-prime search. This is the
 * bottom layer of the engine; everything else builds on it.
 */
#ifndef VFHE_ARITH_NT_H
#define VFHE_ARITH_NT_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * Deterministic Miller-Rabin primality test, exact for all 64-bit inputs.
     *
     * @param n candidate
     * @return true iff @p n is prime
     */
    bool nt_is_prime(uint64_t n);

    /**
     * Modular exponentiation: `base^exp mod mod`.
     *
     * @param base base (reduced mod @p mod internally)
     * @param exp  exponent
     * @param mod  modulus, must be nonzero
     * @return `base^exp mod mod`
     */
    uint64_t nt_power_mod(uint64_t base, uint64_t exp, uint64_t mod);

    /**
     * Modular inverse via Fermat's little theorem.
     *
     * @param a element to invert, `gcd(a, m) == 1`
     * @param m modulus, **must be prime**
     * @return `a^-1 mod m`
     */
    uint64_t nt_inverse_mod(uint64_t a, uint64_t m);

    /**
     * Find a primitive n-th root of unity modulo a prime.
     *
     * @param q prime modulus with `n | q - 1`
     * @param n order; must be a power of two
     * @return w with `w^n == 1` and `w^(n/2) == -1 (mod q)`
     */
    uint64_t nt_gen_root_of_unity(uint64_t q, uint64_t n);

    /**
     * Smallest NTT-friendly prime strictly greater than @p x.
     *
     * Returns the least prime `p > x` with `p == 1 (mod 2n)` so that Z_p contains
     * a primitive 2n-th root of unity (as required by the negacyclic NTT of size
     * n). With @p primitive set, additionally requires `p != 1 (mod 4n)`, which
     * guarantees the 2n-th roots are *exactly* of order 2n (used when the caller
     * needs the root's square to have order n).
     *
     * @param x         lower bound (exclusive)
     * @param n         transform size (power of two)
     * @param primitive require `p % 4n != 1`
     * @return the prime found
     */
    uint64_t nt_next_ntt_prime(uint64_t x, uint64_t n, bool primitive);

#ifdef __cplusplus
}
#endif

#endif // VFHE_ARITH_NT_H
