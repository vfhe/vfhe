// SPDX-License-Identifier: Apache-2.0
/**
 * @file arith/mp.h
 * @brief Multi-precision (big-integer) polynomials and the CRT bridge to RNS.
 *
 * This layer does exact integer arithmetic on polynomials whose coefficients
 * exceed one machine word, stored in base 2^52 digits. Its main job is the
 * CRT bridge: reconstructing a big-integer polynomial from an RNS one
 * (::mp_polynomial_from_poly) and reducing back (::mp_polynomial_to_poly).
 *
 * **Digit type.** On AVX-512 IFMA builds a digit column is a 512-bit lane
 * (::mp_vector_t = `__m512i`, 8 coefficients per vector); on portable builds
 * it is a plain `uint64_t`. This changes the layout of ::mp_scalar, which is
 * therefore opaque to Python -- the FFI reads digits through
 * ::mp_scalar_digit / ::mp_scalar_digit_count and queries the lane count with
 * ::mp_vector_size. ::mp_polynomial rows are always plain `uint64_t` arrays
 * (one digit per coefficient per row) and stay transparent.
 *
 * Coefficients are unsigned; negatives are represented by their radix
 * complement and interpreted modulo the target modulus at reduction time.
 */
#ifndef VFHE_ARITH_MP_H
#define VFHE_ARITH_MP_H

#include <stdint.h>

#include "config.h"
#include "poly.h"

#if VFHE_MP_SIMD
#include <immintrin.h>
/** One digit column: 8 packed base-2^52 digits (AVX-512 build). */
typedef __m512i mp_vector_t;
#else
/** One digit column: a single base-2^52 digit (portable build). */
typedef uint64_t mp_vector_t;
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * Big-integer polynomial: `d` digit rows of `N` coefficients each;
     * `coeffs[j][i]` is digit j (weight 2^52j) of coefficient i.
     */
    typedef struct mp_polynomial
    {
        uint64_t **coeffs; /**< d rows of N base-2^52 digits. */
        uint64_t N;        /**< Number of coefficients. */
        uint64_t d;        /**< Number of digits per coefficient. */
    } *mp_polynomial_t;

    /**
     * Big-integer scalar broadcast across a digit column: `digits[j]` holds
     * digit j replicated in every lane (see file docs for the lane count).
     */
    typedef struct mp_scalar
    {
        mp_vector_t *digits; /**< d digit columns. */
        uint64_t d;          /**< Number of digits. */
    } *mp_scalar_t;

    // --- Digit / lane helpers ------------------------------------------------------

    /** Lanes per ::mp_vector_t: 8 on AVX-512 builds, 1 on portable builds. */
    int mp_vector_size(void);

    /** Broadcast a machine word into a freshly allocated digit column (caller frees). */
    mp_vector_t *mp_vector_load(uint64_t in);

    /**
     * Extract the 52-bit slice starting at bit @p start from an array of 32-bit
     * chunks stored in 64-bit words (helper for building digit schedules).
     */
    uint64_t mp_bit_slice_52(const uint64_t *array, uint64_t start);

    // --- mp_scalar -------------------------------------------------------------------

    /** Load a @p d-digit scalar from base-2^52 digits (broadcast per column). */
    mp_scalar_t mp_scalar_load(const uint64_t *in, uint64_t d);

    /** Free a scalar from ::mp_scalar_load. */
    void mp_scalar_free(mp_scalar_t s);

    /** Number of digits. (FFI accessor; the struct is opaque to Python.) */
    uint64_t mp_scalar_digit_count(const mp_scalar_t s);

    /** Digit @p i as a machine word (lane 0 on SIMD builds). */
    uint64_t mp_scalar_digit(const mp_scalar_t s, uint64_t i);

    /**
     * out = in * m for a single-digit multiplier column @p m, with carry
     * normalization. Requires `out != in` and `out->d >= in->d + 1`.
     */
    void mp_scalar_scale(mp_scalar_t out, const mp_scalar_t in, const mp_vector_t *m);

    /** out = a - b in radix-complement arithmetic (digit-wise, then carries). */
    void mp_scalar_sub(mp_scalar_t out, const mp_scalar_t a, const mp_scalar_t b);

    /** Debug: print the scalar as a hex big integer (lane 0). */
    void mp_scalar_print(const mp_scalar_t x);

    // --- mp_polynomial lifecycle -----------------------------------------------------

    /** Allocate a zeroed N x d big-integer polynomial. */
    mp_polynomial_t mp_polynomial_new(uint64_t N, uint64_t d);

    /** Free rows + struct. */
    void mp_polynomial_free(mp_polynomial_t p);

    /** Zero every digit. */
    void mp_polynomial_zero(mp_polynomial_t poly);

    /** Fill with uniform digits (each < 2^52). */
    void mp_polynomial_randomize(mp_polynomial_t poly);

    // --- mp_polynomial arithmetic ------------------------------------------------------
    // Digit rows are lazily carried: operations may leave digits above 2^52; call
    // ::mp_polynomial_propagate_carry to renormalize where the docs require it.

    /** out = a + b, digit-wise (no carry propagation). */
    void mp_polynomial_add(mp_polynomial_t out, const mp_polynomial_t a, const mp_polynomial_t b);

    /** out = -in (radix complement; carries pending). */
    void mp_polynomial_negate(mp_polynomial_t out, const mp_polynomial_t in);

    /** out = in * X^a in Z[X]/(X^N + 1) (a mod 2N; wrap negates). Inputs carry-free. */
    void mp_polynomial_mul_by_xai(mp_polynomial_t out, const mp_polynomial_t in, uint64_t a);

    /**
     * out += a * b for a sparse polynomial b given as @p size signed monomial
     * exponents (`b[i]`'s low bits = exponent, sign bit = negated monomial).
     * Carries are propagated on exit.
     */
    void mp_polynomial_mul_addto_sparse(mp_polynomial_t out, const mp_polynomial_t a,
                                        const uint64_t *b, uint64_t size);

    /**
     * Digit-local scale by a word, **downward-carry convention**: digit j of
     * @p out receives `lo52(in_j * s)` and digit j-1 accumulates
     * `hi(in_j * s)` (the digit-0 high half is discarded). Exact
     * (out_j = in_j * s) whenever every digit product stays below 2^52.
     * The downward carry treats @p s as a fixed-point fraction, matching the
     * 1/p digit schedules of ::mp_mod_switch_delta_setup. Input must be
     * carry-free. The @c _fixedpoint suffix flags this convention: it is NOT a
     * full multiply (contrast ::mp_polynomial_scale_int_by_scalar).
     */
    void mp_polynomial_scale_fixedpoint(mp_polynomial_t out, const mp_polynomial_t in,
                                        uint64_t scale);

    /** Accumulating ::mp_polynomial_scale_fixedpoint (same convention). */
    void mp_polynomial_scale_fixedpoint_addto(mp_polynomial_t out, const mp_polynomial_t in,
                                              uint64_t scale);

    /**
     * Digit row 0 of @p in scaled by a per-digit column vector, same
     * downward-carry convention as ::mp_polynomial_scale_fixedpoint: out_j gets
     * `lo52(scale_j * in_0)`, out_{j-1} accumulates the high half.
     */
    void mp_polynomial_scale_limb_by_scalar(mp_polynomial_t out, const mp_polynomial_t in,
                                            const mp_vector_t *scale);

    /** out = scale * in for a machine-word residue row @p in (N words). */
    void mp_polynomial_scale_int_by_scalar(mp_polynomial_t out, const uint64_t *in,
                                           const mp_scalar_t scale);

    /** out += scale * in (accumulating variant of the above). */
    void mp_polynomial_scale_int_by_scalar_addto(mp_polynomial_t out, const uint64_t *in,
                                                 const mp_scalar_t scale);

    // --- Carry management and modular reduction -----------------------------------------

    /** Propagate pending carries so every digit is < 2^52 (top carry dropped). */
    void mp_polynomial_propagate_carry(mp_polynomial_t p);

    /** Drop the top @p num_digits digit rows (shrinks d; frees the rows). */
    void mp_polynomial_drop_digits(mp_polynomial_t p, uint64_t num_digits);

    /**
     * In-place Barrett reduction of every coefficient modulo the big modulus
     * @p q, with multiplier @p m = floor(2^k / q) (single digit column) and shift
     * @p k. Coefficients must be < q^2-ish (standard Barrett input range).
     */
    void mp_polynomial_mod_reduce(mp_polynomial_t out, const mp_scalar_t q, const mp_vector_t *m,
                                  uint64_t k);

    /**
     * Precompute the digit schedule of 1/p used by modulus switching (cached in a
     * process-global; re-entrant per (d, p) pair, not thread-safe on first call).
     */
    void mp_mod_switch_delta_setup(uint64_t d, uint64_t p);

    // --- CRT bridge to/from RNS -----------------------------------------------------------

    /**
     * Reconstruct the big-integer polynomial of an RNS polynomial:
     * out = sum_i in_i * PW[i] mod q, where PW[i] = (Q/q_i) * ((Q/q_i)^-1 mod q_i)
     * and (q, m, k) are the Barrett triple for Q (see ::mp_polynomial_mod_reduce).
     * @p in must be ::VFHE_COEFF.
     * @return ::VFHE_OK or ::VFHE_ERR_DOMAIN.
     */
    int mp_polynomial_from_poly(mp_polynomial_t out, const rns_poly_t in, const mp_scalar_t *PW,
                                const mp_scalar_t q, const mp_vector_t *m, uint64_t k);

    /**
     * Reduce a big-integer polynomial into every active limb of @p out
     * (Horner over the digits with 2^52 mod q). Result is ::VFHE_COEFF.
     * @return ::VFHE_OK.
     */
    int mp_polynomial_to_poly(rns_poly_t out, const mp_polynomial_t in);

#ifdef __cplusplus
}
#endif

#endif // VFHE_ARITH_MP_H
