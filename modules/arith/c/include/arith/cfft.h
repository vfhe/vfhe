// SPDX-License-Identifier: Apache-2.0
/**
 * @file arith/cfft.h
 * @brief Complex FFT for CKKS-style encoding.
 *
 * Double-precision radix-2 FFT over arrays stored as *split* real/imaginary
 * halves: an array of `n` complex values occupies `2n` doubles, reals in
 * `[0, n)` and imaginaries in `[n, 2n)`. Twiddle factors are supplied by the
 * caller (CKKS uses non-standard "special" roots generated Python-side in
 * high precision) and packed by ::cfft_load_twiddles_fwd /
 * ::cfft_load_twiddles_inv into the layout the build's kernels expect.
 *
 * Transform conventions match the NTT layer: forward is decimation-in-
 * frequency (natural -> bit-reversed), inverse is decimation-in-time
 * (bit-reversed -> natural), and neither applies the 1/n factor -- callers
 * scale explicitly with ::cfft_scale (see the Python `ComplexPolynomial`).
 *
 * This layer depends only on arith/poly.h (to round results into a ring
 * element); it knows nothing about the RNS tower or the NTT internals.
 */
#ifndef VFHE_ARITH_CFFT_H
#define VFHE_ARITH_CFFT_H

#include <stdint.h>

#include "poly.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * Pack forward twiddles from raw root-of-unity tables.
     *
     * @param rous_real real parts of the 2n roots, natural order
     * @param rous_imag imaginary parts
     * @param size      number of roots provided (the transform size)
     * @return backend-layout table; free-ownership stays with the caller process
     *         (tables live for the ring's lifetime in practice)
     */
    double **cfft_load_twiddles_fwd(const double *rous_real, const double *rous_imag,
                                    uint64_t size);

    /** Pack inverse twiddles; see ::cfft_load_twiddles_fwd. */
    double **cfft_load_twiddles_inv(const double *rous_real, const double *rous_imag,
                                    uint64_t size);

    /**
     * Forward FFT in place over @p n complex values (2n doubles, split layout).
     * Output is in bit-reversed order.
     */
    void cfft_forward(double *x, double *const *ws, uint64_t n);

    /**
     * Inverse FFT in place; input in bit-reversed order, output natural.
     * Does NOT apply the 1/n factor.
     */
    void cfft_inverse(double *x, double *const *ws, uint64_t n);

    /**
     * Bit-reverse permute both halves of a split complex array in place.
     *
     * @param v    2N doubles (split layout)
     * @param N    number of complex values
     * @param prec index bit width (log2 N)
     */
    void cfft_bit_reverse(double *v, uint64_t N, uint32_t prec);

    /** Multiply 2N doubles by @p scale in place (both halves). */
    void cfft_scale(double *v, double scale, uint64_t N);

    /**
     * Round 2N doubles to nearest integers and load them into @p out via
     * ::poly_from_int_array (result is ::VFHE_EVAL).
     * @return ::VFHE_OK
     */
    int cfft_round_to_poly(rns_poly_t out, const double *in);

    /**
     * Batched encode pipeline: for each of @p count rows, bit-reverse, inverse
     * FFT, scale by `delta / n_complex`, and round into the matching output
     * polynomial -- fanned out over a thread pool.
     *
     * @param rows_in   count pointers to 2*n_complex doubles (consumed in place)
     * @param outs      count polynomials to receive the results (::VFHE_EVAL)
     * @param count     number of rows
     * @param n_complex complex values per row
     * @param log_prec  log2(n_complex)
     * @param ws_inv    inverse twiddles from ::cfft_load_twiddles_inv
     * @param delta     CKKS scaling factor applied before rounding
     * @param n_threads worker threads; 0 selects the default (8)
     */
    void cfft_ifft_scale_round_batch(void **rows_in, void **outs, uint64_t count,
                                     uint64_t n_complex, uint32_t log_prec, double *const *ws_inv,
                                     double delta, uint64_t n_threads);

#ifdef __cplusplus
}
#endif

#endif // VFHE_ARITH_CFFT_H
