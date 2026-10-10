// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
#include "fhe.h"

#include <math.h>
#include <stdlib.h>
#include <util.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// The SHIP plaintexts are powers of omega = exp(2 pi i / Q) with the input's
// coefficients as exponents, built as slot vectors and encoded in one batch.

// 2 pi / Q times each centered coefficient of `in`, Q the product of its primes.
static double *angles_of(RNSc_Polynomial in)
{
    RNS_Base base = in->base;
    double modulus = 1.0;
    for (uint64_t j = 0; j < base->l; j++)
        if (in->rns_mask & (1ULL << j))
            modulus *= (double)base->mods[j]->q;
    double *angles = (double *)safe_malloc(base->N * sizeof(double));
    polynomial_RNSc_to_centered_doubles(angles, in, 2.0 * M_PI / modulus);
    return angles;
}

void ship_encode_pi(RNS_Polynomial *out, RNSc_Polynomial a, uint64_t theta, double scale,
                    double **gs_ws, uint32_t log_prec, uint64_t n_threads)
{
    const uint64_t N = a->base->N, M = N / 2;
    assert(M == 1ULL << log_prec);
    double *angles = angles_of(a);
    // pi_1, ..., pi_4 as (real, imaginary) pairs, with exponent -a.
    double *pi[4];
    for (int k = 0; k < 4; k++)
        pi[k] = (double *)safe_aligned_malloc(2 * M * sizeof(double));
    for (uint64_t t = 0; t < M; t++)
    {
        const double low = -angles[t], high = -angles[M + t];
        pi[0][2 * t] = pi[1][2 * t] = cos(low);
        pi[0][2 * t + 1] = sin(low);
        pi[1][2 * t + 1] = -sin(low);
        pi[2][2 * t] = pi[3][2 * t] = cos(high);
        pi[2][2 * t + 1] = sin(high);
        pi[3][2 * t + 1] = -sin(high);
    }
    free(angles);

    // Rot_i shifts right by i; the encoder shifts left, so by -i.
    const uint64_t count = 4 * theta;
    const double **values = (const double **)safe_malloc(count * sizeof(double *));
    uint64_t *lengths = (uint64_t *)safe_malloc(count * sizeof(uint64_t));
    uint64_t *shifts = (uint64_t *)safe_malloc(count * sizeof(uint64_t));
    for (uint64_t i = 0; i < theta; i++)
    {
        for (uint64_t k = 0; k < 4; k++)
        {
            values[4 * i + k] = pi[k];
            lengths[4 * i + k] = M;
            shifts[4 * i + k] = (M - i % M) % M;
        }
    }
    complex_values_ifft_scale_round_to_RNS_batch(out, values, lengths, shifts, count, M, log_prec,
                                                 gs_ws, scale, n_threads);
    free(values);
    free(lengths);
    free(shifts);
    for (int k = 0; k < 4; k++)
        free(pi[k]);
}

void ship_encode_pt0(RNS_Polynomial out, RNSc_Polynomial b, double coefficient_re,
                     double coefficient_im, double scale, double **gs_ws, uint32_t log_prec)
{
    const uint64_t M = b->base->N / 2;
    assert(M == 1ULL << log_prec);
    double *angles = angles_of(b);
    double *values = (double *)safe_aligned_malloc(2 * M * sizeof(double));
    for (uint64_t t = 0; t < M; t++)
    {
        const double re = cos(angles[t]), im = sin(angles[t]);
        values[2 * t] = coefficient_re * re - coefficient_im * im;
        values[2 * t + 1] = coefficient_re * im + coefficient_im * re;
    }
    free(angles);
    const double *rows[1] = {values};
    uint64_t length = M, shift = 0;
    complex_values_ifft_scale_round_to_RNS_batch(&out, rows, &length, &shift, 1, M, log_prec, gs_ws,
                                                 scale, 1);
    free(values);
}
