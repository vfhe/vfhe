// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
// Scalar negacyclic FFT, built into every engine: the whole implementation
// without AVX-512, and the transform for n < 32 with it. Same tables, same
// split layout and same bit-reversed evaluation order as the vector code in
// fft.c, with plain radix-2 stages.
#include <arith.h>

#include "arith_internal.h"

void fft_forward_scalar(double *x, const struct _FFT_Plan *plan)
{
    const uint64_t m = plan->m;
    double *re = x, *im = x + m;
    // h groups at distance t; group i uses ws[h + i].
    for (uint64_t h = 1, t = m >> 1; h < m; h <<= 1, t >>= 1)
    {
        for (uint64_t i = 0; i < h; i++)
        {
            const double wr = plan->ws_re[h + i], wi = plan->ws_im[h + i];
            for (uint64_t j = 2 * i * t; j < 2 * i * t + t; j++)
            {
                const double vr = wr * re[j + t] - wi * im[j + t];
                const double vi = wr * im[j + t] + wi * re[j + t];
                re[j + t] = re[j] - vr;
                im[j + t] = im[j] - vi;
                re[j] += vr;
                im[j] += vi;
            }
        }
    }
}

void fft_inverse_scalar(double *x, const struct _FFT_Plan *plan)
{
    const uint64_t m = plan->m;
    double *re = x, *im = x + m;
    for (uint64_t h = m >> 1, t = 1; h >= 1; h >>= 1, t <<= 1)
    {
        for (uint64_t i = 0; i < h; i++)
        {
            const double wr = plan->ws_re[h + i], wi = plan->wsn_im[h + i];
            for (uint64_t j = 2 * i * t; j < 2 * i * t + t; j++)
            {
                const double sr = re[j] - re[j + t], si = im[j] - im[j + t];
                re[j] += re[j + t];
                im[j] += im[j + t];
                re[j + t] = wr * sr - wi * si;
                im[j + t] = wr * si + wi * sr;
            }
        }
    }
    for (uint64_t j = 0; j < 2 * m; j++)
        x[j] *= plan->inv_m;
}
