// SPDX-License-Identifier: Apache-2.0
/**
 * @file cfft.c
 * @brief Complex FFT kernels for CKKS-style encoding (see arith/cfft.h).
 *
 * Two implementations behind one API: an AVX-512 path processing 8 doubles
 * per vector with per-stage packed twiddles, and a portable scalar path with
 * flat twiddle tables. The variant is fixed at compile time by the same
 * portable/SIMD switch as the rest of the engine (plain AVX-512F suffices
 * here; no IFMA needed).
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <arith/cfft.h>
#include <base.h>

#if defined(__AVX512F__) && !defined(PORTABLE_BUILD) && !defined(PORTABLE)
#define VFHE_CFFT_SIMD 1
#include <immintrin.h>
#else
#define VFHE_CFFT_SIMD 0
#endif

/** Bit-reverse a value of `prec` bits via a nibble lookup. */
static uint32_t bit_reverse_u32(uint32_t x, uint32_t prec)
{
    static const uint8_t lookup[16] = {0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe,
                                       0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf};
    const uint32_t x0 = x & 0xF, x1 = (x >> 4) & 0xF, x2 = (x >> 8) & 0xF, x3 = (x >> 12) & 0xF,
                   x4 = (x >> 16) & 0xF;
    const uint32_t result = (((uint32_t)lookup[x0]) << 16) | (((uint32_t)lookup[x1]) << 12) |
                            (((uint32_t)lookup[x2]) << 8) | (((uint32_t)lookup[x3]) << 4) |
                            ((uint32_t)lookup[x4]);
    return result >> (20 - prec);
}

void cfft_bit_reverse(double *v, uint64_t N, uint32_t prec)
{
    double *tmp = (double *)safe_malloc(N * sizeof(double));
    // Real half.
    for (uint64_t i = 0; i < N; i++)
        tmp[bit_reverse_u32((uint32_t)i, prec)] = v[i];
    memcpy(v, tmp, sizeof(double) * N);
    // Imaginary half.
    for (uint64_t i = 0; i < N; i++)
        tmp[bit_reverse_u32((uint32_t)i, prec)] = v[i + N];
    memcpy(&v[N], tmp, sizeof(double) * N);
    free(tmp);
}

int cfft_round_to_poly(rns_poly_t out, const double *in)
{
    const uint64_t n = out->ring->N;
    uint64_t *tmp = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * n);
    for (uint64_t i = 0; i < n; i++)
    {
        const int64_t r = (int64_t)llrint(in[i]);
        tmp[i] = (uint64_t)r;
    }
    int rc = poly_from_int_array(out, tmp);
    free(tmp);
    return rc;
}

#if VFHE_CFFT_SIMD

// c = a * b (complex, split real/imag vectors)
#define COMPLEX_MULT(c_real, c_imag, a_real, a_imag, b_real, b_imag)                               \
    {                                                                                              \
        c_imag = _mm512_mul_pd(b_real, a_imag);                                                    \
        c_imag = _mm512_fmadd_pd(a_real, b_imag, c_imag);                                          \
        c_real = _mm512_mul_pd(a_imag, b_imag);                                                    \
        c_real = _mm512_fmsub_pd(a_real, b_real, c_real);                                          \
    }

void cfft_scale(double *v, double scale, uint64_t N)
{
    const __m512d scalev = _mm512_set1_pd(scale);
    __m512d *vv = (__m512d *)(v);
    for (uint64_t i = 0; i < (N >> 2); i++)
    {
        vv[i] = _mm512_mul_pd(vv[i], scalev);
    }
}

double **cfft_load_twiddles_fwd(const double *rous_real, const double *rous_imag, uint64_t size)
{
    const uint64_t logn = (uint64_t)log2((double)size);
    double **rous = (double **)safe_malloc(logn * sizeof(double *));
    uint64_t level;
    // Broadcast stages.
    for (level = 0; level < logn - 3; level++)
    {
        const uint64_t h = (1ULL << level);
        rous[level] = (double *)safe_aligned_malloc(2 * sizeof(__m512d) * h);
        __m512d *rouv = (__m512d *)rous[level];
        for (uint64_t i = 0; i < h; i++)
        {
            rouv[2 * i] = _mm512_set1_pd(rous_real[h + i]);
            rouv[2 * i + 1] = _mm512_set1_pd(rous_imag[h + i]);
        }
    }
    { // t = 4 packing
        rous[level] = (double *)safe_aligned_malloc(2 * sizeof(__m512d) * (1ULL << (level - 1)));
        __m512d *rouv = (__m512d *)rous[level];
        const uint64_t h = (1ULL << level);
        for (uint64_t i = 0; i < (1ULL << (level - 1)); i++)
        {
            rouv[2 * i] = _mm512_set_pd(rous_real[h + 2 * i + 1], rous_real[h + 2 * i + 1], 0, 0,
                                        rous_real[h + 2 * i], rous_real[h + 2 * i], 0, 0);
            rouv[2 * i + 1] = _mm512_set_pd(rous_imag[h + 2 * i + 1], rous_imag[h + 2 * i + 1], 0,
                                            0, rous_imag[h + 2 * i], rous_imag[h + 2 * i], 0, 0);
        }
    }
    level++;
    { // t = 2 packing
        rous[level] = (double *)safe_aligned_malloc(2 * sizeof(__m512d) * (1ULL << (level - 2)));
        __m512d *rouv = (__m512d *)rous[level];
        const uint64_t h = (1ULL << level);
        for (uint64_t i = 0; i < (1ULL << (level - 2)); i++)
        {
            rouv[2 * i] = _mm512_set_pd(rous_real[h + 4 * i + 3], 0, rous_real[h + 4 * i + 2], 0,
                                        rous_real[h + 4 * i + 1], 0, rous_real[h + 4 * i], 0);
            rouv[2 * i + 1] =
                _mm512_set_pd(rous_imag[h + 4 * i + 3], 0, rous_imag[h + 4 * i + 2], 0,
                              rous_imag[h + 4 * i + 1], 0, rous_imag[h + 4 * i], 0);
        }
    }
    return rous;
}

double **cfft_load_twiddles_inv(const double *rous_real, const double *rous_imag, uint64_t size)
{
    const uint64_t logn = (uint64_t)log2((double)size), n = size;
    double **rous = (double **)safe_malloc(logn * sizeof(double *));
    uint64_t level = 0;
    {
        const uint64_t h = (n >> (level + 2));
        rous[level] = (double *)safe_aligned_malloc(2 * sizeof(__m512d) * (h >> 2));
        __m512d *rouv = (__m512d *)rous[level];
        for (uint64_t i = 0; i < (h >> 2); i++)
        {
            rouv[2 * i] = _mm512_set_pd(rous_real[h + 4 * i + 3], 1, rous_real[h + 4 * i + 2], 1,
                                        rous_real[h + 4 * i + 1], 1, rous_real[h + 4 * i], 1);
            rouv[2 * i + 1] =
                _mm512_set_pd(rous_imag[h + 4 * i + 3], 0, rous_imag[h + 4 * i + 2], 0,
                              rous_imag[h + 4 * i + 1], 0, rous_imag[h + 4 * i], 0);
        }
    }
    level++;
    {
        const uint64_t h = (n >> (level + 2));
        rous[level] = (double *)safe_aligned_malloc((2 * sizeof(__m512d) * h) >> 1);
        __m512d *rouv = (__m512d *)rous[level];
        for (uint64_t i = 0; i < (h >> 1); i++)
        {
            rouv[2 * i] = _mm512_set_pd(rous_real[h + 2 * i + 1], rous_real[h + 2 * i + 1], 1, 1,
                                        rous_real[h + 2 * i], rous_real[h + 2 * i], 1, 1);
            rouv[2 * i + 1] = _mm512_set_pd(rous_imag[h + 2 * i + 1], rous_imag[h + 2 * i + 1], 0,
                                            0, rous_imag[h + 2 * i], rous_imag[h + 2 * i], 0, 0);
        }
    }
    level++;
    {
        const uint64_t h = (n >> (level + 2));
        rous[level] = (double *)safe_aligned_malloc(2 * sizeof(__m512d) * h);
        __m512d *rouv = (__m512d *)rous[level];
        for (uint64_t i = 0; i < h; i++)
        {
            rouv[2 * i] = _mm512_set_pd(rous_real[h + i], rous_real[h + i], rous_real[h + i],
                                        rous_real[h + i], 1, 1, 1, 1);
            rouv[2 * i + 1] = _mm512_set_pd(rous_imag[h + i], rous_imag[h + i], rous_imag[h + i],
                                            rous_imag[h + i], 0, 0, 0, 0);
        }
    }
    for (level++; level < logn; level++)
    {
        const uint64_t h = (n >> (level + 2));
        rous[level] = (double *)safe_aligned_malloc(2 * sizeof(__m512d) * h);
        __m512d *rouv = (__m512d *)rous[level];
        for (uint64_t i = 0; i < h; i++)
        {
            rouv[2 * i] = _mm512_set1_pd(rous_real[h + i]);
            rouv[2 * i + 1] = _mm512_set1_pd(rous_imag[h + i]);
        }
    }
    return rous;
}

void cfft_forward(double *x, double *const *ws, uint64_t n)
{
    uint64_t t_vec = n >> 3;
    __m512d *real = (__m512d *)x;
    __m512d *imag = &((__m512d *)x)[t_vec];
    __m512d V_real, V_imag, U_real, U_imag;
    uint64_t level;
    for (level = 0; (1ULL << (level + 3)) < n; level++)
    {
        const __m512d *wsv = (const __m512d *)ws[level];
        const uint64_t t = n >> (level + 4);
        for (uint64_t i = 0; i < (1ULL << level); i++)
        {
            const uint64_t slice = 2 * i * t;
            const __m512d w_real = wsv[2 * i];
            const __m512d w_imag = wsv[2 * i + 1];
            for (uint64_t j = slice; j < slice + t; j++)
            {
                COMPLEX_MULT(V_real, V_imag, real[j + t], imag[j + t], w_real, w_imag);
                real[j + t] = _mm512_sub_pd(real[j], V_real);
                imag[j + t] = _mm512_sub_pd(imag[j], V_imag);
                real[j] = _mm512_add_pd(real[j], V_real);
                imag[j] = _mm512_add_pd(imag[j], V_imag);
            }
        }
    }
    { // t = 4
        const __m512i permute_U_idx_t4 = _mm512_set_epi64(3, 2, 1, 0, 3, 2, 1, 0);
        const __m512i permute_V_idx_t4 = _mm512_set_epi64(7, 6, 5, 4, 7, 6, 5, 4);
        const __m512d negate_t4 = _mm512_set_pd(-1, -1, -1, -1, 1, 1, 1, 1);
        const __m512d *wsv = (const __m512d *)ws[level];
        const uint64_t t = n >> (level + 1);
        for (uint64_t i = 0; i < (1ULL << level); i++)
        {
            const uint64_t slice = 2 * i * t >> 3;
            __m512d w_real = wsv[2 * i];
            __m512d w_imag = wsv[2 * i + 1];
            COMPLEX_MULT(V_real, V_imag, real[slice], imag[slice], w_real, w_imag);
            V_real = _mm512_permutexvar_pd(permute_V_idx_t4, V_real);
            V_imag = _mm512_permutexvar_pd(permute_V_idx_t4, V_imag);
            U_real = _mm512_permutexvar_pd(permute_U_idx_t4, real[slice]);
            U_imag = _mm512_permutexvar_pd(permute_U_idx_t4, imag[slice]);
            real[slice] = _mm512_fmadd_pd(V_real, negate_t4, U_real);
            imag[slice] = _mm512_fmadd_pd(V_imag, negate_t4, U_imag);
        }
    }
    level++;
    { // t = 2
        const __m512i permute_U_idx_t2 = _mm512_set_epi64(5, 4, 5, 4, 1, 0, 1, 0);
        const __m512i permute_V_idx_t2 = _mm512_set_epi64(7, 6, 7, 6, 3, 2, 3, 2);
        const __m512d negate_t2 = _mm512_set_pd(-1, -1, 1, 1, -1, -1, 1, 1);
        const __m512d *wsv = (const __m512d *)ws[level];
        const uint64_t t = n >> (level + 1);
        for (uint64_t i = 0; i < (1ULL << (level - 1)); i++)
        {
            const uint64_t slice = 2 * i * t >> 2;
            __m512d w_real = wsv[2 * i];
            __m512d w_imag = wsv[2 * i + 1];
            COMPLEX_MULT(V_real, V_imag, real[slice], imag[slice], w_real, w_imag);
            V_real = _mm512_permutexvar_pd(permute_V_idx_t2, V_real);
            V_imag = _mm512_permutexvar_pd(permute_V_idx_t2, V_imag);
            U_real = _mm512_permutexvar_pd(permute_U_idx_t2, real[slice]);
            U_imag = _mm512_permutexvar_pd(permute_U_idx_t2, imag[slice]);
            real[slice] = _mm512_fmadd_pd(V_real, negate_t2, U_real);
            imag[slice] = _mm512_fmadd_pd(V_imag, negate_t2, U_imag);
        }
    }
    level++;
    { // t = 1
        const __m512i permute_U_idx_t1 = _mm512_set_epi64(6, 6, 4, 4, 2, 2, 0, 0);
        const __m512i permute_V_idx_t1 = _mm512_set_epi64(7, 7, 5, 5, 3, 3, 1, 1);
        const __m512d negate_t1 = _mm512_set_pd(-1, 1, -1, 1, -1, 1, -1, 1);
        const __m512d *wsv = (const __m512d *)ws[level];
        for (uint64_t i = 0; i < (1ULL << (level - 2)); i++)
        {
            const uint64_t slice = i;
            __m512d w_real = wsv[2 * i];
            __m512d w_imag = wsv[2 * i + 1];
            COMPLEX_MULT(V_real, V_imag, real[slice], imag[slice], w_real, w_imag);
            V_real = _mm512_permutexvar_pd(permute_V_idx_t1, V_real);
            V_imag = _mm512_permutexvar_pd(permute_V_idx_t1, V_imag);
            U_real = _mm512_permutexvar_pd(permute_U_idx_t1, real[slice]);
            U_imag = _mm512_permutexvar_pd(permute_U_idx_t1, imag[slice]);
            real[slice] = _mm512_fmadd_pd(V_real, negate_t1, U_real);
            imag[slice] = _mm512_fmadd_pd(V_imag, negate_t1, U_imag);
        }
    }
}

void cfft_inverse(double *x, double *const *ws, uint64_t n)
{
    uint64_t t_vec = n >> 3;
    __m512d *real = (__m512d *)x;
    __m512d *imag = &((__m512d *)x)[t_vec];
    __m512d V_real, V_imag, U_real, U_imag;
    uint64_t level = 0;
    { // t = 1
        const __m512i permute_U_idx_t1 = _mm512_set_epi64(6, 6, 4, 4, 2, 2, 0, 0);
        const __m512i permute_V_idx_t1 = _mm512_set_epi64(7, 7, 5, 5, 3, 3, 1, 1);
        const __m512d negate_t1 = _mm512_set_pd(-1, 1, -1, 1, -1, 1, -1, 1);
        const __m512d *wsv = (const __m512d *)ws[level];
        for (uint64_t i = 0; i < (n >> (level + 3)); i++)
        {
            const uint64_t slice = i;
            __m512d w_real = wsv[2 * i];
            __m512d w_imag = wsv[2 * i + 1];
            V_real = _mm512_permutexvar_pd(permute_V_idx_t1, real[slice]);
            V_imag = _mm512_permutexvar_pd(permute_V_idx_t1, imag[slice]);
            U_real = _mm512_permutexvar_pd(permute_U_idx_t1, real[slice]);
            U_imag = _mm512_permutexvar_pd(permute_U_idx_t1, imag[slice]);
            V_real = _mm512_fmadd_pd(V_real, negate_t1, U_real);
            V_imag = _mm512_fmadd_pd(V_imag, negate_t1, U_imag);
            COMPLEX_MULT(real[slice], imag[slice], V_real, V_imag, w_real, w_imag);
        }
    }
    level++;
    { // t = 2
        const __m512i permute_U_idx_t2 = _mm512_set_epi64(5, 4, 5, 4, 1, 0, 1, 0);
        const __m512i permute_V_idx_t2 = _mm512_set_epi64(7, 6, 7, 6, 3, 2, 3, 2);
        const __m512d negate_t2 = _mm512_set_pd(-1, -1, 1, 1, -1, -1, 1, 1);
        const __m512d *wsv = (const __m512d *)ws[level];
        const uint64_t t = 1ULL << level;
        for (uint64_t i = 0; i < (n >> (level + 2)); i++)
        {
            const uint64_t slice = 2 * i * t >> 2;
            __m512d w_real = wsv[2 * i];
            __m512d w_imag = wsv[2 * i + 1];
            V_real = _mm512_permutexvar_pd(permute_V_idx_t2, real[slice]);
            V_imag = _mm512_permutexvar_pd(permute_V_idx_t2, imag[slice]);
            U_real = _mm512_permutexvar_pd(permute_U_idx_t2, real[slice]);
            U_imag = _mm512_permutexvar_pd(permute_U_idx_t2, imag[slice]);
            V_real = _mm512_fmadd_pd(V_real, negate_t2, U_real);
            V_imag = _mm512_fmadd_pd(V_imag, negate_t2, U_imag);
            COMPLEX_MULT(real[slice], imag[slice], V_real, V_imag, w_real, w_imag);
        }
    }
    level++;
    { // t = 4
        const __m512i permute_U_idx_t4 = _mm512_set_epi64(3, 2, 1, 0, 3, 2, 1, 0);
        const __m512i permute_V_idx_t4 = _mm512_set_epi64(7, 6, 5, 4, 7, 6, 5, 4);
        const __m512d negate_t4 = _mm512_set_pd(-1, -1, -1, -1, 1, 1, 1, 1);
        const __m512d *wsv = (const __m512d *)ws[level];
        const uint64_t t = 1ULL << level;
        for (uint64_t i = 0; i < (n >> (level + 1)); i++)
        {
            const uint64_t slice = 2 * i * t >> 3;
            __m512d w_real = wsv[2 * i];
            __m512d w_imag = wsv[2 * i + 1];
            V_real = _mm512_permutexvar_pd(permute_V_idx_t4, real[slice]);
            V_imag = _mm512_permutexvar_pd(permute_V_idx_t4, imag[slice]);
            U_real = _mm512_permutexvar_pd(permute_U_idx_t4, real[slice]);
            U_imag = _mm512_permutexvar_pd(permute_U_idx_t4, imag[slice]);
            V_real = _mm512_fmadd_pd(V_real, negate_t4, U_real);
            V_imag = _mm512_fmadd_pd(V_imag, negate_t4, U_imag);
            COMPLEX_MULT(real[slice], imag[slice], V_real, V_imag, w_real, w_imag);
        }
    }
    for (level++; (1ULL << level) < n; level++)
    {
        const __m512d *wsv = (const __m512d *)ws[level];
        const uint64_t t = 1ULL << (level - 3);
        for (uint64_t i = 0; i < (n >> (level + 1)); i++)
        {
            const uint64_t slice = 2 * i * t;
            const __m512d w_real = wsv[2 * i];
            const __m512d w_imag = wsv[2 * i + 1];
            for (uint64_t j = slice; j < slice + t; j++)
            {
                V_real = _mm512_sub_pd(real[j], real[j + t]);
                V_imag = _mm512_sub_pd(imag[j], imag[j + t]);
                real[j] = _mm512_add_pd(real[j], real[j + t]);
                imag[j] = _mm512_add_pd(imag[j], imag[j + t]);
                COMPLEX_MULT(real[j + t], imag[j + t], V_real, V_imag, w_real, w_imag);
            }
        }
    }
}

#else // ------------------------- portable scalar path ------------------------

// c = a * b (complex, scalar)
#define COMPLEX_MULT_SCALAR(c_real, c_imag, a_real, a_imag, b_real, b_imag)                        \
    {                                                                                              \
        c_imag = (a_real * b_imag + b_real * a_imag);                                              \
        c_real = (a_real * b_real - a_imag * b_imag);                                              \
    }

void cfft_scale(double *v, double scale, uint64_t N)
{
    for (uint64_t i = 0; i < 2 * N; i++)
        v[i] *= scale;
}

double **cfft_load_twiddles_fwd(const double *rous_real, const double *rous_imag, uint64_t size)
{
    // Portable layout: two flat arrays (real, imag), indexed by stage + offset.
    double **rous = (double **)safe_malloc(2 * sizeof(double *));
    rous[0] = (double *)safe_aligned_malloc(sizeof(double) * size);
    rous[1] = (double *)safe_aligned_malloc(sizeof(double) * size);
    memcpy(rous[0], rous_real, size * sizeof(double));
    memcpy(rous[1], rous_imag, size * sizeof(double));
    return rous;
}

double **cfft_load_twiddles_inv(const double *rous_real, const double *rous_imag, uint64_t size)
{
    double **rous = (double **)safe_malloc(2 * sizeof(double *));
    rous[0] = (double *)safe_aligned_malloc(sizeof(double) * size);
    rous[1] = (double *)safe_aligned_malloc(sizeof(double) * size);
    memcpy(rous[0], rous_real, size * sizeof(double));
    memcpy(rous[1], rous_imag, size * sizeof(double));
    return rous;
}

void cfft_forward(double *x, double *const *ws, uint64_t n)
{
    uint64_t t = n, m = 1;
    double V_real, V_imag;
    double *real = x, *imag = &x[n];
    while (m < n)
    {
        t >>= 1;
        for (uint64_t i = 0; i < m; i++)
        {
            const uint64_t j1 = 2 * i * t;
            const uint64_t j2 = j1 + t;
            const double w_real = ws[0][m + i], w_imag = ws[1][m + i];
            for (uint64_t j = j1; j < j2; j++)
            {
                COMPLEX_MULT_SCALAR(V_real, V_imag, real[j + t], imag[j + t], w_real, w_imag);
                real[j + t] = real[j] - V_real;
                imag[j + t] = imag[j] - V_imag;
                real[j] += V_real;
                imag[j] += V_imag;
            }
        }
        m <<= 1;
    }
}

void cfft_inverse(double *x, double *const *ws, uint64_t n)
{
    uint64_t t = 1, m = n;
    double V_real, V_imag;
    double *real = x, *imag = &x[n];
    while (m > 1)
    {
        uint64_t j1 = 0, h = m >> 1;
        for (uint64_t i = 0; i < h; i++)
        {
            uint64_t j2 = j1 + t;
            const double w_real = ws[0][h + i], w_imag = ws[1][h + i];
            for (uint64_t j = j1; j < j2; j++)
            {
                V_real = real[j] - real[j + t];
                V_imag = imag[j] - imag[j + t];
                real[j] += real[j + t];
                imag[j] += imag[j + t];
                COMPLEX_MULT_SCALAR(real[j + t], imag[j + t], V_real, V_imag, w_real, w_imag);
            }
            j1 += 2 * t;
        }
        t <<= 1;
        m >>= 1;
    }
}

#endif // VFHE_CFFT_SIMD
