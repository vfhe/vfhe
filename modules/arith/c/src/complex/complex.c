// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include <rns-rlwe.h>
#include <engine.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if VFHE_HAVE_AVX512F
#include <immintrin.h>
#endif

static uint32_t reverse_bits(uint32_t x, uint32_t bits)
{
    uint32_t r = 0;
    for (uint32_t b = 0; b < bits; b++)
        r |= ((x >> b) & 1u) << (bits - 1 - b);
    return r;
}

// zeta^(5^j) is zeta^(4k+1) for k = (5^j mod 4N - 1) / 4: every power of 5 is
// 1 mod 4, and the powers of 5 below 4N are exactly the residues 1 mod 4.
void complex_slot_positions(uint32_t *out, uint64_t N)
{
    uint32_t bits = 0;
    while ((1ULL << bits) < N)
        bits++;
    const uint64_t mask = 4 * N - 1;
    uint64_t e = 1;
    for (uint64_t j = 0; j < N; j++)
    {
        out[j] = reverse_bits((uint32_t)((e - 1) / 4), bits);
        e = (e * 5) & mask;
    }
}

void complex_poly_fft(double *x, FFT_Plan plan, const uint32_t *slots)
{
    const uint64_t N = plan->m;
    const size_t bytes = 2 * N * sizeof(double);
    double *tmp = (double *)mempool_aligned_malloc(bytes);
    fft_forward(x, plan);
    for (uint64_t j = 0; j < N; j++)
    {
        tmp[j] = x[slots[j]];
        tmp[N + j] = x[N + slots[j]];
    }
    memcpy(x, tmp, bytes);
    mempool_free(tmp, bytes);
}

void complex_poly_ifft(double *x, FFT_Plan plan, const uint32_t *slots)
{
    const uint64_t N = plan->m;
    const size_t bytes = 2 * N * sizeof(double);
    double *tmp = (double *)mempool_aligned_malloc(bytes);
    for (uint64_t j = 0; j < N; j++)
    {
        tmp[slots[j]] = x[j];
        tmp[N + slots[j]] = x[N + j];
    }
    fft_inverse(tmp, plan);
    memcpy(x, tmp, bytes);
    mempool_free(tmp, bytes);
}

void complex_poly_to_interleaved(double *out, const double *in, uint64_t N)
{
    for (size_t i = 0; i < N; i++)
    {
        out[2 * i] = in[i];
        out[2 * i + 1] = in[i + N];
    }
}

void complex_poly_round_to_RNS(RNS_Polynomial out, double *in, uint64_t N)
{
    (void)N;
    const uint64_t n = out->base->N;
    uint64_t *tmp = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * n);
    size_t i = 0;
#if VFHE_HAVE_AVX512IFMA
    // AVX-512DQ (on every IFMA CPU) rounds 8 doubles at a time, to nearest
    // even like llrint.
    for (; i + 8 <= n; i += 8)
        _mm512_storeu_si512((void *)(tmp + i), _mm512_cvt_roundpd_epi64(_mm512_loadu_pd(in + i),
                                                                        _MM_FROUND_TO_NEAREST_INT |
                                                                            _MM_FROUND_NO_EXC));
#endif
    for (; i < n; i++)
    {
        const int64_t r = (int64_t)llrint(in[i]);
        tmp[i] = (uint64_t)r;
    }
    int_array_to_RNS(out, tmp);
    free(tmp);
}

static void complex_poly_scale_double_scalar(double *v, double scale, uint64_t N)
{
    for (size_t i = 0; i < 2 * N; i++)
        v[i] *= scale;
}

#if VFHE_HAVE_AVX512F
// Shortest length the vector scaling handles (a vector holds 8 real or 8
// imaginary parts).
#define COMPLEX_MIN_VECTOR_LEN 8

void complex_poly_scale_double(double *v, double scale, uint64_t N)
{
    if (N < COMPLEX_MIN_VECTOR_LEN)
    {
        complex_poly_scale_double_scalar(v, scale, N);
        return;
    }
    const __m512d scalev = _mm512_set1_pd(scale);
    __m512d *vv = (__m512d *)(v);
    for (size_t i = 0; i < (N >> 2); i++)
    {
        vv[i] = _mm512_mul_pd(vv[i], scalev);
    }
}

#else

void complex_poly_scale_double(double *v, double scale, uint64_t N)
{
    complex_poly_scale_double_scalar(v, scale, N);
}

#endif

typedef struct
{
    void **rows_in;
    void **outs_rns;
    FFT_Plan plan;
    const uint32_t *slots;
    double temp_delta;
} ComplexBatch;

static void complex_batch_item(void *ctx, uint64_t i)
{
    const ComplexBatch *a = (const ComplexBatch *)ctx;
    const uint64_t N = a->plan->m;
    const size_t bytes = 2 * N * sizeof(double);
    const double *in = (const double *)a->rows_in[i];
    double *row = (double *)mempool_aligned_malloc(bytes);
    for (uint64_t j = 0; j < N; j++)
    {
        row[a->slots[j]] = a->temp_delta * in[j];
        row[N + a->slots[j]] = a->temp_delta * in[N + j];
    }
    fft_inverse(row, a->plan);
    complex_poly_round_to_RNS((RNS_Polynomial)a->outs_rns[i], row, N);
    mempool_free(row, bytes);
}

void complex_polys_ifft_scale_round_to_RNS_batch(void **rows_in, void **outs_rns, uint64_t count,
                                                 FFT_Plan plan, const uint32_t *slots,
                                                 double temp_delta)
{
    ComplexBatch batch = {rows_in, outs_rns, plan, slots, temp_delta};
    vfhe_parallel_for(count, 0, complex_batch_item, &batch);
}

typedef struct
{
    const double *const *values;
    const uint64_t *lengths;
    const uint64_t *shifts;
    RNS_Polynomial *outs;
    FFT_Plan plan;
    const uint32_t *slots;
    double scale;
} ComplexValuesBatch;

static void complex_values_batch_item(void *ctx, uint64_t k)
{
    const ComplexValuesBatch *a = (const ComplexValuesBatch *)ctx;
    const uint64_t N = a->plan->m, mask = a->lengths[k] - 1;
    const uint64_t shift = a->shifts[k] & mask;
    const double *in = a->values[k];
    const double scale = a->scale;
    const size_t bytes = 2 * N * sizeof(double);
    double *row = (double *)mempool_aligned_malloc(bytes);
    // Scattered straight to the positions fft_inverse expects, scaled on the
    // way.
    for (uint64_t i = 0; i < N; i++)
    {
        const uint64_t j = (i + shift) & mask;
        const uint32_t r = a->slots[i];
        row[r] = scale * in[2 * j];
        row[r + N] = scale * in[2 * j + 1];
    }
    fft_inverse(row, a->plan);
    complex_poly_round_to_RNS(a->outs[k], row, N);
    mempool_free(row, bytes);
}

void complex_values_ifft_scale_round_to_RNS_batch(RNS_Polynomial *outs, const double *const *values,
                                                  const uint64_t *lengths, const uint64_t *shifts,
                                                  uint64_t count, FFT_Plan plan,
                                                  const uint32_t *slots, double scale,
                                                  uint64_t n_threads)
{
    ComplexValuesBatch batch = {values, lengths, shifts, outs, plan, slots, scale};
    vfhe_parallel_for(count, n_threads, complex_values_batch_item, &batch);
}
