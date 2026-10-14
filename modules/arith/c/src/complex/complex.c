// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
#include <rns-rlwe.h>
#include "arith_internal.h"
#include <memory_pool.h>
#include <parallel.h>
#include <x86_64.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if VFHE_HAVE_AVX512F
#include <immintrin.h>
#endif

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
        out[j] = (uint32_t)arith_bit_reverse((e - 1) / 4, bits);
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

// Bits per limb when a coefficient is too large for one int64_t: the limbs
// below the top one are at most 2^61 in magnitude, the top one at most 2^62.
#define ROUND_LIMB_BITS 62
// A double is below 2^1024, so 17 limbs of 62 bits hold any of them.
#define ROUND_MAX_LIMBS 17

// Splits the rounding of each of the n doubles of `in` into `count` limbs,
// round(v) = sum over j of limbs[j][k] 2^(j ROUND_LIMB_BITS), from the top:
// c = rint(r 2^-(j bits)), then r - c 2^(j bits) in one fused operation, which
// is exact (a multiple of r's last place, and no larger than r). The last remainder is rounded to
// nearest even, which rounds v the same way since the limbs above it add a
// multiple of 2^ROUND_LIMB_BITS. A NaN or infinite value is split as 0.
// Returns whether every value was finite.
static bool round_split_limbs(int64_t *const *limbs, uint64_t count, const double *in, uint64_t n)
{
    double down[ROUND_MAX_LIMBS], up[ROUND_MAX_LIMBS];
    for (uint64_t j = 0; j < count; j++)
    {
        down[j] = ldexp(1.0, -(int)(j * ROUND_LIMB_BITS));
        up[j] = ldexp(1.0, (int)(j * ROUND_LIMB_BITS));
    }
    bool finite = true;
    uint64_t i = 0;
#if VFHE_HAVE_AVX512IFMA
    const __m512d infinity = _mm512_set1_pd(INFINITY);
    __mmask8 all_finite = 0xff;
    for (; i + 8 <= n; i += 8)
    {
        const __m512d x = _mm512_loadu_pd(in + i);
        const __mmask8 ok = _mm512_cmp_pd_mask(_mm512_abs_pd(x), infinity, _CMP_LT_OQ);
        all_finite &= ok;
        __m512d r = _mm512_maskz_mov_pd(ok, x);
        for (uint64_t j = count - 1; j > 0; j--)
        {
            const __m512d c = _mm512_roundscale_pd(_mm512_mul_pd(r, _mm512_set1_pd(down[j])),
                                                   _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            r = _mm512_fnmadd_pd(c, _mm512_set1_pd(up[j]), r);
            _mm512_storeu_si512((void *)(limbs[j] + i),
                                _mm512_cvtt_roundpd_epi64(c, _MM_FROUND_NO_EXC));
        }
        _mm512_storeu_si512(
            (void *)(limbs[0] + i),
            _mm512_cvt_roundpd_epi64(r, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }
    finite = all_finite == 0xff;
#endif
    for (; i < n; i++)
    {
        double r = isfinite(in[i]) ? in[i] : 0.0;
        finite &= isfinite(in[i]);
        for (uint64_t j = count - 1; j > 0; j--)
        {
            const double c = rint(r * down[j]);
            // Fused, like the vector body: c 2^(j bits) alone can reach 2^1024.
            r = fma(-c, up[j], r);
            limbs[j][i] = (int64_t)c;
        }
        limbs[0][i] = (int64_t)llrint(r);
    }
    return finite;
}

// The rounding of `in` (n values, some at or beyond 2^63 in magnitude, which
// is already an integer) written to `out` in limbs. Returns false when a value
// is NaN or infinite, writing 0 for it.
static bool round_large_to_RNS(RNS_Polynomial out, const double *in, uint64_t n)
{
    // The largest finite magnitude.
    double largest = 0.0;
    uint64_t i = 0;
#if VFHE_HAVE_AVX512IFMA
    const __m512d infinity = _mm512_set1_pd(INFINITY);
    __m512d top = _mm512_setzero_pd();
    for (; i + 8 <= n; i += 8)
    {
        const __m512d x = _mm512_abs_pd(_mm512_loadu_pd(in + i));
        top = _mm512_mask_max_pd(top, _mm512_cmp_pd_mask(x, infinity, _CMP_LT_OQ), top, x);
    }
    largest = _mm512_reduce_max_pd(top);
#endif
    for (; i < n; i++)
        largest = isfinite(in[i]) && fabs(in[i]) > largest ? fabs(in[i]) : largest;
    // The top limb, after (count - 1) splits off ROUND_LIMB_BITS each, must stay
    // below 2^ROUND_LIMB_BITS.
    uint64_t count = 1;
    while (largest >= ldexp(1.0, (int)(count * ROUND_LIMB_BITS)))
        count++;
    int64_t *limbs[ROUND_MAX_LIMBS];
    int64_t *storage = (int64_t *)safe_aligned_malloc(count * n * sizeof(int64_t));
    for (uint64_t j = 0; j < count; j++)
        limbs[j] = storage + j * n;
    const bool finite = round_split_limbs(limbs, count, in, n);
    int_limbs_to_RNS(out, limbs, count, ROUND_LIMB_BITS);
    free(storage);
    return finite;
}

bool complex_poly_round_to_RNS(RNS_Polynomial out, double *in, uint64_t N)
{
    (void)N;
    const uint64_t n = out->base->N;
    uint64_t *tmp = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * n);
    // Whether every |in[i]| < 2^63, so that its rounding fits an int64_t.
    bool fits = true;
    size_t i = 0;
#if VFHE_HAVE_AVX512IFMA
    // AVX-512DQ (on every IFMA CPU) rounds 8 doubles at a time, to nearest
    // even like llrint.
    const __m512d limit = _mm512_set1_pd(0x1p63);
    __mmask8 outside = 0;
    for (; i + 8 <= n; i += 8)
    {
        const __m512d x = _mm512_loadu_pd(in + i);
        outside |= _mm512_cmp_pd_mask(_mm512_abs_pd(x), limit, _CMP_NLT_UQ);
        _mm512_storeu_si512(
            (void *)(tmp + i),
            _mm512_cvt_roundpd_epi64(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }
    fits = outside == 0;
#endif
    for (; i < n; i++)
    {
        fits &= fabs(in[i]) < 0x1p63;
        tmp[i] = (uint64_t)llrint(in[i]);
    }
    bool finite = true;
    if (fits)
        int_array_to_RNS(out, tmp);
    else
        finite = round_large_to_RNS(out, in, n);
    free(tmp);
    return finite;
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
    bool *finite;
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
    if (!complex_poly_round_to_RNS((RNS_Polynomial)a->outs_rns[i], row, N))
        __atomic_store_n(a->finite, false, __ATOMIC_RELAXED);
    mempool_free(row, bytes);
}

bool complex_polys_ifft_scale_round_to_RNS_batch(void **rows_in, void **outs_rns, uint64_t count,
                                                 FFT_Plan plan, const uint32_t *slots,
                                                 double temp_delta)
{
    bool finite = true;
    ComplexBatch batch = {rows_in, outs_rns, plan, slots, temp_delta, &finite};
    vfhe_parallel_for(count, 0, complex_batch_item, &batch);
    return finite;
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
    bool *finite;
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
    if (!complex_poly_round_to_RNS(a->outs[k], row, N))
        __atomic_store_n(a->finite, false, __ATOMIC_RELAXED);
    mempool_free(row, bytes);
}

bool complex_values_ifft_scale_round_to_RNS_batch(RNS_Polynomial *outs, const double *const *values,
                                                  const uint64_t *lengths, const uint64_t *shifts,
                                                  uint64_t count, FFT_Plan plan,
                                                  const uint32_t *slots, double scale,
                                                  uint64_t n_threads)
{
    bool finite = true;
    ComplexValuesBatch batch = {values, lengths, shifts, outs, plan, slots, scale, &finite};
    vfhe_parallel_for(count, n_threads, complex_values_batch_item, &batch);
    return finite;
}
