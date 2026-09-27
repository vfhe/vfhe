// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include <arith.h>

#include "arith_internal.h"

/* The two passes the exact base conversion adds to the fast one.
 *
 * `polynomial_base_conversion_RNSc` writes `x + u * M` and, when asked to be
 * exact, removes the `u * M`. Recovering `u` is a sum of one term per input
 * prime; removing it is a table lookup and a modular subtract per output
 * prime. Neither fits the `mod_eltwise_*` families -- one produces doubles
 * from residues, the other consumes them -- so both live here.
 *
 * `u` is recovered in floating point, and that is a property of the algorithm,
 * not of this file: the caller's margin (see `arith.h`) is what makes the
 * truncation exact. Both engines compute the same sum in the same order, so
 * the recovered `u` is identical across engines.
 *
 * As in `mod_w32.c`, every entry point accepts every length: the vector bodies
 * step a full lane group, so a short array routes to the scalar one.
 */

#define CVT_LANES 8
#define CVT_MIN_VECTOR_LEN 8

/* --- scalar bodies --- */

static void cvt_acc_scalar(double *acc, const uint64_t *in, double inv_q, uint64_t n)
{
    for (uint64_t i = 0; i < n; i++)
        acc[i] += (double)in[i] * inv_q;
}

static void cvt_acc_w32_scalar(double *acc, const uint32_t *in, double inv_q, uint64_t n)
{
    for (uint64_t i = 0; i < n; i++)
        acc[i] += (double)in[i] * inv_q;
}

static void cvt_sub_indexed_scalar(uint64_t *out, const double *index, const uint64_t *table,
                                   uint64_t n, uint64_t q)
{
    for (uint64_t i = 0; i < n; i++)
        out[i] = sub_modq(out[i], table[(size_t)index[i]], q);
}

static void cvt_sub_indexed_w32_scalar(uint32_t *out, const double *index, const uint64_t *table,
                                       uint64_t n, uint64_t q)
{
    for (uint64_t i = 0; i < n; i++)
        out[i] = (uint32_t)sub_modq(out[i], table[(size_t)index[i]], q);
}

#if VFHE_HAVE_AVX512IFMA

/* --- vector bodies --- */

static void cvt_acc_vec(double *acc, const uint64_t *in, double inv_q, uint64_t n)
{
    const __m512d s = _mm512_set1_pd(inv_q);
    uint64_t i = 0;
    for (; i + CVT_LANES <= n; i += CVT_LANES)
    {
        // Residues are below 2^62, so the conversion is exact for a 52-bit
        // prime and correctly rounded above it -- either way far inside the
        // margin the caller leaves.
        const __m512d v = _mm512_cvtepu64_pd(_mm512_loadu_si512((const void *)(in + i)));
        _mm512_storeu_pd(acc + i, _mm512_fmadd_pd(v, s, _mm512_loadu_pd(acc + i)));
    }
    cvt_acc_scalar(acc + i, in + i, inv_q, n - i);
}

static void cvt_acc_w32_vec(double *acc, const uint32_t *in, double inv_q, uint64_t n)
{
    const __m512d s = _mm512_set1_pd(inv_q);
    uint64_t i = 0;
    for (; i + CVT_LANES <= n; i += CVT_LANES)
    {
        const __m512d v = _mm512_cvtepu32_pd(_mm256_loadu_si256((const __m256i *)(in + i)));
        _mm512_storeu_pd(acc + i, _mm512_fmadd_pd(v, s, _mm512_loadu_pd(acc + i)));
    }
    cvt_acc_w32_scalar(acc + i, in + i, inv_q, n - i);
}

/* `index` holds `u + x/M` with `u` below `len`, so truncating toward zero
   gives the entry to subtract.
 *
 * The table is `len` entries, one per possible overflow, which is at most the
 * number of input primes plus one -- small enough to sit in registers. Held
 * there, the lookup is a permute rather than a gather, and that is the whole
 * reason this kernel beats the obvious scalar loop: a gather costs more than
 * the L1 loads it replaces. Only a table too wide for two vectors falls back
 * to one. */
typedef struct
{
    __m512i lo, hi;
    const uint64_t *table;
    uint32_t len;
} cvt_lookup;

static inline cvt_lookup cvt_lookup_load(const uint64_t *table, uint32_t len)
{
    cvt_lookup t = {_mm512_setzero_si512(), _mm512_setzero_si512(), table, len};
    if (len <= 8)
        t.lo = _mm512_maskz_loadu_epi64((__mmask8)((1u << len) - 1u), table);
    else if (len <= 16)
    {
        t.lo = _mm512_loadu_si512((const void *)table);
        t.hi = _mm512_maskz_loadu_epi64((__mmask8)((1u << (len - 8)) - 1u), table + 8);
    }
    return t;
}

static inline __m512i cvt_pick(const cvt_lookup *t, const double *index)
{
    const __m512i idx = _mm512_cvttpd_epi64(_mm512_loadu_pd(index));
    if (t->len <= 8)
        return _mm512_permutexvar_epi64(idx, t->lo);
    if (t->len <= 16)
        return _mm512_permutex2var_epi64(t->lo, idx, t->hi);
    return _mm512_i64gather_epi64(idx, (const long long *)t->table, 8);
}

static inline __m512i cvt_sub_modq(__m512i a, __m512i b, __m512i q)
{
    const __m512i d = _mm512_sub_epi64(a, b);
    return _mm512_min_epu64(d, _mm512_add_epi64(d, q));
}

static void cvt_sub_indexed_vec(uint64_t *out, const double *index, const uint64_t *table,
                                uint32_t len, uint64_t n, uint64_t q)
{
    const cvt_lookup t = cvt_lookup_load(table, len);
    const __m512i qv = _mm512_set1_epi64((long long)q);
    uint64_t i = 0;
    for (; i + CVT_LANES <= n; i += CVT_LANES)
    {
        const __m512i a = _mm512_loadu_si512((const void *)(out + i));
        _mm512_storeu_si512((void *)(out + i), cvt_sub_modq(a, cvt_pick(&t, index + i), qv));
    }
    cvt_sub_indexed_scalar(out + i, index + i, table, n - i, q);
}

static void cvt_sub_indexed_w32_vec(uint32_t *out, const double *index, const uint64_t *table,
                                    uint32_t len, uint64_t n, uint64_t q)
{
    const cvt_lookup t = cvt_lookup_load(table, len);
    const __m512i qv = _mm512_set1_epi64((long long)q);
    uint64_t i = 0;
    for (; i + CVT_LANES <= n; i += CVT_LANES)
    {
        const __m512i a = _mm512_cvtepu32_epi64(_mm256_loadu_si256((const __m256i *)(out + i)));
        const __m512i r = cvt_sub_modq(a, cvt_pick(&t, index + i), qv);
        _mm256_storeu_si256((__m256i *)(out + i), _mm512_cvtepi64_epi32(r));
    }
    cvt_sub_indexed_w32_scalar(out + i, index + i, table, n - i, q);
}

#define CVT_RUN(vec_call, scalar_call)                                                             \
    do                                                                                             \
    {                                                                                              \
        if (n < CVT_MIN_VECTOR_LEN)                                                                \
            scalar_call;                                                                           \
        else                                                                                       \
            vec_call;                                                                              \
    } while (0)
#else
#define CVT_RUN(vec_call, scalar_call)                                                             \
    do                                                                                             \
    {                                                                                              \
        scalar_call;                                                                               \
    } while (0)
#endif

void mod_basecvt_accumulate(double *acc, const uint64_t *in, double inv_q, uint64_t n)
{
    CVT_RUN(cvt_acc_vec(acc, in, inv_q, n), cvt_acc_scalar(acc, in, inv_q, n));
}

void mod_basecvt_accumulate_w32(double *acc, const uint32_t *in, double inv_q, uint64_t n)
{
    CVT_RUN(cvt_acc_w32_vec(acc, in, inv_q, n), cvt_acc_w32_scalar(acc, in, inv_q, n));
}

void mod_basecvt_sub_indexed(uint64_t *out, const double *index, const uint64_t *table,
                             uint32_t len, uint64_t n, Modulus mod)
{
    const uint64_t q = mod->q;
    CVT_RUN(cvt_sub_indexed_vec(out, index, table, len, n, q),
            cvt_sub_indexed_scalar(out, index, table, n, q));
}

void mod_basecvt_sub_indexed_w32(uint32_t *out, const double *index, const uint64_t *table,
                                 uint32_t len, uint64_t n, Modulus mod)
{
    const uint64_t q = mod->q;
    CVT_RUN(cvt_sub_indexed_w32_vec(out, index, table, len, n, q),
            cvt_sub_indexed_w32_scalar(out, index, table, n, q));
}
