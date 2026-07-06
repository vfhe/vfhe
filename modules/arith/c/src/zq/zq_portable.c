// SPDX-License-Identifier: Apache-2.0
/**
 * @file zq_portable.c
 * @brief Portable scalar element-wise kernels (the baseline zq_ops table).
 *
 * Plain C11, correct for any prime q < 2^63 on any target. Serves as the
 * only backend on portable builds and as the reference implementation the
 * SIMD backends are checked against.
 */
#include <stddef.h>

#include <arith/zq.h>

#include "zq_backends.h"

static void p_mul(uint64_t *out, const uint64_t *a, const uint64_t *b, uint64_t n, const zq_ctx *zq)
{
    for (size_t i = 0; i < n; i++)
        out[i] = zq_reduce_u128((unsigned __int128)a[i] * b[i], zq);
}

static void p_mul_addto(uint64_t *out, const uint64_t *a, const uint64_t *b, uint64_t n,
                        const zq_ctx *zq)
{
    for (size_t i = 0; i < n; i++)
    {
        uint64_t prod = zq_reduce_u128((unsigned __int128)a[i] * b[i], zq);
        out[i] = zq_scalar_add(out[i], prod, zq->q);
    }
}

static void p_mul_subto(uint64_t *out, const uint64_t *a, const uint64_t *b, uint64_t n,
                        const zq_ctx *zq)
{
    for (size_t i = 0; i < n; i++)
    {
        uint64_t prod = zq_reduce_u128((unsigned __int128)a[i] * b[i], zq);
        out[i] = zq_scalar_sub(out[i], prod, zq->q);
    }
}

static void p_scale(uint64_t *out, const uint64_t *a, uint64_t s, uint64_t n, const zq_ctx *zq)
{
    const uint64_t sr = s % zq->q;
    for (size_t i = 0; i < n; i++)
        out[i] = zq_reduce_u128((unsigned __int128)a[i] * sr, zq);
}

static void p_scale_addto(uint64_t *out, const uint64_t *a, uint64_t s, uint64_t n,
                          const zq_ctx *zq)
{
    const uint64_t sr = s % zq->q;
    for (size_t i = 0; i < n; i++)
    {
        uint64_t prod = zq_reduce_u128((unsigned __int128)a[i] * sr, zq);
        out[i] = zq_scalar_add(out[i] % zq->q, prod, zq->q);
    }
}

static void p_add(uint64_t *out, const uint64_t *a, const uint64_t *b, uint64_t n, const zq_ctx *zq)
{
    for (size_t i = 0; i < n; i++)
        out[i] = (a[i] + b[i]) % zq->q;
}

static void p_sub(uint64_t *out, const uint64_t *a, const uint64_t *b, uint64_t n, const zq_ctx *zq)
{
    for (size_t i = 0; i < n; i++)
        out[i] = (a[i] + zq->q - b[i]) % zq->q;
}

static void p_negate(uint64_t *out, const uint64_t *a, uint64_t n, const zq_ctx *zq)
{
    for (size_t i = 0; i < n; i++)
        out[i] = (zq->q - (a[i] % zq->q)) % zq->q;
}

static void p_add_scalar(uint64_t *out, const uint64_t *a, uint64_t s, uint64_t n, const zq_ctx *zq)
{
    const uint64_t sr = s % zq->q;
    for (size_t i = 0; i < n; i++)
        out[i] = (a[i] + sr) % zq->q;
}

static void p_sub_scalar(uint64_t *out, const uint64_t *a, uint64_t s, uint64_t n, const zq_ctx *zq)
{
    const uint64_t sr = s % zq->q;
    for (size_t i = 0; i < n; i++)
        out[i] = (a[i] + zq->q - sr) % zq->q;
}

static void p_reduce(uint64_t *out, const uint64_t *a, uint64_t n, const zq_ctx *zq)
{
    for (size_t i = 0; i < n; i++)
        out[i] = a[i] % zq->q;
}

static void p_reduce_signed(uint64_t *out, const int64_t *a, uint64_t n, const zq_ctx *zq)
{
    const uint64_t q = zq->q;
    for (size_t i = 0; i < n; i++)
    {
        int64_t val = a[i];
        uint64_t abs_val = (val < 0) ? -(uint64_t)val : (uint64_t)val;
        uint64_t r = zq_reduce_u128(abs_val, zq);
        out[i] = (val < 0) ? ((r == 0) ? 0 : q - r) : r;
    }
}

static void p_reduce_wide(uint64_t *out, const uint64_t *hi, const uint64_t *lo, uint64_t n,
                          const zq_ctx *zq)
{
    for (size_t i = 0; i < n; i++)
    {
        unsigned __int128 val = ((unsigned __int128)hi[i] << 64) | lo[i];
        out[i] = zq_reduce_u128(val, zq);
    }
}

const zq_ops zq_ops_portable = {
    .mul = p_mul,
    .mul_addto = p_mul_addto,
    .mul_subto = p_mul_subto,
    .scale = p_scale,
    .scale_addto = p_scale_addto,
    .add = p_add,
    .sub = p_sub,
    .negate = p_negate,
    .add_scalar = p_add_scalar,
    .sub_scalar = p_sub_scalar,
    .reduce = p_reduce,
    .reduce_signed = p_reduce_signed,
    .reduce_wide = p_reduce_wide,
};
