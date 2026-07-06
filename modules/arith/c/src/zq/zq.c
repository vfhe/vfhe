// SPDX-License-Identifier: Apache-2.0
/**
 * @file zq.c
 * @brief zq_ctx construction and scalar wide reduction (see arith/zq.h).
 *
 * All constants that the element-wise kernels and the NTT need are computed
 * here, once per prime. The kernels themselves live in the backend files
 * (zq_portable.c, zq_avx512_*.c); this file only selects among them.
 */
#include <stdlib.h>

#include <arith/error.h>
#include <arith/zq.h>
#include <base.h>

#include "zq_backends.h"

void zq_ctx_init(zq_ctx *zq, uint64_t q)
{
    zq->q = q;

    uint32_t q_bits = 0;
    for (uint64_t t = q; t > 0; t >>= 1)
        q_bits++;
    zq->q_bits = q_bits;

#if VFHE_MP_SIMD
    // Barrett pair tuned for the IFMA tier: m has ~51 significant bits so its
    // low 52 bits fit a single IFMA multiplier. Clamped to k >= 64 so the
    // wide-reduction shift (k - 64) below stays well defined for tiny primes.
    uint64_t k = 50 + q_bits;
    if (k < 64)
        k = 64;
    uint64_t m = (uint64_t)(((unsigned __int128)1 << k) / q);
#else
    // Portable Barrett pair: grow k until m has its top bit set (maximum
    // precision for the two-word q_hat computation in zq_reduce_u128).
    uint64_t k = 64;
    unsigned __int128 m128 = ((unsigned __int128)1 << k) / q;
    while (m128 < (1ULL << 63))
    {
        k++;
        m128 = ((unsigned __int128)1 << k) / q;
    }
    uint64_t m = (uint64_t)m128;
#endif

    zq->barrett_k = k;
    zq->barrett_m = m;
    zq->barrett_m52 = (k - 52 >= 64) ? 0 : (m >> (k - 52));
    zq->ifma_barr_lo = m & ((1ULL << 52) - 1);
    zq->ifma_shift = k - 52;

    // Generic vector Barrett pair (32/64-bit tiers): approximates x/q from the
    // top ~62 bits of the product. Hoisted here; the old kernels recomputed
    // this 128-bit division on every call.
    zq->prod_right_shift = q_bits - 2;
    zq->barr_lo = (uint64_t)((((unsigned __int128)1) << (q_bits + 62)) / q);

    // Powers of the digit radices modulo q, for wide reductions and the
    // multi-precision CRT bridge.
    zq->w52_1 = (uint64_t)(((unsigned __int128)1 << 52) % q);
    zq->w52_2 = (uint64_t)(((unsigned __int128)1 << 104) % q);
    zq->w64 = (uint64_t)(((unsigned __int128)1 << 64) % q);

#if VFHE_MP_SIMD
    if (q < (1ULL << 32))
    {
        zq->ops = &zq_ops_avx512_32;
    }
    else if (q < (1ULL << 50))
    {
        zq->ops = &zq_ops_avx512_50;
    }
    else
    {
        zq->ops = &zq_ops_avx512_64;
    }
#else
    zq->ops = &zq_ops_portable;
#endif
}

zq_ctx *zq_ctx_new(uint64_t q)
{
    zq_ctx *zq = (zq_ctx *)safe_malloc(sizeof(*zq));
    zq_ctx_init(zq, q);
    return zq;
}

void zq_ctx_free(zq_ctx *zq) { free(zq); }

uint64_t zq_reduce_u128(unsigned __int128 x, const zq_ctx *zq)
{
    const uint64_t q = zq->q;

    // Fast path: inputs below 2^52 need one small Barrett step.
    if (x < (1ULL << 52))
    {
        uint64_t q_hat = (uint64_t)((x * zq->barrett_m52) >> 52);
        uint64_t res = (uint64_t)x - q_hat * q;
        if (res >= q)
            res -= q;
        return res;
    }

    // General path: fold the 128-bit value into ~64+eps bits using the radix
    // powers 2^52 and 2^104 mod q, then one full-precision Barrett step.
    uint64_t x0 = (uint64_t)x & ((1ULL << 52) - 1);
    uint64_t x1 = (uint64_t)(x >> 52) & ((1ULL << 52) - 1);
    uint64_t x2 = (uint64_t)(x >> 104);

    unsigned __int128 reduced =
        x0 + (unsigned __int128)x1 * zq->w52_1 + (unsigned __int128)x2 * zq->w52_2;

    uint64_t r_lo = (uint64_t)reduced;
    uint64_t r_hi = (uint64_t)(reduced >> 64);
    unsigned __int128 r_lo_m = (unsigned __int128)r_lo * zq->barrett_m;
    unsigned __int128 r_hi_m = (unsigned __int128)r_hi * zq->barrett_m;
    unsigned __int128 prod_hi = r_hi_m + (r_lo_m >> 64);

    uint64_t q_hat = (uint64_t)(prod_hi >> (zq->barrett_k - 64));
    uint64_t res = (uint64_t)reduced - q_hat * q;

    while (res >= q)
        res -= q;

    return res;
}
