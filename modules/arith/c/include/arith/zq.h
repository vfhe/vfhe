// SPDX-License-Identifier: Apache-2.0
/**
 * @file arith/zq.h
 * @brief Arithmetic in Z_q for a single prime: context, scalar ops, and the
 *        element-wise kernel vtable.
 *
 * A ::zq_ctx captures exactly one concern: *reducing modulo one prime q*. It
 * holds q, the Barrett constants derived from it, and a pointer to the kernel
 * table (::zq_ops) selected once at creation time based on the bit width of q:
 *
 *  - `q < 2^32`: 32-bit tier -- `mullo` products fit 64 bits.
 *  - `q < 2^50`: 50-bit tier -- AVX-512 IFMA 52-bit multiply-add.
 *  - otherwise : 64-bit tier -- emulated 128-bit products.
 *
 * On portable builds a single scalar table serves all tiers. Call sites never
 * branch on the tier; they go through the `zq_arr_*` wrappers below, which
 * dispatch through the table (one indirect call amortized over an array of
 * `n` elements).
 *
 * **Contract (all backends, all tiers):** array inputs are residues in
 * `[0, q)` and outputs are residues in `[0, q)`. Lazy `[0, 2q)` states are
 * strictly internal to a kernel. Arrays passed to `zq_arr_*` must hold `n`
 * elements; on SIMD builds `n` must be a multiple of 8 and pointers should be
 * 64-byte aligned (allocations from the engine always are).
 *
 * Adding a backend (e.g. NEON) means writing one file that exports one
 * ::zq_ops table -- no call site changes.
 */
#ifndef VFHE_ARITH_ZQ_H
#define VFHE_ARITH_ZQ_H

#include <stdint.h>

#include "config.h"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct zq_ctx zq_ctx;

    /**
     * Element-wise kernel table for one reduction backend.
     *
     * Each function operates on length-@p n arrays of residues in `[0, q)` and
     * writes residues in `[0, q)`. `out` may alias inputs unless noted. The
     * final ::zq_ctx parameter carries the constants the kernel needs.
     */
    typedef struct zq_ops
    {
        /** out[i] = a[i] * b[i] mod q */
        void (*mul)(uint64_t *out, const uint64_t *a, const uint64_t *b, uint64_t n,
                    const zq_ctx *zq);
        /** out[i] += a[i] * b[i] mod q (fused accumulate: one pass over memory) */
        void (*mul_addto)(uint64_t *out, const uint64_t *a, const uint64_t *b, uint64_t n,
                          const zq_ctx *zq);
        /** out[i] -= a[i] * b[i] mod q (fused) */
        void (*mul_subto)(uint64_t *out, const uint64_t *a, const uint64_t *b, uint64_t n,
                          const zq_ctx *zq);
        /** out[i] = a[i] * s mod q (s reduced internally) */
        void (*scale)(uint64_t *out, const uint64_t *a, uint64_t s, uint64_t n, const zq_ctx *zq);
        /** out[i] += a[i] * s mod q */
        void (*scale_addto)(uint64_t *out, const uint64_t *a, uint64_t s, uint64_t n,
                            const zq_ctx *zq);
        /** out[i] = a[i] + b[i] mod q */
        void (*add)(uint64_t *out, const uint64_t *a, const uint64_t *b, uint64_t n,
                    const zq_ctx *zq);
        /** out[i] = a[i] - b[i] mod q */
        void (*sub)(uint64_t *out, const uint64_t *a, const uint64_t *b, uint64_t n,
                    const zq_ctx *zq);
        /** out[i] = -a[i] mod q */
        void (*negate)(uint64_t *out, const uint64_t *a, uint64_t n, const zq_ctx *zq);
        /** out[i] = a[i] + s mod q (s reduced internally) */
        void (*add_scalar)(uint64_t *out, const uint64_t *a, uint64_t s, uint64_t n,
                           const zq_ctx *zq);
        /** out[i] = a[i] - s mod q */
        void (*sub_scalar)(uint64_t *out, const uint64_t *a, uint64_t s, uint64_t n,
                           const zq_ctx *zq);
        /** out[i] = a[i] mod q, a[i] arbitrary 64-bit */
        void (*reduce)(uint64_t *out, const uint64_t *a, uint64_t n, const zq_ctx *zq);
        /** out[i] = a[i] mod q, a[i] signed 64-bit (negatives map to q - |a| mod q) */
        void (*reduce_signed)(uint64_t *out, const int64_t *a, uint64_t n, const zq_ctx *zq);
        /** out[i] = (hi[i]*2^64 + lo[i]) mod q -- 128-bit inputs split into two arrays */
        void (*reduce_wide)(uint64_t *out, const uint64_t *hi, const uint64_t *lo, uint64_t n,
                            const zq_ctx *zq);
    } zq_ops;

    /**
     * Reduction context for one prime q.
     *
     * All fields are derived from q at ::zq_ctx_init time and constant afterwards;
     * treat the struct as read-only. It is cheap (a few words) and embeddable by
     * value (see ::ring_limb).
     *
     * Field groups, by consumer:
     *  - `q`, `q_bits`: everyone.
     *  - `barrett_k/m/m52`: scalar wide reduction (::zq_reduce_u128).
     *  - `barr_lo`, `prod_right_shift`, `w64`: 32/64-bit tier vector kernels.
     *  - `ifma_barr_lo`, `ifma_shift`: 50-bit tier (IFMA) vector kernels & NTT.
     *  - `w52_1`, `w52_2` (= 2^52 mod q, 2^104 mod q): wide reduction and the
     *    multi-precision CRT bridge (arith/mp.h).
     */
    struct zq_ctx
    {
        uint64_t q;                /**< The prime modulus. */
        uint32_t q_bits;           /**< Bit length of q. */
        uint64_t barrett_k;        /**< Barrett shift: m approximates 2^k / q. */
        uint64_t barrett_m;        /**< Barrett multiplier floor(2^k / q). */
        uint64_t barrett_m52;      /**< m >> (k - 52), for 52-bit inputs. */
        uint64_t barr_lo;          /**< floor(2^(q_bits + 62) / q), generic vector Barrett. */
        uint64_t prod_right_shift; /**< q_bits - 2, shift paired with barr_lo. */
        uint64_t ifma_barr_lo;     /**< Low 52 bits of barrett_m (IFMA tier). */
        uint64_t ifma_shift;       /**< barrett_k - 52 (IFMA tier). */
        uint64_t w52_1;            /**< 2^52  mod q. */
        uint64_t w52_2;            /**< 2^104 mod q. */
        uint64_t w64;              /**< 2^64  mod q. */
        const zq_ops *ops;         /**< Kernel table, selected once at init. */
    };

    /**
     * Initialize a caller-owned context for prime @p q.
     *
     * Computes every derived constant and installs the kernel table for q's bit
     * tier (or the portable table on non-SIMD builds). O(1); no allocation.
     *
     * @param zq context to fill
     * @param q  prime modulus, `2 < q < 2^63`
     */
    void zq_ctx_init(zq_ctx *zq, uint64_t q);

    /** Heap-allocating variant of ::zq_ctx_init. Free with ::zq_ctx_free. */
    zq_ctx *zq_ctx_new(uint64_t q);

    /** Free a context obtained from ::zq_ctx_new. NULL is a no-op. */
    void zq_ctx_free(zq_ctx *zq);

    // --- Scalar operations (single residue) -------------------------------------

    /** `a + b mod q` for `a, b` in `[0, q)`. */
    static inline uint64_t zq_scalar_add(uint64_t a, uint64_t b, uint64_t q)
    {
        uint64_t sum = a + b;
        return sum >= q ? sum - q : sum;
    }

    /** `a - b mod q` for `a, b` in `[0, q)`. */
    static inline uint64_t zq_scalar_sub(uint64_t a, uint64_t b, uint64_t q)
    {
        return a >= b ? a - b : a + q - b;
    }

    /** `-a mod q` for `a` in `[0, q)`. */
    static inline uint64_t zq_scalar_negate(uint64_t a, uint64_t q) { return a == 0 ? 0 : q - a; }

    /**
     * Reduce a 128-bit value modulo q (Barrett, using the context constants).
     *
     * @param x  value in `[0, 2^128)`
     * @param zq context for q
     * @return `x mod q`
     */
    uint64_t zq_reduce_u128(unsigned __int128 x, const zq_ctx *zq);

    /** `a * b mod q` for `a, b` in `[0, 2^64)`. */
    static inline uint64_t zq_scalar_mul(uint64_t a, uint64_t b, const zq_ctx *zq)
    {
        return zq_reduce_u128((unsigned __int128)a * b, zq);
    }

    // --- Element-wise wrappers (dispatch through the kernel table) --------------
    // One indirect call per *array*, not per element; identical cost to a direct
    // call once n is more than a handful of elements.

    static inline void zq_arr_mul(const zq_ctx *zq, uint64_t *out, const uint64_t *a,
                                  const uint64_t *b, uint64_t n)
    {
        zq->ops->mul(out, a, b, n, zq);
    }
    static inline void zq_arr_mul_addto(const zq_ctx *zq, uint64_t *out, const uint64_t *a,
                                        const uint64_t *b, uint64_t n)
    {
        zq->ops->mul_addto(out, a, b, n, zq);
    }
    static inline void zq_arr_mul_subto(const zq_ctx *zq, uint64_t *out, const uint64_t *a,
                                        const uint64_t *b, uint64_t n)
    {
        zq->ops->mul_subto(out, a, b, n, zq);
    }
    static inline void zq_arr_scale(const zq_ctx *zq, uint64_t *out, const uint64_t *a, uint64_t s,
                                    uint64_t n)
    {
        zq->ops->scale(out, a, s, n, zq);
    }
    static inline void zq_arr_scale_addto(const zq_ctx *zq, uint64_t *out, const uint64_t *a,
                                          uint64_t s, uint64_t n)
    {
        zq->ops->scale_addto(out, a, s, n, zq);
    }
    static inline void zq_arr_add(const zq_ctx *zq, uint64_t *out, const uint64_t *a,
                                  const uint64_t *b, uint64_t n)
    {
        zq->ops->add(out, a, b, n, zq);
    }
    static inline void zq_arr_sub(const zq_ctx *zq, uint64_t *out, const uint64_t *a,
                                  const uint64_t *b, uint64_t n)
    {
        zq->ops->sub(out, a, b, n, zq);
    }
    static inline void zq_arr_negate(const zq_ctx *zq, uint64_t *out, const uint64_t *a, uint64_t n)
    {
        zq->ops->negate(out, a, n, zq);
    }
    static inline void zq_arr_add_scalar(const zq_ctx *zq, uint64_t *out, const uint64_t *a,
                                         uint64_t s, uint64_t n)
    {
        zq->ops->add_scalar(out, a, s, n, zq);
    }
    static inline void zq_arr_sub_scalar(const zq_ctx *zq, uint64_t *out, const uint64_t *a,
                                         uint64_t s, uint64_t n)
    {
        zq->ops->sub_scalar(out, a, s, n, zq);
    }
    static inline void zq_arr_reduce(const zq_ctx *zq, uint64_t *out, const uint64_t *a, uint64_t n)
    {
        zq->ops->reduce(out, a, n, zq);
    }
    static inline void zq_arr_reduce_signed(const zq_ctx *zq, uint64_t *out, const int64_t *a,
                                            uint64_t n)
    {
        zq->ops->reduce_signed(out, a, n, zq);
    }
    static inline void zq_arr_reduce_wide(const zq_ctx *zq, uint64_t *out, const uint64_t *hi,
                                          const uint64_t *lo, uint64_t n)
    {
        zq->ops->reduce_wide(out, hi, lo, n, zq);
    }

#ifdef __cplusplus
}
#endif

#endif // VFHE_ARITH_ZQ_H
