// SPDX-License-Identifier: Apache-2.0
/**
 * @file arith/ring.h
 * @brief Ring context: RNS prime pool x ring layout x multiplication strategy.
 *
 * A ::ring_ctx describes the ambient ring `R_Q = Z_Q[X]/(X^N + 1)` with `Q`
 * split into an RNS pool of NTT-friendly primes ("limbs"). It composes three
 * independent concerns, kept explicit:
 *
 *  1. **Ring layout** -- degree `N` and `split_degree`. When the primes only
 *     support a transform of size `poly_size = N / split_degree` (their 2n-th
 *     roots don't reach 2N), the ring uses an *incomplete* NTT: each limb row
 *     is a `split_degree x poly_size` block matrix of evaluations, and
 *     multiplication needs schoolbook cross terms between blocks.
 *  2. **RNS pool** -- `l` limbs, each a ::ring_limb owning its prime's
 *     ::zq_ctx, ::ntt_plan and twist table. Polynomials select a subset of
 *     the pool with a bit mask (see arith/poly.h).
 *  3. **Multiplication strategy** -- `mul_limb`, chosen once at creation:
 *     plain pointwise multiplication when `split_degree == 1`, or the
 *     split-degree schoolbook kernel otherwise. Call sites are strategy-blind.
 *
 * The pool is append-only: ::ring_extend adds limbs without invalidating any
 * existing limb pointer (limbs are individually allocated), so live
 * polynomials remain valid while the pool grows.
 */
#ifndef VFHE_ARITH_RING_H
#define VFHE_ARITH_RING_H

#include <stdint.h>

#include "ntt.h"
#include "zq.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /** Accumulation mode for fused multiply kernels: out (op)= a * b. */
    typedef enum vfhe_acc
    {
        VFHE_ACC_SET = 0, /**< out  = a * b */
        VFHE_ACC_ADD = 1, /**< out += a * b */
        VFHE_ACC_SUB = 2, /**< out -= a * b */
    } vfhe_acc;

    /**
     * One RNS limb: everything the engine knows about one prime of the pool.
     *
     * Limbs are allocated individually and never move, so `&limb->zq` and
     * `&limb->ntt` may be cached across ::ring_extend calls.
     */
    typedef struct ring_limb
    {
        uint64_t q;   /**< The limb's prime. */
        zq_ctx zq;    /**< Reduction context for q. */
        ntt_plan ntt; /**< Transform plan of size poly_size over q. */
        /**
         * Twist table: the first poly_size powers of the plan's 2*poly_size-th
         * root of unity, in bit-reversed order. Used by the split-degree
         * multiplication kernel to fold cross-block products back into the ring,
         * and exposed to callers (slot bookkeeping) via ::ring_twist.
         */
        uint64_t *twist;
    } ring_limb;

    typedef struct ring_ctx ring_ctx;

    /**
     * Per-limb negacyclic multiplication kernel (strategy; see file docs).
     *
     * Computes `out (acc)= a * b` for one limb's row of `N` residues in
     * evaluation form. @p scratch must hold `poly_size` elements when
     * `split_degree > 1` and may be NULL otherwise. `out` must not alias
     * `a` or `b`.
     */
    typedef void (*ring_mul_fn)(uint64_t *out, const uint64_t *a, const uint64_t *b,
                                const ring_limb *limb, const ring_ctx *r, vfhe_acc acc,
                                uint64_t *scratch);

    /** Ring context. Read-only after creation except through ::ring_extend. */
    struct ring_ctx
    {
        uint64_t N;            /**< Ring degree (power of two). */
        uint64_t split_degree; /**< Blocks per limb row (power of two, >= 1). */
        uint64_t poly_size;    /**< NTT size = N / split_degree. */
        uint64_t l;            /**< Number of limbs currently in the pool (<= 64). */
        ring_limb **limbs;     /**< Pool; limbs[i] is stable for the ring's lifetime. */
        ring_mul_fn mul_limb;  /**< Multiplication strategy, fixed at creation. */
    };

    typedef ring_ctx *ring_t;

    /**
     * Create a ring context.
     *
     * Builds one limb (zq context, NTT plan of size `N / split_degree`, twist
     * table) per prime. Primes must each satisfy `2 * (N / split_degree) | q - 1`.
     *
     * @param primes       array of @p l NTT-friendly primes
     * @param split_degree ring-splitting factor (power of two; 1 = full NTT)
     * @param N            ring degree (power of two, multiple of split_degree)
     * @param l            number of primes (1..64)
     * @return the ring, or NULL on invalid arguments
     */
    ring_t ring_new(const uint64_t *primes, uint64_t split_degree, uint64_t N, uint64_t l);

    /**
     * Append primes to the pool. Existing limbs, masks and polynomials that were
     * allocated with headroom (see ::poly_new) remain valid.
     *
     * @param r      ring
     * @param primes new primes
     * @param count  how many
     * @return ::VFHE_OK, or ::VFHE_ERR_ARG if the pool would exceed 64 limbs
     */
    int ring_extend(ring_t r, const uint64_t *primes, uint64_t count);

    /** Free the ring and every limb it owns. NULL is a no-op. */
    void ring_free(ring_t r);

    /** Pool size (number of limbs). */
    uint64_t ring_level_count(const ring_ctx *r);

    /** Limb @p i of the pool (borrowed pointer; stable for the ring's lifetime). */
    const ring_limb *ring_limb_at(const ring_ctx *r, uint64_t i);

    /** Limb @p i's twist table (`poly_size` entries; see ::ring_limb.twist). */
    const uint64_t *ring_twist(const ring_ctx *r, uint64_t i);

#ifdef __cplusplus
}
#endif

#endif // VFHE_ARITH_RING_H
