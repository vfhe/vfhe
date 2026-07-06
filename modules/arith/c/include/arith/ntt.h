// SPDX-License-Identifier: Apache-2.0
/**
 * @file arith/ntt.h
 * @brief Negacyclic number-theoretic transform plan for one prime.
 *
 * An ::ntt_plan is the FFTW-style bundle of everything one forward/inverse
 * transform of size n over Z_q needs: the (inverse) root of unity, the
 * precomputed twiddle tables *in the backend's preferred memory layout*, and
 * the transform entry points themselves. Table layout is a property of the
 * execution strategy, so tables and butterflies live and die together in the
 * backend; the plan only sees `void *` table handles plus function pointers.
 *
 * The plan does not own its ::zq_ctx -- it is injected at init time and must
 * outlive the plan (dependency points downward: ntt -> zq).
 *
 * Transforms use the standard decimation conventions: forward is
 * Cooley-Tukey, natural -> bit-reversed order; inverse is Gentleman-Sande,
 * bit-reversed -> natural order, with the 1/n factor folded in. A forward
 * followed by an inverse is the identity on `[0, q)^n`.
 */
#ifndef VFHE_ARITH_NTT_H
#define VFHE_ARITH_NTT_H

#include <stdint.h>

#include "config.h"
#include "zq.h"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct ntt_plan ntt_plan;

    /**
     * Transform plan for size @c n over the prime of @c zq.
     *
     * Created with ::ntt_plan_init, destroyed with ::ntt_plan_clear. Read-only
     * after creation and safe to share between threads for concurrent transforms.
     */
    struct ntt_plan
    {
        uint64_t n;        /**< Transform size (power of two). */
        uint32_t log_n;    /**< log2(n). */
        const zq_ctx *zq;  /**< Injected reduction context (not owned). */
        uint64_t root;     /**< Primitive 2n-th root of unity w (w^n = -1 mod q). */
        uint64_t inv_root; /**< w^-1 mod q. */
        void **tw_fwd;     /**< Forward twiddles (backend-owned layout). */
        void **tw_fwd_pre; /**< Forward Barrett-preconditioned twiddles (may be NULL). */
        void **tw_inv;     /**< Inverse twiddles. */
        void **tw_inv_pre; /**< Inverse preconditioned twiddles (may be NULL). */
        /** Forward transform: out = NTT(in). `out == in` is allowed. */
        void (*forward)(uint64_t *out, const uint64_t *in, const ntt_plan *plan);
        /** Inverse transform: out = NTT^-1(in), including the 1/n factor. */
        void (*inverse)(uint64_t *out, const uint64_t *in, const ntt_plan *plan);
        /** Backend-specific table destructor (used by ::ntt_plan_clear). */
        void (*free_tables)(ntt_plan *plan);
    };

    /**
     * Initialize a caller-owned plan.
     *
     * Deterministically finds a primitive 2n-th root of unity mod q (via
     * ::nt_gen_root_of_unity), generates the twiddle factors, and hands them to
     * the backend matching `zq->ops` for packing. O(n log n) precomputation.
     *
     * @param plan plan to fill
     * @param n    transform size; power of two, and `2n | q - 1`
     * @param zq   reduction context for the target prime; must outlive the plan
     * @return ::VFHE_OK, or ::VFHE_ERR_ARG if n is not a power of two
     */
    int ntt_plan_init(ntt_plan *plan, uint64_t n, const zq_ctx *zq);

    /** Release the plan's twiddle tables (not the injected zq_ctx). */
    void ntt_plan_clear(ntt_plan *plan);

    /** Convenience: forward transform through the plan's function pointer. */
    static inline void ntt_forward(const ntt_plan *plan, uint64_t *out, const uint64_t *in)
    {
        plan->forward(out, in, plan);
    }

    /** Convenience: inverse transform through the plan's function pointer. */
    static inline void ntt_inverse(const ntt_plan *plan, uint64_t *out, const uint64_t *in)
    {
        plan->inverse(out, in, plan);
    }

#ifdef __cplusplus
}
#endif

#endif // VFHE_ARITH_NTT_H
