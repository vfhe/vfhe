// SPDX-License-Identifier: Apache-2.0
/**
 * @file arith/tower.h
 * @brief Operations on the RNS tower: limb-set utilities, base conversion,
 *        rescaling (division by limb primes), and scaled lifting.
 *
 * "Tower" operations change *which* limbs a polynomial lives on -- they walk
 * up and down the chain of quotient rings R_{Q_l} obtained by dropping or
 * adding RNS primes. They all operate in ::VFHE_COEFF domain: coefficient
 * residues are what CRT-style reconstruction reasons about.
 *
 * Limb sets are bit masks over the ring's pool (bit i = limb i), the same
 * masks stored in ::rns_poly. Dropping a limb via the division operations
 * zeroes its row and clears its bit -- no reallocation, so a dropped level
 * can later be re-extended into the same polynomial.
 */
#ifndef VFHE_ARITH_TOWER_H
#define VFHE_ARITH_TOWER_H

#include <stdint.h>

#include "poly.h"
#include "ring.h"

#ifdef __cplusplus
extern "C"
{
#endif

    // --- Limb-set (mask) utilities ------------------------------------------------

    /** Number of active limbs in @p mask (popcount). */
    static inline uint64_t limbset_count(uint64_t mask)
    {
        uint64_t count = 0;
        while (mask)
        {
            count += (mask & 1);
            mask >>= 1;
        }
        return count;
    }

    /** Pool index of the @p i-th active limb (0-based), or -1 if fewer are set. */
    int limbset_nth(uint64_t mask, uint64_t i);

    /** Pool index of the highest active limb, or -1 if the mask is empty. */
    int limbset_last(uint64_t mask);

    // --- CRT constants -------------------------------------------------------------

    /**
     * out[i] = (Q / p[i])^-1 mod p[i] for Q = prod p[j] -- the CRT interpolation
     * coefficients of the prime list.
     *
     * @param out l results
     * @param p   l pairwise-distinct primes
     * @param l   count
     */
    void tower_qhat_array(uint64_t *out, const uint64_t *p, uint64_t l);

    // --- Fast base conversion -------------------------------------------------------

    /**
     * Precomputed constants for one (in_mask -> out_mask) fast base extension
     * over a specific ring: the source limbs D, the strictly-new target limbs P,
     * the interpolation inverses Dhat, and the cross-products D mod p. Build once
     * per conversion and reuse (see ::baseconv_plan_new); the per-polynomial work
     * is then w*(v+1) element-wise passes.
     */
    typedef struct baseconv_plan
    {
        uint64_t in_mask;   /**< Source limb set. */
        uint64_t out_mask;  /**< Target limb set. */
        uint32_t w;         /**< |D| = active source limbs. */
        uint32_t v;         /**< |P| = limbs to extend into (out \ in). */
        uint32_t *D;        /**< Pool indices of source limbs. */
        uint32_t *P;        /**< Pool indices of new target limbs. */
        uint64_t *Dhat;     /**< (D/q_j)^-1 mod q_j per source limb. */
        uint64_t **D_mod_p; /**< (D/q_j) mod p_i per (target, source) pair. */
    } *baseconv_plan_t;

    /**
     * Build a base-conversion plan for @p ring from @p in_mask to @p out_mask.
     * O(w^2 * (v + 1)) scalar work.
     *
     * @return the plan, or NULL on allocation failure
     */
    baseconv_plan_t baseconv_plan_new(const ring_ctx *ring, uint64_t in_mask, uint64_t out_mask);

    /** Free a plan. NULL is a no-op. */
    void baseconv_plan_free(baseconv_plan_t plan);

    /**
     * Fast base extension: reconstruct @p in (on plan->in_mask) approximately on
     * plan->out_mask, writing @p out. Shared limbs are copied; new limbs receive
     * the standard fast-base-conversion sum (exact up to the usual small
     * multiple-of-Q offset). If @p plan is NULL a throwaway plan is built.
     *
     * @return ::VFHE_OK; ::VFHE_ERR_DOMAIN if @p in is not ::VFHE_COEFF;
     *         ::VFHE_ERR_ARG if the plan's masks do not match the polynomials'.
     */
    int poly_base_convert(rns_poly_t out, const rns_poly_t in, const baseconv_plan_t plan);

    // --- Rescaling: exact division by limb primes -----------------------------------

    /**
     * In-place floor division by the product of the limbs in @p divide_mask:
     * p <- floor(p / prod q_i), limb by limb. Each divided limb's row is zeroed
     * and removed from the mask.
     * @return ::VFHE_OK or ::VFHE_ERR_DOMAIN.
     */
    int poly_div_floor(rns_poly_t p, uint64_t divide_mask);

    /** ::poly_div_floor with centered rounding instead of floor. */
    int poly_div_round(rns_poly_t p, uint64_t divide_mask);

    /** ::poly_div_floor by the highest active limb only (drop one level). */
    int poly_div_floor_last(rns_poly_t p);

    /** ::poly_div_round by the highest active limb only. */
    int poly_div_round_last(rns_poly_t p);

    // --- Projection / lifting --------------------------------------------------------

    /**
     * Project onto @p out's (smaller) limb set: copy the rows @p out keeps.
     * The natural ring hom R_Q -> R_{Q'} for Q' | Q.
     * @return ::VFHE_OK or ::VFHE_ERR_DOMAIN.
     */
    int poly_mod_reduce(rns_poly_t out, const rns_poly_t in);

    /**
     * Lift the single residue row @p src_limb of @p in into every active limb of
     * @p out (reducing it mod each target prime). Interprets that row as a small
     * integer polynomial.
     * @return ::VFHE_OK or ::VFHE_ERR_DOMAIN.
     */
    int poly_lift_residue(rns_poly_t out, const rns_poly_t in, uint64_t src_limb);

    /**
     * Per-limb scaling factors for ::poly_scaled_lift: delta[i] = prod of the
     * primes in `out_mask \ in_mask`, mod q_i, for every i in out_mask that is
     * also in in_mask (0 elsewhere). @p delta_out must hold ring->l entries.
     */
    void tower_scaling_factors(uint64_t *delta_out, const ring_ctx *ring, uint64_t in_mask,
                               uint64_t out_mask);

    /**
     * Scaled lift into a larger limb set: out_i = in_i * delta_i on shared limbs,
     * zero on the new limbs -- i.e. multiply by the modulus ratio while lifting,
     * so the value re-enters the larger ring at the right scale. Pass @p delta
     * from ::tower_scaling_factors, or NULL to compute it on the fly.
     * @return ::VFHE_OK or ::VFHE_ERR_DOMAIN.
     */
    int poly_scaled_lift(rns_poly_t out, const rns_poly_t in, const uint64_t *delta);

#ifdef __cplusplus
}
#endif

#endif // VFHE_ARITH_TOWER_H
