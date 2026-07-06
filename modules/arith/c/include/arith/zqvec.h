// SPDX-License-Identifier: Apache-2.0
/**
 * @file arith/zqvec.h
 * @brief Vectors of RNS scalars (one residue row per limb, no ring structure).
 *
 * A ::zqvec is a length-n vector over Z_Q with Q in RNS form -- the flat
 * data type used by protocol layers (commitments, sumcheck transcripts) that
 * need RNS arithmetic on vectors without any polynomial/NTT semantics.
 * It borrows the limbs' reduction contexts from a ::ring_ctx but never
 * touches twiddles or transforms; its dependency really is just arith/zq.h.
 */
#ifndef VFHE_ARITH_ZQVEC_H
#define VFHE_ARITH_ZQVEC_H

#include <stdint.h>

#include "ring.h"
#include "zq.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /** Length-n vector with one residue row per limb (rows 64-byte aligned). */
    typedef struct zqvec
    {
        uint64_t **rows;         /**< l rows of n residues. */
        uint64_t n;              /**< Vector length. */
        uint64_t l;              /**< Number of limbs. */
        const zq_ctx *const *zq; /**< l borrowed reduction contexts. */
    } *zqvec_t;

    /**
     * Allocate a vector over the first @p l limbs of @p ring (uninitialized).
     *
     * @param ring source of the reduction contexts (borrowed; must outlive it)
     * @param n    vector length
     * @param l    number of limbs (<= ring->l)
     */
    zqvec_t zqvec_new(const ring_ctx *ring, uint64_t n, uint64_t l);

    /** Free the vector. NULL is a no-op. */
    void zqvec_free(zqvec_t v);

    /** out = a + b, element-wise per limb. */
    void zqvec_add(zqvec_t out, const zqvec_t a, const zqvec_t b);

    /** out = a - b. */
    void zqvec_sub(zqvec_t out, const zqvec_t a, const zqvec_t b);

    /** out = a * s for a machine-word scalar (reduced per limb). */
    void zqvec_scale(zqvec_t out, const zqvec_t a, uint64_t s);

#ifdef __cplusplus
}
#endif

#endif // VFHE_ARITH_ZQVEC_H
