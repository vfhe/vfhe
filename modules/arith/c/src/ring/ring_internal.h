// SPDX-License-Identifier: Apache-2.0
/**
 * @file ring_internal.h
 * @brief Internal declarations of the multiplication strategies (ring_mul.c).
 */
#ifndef VFHE_RING_INTERNAL_H
#define VFHE_RING_INTERNAL_H

#include <arith/ring.h>

/** Pointwise strategy for split_degree == 1 (complete NTT). */
void ring_mul_split1(uint64_t *out, const uint64_t *a, const uint64_t *b, const ring_limb *limb,
                     const ring_ctx *r, vfhe_acc acc, uint64_t *scratch);

/** Split-degree schoolbook strategy (incomplete NTT, cross-term twists). */
void ring_mul_splitk(uint64_t *out, const uint64_t *a, const uint64_t *b, const ring_limb *limb,
                     const ring_ctx *r, vfhe_acc acc, uint64_t *scratch);

#endif // VFHE_RING_INTERNAL_H
