// SPDX-License-Identifier: Apache-2.0
/**
 * @file poly_digest.c
 * @brief BLAKE3 digest of RNS polynomials.
 *
 * Deliberately the only file in the arith module that includes blake3.h:
 * hashing is a transcript/protocol concern, not ring algebra, and consumers
 * that want arithmetic without the dependency can drop this one file.
 */
#include <blake3.h>

#include <arith/poly.h>
#include <base.h>

void poly_digest(uint64_t *out, const rns_poly_t p)
{
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    for (uint64_t i = 0; i < p->ring->l; i++)
    {
        if (p->mask & (1ULL << i))
        {
            blake3_hasher_update(&hasher, p->limb[i], p->ring->N * sizeof(uint64_t));
        }
    }
    blake3_hasher_finalize(&hasher, (uint8_t *)out, BLAKE3_OUT_LEN);
}

uint64_t *poly_digest_alloc(const rns_poly_t p)
{
    uint64_t *out = (uint64_t *)safe_malloc(VFHE_POLY_DIGEST_WORDS * sizeof(uint64_t));
    poly_digest(out, p);
    return out;
}
