// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
// The unfolded loop's groups and their combined keys [ZYL+17, BMMP18], shared
// by the blind rotation in cggi16.c. Not part of the library's API.
#pragma once
#include "fhe.h"

// One blind rotation's key and input, read-only and shared by every thread.
typedef struct
{
    const uint64_t *a;
    uint64_t n, unfolding, groups, keys_per_group, two_n;
    RNS_MLWE *const *bk;
    int all_patterns;
    ZYL17_Combination combination;
    ZYL17_MonomialTable monomials;
    uint64_t ell, log_base, r;
    ArithRing key_ring;
} UnfoldedRotation;

uint64_t zyl17_group_size(const UnfoldedRotation *R, uint64_t g);
// Whether group g's key is combined: everything but a [BMMP18] group of one
// coefficient, whose single key multiplies the accumulator's X^e - 1 multiple.
int zyl17_group_is_combined(const UnfoldedRotation *R, uint64_t g);
// Whether any group of the rotation is combined.
int zyl17_needs_combination(const UnfoldedRotation *R);
// e_j of group g: the sum, mod 2N, of the exponents whose bit is set in j.
uint64_t zyl17_pattern_exponent(const UnfoldedRotation *R, uint64_t g, uint64_t j);
// X^m in the mul domain, 0 < m < N.
const ArithElement *zyl17_monomial(ZYL17_MonomialTable table, uint64_t m);

// A group's combined key, laid out as an MGSW key in the mul domain; `zero`
// when every factor vanished, so the group leaves the accumulator unchanged.
typedef struct
{
    RNS_MLWE *rows;
    int zero;
} CombinedKey;

void zyl17_combined_key_init(CombinedKey *ck, const UnfoldedRotation *R);
void zyl17_combined_key_free(CombinedKey *ck, const UnfoldedRotation *R);

// Per-thread elements of the key's ring for zyl17_combine_group.
typedef struct
{
    ArithElement one, factor;
} CombineScratch;

void zyl17_scratch_init(CombineScratch *s, const UnfoldedRotation *R);
void zyl17_scratch_free(CombineScratch *s, const UnfoldedRotation *R);

// out <- sum_j f_j BK_j over group g, by the rotation's combination; a no-op
// for a group that is not combined.
void zyl17_combine_group(const UnfoldedRotation *R, uint64_t g, CombinedKey *out,
                         CombineScratch *s);
