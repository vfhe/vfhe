// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
//
// What the files of the extension-field fold kernels share: the decoded shape,
// one call's job, the constants and the per-degree instantiation of a body.
// field_vector_fold.c describes the kernels.
#ifndef VFHE_FIELD_VECTOR_FOLD_H
#define VFHE_FIELD_VECTOR_FOLD_H

#include <arith.h>
#include <stdbool.h>
#include "arith_internal.h"
#include "../vec_chunks.h"

#define FIELD_FOLD_INLINE static inline __attribute__((always_inline))

// The pairs in one IFMA group. Chunks start at multiples of it, so that every
// chunk starts on a whole group, whichever body runs.
#define FIELD_FOLD_GROUP 8

// The most coefficients the unreduced product of two elements has: those of
// x^0 .. x^(2d - 2).
#define FIELD_FOLD_MAX_UNREDUCED (2 * FIELD_FOLD_MAX_DEGREE - 1)

// The shape word array of field_vec_fold_product_sums, decoded. The products of
// sum s are first_product[s] .. first_product[s + 1] - 1; `used` lists the
// tables some product reads, in increasing order, and `is_used` marks them.
typedef struct
{
    uint64_t sums;
    uint64_t nodes;
    uint64_t first_product[FIELD_FOLD_MAX_SUMS + 1];
    uint64_t negated[FIELD_FOLD_MAX_PRODUCTS];
    uint64_t factor_count[FIELD_FOLD_MAX_PRODUCTS];
    uint64_t factors[FIELD_FOLD_MAX_PRODUCTS][FIELD_FOLD_MAX_FACTORS];
    uint64_t used[FIELD_FOLD_MAX_TABLES];
    uint64_t used_count;
    bool is_used[FIELD_FOLD_MAX_TABLES];
} FieldProductShape;

typedef struct FieldFoldJob FieldFoldJob;

// One call of an entry point, as its chunks see it: the operands, the decoded
// shape, and `range`, the body that adds up the pairs [begin, end) and writes
// the result into sums[0 .. slots - 1], one element of d words for each sum
// and node. In the "fold and" variants the pairs are those of the folded
// tables.
struct FieldFoldJob
{
    void (*range)(const FieldFoldJob *job, uint64_t begin, uint64_t end, uint64_t *const *sums);
    VecChunks chunks;
    uint64_t total; // pairs, [0, total)
    const FieldVector *tables;
    uint64_t count;
    const FieldVector *outs; // the folds, or NULL
    FieldVector weights;     // NULL for weights of one
    const uint64_t *r;       // the fold's element, with outs
    const FieldProductShape *shape;
    uint64_t slots;     // sums * nodes
    uint64_t *partials; // `slots` elements per chunk
};

// The two ranges each body provides, as FieldFoldJob.range: the sums of the
// pairs [begin, end) of the tables, and the same for the "fold and" variant.
#if VFHE_HAVE_AVX512IFMA
// Whether the IFMA body takes vectors over v's field; the element-wise body
// takes the others.
bool field_fold_ifma_applies(const FieldVector v);
void field_fold_ifma_sums_range(const FieldFoldJob *job, uint64_t begin, uint64_t end,
                                uint64_t *const *sums);
void field_fold_ifma_fold_sums_range(const FieldFoldJob *job, uint64_t begin, uint64_t end,
                                     uint64_t *const *sums);
void field_fold_eltwise_sums_range(const FieldFoldJob *job, uint64_t begin, uint64_t end,
                                   uint64_t *const *sums);
void field_fold_eltwise_fold_sums_range(const FieldFoldJob *job, uint64_t begin, uint64_t end,
                                        uint64_t *const *sums);
#else
void field_fold_scalar_sums_range(const FieldFoldJob *job, uint64_t begin, uint64_t end,
                                  uint64_t *const *sums);
void field_fold_scalar_fold_sums_range(const FieldFoldJob *job, uint64_t begin, uint64_t end,
                                       uint64_t *const *sums);
#endif

// Per-degree instances.
//
// Each body is instantiated for the degrees 1, 2, 4 and 8, where the compiler
// unrolls the coefficient loops, and once for the degree read at run time
// (DEG 0). Each instance is its own function, so that a call's stack frame
// holds one degree's arrays.
#define FIELD_FOLD_BODY_AT(BODY, DEG)                                                              \
    static __attribute__((noinline)) void BODY##_##DEG(                                            \
        uint64_t *const *sums, const FieldFoldJob *job, uint64_t begin, uint64_t end, uint64_t d)  \
    {                                                                                              \
        BODY(sums, job, begin, end, DEG == 0 ? d : DEG);                                           \
    }
#define FIELD_FOLD_BODIES(BODY)                                                                    \
    FIELD_FOLD_BODY_AT(BODY, 0)                                                                    \
    FIELD_FOLD_BODY_AT(BODY, 1)                                                                    \
    FIELD_FOLD_BODY_AT(BODY, 2)                                                                    \
    FIELD_FOLD_BODY_AT(BODY, 4)                                                                    \
    FIELD_FOLD_BODY_AT(BODY, 8)

// Call BODY's instance for degree d, or its run-time-degree instance for any
// other d.
#define FIELD_FOLD_CALL(BODY, d, ...)                                                              \
    do                                                                                             \
    {                                                                                              \
        switch (d)                                                                                 \
        {                                                                                          \
        case 1:                                                                                    \
            BODY##_1(__VA_ARGS__, 1);                                                              \
            break;                                                                                 \
        case 2:                                                                                    \
            BODY##_2(__VA_ARGS__, 2);                                                              \
            break;                                                                                 \
        case 4:                                                                                    \
            BODY##_4(__VA_ARGS__, 4);                                                              \
            break;                                                                                 \
        case 8:                                                                                    \
            BODY##_8(__VA_ARGS__, 8);                                                              \
            break;                                                                                 \
        default:                                                                                   \
            BODY##_0(__VA_ARGS__, d);                                                              \
            break;                                                                                 \
        }                                                                                          \
    } while (0)

#endif // VFHE_FIELD_VECTOR_FOLD_H
