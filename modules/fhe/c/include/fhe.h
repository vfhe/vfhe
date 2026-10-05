// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <mlwe.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // How an unfolded group's keys are combined into one key, sum_j f_j BK_j
    // with f_j = X^e_j or X^e_j - 1 (see cggi16_blind_rotate). All three give
    // the same key; they trade transforms against memory:
    typedef enum
    {
        // Transform each factor f_j and multiply it into every row of BK_j:
        // one forward transform per pattern. Keys in the mul domain.
        ZYL17_COMBINE_EVALUATION = 0,
        // The same with the transforms of X^m read from a table
        // (zyl17_monomial_table_new): no transform at all, at N elements of
        // the key's ring of memory. Keys in the mul domain.
        ZYL17_COMBINE_PRECOMPUTED = 1,
        // Multiply BK_j by f_j in the canonical domain, where it is a
        // rotation, and transform the sum: one transform per row of the
        // combined key whatever the number of patterns. Keys of combined
        // groups canonical; keys of [BMMP18] single-coefficient groups, which
        // are not combined, in the mul domain. The data-parallel rotation
        // forms no combined key and reads every key in the mul domain.
        ZYL17_COMBINE_COEFFICIENT = 2,
    } ZYL17_Combination;

    // The mul-domain images of X^m, 0 <= m < N, over one ring, for
    // ZYL17_COMBINE_PRECOMPUTED (X^(m+N) is the negation of X^m). Depends on
    // the ring only, so one table serves every key over it; built with up to
    // `n_threads` threads (0 for the library limit). Read-only once built.
    typedef struct _ZYL17_MonomialTable *ZYL17_MonomialTable;
    ZYL17_MonomialTable zyl17_monomial_table_new(ArithRing ring, uint64_t n_threads);
    void zyl17_monomial_table_free(ZYL17_MonomialTable table);

    // How one blind rotation uses more than one thread. Both give the
    // sequential result, bit for bit.
    typedef enum
    {
        // Helpers combine the keys of the groups ahead while one thread runs
        // the chain of external products: the chain stays sequential, so this
        // gains only what the combinations cost.
        CGGI16_PIPELINE = 0,
        // Every thread works on every group: the gadget digits, the products
        // of the digits with each pattern's key and the final rescale are
        // split into tasks, with a barrier between them. The combined key is
        // never formed -- each pattern's product is taken with its own key and
        // multiplied by its factor afterwards -- so every key is read once.
        CGGI16_DATA_PARALLEL = 1,
    } CGGI16_Parallelism;

    // Blind rotation [CGGI16] with the loop unfolded over groups of
    // `unfolding` mask coefficients [ZYL+17, BMMP18]: acc <- X^<a,s> * acc,
    // for a binary secret s, using one external product per group.
    //
    // Group g covers coefficients g*unfolding .. g*unfolding + u_g - 1, where
    // u_g = unfolding except for a shorter last group when `unfolding` does
    // not divide `n`. Its keys are indexed by the bit patterns j of u_g bits
    // (bit t standing for coefficient g*unfolding + t), and key j is an MGSW
    // encryption of the indicator prod_t (s_t if bit t of j is set, else
    // 1 - s_t). With e_j = sum of the a_t whose bit is set in j, mod 2N:
    //
    //   all_patterns != 0 [ZYL+17]: the group holds the 2^u_g keys j = 0 ..
    //     2^u_g - 1, and acc <- (sum_j X^e_j BK_j) (x) acc.
    //   all_patterns == 0 [BMMP18]: the indicators sum to 1, so key 0 is left
    //     out: the group holds the 2^u_g - 1 keys j = 1 .. 2^u_g - 1 (stored
    //     in that order), and acc <- acc + (sum_j (X^e_j - 1) BK_j) (x) acc.
    //     With unfolding = 1 this is the [CGGI16] loop, one key per
    //     coefficient encrypting s_i; a group of one coefficient multiplies
    //     the accumulator by X^e - 1 instead of combining.
    //
    // `bk` lists the keys group after group, each an MGSW key -- (r + 1) *
    // `ell` samples over the key's ring, in the domain `combination` asks
    // for, with `log_base` the gadget they were encrypted against.
    // `monomials` is the table for ZYL17_COMBINE_PRECOMPUTED, over the key's
    // ring, and is not read otherwise (NULL there falls back to evaluation).
    // `acc` is canonical on entry and on return, over the ring the external
    // products rescale to. `a` holds `n` exponents below 2N.
    //
    // With more than one thread (`n_threads`, 0 for the library limit), the
    // rotation is parallel the way `parallelism` says; the result does not
    // depend on the thread count, on `parallelism` or on `combination`
    // (which the data-parallel rotation does not use: it forms no combined
    // key). Reads `bk`, `monomials` and `a` only, so several calls may share
    // a key.
    void cggi16_blind_rotate(RNSc_MLWE acc, const uint64_t *a, uint64_t n, RNS_MLWE *const *bk,
                             uint64_t unfolding, int all_patterns, ZYL17_Combination combination,
                             ZYL17_MonomialTable monomials, uint64_t ell, uint64_t log_base,
                             CGGI16_Parallelism parallelism, uint64_t n_threads);

    // `count` independent blind rotations against one key, in parallel over
    // the rotations: acc[k] is rotated by the row a[k*n .. k*n + n - 1]. Each
    // rotation runs on one thread, so the result is that of `count` calls to
    // cggi16_blind_rotate.
    void cggi16_blind_rotate_batch(RNSc_MLWE *acc, const uint64_t *a, uint64_t count, uint64_t n,
                                   RNS_MLWE *const *bk, uint64_t unfolding, int all_patterns,
                                   ZYL17_Combination combination, ZYL17_MonomialTable monomials,
                                   uint64_t ell, uint64_t log_base, uint64_t n_threads);

#ifdef __cplusplus
}
#endif
