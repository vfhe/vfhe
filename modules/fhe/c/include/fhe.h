// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <mlwe.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // Blind rotation [CGGI16]: acc <- X^<a,s> * acc, for a binary secret s,
    // with the loop unfolded [ZYL+17, BMMP18] so that each step consumes
    // `unfolding` mask coefficients at the price of one gadget decomposition
    // of acc.
    //
    // Step k consumes coefficients k*unfolding .. k*unfolding + u_k - 1, where
    // u_k = unfolding except for a shorter last step when `unfolding` does not
    // divide `n`. Its keys are indexed by the non-zero bit patterns j of u_k
    // bits (bit t standing for coefficient k*unfolding + t), stored for
    // j = 1 .. 2^u_k - 1 in that order, and key j is an MGSW encryption of the
    // indicator prod_t (s_t if bit t of j is set, else 1 - s_t). With e_j the
    // sum of the a_t whose bit is set in j, mod 2N, the step computes
    //
    //   acc <- acc + (sum_j (X^e_j - 1) BK_j) (x) acc          [BMMP18, Alg. 1]
    //
    // which for unfolding = 1 is the [CGGI16] step, one key per coefficient
    // encrypting s_i.
    //
    // `bk` lists the keys step after step, each an MGSW key -- (r + 1) * `ell`
    // samples in the mul domain over the key's ring, with `log_base` the gadget
    // they were encrypted against. `acc` is canonical on entry and on return,
    // over the ring the products rescale to. `a` holds `n` exponents below 2N.
    //
    // Runs on up to `n_threads` threads (0 for the library limit), all of
    // them working on every step; the result does not depend on the thread
    // count. Reads `bk` and `a` only, so several calls may share a key.
    void cggi16_blind_rotate(RNSc_MLWE acc, const uint64_t *a, uint64_t n, RNS_MLWE *const *bk,
                             uint64_t unfolding, uint64_t ell, uint64_t log_base,
                             uint64_t n_threads);

    // `count` independent blind rotations against one key, in parallel over
    // the rotations, each on one thread: acc[k] is rotated by the row
    // a[k*n .. k*n + n - 1]. The result is that of `count` calls to
    // cggi16_blind_rotate.
    void cggi16_blind_rotate_batch(RNSc_MLWE *acc, const uint64_t *a, uint64_t count, uint64_t n,
                                   RNS_MLWE *const *bk, uint64_t unfolding, uint64_t ell,
                                   uint64_t log_base, uint64_t n_threads);

#ifdef __cplusplus
}
#endif
