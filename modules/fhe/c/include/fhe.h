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
    // samples in the mul domain over the key's ring, with `gadget_params`
    // those they were encrypted with, as in
    // `gadget_mul_addto_polynomial`. `acc` is canonical on entry and on return,
    // over the ring the products rescale to. `a` holds `n` exponents below 2N.
    //
    // Runs on up to `n_threads` threads (0 for the library limit), all of
    // them working on every step; the result does not depend on the thread
    // count. Reads `bk` and `a` only, so several calls may share a key.
    void cggi16_blind_rotate(RNSc_MLWE acc, const uint64_t *a, uint64_t n, RNS_MLWE *const *bk,
                             uint64_t unfolding, uint64_t ell, const GadgetParams *gadget_params,
                             uint64_t n_threads);

    // `count` independent blind rotations against one key, in parallel over
    // the rotations, each on one thread: acc[k] is rotated by the row
    // a[k*n .. k*n + n - 1]. The result is that of `count` calls to
    // cggi16_blind_rotate.
    void cggi16_blind_rotate_batch(RNSc_MLWE *acc, const uint64_t *a, uint64_t count, uint64_t n,
                                   RNS_MLWE *const *bk, uint64_t unfolding, uint64_t ell,
                                   const GadgetParams *gadget_params, uint64_t n_threads);

    // The sparse-amortized blind rotation [GP25]: n accumulators acc[0..n-1],
    // samples over one ring R_N, canonical on entry and on return, one per
    // coefficient of an input sample over R_n = Z[Y]/(Y^n + 1). Exponents are
    // mod 2N, and sigma is the automorphism X -> X^-1 of R_N, applied through
    // the key-switch key `aut`.
    //
    // MGSW keys are passed as in `cggi16_blind_rotate`: each one (r + 1) *
    // `ell` samples in the mul domain over the key's ring, encrypting a bit,
    // with `gadget_params` its gadget's. Each call runs on up to
    // `n_threads` threads (0: the library limit), the result does not depend
    // on the thread count, and the keys are only read.

    // Rotates the accumulators by d, 0 <= d <= n, given as the MGSW
    // encryptions of its `n_bits` bits, least significant first
    // (2^(n_bits - 1) <= n):
    //
    //   acc[k] <- acc[k - d]                for k >= d,
    //   acc[k] <- sigma(acc[k - d + n])     for k <  d.
    void gp25_rotate(RNSc_MLWE *acc, uint64_t n, RNS_MLWE *const *bits, uint64_t n_bits,
                     RNS_MLWE_KS_Key aut, uint64_t ell, const GadgetParams *gadget_params,
                     uint64_t n_threads);

    // acc[k] <- acc[k] * X^a[k] if `sign` encrypts 0, acc[k] * X^-a[k] if it
    // encrypts 1; with `sign` NULL, acc[k] * X^a[k]. a[k] < 2N.
    void gp25_multiply_by_signed_monomials(RNSc_MLWE *acc, uint64_t n, const uint64_t *a,
                                           RNS_MLWE *sign, uint64_t ell,
                                           const GadgetParams *gadget_params, uint64_t n_threads);

    // The whole rotation, for an input key of `rank` components s_i with h
    // non-zero coefficients each, in {-1, 1}. `a` holds the input's masks
    // mod 2N, component after component (rank * n values), and on return
    //
    //   acc[k] <- sigma^rank(acc[k]) * X^((-1)^(rank + 1) * c_k),
    //
    // c_k the coefficient k of sum_i a_i * s_i in Z_2N[Y]/(Y^n + 1).
    //
    // Component i's coefficients j_1 > ... > j_h are consumed through its h + 1
    // gaps, n - j_1, j_1 - j_2, ..., j_h: gap t is a `gp25_rotate` by it, and
    // the first h gaps are each followed by a
    // `gp25_multiply_by_signed_monomials` by the mask, with the key of that
    // coefficient's sign (negated masks for odd i). Every gap is given
    // `gap_bits` bits (2^(gap_bits - 1) <= n); `gap_keys` lists their keys
    // component after component, gap after gap, least significant bit first;
    // and `sign_keys` the h signs of each component in turn -- or NULL for a
    // binary key, where every sign is +1.
    void gp25_blind_rotate(RNSc_MLWE *acc, uint64_t n, const uint64_t *a, uint64_t rank, uint64_t h,
                           uint64_t gap_bits, RNS_MLWE *const *gap_keys, RNS_MLWE *const *sign_keys,
                           RNS_MLWE_KS_Key aut, uint64_t ell, const GadgetParams *gadget_params,
                           uint64_t n_threads);

    // Encoders for the SHIP bootstrap [CHKS25, Alg. 1]. The input polynomial is
    // canonical, over a ring of modulus Q; omega = exp(2 pi i / Q) and
    // M = N/2. Each output is a vector of M slots encoded at `scale` into the
    // ring `out` was allocated over, in the NTT domain, as
    // complex_values_ifft_scale_round_to_RNS_batch does with the complex
    // ring's `plan` (length N) and `slots`.

    // The plaintexts of the column step for a sample with mask `a` (linear
    // decryption b - a s): out[4 i + k - 1] = Rot_i(pi_k) for i < theta and
    // k = 1..4, with (Rot_i v)_t = v_(t - i mod M), pi_1 = (omega^(-a_t))_(t < M),
    // pi_3 = (omega^(-a_(M + t)))_(t < M), and pi_2, pi_4 their conjugates.
    // Up to `n_threads` threads (0: the library limit).
    void ship_encode_pi(RNS_Polynomial *out, RNSc_Polynomial a, uint64_t theta, double scale,
                        FFT_Plan plan, const uint32_t *slots, uint64_t n_threads);
    // `out` encodes (c omega^(b_t))_(t < M), with c = coefficient_re + i coefficient_im.
    void ship_encode_pt0(RNS_Polynomial out, RNSc_Polynomial b, double coefficient_re,
                         double coefficient_im, double scale, FFT_Plan plan, const uint32_t *slots);

#ifdef __cplusplus
}
#endif
