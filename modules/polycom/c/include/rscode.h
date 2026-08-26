// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <arith.h>
#include <arith_generic.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // Reed-Solomon code kernels over R_q for the basefold commitment
    // (polycom.md §2). The code is *interleaved*: it acts on a vector of
    // `degree` ring elements coefficient-slot-wise and per RNS prime, so each
    // (prime, coefficient slot) pair carries an independent RS codeword of
    // length `size` over Z_p. Ring elements are treated as vectors of Z_p
    // values, so the transform is oblivious to their NTT representation — but
    // it reads `coeffs` directly, so every entry must be in the *same*
    // representation (the callers normalize to RNS/NTT form first).
    //
    // A codeword is the message polynomial evaluated at the `size` points of
    // arith's negacyclic NTT of that length: with psi the 2*size-th root of
    // unity `ntt_new_plan` picks, position p holds P(psi^(2*brv(p)+1)) where
    // brv reverses the log2(size) index bits (`ntt_forward` is CT_NR: natural
    // in, bit-reversed out). Two consequences the fold relies on:
    //
    //   - positions 2i and 2i+1 hold P(x_i) and P(-x_i) for
    //     x_i = psi^(2*brv_{size/2}(i)+1) — the +/- pairs are *adjacent*;
    //   - x_i^2 is the i-th evaluation point of the half-length code, because
    //     `ntt_new_plan` derives psi from the smallest quadratic non-residue
    //     mod p, a choice independent of the length, so the roots of
    //     successive lengths satisfy psi_{n/2} = psi_n^2.
    //
    // `plans` is indexed by *global* RNS prime index (like RNS_Polynomial's
    // `coeffs`), with NULL in the slots the mask excludes.

    // One NTT_Plan per active prime of `rns_mask`, for codewords of length
    // `size`; the array owns its plans and is one past `rns_mask`'s highest set
    // bit long -- deliberately sized by the mask rather than by `base->l`, which
    // grows in place whenever another ring of the same (N, split_degree)
    // introduces a prime and so cannot be re-read at free time. The plans
    // *borrow* the RNS_Base's moduli, so they must not outlive it.
    NTT_Plan *rs_new_plans(RNS_Base base, uint64_t rns_mask, uint64_t size);

    // Free an array from rs_new_plans. `count` is its allocation length, i.e.
    // one past the highest set bit of the `rns_mask` it was built with
    // (`Ring.rns_rows` on the Python side). Leaves the borrowed moduli alone.
    void rs_free_plans(NTT_Plan *plans, uint64_t count);

    // The 2*size-th root of unity the plan at global prime index `index`
    // transforms with — the value the fold's twists are powers of.
    uint64_t rs_plans_root(NTT_Plan *plans, uint64_t index);

    // out[0..size) = the codeword of the length-`degree` message in[0..degree)
    // (zero-padded). Out and in must not alias.
    void rs_encode(ArithElement *out, ArithElement *in, uint64_t size, uint64_t degree,
                   NTT_Plan *plans);

    // The inverse transform plus the degree check that makes it a decoder:
    // returns 1 when every codeword's coefficients above `degree` vanish (so
    // `in` is a codeword of this code), 0 otherwise. Writes the recovered
    // message to out[0..degree) when `out` is non-NULL; a failed check leaves
    // the output partially written.
    int rs_decode(ArithElement *out, ArithElement *in, uint64_t size, uint64_t degree,
                  NTT_Plan *plans);

#ifdef __cplusplus
}
#endif
