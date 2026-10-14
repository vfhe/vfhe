// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
// Python-facing ABI of the fhe engine. CKKS and BFV reuse the arith + mlwe
// surfaces; declared here are the GP25 and CGGI16 blind rotations
// (fhe/c/src/gp25.c and cggi16.c) and SHIP's encoders (ship.c), all
// prototyped in fhe.h.
// Handles/handle-matrices are passed as `void *` (cffi converts any pointer or
// array cdata).

uint64_t gp25_rotation_layers(uint64_t n, uint64_t n_bits, uint64_t bits_per_layer, uint64_t *bits);
void gp25_rotate(void *acc, uint64_t n, void *keys, uint64_t n_bits, uint64_t bits_per_layer,
                 void *aut, uint64_t ell, const void *gadget_params, uint64_t n_threads);
void gp25_multiply_by_signed_monomials(void *acc, uint64_t n, uint64_t *a, void *sign, uint64_t ell,
                                       const void *gadget_params, uint64_t n_threads);
void gp25_blind_rotate(void *acc, uint64_t n, uint64_t *a, uint64_t rank, uint64_t h,
                       uint64_t gap_bits, uint64_t bits_per_layer, void *gap_keys, void *sign_keys,
                       void *aut, uint64_t ell, const void *gadget_params, uint64_t n_threads);
void cggi16_blind_rotate(void *acc, uint64_t *a, uint64_t n, void *bk, uint64_t unfolding,
                         uint64_t ell, const void *gadget_params, uint64_t n_threads);
void cggi16_blind_rotate_batch(void *acc, uint64_t *a, uint64_t count, uint64_t n, void *bk,
                               uint64_t unfolding, uint64_t ell, const void *gadget_params,
                               uint64_t n_threads);
bool ship_encode_pi(void *out, void *a, uint64_t theta, double scale, void *plan,
                    const uint32_t *slots, uint64_t n_threads);
bool ship_encode_pt0(void *out, void *b, double coefficient_re, double coefficient_im, double scale,
                     void *plan, const uint32_t *slots);
