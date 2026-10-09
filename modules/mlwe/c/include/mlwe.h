// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <arith.h>
#include <arith_generic.h>

#ifdef __cplusplus
extern "C"
{
#endif
    void gen_sparse_ternary_array_modq(uint64_t *out, uint64_t size, uint64_t h, uint64_t q);
    /* LWE
     *
     * An LWE key or sample lives over the primes of `base` that `mask`
     * selects, one limb per prime: limb i is the residue modulo the prime at
     * the i-th set bit of `mask`, in ascending base index, and `l` is the
     * number of set bits. A key and the samples it touches share one mask.
     *
     * A key's `sigma` is the standard deviation of the noise it encrypts with.
     * A key without one (LWE_NO_SIGMA, what lwe_alloc_key leaves) decrypts but
     * must not encrypt: lwe_sample and everything built on it require
     * `sigma >= 0`.
     */

#define LWE_NO_SIGMA (-1.0)

    typedef struct _LWE_Key
    {
        uint64_t **s;
        uint64_t n, l, mask;
        RNS_Base base;
        double sigma;
    } *LWE_Key;

    typedef struct _LWE
    {
        uint64_t **a;
        uint64_t *b;
        uint64_t n, l, mask;
        RNS_Base base;
    } *LWE;

    // mlwe rns
    /* MLWE RNS */

    typedef struct _RNS_MLWE_Key
    {
        // The secret, one element per module component, held in the mul
        // domain where every use of it multiplies.
        ArithElement *s;
        ArithRing ring;
        uint64_t N, l, r;
        double sigma;
    } *RNS_MLWE_Key;

    // A module-LWE sample: `r` mask components and a body, over one ring.
    //
    // Every component of a sample is in the same domain, which the elements
    // themselves carry: `mlwe_domain` reads it, and the whole-sample
    // conversions move all of them together. A caller that puts components in
    // different domains breaks that invariant, and the arithmetic will refuse
    // them.
    typedef struct _MLWE
    {
        ArithElement *a, b;
        uint64_t r;
        ArithRing ring;
    } *MLWE;

    // Aliases that let a signature say which domain it expects its argument
    // in. They are the same type, so this is documentation, not enforcement:
    // check `mlwe_domain` where it matters.
    typedef MLWE RNS_MLWE;
    typedef MLWE RNSc_MLWE;

    ArithDomain mlwe_domain(MLWE c);
    MLWE mlwe_alloc_sample(ArithRing ring, uint64_t r);

    // A key-switch key: one gadget-decomposed key array per input component
    // (NULL marks a component that keeps the target key and passes through),
    // and the ring the key lives in. A key switch must accumulate in that
    // ring -- the caller's `out` is only guaranteed to be allocated for its
    // own, narrower one -- so it allocates its scratch there and copies the
    // finished result out. The key stays immutable and therefore shareable:
    // gp25 hands one key to every thread of a parallel bootstrap.
    //
    // `log_base` is the gadget the keys were generated for, and the key switch
    // decomposes against it: 0 for the RNS gadget (one key per prime), or the
    // radix base's log for the radix one (one key per prime and digit; see
    // `gadget_radix_digits`). A key generated for one and read as the other
    // decrypts to garbage, which is why it travels with the key rather than
    // with the call. `balanced` picks the RNS gadget's digit, which the keys
    // do not depend on; it travels with them only because they are all a key
    // switch is given.
    typedef struct _RNS_MLWE_KS_Key
    {
        RNS_MLWE **s;
        uint64_t count;
        uint64_t mask;
        uint64_t log_base;
        bool balanced;
        ArithRing ring;
    } *RNS_MLWE_KS_Key;

    // mlwe rns
    RNS_MLWE_Key mlwe_alloc_RNS_key_special_primes(uint64_t N, uint64_t r, uint64_t l,
                                                   uint64_t special_primes, RNS_Base base,
                                                   double sigma);
    void free_polynomial_array(uint64_t size, IntPolynomial *p);
    RNS_MLWE_Key mlwe_get_RNS_key_from_array(uint64_t N, uint64_t r, uint64_t l, uint64_t *array,
                                             RNS_Base base, double sigma);
    RNS_MLWE_Key mlwe_alloc_key(ArithRing ring, uint64_t r, uint64_t l, double sigma);
    RNS_MLWE_Key mlwe_alloc_RNS_key(uint64_t N, uint64_t r, uint64_t l, RNS_Base base,
                                    double sigma);
    void free_RNS_mlwe_sample(RNS_MLWE c);
    void free_mlwe_RNS_key(RNS_MLWE_Key key);
    LWE mlwe_extract_LWE(RNSc_MLWE in, uint64_t idx);
    RNS_MLWE_Key mlwe_new_RNS_gaussian_key(uint64_t N, uint64_t r, uint64_t l, double key_sigma,
                                           RNS_Base base, double sigma);
    RNS_MLWE mlwe_alloc_RNS_sample(uint64_t N, uint64_t r, uint64_t mask, RNS_Base base);
    RNSc_MLWE mlwe_alloc_RNSc_sample(uint64_t N, uint64_t r, uint64_t mask, RNS_Base base);
    RNS_MLWE mlwe_new_RNS_sample(RNS_MLWE_Key key, uint64_t *m, uint64_t p);
    void mlwe_RNS_sample_of_zero(RNS_MLWE out, RNS_MLWE_Key key);
    void mlwe_RNSc_sample_of_zero(RNSc_MLWE out, RNS_MLWE_Key key);
    RNS_MLWE mlwe_new_RNS_sample_of_zero(RNS_MLWE_Key key);
    RNSc_MLWE mlwe_new_RNSc_sample_of_zero(RNS_MLWE_Key key);
    RNS_MLWE mlwe_new_RNS_trivial_sample_of_zero(uint64_t N, uint64_t r, uint64_t mask,
                                                 RNS_Base base);
    void mlwe_RNS_linear_decrypt(ArithElement *out, RNS_MLWE in, RNS_MLWE_Key key);
    void mlwe_RNSc_to_RNS(RNS_MLWE out, RNSc_MLWE in);
    void mlwe_RNS_to_RNSc(RNSc_MLWE out, RNS_MLWE in);

    void mlwe_RNS_trivial_sample_of_zero(RNS_MLWE out);
    void mlwe_copy_RNS_sample(RNS_MLWE out, RNS_MLWE in);
    void mlwe_copy_RNSc_sample(RNSc_MLWE out, RNSc_MLWE in);
    void mlwe_RNSc_sample(RNSc_MLWE out, RNS_MLWE_Key key, const ArithElement *m);
    // The same with the mask drawn from `seed`: component i of `a` is stream i
    // (arith_sample_uniform_seeded), so the sample can be stored as its seed
    // and `b`. Use a fresh seed per sample.
    void mlwe_RNS_sample_of_zero_seeded(RNS_MLWE out, RNS_MLWE_Key key, const uint8_t *seed,
                                        uint64_t seed_len);
    void mlwe_RNSc_sample_seeded(RNSc_MLWE out, RNS_MLWE_Key key, const ArithElement *m,
                                 const uint8_t *seed, uint64_t seed_len);
    // Fresh samples under `key` of every message times every scalar,
    // message-major: out[i * n_scales + k] encrypts msgs[i] scaled by
    // scales[k], a scalar being one value per component as arith_scalar_new
    // takes it. The messages are in the mul domain over `ring`, which the
    // samples are allocated over at the key's rank, owned by the caller and
    // left in the mul domain. Sample i's mask is expanded from the 32 bytes at
    // mask_seeds[32 * i], which this fills, as mlwe_RNS_sample_of_zero_seeded
    // expands it, and its noise from a secret seed of its own
    // (prng_normal_seeded). Every seed is drawn here, on the calling thread
    // and in order, so the samples do not depend on how many of the up to
    // `n_threads` threads (0: the library limit) draw them.
    void mlwe_RNS_sample_scaled_batch(RNS_MLWE *out, ArithRing ring, RNS_MLWE_Key key,
                                      const ArithElement *msgs, uint64_t n_msgs,
                                      uint64_t *const *scales, uint64_t n_scales,
                                      uint8_t *mask_seeds, uint64_t n_threads);
    void mlwe_scale_RNS_mlwe_RNS(RNS_MLWE c, const uint64_t *per_component);
    void mlwe_add_RNSc_sample(RNSc_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2);
    void mlwe_add_RNS_sample(RNS_MLWE out, RNS_MLWE in1, RNS_MLWE in2);
    void mlwe_sub_RNSc_sample(RNSc_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2);
    void mlwe_RNSc_mul_by_xai(RNSc_MLWE out, RNSc_MLWE in, uint64_t a);
    void mlwe_RNSc_mul_by_xai_minus1(RNSc_MLWE out, RNSc_MLWE in, uint64_t a);
    void mlwe_RNS_mul_addto_by_poly(RNS_MLWE out, RNS_MLWE in, const ArithElement *poly);
    void mlwe_RNS_mul_subto_by_poly(RNS_MLWE out, RNS_MLWE in, const ArithElement *poly);
    // out = sum_i coeff[i] * in[i], with plaintext coefficients, all in the mul
    // domain over out's ring and rank; `out` must not alias any in[i]. A NULL
    // coeff[i].handle skips that term; with no terms, out = 0.
    void mlwe_RNS_linear_combination(RNS_MLWE out, RNS_MLWE *in, const ArithElement *coeff,
                                     uint64_t n);
    // out[j] = sum_i coeff[j * n_in + i] * in[i] for j < n_out: a plaintext
    // matrix times a vector of samples. Up to `n_threads` threads (0: the
    // library limit, see vfhe_threads_for).
    void mlwe_RNS_linear_combinations(RNS_MLWE *out, RNS_MLWE *in, const ArithElement *coeff,
                                      uint64_t n_out, uint64_t n_in, uint64_t n_threads);
    void mlwe_scale_RNSc_mlwe(RNSc_MLWE c, uint64_t scale);
    void mlwe_addto_RNSc_sample(RNSc_MLWE out, RNSc_MLWE in);
    RNS_MLWE *mlwe_alloc_RNS_sample_array(uint64_t size, uint64_t N, uint64_t r, uint64_t mask,
                                          RNS_Base base);
    RNS_MLWE *mlwe_alloc_RNS_sample_array2(uint64_t size, RNS_MLWE c);
    void free_RNS_mlwe_array(uint64_t size, RNS_MLWE *v);
    void free_mlwe_RNS_sample(void *p);
    void mlwe_scale_RNS_mlwe_addto(RNS_MLWE out, RNS_MLWE in, uint64_t scale);
    void mlwe_RNS_mul_by_poly(RNS_MLWE out, RNS_MLWE in, const ArithElement *poly);
    void mlwe_add_RNSc_polynomial(RNSc_MLWE out, RNSc_MLWE in1, const ArithElement *in2);
    void mlwe_sub_RNSc_polynomial(RNSc_MLWE out, RNSc_MLWE in1, const ArithElement *in2);
    void mlwe_RNS_add_polynomial(RNS_MLWE out, RNS_MLWE in1, const ArithElement *in2);
    void mlwe_RNS_sub_polynomial(RNS_MLWE out, RNS_MLWE in1, const ArithElement *in2);

    // Rank of the extended (not-yet-relinearized) product of two rank-r ciphertexts:
    // r*(r+1)/2 quadratic components plus r linear ones.
    uint64_t mlwe_extended_rank(uint64_t r);
    void mlwe_tensor_product(ArithElement *out, RNS_MLWE in1, RNS_MLWE in2);
    void mlwe_multiply(RNS_MLWE out, RNS_MLWE in1, RNS_MLWE in2, RNS_MLWE_KS_Key ksk);
    // out[i] = in1[i] * in2[i] for i < n, as mlwe_multiply with `ksk` (NULL:
    // extended products). Inputs not in the mul domain are converted in place,
    // so each must appear only once; inputs in the mul domain may repeat. Up
    // to `n_threads` threads (0: the library limit).
    void mlwe_multiply_batch(RNS_MLWE *out, RNS_MLWE *in1, RNS_MLWE *in2, RNS_MLWE_KS_Key ksk,
                             uint64_t n, uint64_t n_threads);
    // mlwe_multiply_batch with `ksk`, then mlwe_round_division of each product
    // to `to`, a quotient of the inputs' ring. The outputs are canonical. Same
    // input rules.
    void mlwe_multiply_round_division_batch(RNSc_MLWE *out, RNS_MLWE *in1, RNS_MLWE *in2,
                                            RNS_MLWE_KS_Key ksk, ArithRing to, uint64_t n,
                                            uint64_t n_threads);

    RNS_MLWE_Key mlwe_new_RNS_key_from_array(uint64_t *array, uint64_t N, uint64_t r, uint64_t l,
                                             RNS_Base base, double sigma);
    void mlwe_copy_array(RNS_MLWE *out, RNS_MLWE *in, uint64_t size);
    RNS_MLWE *mlwe_create_copy_array(RNS_MLWE *in, uint64_t size);

    /* Key switching */

    // Wrap per-component gadget key arrays (borrowed, not deep-copied) into a
    // key-switch key, deriving the key's ring from its first real component.
    // `log_base` is the gadget the arrays were generated against (0 for the
    // RNS gadget), and each array must hold exactly as many keys as that
    // gadget decomposes an element of the key's ring into.
    RNS_MLWE_KS_Key mlwe_new_RNS_ks_key(RNS_MLWE **s, uint64_t count, uint64_t log_base,
                                        bool balanced);
    void free_mlwe_RNS_ks_key(RNS_MLWE_KS_Key key);
    void mlwe_RNSc_GHS_hybrid_keyswitch(RNSc_MLWE out, RNSc_MLWE in, RNS_MLWE_KS_Key ksk,
                                        uint64_t lvl);
    void mlwe_automorphism_RNSc_GHS(RNSc_MLWE out, RNSc_MLWE in, uint64_t gen, RNS_MLWE_KS_Key ksk,
                                    uint64_t lvl);
    // Hoisted automorphisms: several automorphisms of one sample, decomposing
    // it only once. mlwe_hoist copies `in` and decomposes it for the gadget
    // and ring of `ksk`; each mlwe_automorphism_RNSc_GHS_hoisted then only
    // computes the key products, and matches mlwe_automorphism_RNSc_GHS up to
    // noise. Its key must have the same gadget, ring and pass-through
    // components as `ksk` (as automorphism keys generated together for one
    // level do); otherwise it returns -1 and leaves `out` untouched. A hoisted
    // sample is read-only and may be shared between threads.
    typedef struct _MLWE_Hoisted *MLWE_Hoisted;
    MLWE_Hoisted mlwe_hoist(RNSc_MLWE in, RNS_MLWE_KS_Key ksk);
    void free_mlwe_hoisted(MLWE_Hoisted h);
    int mlwe_automorphism_RNSc_GHS_hoisted(RNSc_MLWE out, MLWE_Hoisted h, uint64_t gen,
                                           RNS_MLWE_KS_Key ksk, uint64_t lvl);
    // out[i] = Aut_gens[i](sample of h) for i < n, on up to `n_threads`
    // threads (0: the library limit). Returns -1, writing nothing, if any key
    // does not fit `h`.
    int mlwe_automorphisms_RNSc_GHS_hoisted(RNSc_MLWE *out, MLWE_Hoisted h, const uint64_t *gens,
                                            RNS_MLWE_KS_Key *ksks, uint64_t n, uint64_t lvl,
                                            uint64_t n_threads);
    // out[i] = Aut_gens[i](in[i]) for i < n, canonical, on up to `n_threads`
    // threads (0: the library limit). Inputs may be in either domain and are
    // not modified. gens[i] == 1 copies, and ksks[i] may then be NULL.
    void mlwe_automorphism_RNSc_GHS_batch(RNSc_MLWE *out, RNSc_MLWE *in, const uint64_t *gens,
                                          RNS_MLWE_KS_Key *ksks, uint64_t n, uint64_t lvl,
                                          uint64_t n_threads);
    // out = sum_i Aut_gens[i](in[i]) for i < n, canonical, on up to
    // `n_threads` threads. The key-switch products of all terms are
    // accumulated in the keys' ring and divided down once: one inverse
    // transform, one round division and one rounding error for the whole sum.
    // The keys must share a ring and pass-through components, as automorphism
    // keys for one level do; otherwise returns -1 and leaves `out` untouched.
    // gens[i] == 1 adds in[i] as is, and ksks[i] may then be NULL. Inputs are
    // over out's ring and rank, in either domain, and are not modified.
    int mlwe_automorphism_sum_RNSc_GHS(RNSc_MLWE out, RNS_MLWE *in, const uint64_t *gens,
                                       RNS_MLWE_KS_Key *ksks, uint64_t n, uint64_t n_threads);
    void mlwe_partial_trace(RNSc_MLWE out, RNSc_MLWE in, uint64_t *gens, RNS_MLWE_KS_Key *ksks,
                            uint64_t size, uint64_t lvl);
    // The trace Tr_(K/Q) of in's message (canonical): `out` decrypts to N times
    // its constant coefficient. Evaluated over the tower of power-of-two
    // cyclotomics, one automorphism per level [AP13]: ksks[i] is the key for
    // 2^(log2 N - i) + 1, i < log2 N.
    void mlwe_trace(RNSc_MLWE out, RNSc_MLWE in, RNS_MLWE_KS_Key *ksks, uint64_t lvl);
    // The normalized trace (n / N) Tr_(K/K_n) [AP13] of `in` (canonical), K_n
    // the subring Z[X^(N/n)], each level halved before its automorphism after
    // [LY26]: `out` decrypts to the coefficients of in's message at the
    // multiples of N / n and to zero elsewhere (n = 1: the constant
    // coefficient alone), with the noise of log2(N / n) key switches, which
    // later levels do not amplify. n is a power of two dividing N; ksks[j] is
    // the automorphism key for 2^(j + 1) + 1, as for mlwe_trace_pack, and only
    // those with j >= log2(n) are read. `out` may alias `in`.
    void mlwe_normalized_trace(RNSc_MLWE out, RNSc_MLWE in, uint64_t n, RNS_MLWE_KS_Key *ksks);
    // Packs `size` LWE samples into one: out decrypts to sum_k m_k X^k, m_k
    // what in[k] decrypts to (size <= N). The samples live over the same
    // primes as `out`'s ring -- by value: they may come from a ring of
    // another dimension -- under a key whose coefficients `ksk` switches from:
    // component i of `ksk` encrypts the i-th coefficient, against the key's
    // gadget.
    void mlwe_full_packing_keyswitch(RNS_MLWE out, LWE *in, uint64_t size, RNS_MLWE_KS_Key ksk);
    // Packs the constant coefficients of the 2^ell canonical samples of `vec`
    // into vec[0]: it decrypts to m_k at k * N / 2^ell, m_k the constant
    // coefficient of what vec[k] decrypts to, and to junk elsewhere. The values
    // keep their scale, and each level adds one key switch's noise, which the
    // levels after it do not amplify. ksks[j] is the automorphism key for
    // 2^(j + 1) + 1 at the
    // samples' level, j < ell. The other samples of `vec` are overwritten.
    void mlwe_trace_pack(RNSc_MLWE *vec, uint64_t ell, RNS_MLWE_KS_Key *ksks);

    // Subring maps between R_N = Z[X]/(X^N + 1) and R_n = Z[Y]/(Y^n + 1),
    // N = k * n, R_n embedded as Y = X^k. Both move coefficients only, so they
    // add no noise; the input is canonical and the output is written
    // canonical over its own ring, which must have the input ring's primes
    // (by value: the two dimensions have bases of their own).
    //
    // mlwe_project_subring: `in` over R_N of rank r, `out` over R_n of rank
    // r * k. Component i * k + j of `out` holds coefficients j + k * m of
    // component i of `in` (at m), the body coefficients k * m. Its linear
    // decryption under the key whose components i * k + j are s_i^(0) for
    // j = 0 and Y * s_i^(k - j) otherwise -- s_i^(l) being coefficients
    // l + k * m of s_i -- is coefficients k * m of the linear decryption of
    // `in` under s: the subring part of the message, and the rest dropped.
    void mlwe_project_subring(MLWE out, MLWE in);
    // mlwe_embed_subring: `count` <= k samples in[i] over R_n, one ring and
    // rank r, and `out` over R_N of rank r: out = sum_i X^i in[i](X^k), so it
    // decrypts under s(X^k) to sum_i X^i m_i(X^k) -- coefficient i + k * m
    // holds coefficient m of m_i, and the coefficients of missing inputs are
    // zero. With one input it is the embedding Y = X^k, and with k the inverse
    // of mlwe_project_subring's split of the mask.
    void mlwe_embed_subring(MLWE out, MLWE *in, uint64_t count);
    // Ring switching [GHPS12], down by reading the sample as one of rank r * k
    // over the subring before the key switch: mlwe_project_subring of in[0]
    // (if its dimension is at least `out`'s; `count` is then 1) or
    // mlwe_embed_subring of the `count` inputs, into a sample over `out`'s
    // ring, then a key switch with `ksk`, whose components switch from the
    // key that map leaves the sample under (one per component it produces,
    // r * k down and r up) to `out`'s key, at any rank. `ksk` belongs to the
    // level of `out`'s ring.
    void mlwe_ring_switch(RNSc_MLWE out, RNSc_MLWE *in, uint64_t count, RNS_MLWE_KS_Key ksk);

    void mlwe_round_division(RNSc_MLWE out, ArithRing to);
    // Reduces every component into `to`, a quotient of the sample's ring, in
    // place and in either domain. Unlike mlwe_round_division, the value is not
    // divided: this is the CKKS level drop, and it is not valid for BFV.
    void mlwe_mod_reduce(MLWE c, ArithRing to);
    // mlwe_round_division of each of the n distinct samples, in place,
    // converting them to the canonical domain first if needed. Up to
    // `n_threads` threads (0: the library limit).
    void mlwe_round_division_batch(RNSc_MLWE *io, ArithRing to, uint64_t n, uint64_t n_threads);
    // Moves each of the n distinct samples to the mul domain in place (a no-op
    // for those already there). Up to `n_threads` threads.
    void mlwe_RNSc_to_RNS_batch(RNS_MLWE *io, uint64_t n, uint64_t n_threads);

    // Gadget decomposition products: decompose `poly` against the gadget
    // `log_base` names -- the RNS one (one digit per prime) when it is 0, the
    // radix one (one digit per prime and power of 2^log_base) otherwise -- and
    // accumulate the products with the matching keys into `out`. `ksk` must
    // hold one key per digit, prime-major, generated against the same gadget.
    // `balanced` centers the RNS gadget's digit, in (-p_j/2, p_j/2].
    void gadget_mul_addto_polynomial(RNS_MLWE out, RNS_MLWE *ksk, const ArithElement *poly,
                                     uint64_t log_base, bool balanced);
    void gadget_mul_subto_polynomial(RNS_MLWE out, RNS_MLWE *ksk, const ArithElement *poly,
                                     uint64_t log_base, bool balanced);

    // How many base-2^log_base digits the radix gadget takes for one residue
    // modulo `prime`: enough to cover every value below it.
    uint64_t gadget_radix_digits(uint64_t prime, uint64_t log_base);

    // A stored gadget decomposition of one element: `n` digits over the key's
    // ring, in key order, ready to multiply (in the mul domain, or canonical
    // when the ring is not fully split).
    typedef struct
    {
        ArithElement *digit;
        uint64_t n;
    } GadgetDigits;

    // Decomposes `poly` as gadget_mul_*to_polynomial would, keeping the
    // digits. `ksk` only fixes the key ring and the gadget, so any key array
    // with both will do.
    void gadget_decompose(GadgetDigits *out, RNS_MLWE *ksk, const ArithElement *poly,
                          uint64_t log_base, bool balanced);
    void gadget_digits_free(GadgetDigits *digits);
    // Digit `i` (in key order) of the same decomposition, for a caller that
    // computes the digits in parallel: written into `out`, an element of the
    // key ring, in the domain gadget_decompose leaves them in. Reads `ksk` and
    // `poly` only.
    void gadget_decompose_digit(ArithElement *out, RNS_MLWE *ksk, const ArithElement *poly,
                                uint64_t i, uint64_t log_base, bool balanced);
    // out -= sum_i Aut_gen(digit_i) * ksk[i]: the gadget product of
    // Aut_gen(poly), because permuted digits are a valid decomposition of the
    // permuted element (equal to decomposing Aut_gen(poly) up to the choice of
    // digit representatives). `gen` is odd and below 2N; 1 is the identity.
    void gadget_mul_subto_automorphism(RNS_MLWE out, RNS_MLWE *ksk, const GadgetDigits *digits,
                                       uint64_t gen);

    // `ell` is the number of gadget keys per component of the MGSW key -- one
    // per prime for the RNS gadget (`log_base` 0), one per prime and digit for
    // the radix one -- and `log_base` and `balanced` are as in
    // `gadget_mul_addto_polynomial`.
    void mgsw_external_product(RNS_MLWE out, RNS_MLWE *mgsw, RNSc_MLWE in, uint64_t ell,
                               uint64_t special_primes, uint64_t log_base, bool balanced);
    void mgsw_CMUX(RNS_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2, RNS_MLWE *mgsw, uint64_t ell,
                   uint64_t special_primes, uint64_t log_base, bool balanced);
    void mgsw_NCMUX(RNS_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2, RNS_MLWE *mgsw, RNS_MLWE_KS_Key ksk,
                    uint64_t ell, uint64_t special_primes, uint64_t log_base, bool balanced);
    void mgsw_CMUX_to_coeff(RNS_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2, RNS_MLWE *mgsw,
                            uint64_t ell, uint64_t special_primes, uint64_t log_base,
                            bool balanced);
    void mgsw_NCMUX_to_coeff(RNS_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2, RNS_MLWE *mgsw,
                             RNS_MLWE_KS_Key ksk, uint64_t ell, uint64_t special_primes,
                             uint64_t log_base, bool balanced);

    // MGSW keys as arrays of (r + 1) * `width` samples, row j * width + k
    // encrypting mu_j * g_k for the gadget elements g_k, with mu_j = -s_j * m
    // for j < r and mu_r = m: the layout of `mgsw_external_product`'s key.
    // Rows are in the mul domain over the key's ring; what these allocate the
    // caller owns.

    // The noiseless MGSW of `msg` (in the mul domain over `ring`): row j * width + k
    // is zero but for component j (the body for j = r), which is msg times
    // the gadget element `scales[k]` (one value per component, as
    // arith_scalar_new takes it). It decrypts as m * G under any key of rank r.
    void mgsw_trivial(RNS_MLWE *out, ArithRing ring, uint64_t r, const ArithElement *msg,
                      uint64_t *const *scales, uint64_t width);

    // The internal product: out[i] = a (x) b[i] for each of the `rows` rows of
    // an MGSW key b, so `out` is an MGSW key of the product of the two
    // messages, laid out as b and over b's ring. `a` (with `ell` gadget keys
    // per component) must consume samples over that ring: with special
    // primes, b's rows live over its key ring, so `a` belongs to a scheme
    // where that ring is a level. Up to `n_threads` threads.
    void mgsw_internal_product(RNS_MLWE *out, RNS_MLWE *a, uint64_t ell, RNS_MLWE *b, uint64_t rows,
                               uint64_t log_base, bool balanced, uint64_t n_threads);

    // The MGSW key of Aut_gen(m) from that of m (`in`, gadget `width`):
    // its m rows go through the automorphism and `aut` (Aut_gen(s) -> s),
    // and the -s_j m rows are rebuilt from them through the relinearization
    // key `rlk` (the quadratic terms -(s_p s_q) -> s, as for a product). Both
    // keys consume samples over the rows' ring. Up to `n_threads` threads.
    void mgsw_automorphism(RNS_MLWE *out, RNS_MLWE *in, uint64_t width, uint64_t gen,
                           RNS_MLWE_KS_Key aut, RNS_MLWE_KS_Key rlk, uint64_t n_threads);

    // lwe
    LWE_Key lwe_alloc_key(uint64_t n, uint64_t mask, RNS_Base base);
    LWE lwe_alloc_sample(uint64_t n, uint64_t mask, RNS_Base base);
    void free_lwe_sample(LWE c);
    LWE_Key lwe_new_key(uint64_t n, uint64_t mask, RNS_Base base, double sec_sigma,
                        double err_sigma);
    LWE_Key lwe_new_sparse_ternary_key(uint64_t n, uint64_t mask, RNS_Base base, uint64_t h,
                                       double err_sigma);
    void lwe_sample(LWE c, uint64_t *m, LWE_Key key);
    LWE lwe_new_sample(uint64_t *m, LWE_Key key);
    LWE lwe_new_trivial_sample(uint64_t *m, uint64_t n, uint64_t mask, RNS_Base base);
    void lwe_linear_decrypt(uint64_t *out, LWE c, LWE_Key key);
    void lwe_subto(LWE out, LWE in);

#ifdef __cplusplus
}
#endif
