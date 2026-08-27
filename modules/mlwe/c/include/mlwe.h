// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <arith.h>
#include <arith_generic.h>

#ifdef __cplusplus
extern "C"
{
#endif
    /* LWE */

    typedef struct _LWE_Key
    {
        uint64_t **s;
        uint64_t n, l;
        RNS_Base base;
        double sigma;
    } *LWE_Key;

    typedef struct _LWE
    {
        uint64_t **a;
        uint64_t *b;
        uint64_t n, l;
        RNS_Base base;
    } *LWE;

    typedef struct _LWE_KS_Key
    {
        LWE ***s;
        uint64_t base_bit, t;
    } *LWE_KS_Key;

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
    typedef struct _RNS_MLWE_KS_Key
    {
        RNS_MLWE **s;
        uint64_t count;
        uint64_t mask;
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
    void mlwe_RNS_phase(ArithElement *out, RNS_MLWE in, RNS_MLWE_Key key);
    void mlwe_RNSc_to_RNS(RNS_MLWE out, RNSc_MLWE in);
    void mlwe_RNS_to_RNSc(RNSc_MLWE out, RNS_MLWE in);

    void mlwe_RNS_trivial_sample_of_zero(RNS_MLWE out);
    void mlwe_copy_RNS_sample(RNS_MLWE out, RNS_MLWE in);
    void mlwe_copy_RNSc_sample(RNSc_MLWE out, RNSc_MLWE in);
    void mlwe_RNSc_sample(RNSc_MLWE out, RNS_MLWE_Key key, const ArithElement *m);
    void mlwe_scale_RNS_mlwe_RNS(RNS_MLWE c, const uint64_t *per_component);
    void mlwe_add_RNSc_sample(RNSc_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2);
    void mlwe_add_RNS_sample(RNS_MLWE out, RNS_MLWE in1, RNS_MLWE in2);
    void mlwe_sub_RNSc_sample(RNSc_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2);
    void mlwe_RNSc_mul_by_xai(RNSc_MLWE out, RNSc_MLWE in, uint64_t a);
    void mlwe_RNSc_mul_by_xai_minus1(RNSc_MLWE out, RNSc_MLWE in, uint64_t a);
    void mlwe_RNS_mul_addto_by_poly(RNS_MLWE out, RNS_MLWE in, const ArithElement *poly);
    void mlwe_RNS_mul_subto_by_poly(RNS_MLWE out, RNS_MLWE in, const ArithElement *poly);
    void mlwe_automorphism_RNSc_GHS(RNSc_MLWE out, RNSc_MLWE in, uint64_t gen, RNS_MLWE_KS_Key ksk,
                                    uint64_t lvl);
    void mlwe_scale_RNSc_mlwe(RNSc_MLWE c, uint64_t scale);
    void mlwe_RNSc_mod_switch(RNSc_MLWE c, uint64_t q);
    void mlwe_addto_RNSc_sample(RNSc_MLWE out, RNSc_MLWE in);
    RNS_MLWE *mlwe_alloc_RNS_sample_array(uint64_t size, uint64_t N, uint64_t r, uint64_t mask,
                                          RNS_Base base);
    RNS_MLWE *mlwe_alloc_RNS_sample_array2(uint64_t size, RNS_MLWE c);
    void free_RNS_mlwe_array(uint64_t size, RNS_MLWE *v);
    void free_mlwe_RNS_sample(void *p);
    void mlwe_scale_RNS_mlwe_addto(RNS_MLWE out, RNS_MLWE in, uint64_t scale);
    void mlwe_RNS_mul_by_poly(RNS_MLWE out, RNS_MLWE in, const ArithElement *poly);
    void mlwe_RNSc_extract_lwe(uint64_t *out, RNSc_MLWE in, uint64_t idx);
    void mlwe_add_RNSc_polynomial(RNSc_MLWE out, RNSc_MLWE in1, const ArithElement *in2);
    void mlwe_sub_RNSc_polynomial(RNSc_MLWE out, RNSc_MLWE in1, const ArithElement *in2);
    void mlwe_RNS_add_polynomial(RNS_MLWE out, RNS_MLWE in1, const ArithElement *in2);
    void mlwe_RNS_sub_polynomial(RNS_MLWE out, RNS_MLWE in1, const ArithElement *in2);

    // Rank of the extended (not-yet-relinearized) product of two rank-r ciphertexts:
    // r*(r+1)/2 quadratic components plus r linear ones.
    uint64_t mlwe_extended_rank(uint64_t r);
    void mlwe_tensor_product(ArithElement *out, RNS_MLWE in1, RNS_MLWE in2);
    void mlwe_multiply(RNS_MLWE out, RNS_MLWE in1, RNS_MLWE in2, RNS_MLWE_KS_Key ksk);

    RNS_MLWE_Key mlwe_new_RNS_key_from_array(uint64_t *array, uint64_t N, uint64_t r, uint64_t l,
                                             RNS_Base base, double sigma);
    void mlwe_copy_array(RNS_MLWE *out, RNS_MLWE *in, uint64_t size);
    RNS_MLWE *mlwe_create_copy_array(RNS_MLWE *in, uint64_t size);

    // Wrap per-component gadget key arrays (borrowed, not deep-copied) into a
    // key-switch key, deriving the key's ring from its first real component
    // and allocating the accumulator in it.
    RNS_MLWE_KS_Key mlwe_new_RNS_ks_key(RNS_MLWE **s, uint64_t count);
    void free_mlwe_RNS_ks_key(RNS_MLWE_KS_Key key);
    void mlwe_RNSc_GHS_hybrid_keyswitch(RNSc_MLWE out, RNSc_MLWE in, RNS_MLWE_KS_Key ksk,
                                        uint64_t lvl);
    void mlwe_partial_trace(RNSc_MLWE out, RNSc_MLWE in, uint64_t *gens, RNS_MLWE_KS_Key *ksks,
                            uint64_t size, uint64_t lvl);
    void mlwe_trace(RNSc_MLWE out, RNSc_MLWE in, RNS_MLWE_KS_Key *ksks, uint64_t lvl);
    void mlwe_full_packing_keyswitch(RNS_MLWE out, LWE *in, uint64_t size, RNS_MLWE_KS_Key ksk,
                                     uint64_t lvl);
    void mlwe_full_packing_keyswitch_scaled(RNSc_MLWE *vec, uint64_t ell, RNS_MLWE_KS_Key *ksks,
                                            uint64_t lvl);
    void mlwe_round_division(RNSc_MLWE out, ArithRing to);

    // gadget decomposition products
    void gadget_mul_addto_polynomial(RNS_MLWE out, RNS_MLWE *ksk, const ArithElement *poly);
    void gadget_mul_subto_polynomial(RNS_MLWE out, RNS_MLWE *ksk, const ArithElement *poly);

    void mgsw_external_product(RNS_MLWE out, RNS_MLWE *mgsw, RNSc_MLWE in, uint64_t ell,
                               uint64_t special_primes);
    void mgsw_CMUX(RNS_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2, RNS_MLWE *mgsw, uint64_t ell,
                   uint64_t special_primes);
    void mgsw_NCMUX(RNS_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2, RNS_MLWE *mgsw, RNS_MLWE_KS_Key ksk,
                    uint64_t ell, uint64_t special_primes);
    void mgsw_CMUX_to_coeff(RNS_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2, RNS_MLWE *mgsw,
                            uint64_t ell, uint64_t special_primes);
    void mgsw_NCMUX_to_coeff(RNS_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2, RNS_MLWE *mgsw,
                             RNS_MLWE_KS_Key ksk, uint64_t ell, uint64_t special_primes);
    void gp25_RGSW_monomial_mul(RNS_MLWE *p0, uint64_t in_N, RNS_MLWE **e, uint64_t r_prec,
                                RNS_MLWE_KS_Key ksk, uint64_t ell, uint64_t special_primes);
    void gp25_RGSW_monomial_mul_mt(RNS_MLWE *p0, uint64_t in_N, RNS_MLWE **e, uint64_t r_prec,
                                   RNS_MLWE_KS_Key ksk, uint64_t ell, uint64_t special_primes,
                                   uint64_t num_threads);
    void gp25_sub_a_mt(RNS_MLWE *p0, uint64_t in_N, uint64_t *a, RNS_MLWE *s_sign, uint64_t ell,
                       uint64_t special_primes, uint64_t N, uint64_t num_threads);

    // lwe
    LWE_Key lwe_alloc_key(uint64_t n, uint64_t l, RNS_Base base);
    LWE lwe_alloc_sample(uint64_t n, uint64_t l, RNS_Base base);
    void free_lwe_sample(LWE c);
    LWE_Key lwe_new_key(uint64_t n, uint64_t l, RNS_Base base, double sec_sigma, double err_sigma);
    LWE_Key lwe_new_sparse_ternary_key(uint64_t n, uint64_t l, RNS_Base base, uint64_t h,
                                       double err_sigma);
    void lwe_sample(LWE c, uint64_t *m, LWE_Key key);
    LWE lwe_new_sample(uint64_t *m, LWE_Key key);
    LWE lwe_new_trivial_sample(uint64_t *m, uint64_t n, uint64_t l, RNS_Base base);
    void lwe_phase(uint64_t *out, LWE c, LWE_Key key);
    void lwe_subto(LWE out, LWE in);
    LWE_KS_Key lwe_new_KS_key(LWE_Key out_key, LWE_Key in_key, uint64_t t, uint64_t base_bit);
    void lwe_keyswitch(LWE out, LWE in, LWE_KS_Key ks_key);

#ifdef __cplusplus
}
#endif
