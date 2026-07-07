#pragma once
#include <arith.h>

#ifdef __cplusplus
extern "C"
{
#endif
    /* LWE */

    typedef struct _LWE_Key
    {
        uint64_t **s;
        uint64_t n, l;
        incNTT ntt;
        double sigma;
    } *LWE_Key;

    typedef struct _LWE
    {
        uint64_t **a;
        uint64_t *b;
        uint64_t n, l;
        incNTT ntt;
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
        IntPolynomial *s;
        RNS_Polynomial *s_RNS;
        uint64_t N, l, r;
        double sigma;
    } *RNS_MLWE_Key;

    typedef struct _RNS_MLWE
    {
        RNS_Polynomial *a, b;
        uint64_t r;
    } *RNS_MLWE;

    typedef struct _RNSc_MLWE
    {
        RNSc_Polynomial *a, b;
        uint64_t r;
    } *RNSc_MLWE;

    typedef struct _RNS_MLWE_KS_Key
    {
        RNS_MLWE **s;
        uint64_t ell;
    } *RNS_MLWE_KS_Key;

    // mlwe rns
    RNS_MLWE_Key mlwe_alloc_RNS_key_special_primes(uint64_t N, uint64_t r, uint64_t l,
                                                   uint64_t special_primes, incNTT ntt,
                                                   double sigma);
    void free_polynomial_array(uint64_t size, IntPolynomial *p);
    RNS_MLWE_Key mlwe_get_RNS_key_from_array(uint64_t N, uint64_t r, uint64_t l, uint64_t *array,
                                             incNTT ntt, double sigma);
    RNS_MLWE_Key mlwe_alloc_RNS_key(uint64_t N, uint64_t r, uint64_t l, incNTT ntt, double sigma);
    void free_RNS_mlwe_sample(RNS_MLWE c);
    void free_mlwe_RNS_key(RNS_MLWE_Key key);
    LWE mlwe_extract_LWE(RNSc_MLWE in, uint64_t idx);
    RNS_MLWE_Key mlwe_new_RNS_gaussian_key(uint64_t N, uint64_t r, uint64_t l, double key_sigma,
                                           incNTT ntt, double sigma);
    RNS_MLWE mlwe_alloc_RNS_sample(uint64_t N, uint64_t r, uint64_t mask, incNTT ntt);
    RNSc_MLWE mlwe_alloc_RNSc_sample(uint64_t N, uint64_t r, uint64_t mask, incNTT ntt);
    RNS_MLWE mlwe_new_RNS_sample(RNS_MLWE_Key key, uint64_t *m, uint64_t p);
    void mlwe_RNS_sample_of_zero(RNS_MLWE out, RNS_MLWE_Key key);
    void mlwe_RNSc_sample_of_zero(RNSc_MLWE out, RNS_MLWE_Key key);
    RNS_MLWE mlwe_new_RNS_sample_of_zero(RNS_MLWE_Key key);
    RNSc_MLWE mlwe_new_RNSc_sample_of_zero(RNS_MLWE_Key key);
    RNS_MLWE mlwe_new_RNS_trivial_sample_of_zero(uint64_t N, uint64_t r, uint64_t mask, incNTT ntt);
    void mlwe_RNS_phase(RNS_Polynomial out, RNS_MLWE in, RNS_MLWE_Key key);
    void mlwe_RNSc_to_RNS(RNS_MLWE out, RNSc_MLWE in);
    void mlwe_RNS_to_RNSc(RNSc_MLWE out, RNS_MLWE in);

    void mlwe_RNS_trivial_sample_of_zero(RNS_MLWE out);
    void mlwe_copy_RNS_sample(RNS_MLWE out, RNS_MLWE in);
    void mlwe_copy_RNSc_sample(RNSc_MLWE out, RNSc_MLWE in);
    void mlwe_RNSc_sample(RNSc_MLWE out, RNS_MLWE_Key key, RNSc_Polynomial m);
    void mlwe_scale_RNS_mlwe_RNS(RNS_MLWE c, uint64_t *scale);
    void mlwe_add_RNSc_sample(RNSc_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2);
    void mlwe_add_RNS_sample(RNS_MLWE out, RNS_MLWE in1, RNS_MLWE in2);
    void mlwe_sub_RNSc_sample(RNSc_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2);
    void mlwe_RNSc_mul_by_xai(RNSc_MLWE out, RNSc_MLWE in, uint64_t a);
    void mlwe_RNSc_mul_by_xai_minus1(RNSc_MLWE out, RNSc_MLWE in, uint64_t a);
    void mlwe_RNS_mul_addto_by_poly(RNS_MLWE out, RNS_MLWE in, RNS_Polynomial poly);
    void mlwe_RNS_mul_subto_by_poly(RNS_MLWE out, RNS_MLWE in, RNS_Polynomial poly);
    void mlwe_automorphism_RNSc_GHS(RNSc_MLWE out, RNSc_MLWE in, uint64_t gen, RNS_MLWE **ksk,
                                    uint64_t lvl);
    void mlwe_scale_RNSc_mlwe(RNSc_MLWE c, uint64_t scale);
    void mlwe_RNSc_mod_switch(RNSc_MLWE c, uint64_t q);
    void mlwe_addto_RNSc_sample(RNSc_MLWE out, RNSc_MLWE in);
    RNS_MLWE *mlwe_alloc_RNS_sample_array(uint64_t size, uint64_t N, uint64_t r, uint64_t mask,
                                          incNTT ntt);
    RNS_MLWE *mlwe_alloc_RNS_sample_array2(uint64_t size, RNS_MLWE c);
    void free_RNS_mlwe_array(uint64_t size, RNS_MLWE *v);
    void free_mlwe_RNS_sample(void *p);
    void mlwe_scale_RNS_mlwe_addto(RNS_MLWE out, RNS_MLWE in, uint64_t scale);
    void mlwe_RNS_mul_by_poly(RNS_MLWE out, RNS_MLWE in, RNS_Polynomial poly);
    void mlwe_RNSc_extract_lwe(uint64_t *out, RNSc_MLWE in, uint64_t idx);
    void mlwe_add_RNSc_polynomial(RNSc_MLWE out, RNSc_MLWE in1, RNSc_Polynomial in2);
    void mlwe_sub_RNSc_polynomial(RNSc_MLWE out, RNSc_MLWE in1, RNSc_Polynomial in2);
    void mlwe_RNS_add_polynomial(RNS_MLWE out, RNS_MLWE in1, RNS_Polynomial in2);
    void mlwe_RNS_sub_polynomial(RNS_MLWE out, RNS_MLWE in1, RNS_Polynomial in2);

    void mlwe_discrete_convolution(RNS_Polynomial *out, RNS_MLWE in1, RNS_MLWE in2);
    void mlwe_multiply(RNS_MLWE out, RNS_MLWE in1, RNS_MLWE in2, RNS_MLWE **ksk);

    RNS_MLWE_Key mlwe_new_RNS_key_from_array(uint64_t *array, uint64_t N, uint64_t r, uint64_t l,
                                             incNTT ntt, double sigma);
    void mlwe_copy_array(RNS_MLWE *out, RNS_MLWE *in, uint64_t size);
    RNS_MLWE *mlwe_create_copy_array(RNS_MLWE *in, uint64_t size);

    void mlwe_RNSc_GHS_hybrid_keyswitch(RNSc_MLWE out, RNSc_MLWE in, RNS_MLWE **ksk, uint64_t lvl);
    void mlwe_partial_trace(RNSc_MLWE out, RNSc_MLWE in, uint64_t *gens, RNS_MLWE ***ksks,
                            uint64_t size, uint64_t lvl);
    void mlwe_trace(RNSc_MLWE out, RNSc_MLWE in, RNS_MLWE ***ksks, uint64_t lvl);
    void mlwe_full_packing_keyswitch(RNS_MLWE out, LWE *in, uint64_t size, RNS_MLWE **ksk,
                                     uint64_t lvl);
    void mlwe_full_packing_keyswitch_scaled(RNSc_MLWE *vec, uint64_t ell, RNS_MLWE ***ksks,
                                            uint64_t lvl);
    void mlwe_round_division_RNSc(RNSc_MLWE out, uint64_t divide_mask);

    // gadget decomposition products
    void gadget_mul_addto_polynomial(RNS_MLWE out, RNS_MLWE *ksk, RNSc_Polynomial poly);
    void gadget_mul_subto_polynomial(RNS_MLWE out, RNS_MLWE *ksk, RNSc_Polynomial poly);

    void mgsw_external_product(RNS_MLWE out, RNS_MLWE *mgsw, RNSc_MLWE in, uint64_t ell,
                               uint64_t special_primes);
    void mgsw_CMUX(RNS_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2, RNS_MLWE *mgsw, uint64_t ell,
                   uint64_t special_primes);
    void mgsw_NCMUX(RNS_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2, RNS_MLWE *mgsw, RNS_MLWE **ksk,
                    uint64_t ell, uint64_t special_primes);
    void mgsw_CMUX_to_coeff(RNS_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2, RNS_MLWE *mgsw,
                            uint64_t ell, uint64_t special_primes);
    void mgsw_NCMUX_to_coeff(RNS_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2, RNS_MLWE *mgsw,
                             RNS_MLWE **ksk, uint64_t ell, uint64_t special_primes);
    void gp25_RGSW_monomial_mul(RNS_MLWE *p0, uint64_t in_N, RNS_MLWE **e, uint64_t r_prec,
                                RNS_MLWE **ksk, uint64_t ell, uint64_t special_primes);
    void gp25_RGSW_monomial_mul_mt(RNS_MLWE *p0, uint64_t in_N, RNS_MLWE **e, uint64_t r_prec,
                                   RNS_MLWE **ksk, uint64_t ell, uint64_t special_primes,
                                   uint64_t num_threads);
    void gp25_sub_a_mt(RNS_MLWE *p0, uint64_t in_N, uint64_t *a, RNS_MLWE *s_sign, uint64_t ell,
                       uint64_t special_primes, uint64_t N, uint64_t num_threads);

    // lwe
    LWE_Key lwe_alloc_key(uint64_t n, uint64_t l, incNTT ntt);
    LWE lwe_alloc_sample(uint64_t n, uint64_t l, incNTT ntt);
    void free_lwe_sample(LWE c);
    LWE_Key lwe_new_key(uint64_t n, uint64_t l, incNTT ntt, double sec_sigma, double err_sigma);
    LWE_Key lwe_new_sparse_ternary_key(uint64_t n, uint64_t l, incNTT ntt, uint64_t h,
                                       double err_sigma);
    void lwe_sample(LWE c, uint64_t *m, LWE_Key key);
    LWE lwe_new_sample(uint64_t *m, LWE_Key key);
    LWE lwe_new_trivial_sample(uint64_t *m, uint64_t n, uint64_t l, incNTT ntt);
    void lwe_phase(uint64_t *out, LWE c, LWE_Key key);
    void lwe_subto(LWE out, LWE in);
    LWE_KS_Key lwe_new_KS_key(LWE_Key out_key, LWE_Key in_key, uint64_t t, uint64_t base_bit);
    void lwe_keyswitch(LWE out, LWE in, LWE_KS_Key ks_key);

#ifdef __cplusplus
}
#endif
