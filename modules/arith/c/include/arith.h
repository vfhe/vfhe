#ifndef __NTT_H__
#define __NTT_H__

#ifdef __AVX512IFMA__
#include <immintrin.h>
#endif
#include <stdbool.h>
#include <stdint.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#if defined(__AVX512IFMA__) && !defined(PORTABLE_BUILD) && !defined(PORTABLE)
typedef __m512i mp_vector_t;
#else
typedef uint64_t mp_vector_t;
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct _NTT_proc
    {
        uint64_t n;
        uint64_t q;
        void **ws_fwd;
        void **w_precon_fwd;
        void **ws_inv;
        void **w_precon_inv;
        uint64_t root_of_unity;
        uint64_t inv_root_of_unity;
        uint64_t k;
        uint64_t m;
        uint64_t m52;
        uint64_t ifma_barr_lo;
        uint64_t ifma_prod_right_shift;
        uint64_t mp_w1;
        uint64_t mp_w2;
    } *NTT_proc;

    typedef struct _incNTT
    {
        NTT_proc *ntt;
        uint64_t split_degree;
        uint64_t **w;
        uint64_t N, l;
    } *incNTT;

    void incNTT_extend_with_primes(incNTT ntt, uint64_t *new_primes, uint64_t count);

    static inline uint64_t rns_mask_to_l(uint64_t mask)
    {
        uint64_t count = 0;
        while (mask)
        {
            count += (mask & 1);
            mask >>= 1;
        }
        return count;
    }

    int rns_mask_get_active_index(uint64_t mask, uint64_t i);
    int rns_mask_get_last_active_index(uint64_t mask);

    typedef struct _RNS_Polynomial
    {
        uint64_t **coeffs;
        incNTT ntt;
        uint64_t rns_mask;
        uint64_t allocated_l;
    } *RNS_Polynomial;

    /* RNS polynomial in coefficient representation*/
    typedef struct _RNSc_Polynomial
    {
        uint64_t **coeffs;
        incNTT ntt;
        uint64_t rns_mask;
        uint64_t allocated_l;
    } *RNSc_Polynomial;

    typedef struct _ZqVector
    {
        uint64_t **elements;
        uint64_t n, l;
        NTT_proc *ntt;
    } *ZqVector;

    typedef struct _IntPolynomial
    {
        uint64_t *coeffs;
        uint64_t N;
    } *IntPolynomial;

#ifdef __AVX512IFMA__
    void ntt_CT_NR(__m512i *x, __m512i **ws, __m512i **w_precon, uint64_t n, uint64_t q,
                   NTT_proc proc);
    void ntt_GS_RN(__m512i *x, __m512i **ws, __m512i **w_precon, uint64_t n, uint64_t q,
                   NTT_proc proc);
#endif

#if defined(__AVX512IFMA__) && !defined(PORTABLE_BUILD) && !defined(PORTABLE)
    void ntt_precompute_fwd(uint64_t n, uint64_t q, uint64_t root_of_unity, __m512i ***out_ws,
                            __m512i ***out_w_precon);
    void ntt_precompute_inv(uint64_t n, uint64_t q, uint64_t inv_root_of_unity, __m512i ***out_ws,
                            __m512i ***out_w_precon);
    void ntt_free_precompute(__m512i **ws, __m512i **w_precon, uint64_t n);
#else
void ntt_precompute_fwd(uint64_t n, uint64_t q, uint64_t root_of_unity, uint64_t ***out_ws,
                        uint64_t ***out_w_precon);
void ntt_precompute_inv(uint64_t n, uint64_t q, uint64_t inv_root_of_unity, uint64_t ***out_ws,
                        uint64_t ***out_w_precon);
void ntt_free_precompute(uint64_t **ws, uint64_t **w_precon, uint64_t n);
#endif

    void ntt_forward(uint64_t *out, uint64_t *in, NTT_proc proc);
    void ntt_reverse(uint64_t *out, uint64_t *in, NTT_proc proc);

    uint64_t add_modq(uint64_t a, uint64_t b, uint64_t q);
    uint64_t sub_modq(uint64_t a, uint64_t b, uint64_t q);
    uint64_t negate_modq(uint64_t a, uint64_t q);
    uint64_t mul_modq(uint64_t a, uint64_t b, NTT_proc proc);
    uint64_t modq(unsigned __int128 x, NTT_proc proc);

    void mod_eltwise_mul(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc);
    void mod_eltwise_mul_addto(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n,
                               NTT_proc proc);
    void mod_eltwise_mul_subto(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n,
                               NTT_proc proc);
    void mod_eltwise_scale(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, NTT_proc proc);
    void mod_eltwise_fma(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, NTT_proc proc);
    void mod_eltwise_add_scalar(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n,
                                NTT_proc proc);
    void mod_eltwise_sub_scalar(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n,
                                NTT_proc proc);
    void mod_eltwise_negate(uint64_t *out, uint64_t *in, uint64_t n, NTT_proc proc);
    void mod_eltwise_add(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc);
    void mod_eltwise_sub(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc);
    void mod_eltwise_reduce(uint64_t *out, uint64_t *in, uint64_t n, NTT_proc proc);
    void mod_eltwise_reduce_signed(uint64_t *out, int64_t *in, uint64_t n, NTT_proc proc);
    void mod_reduce_array_mp(uint64_t *out, uint64_t *in_high, uint64_t *in_low, uint64_t n,
                             NTT_proc proc);

    NTT_proc ntt_new_proc(uint64_t n, uint64_t q);
    void ntt_free_proc(NTT_proc proc);

    uint64_t power_mod(uint64_t base, uint64_t exp, uint64_t mod);
    uint64_t inverse_mod(uint64_t a, uint64_t m);
    uint64_t inverse_mod_eea(uint64_t a, uint64_t p);
    bool is_prime(uint64_t n);
    uint64_t generate_Nth_root_of_unity(uint64_t q, uint64_t n);
    uint64_t next_special_prime(uint64_t x, uint64_t n, bool primitive);

    // field arithmetic
    NTT_proc field_new_proc(uint64_t q);
    void field_ext_add(uint64_t *c, const uint64_t *a, const uint64_t *b, uint64_t d, uint64_t q);
    void field_ext_sub(uint64_t *c, const uint64_t *a, const uint64_t *b, uint64_t d, uint64_t q);
    void field_ext_neg(uint64_t *c, const uint64_t *a, uint64_t d, uint64_t q);
    void field_ext_mul(uint64_t *c, const uint64_t *a, const uint64_t *b, uint64_t d, uint64_t w,
                       NTT_proc proc);
    void field_ext_pow(uint64_t *res, const uint64_t *base, uint64_t exp_lo, uint64_t exp_hi,
                       uint64_t d, uint64_t w, NTT_proc proc);
    int field_ext_inv(uint64_t *ainv, const uint64_t *a, uint64_t d, uint64_t w, NTT_proc proc);
    void field_sample_random_element(uint64_t *a, const uint8_t *seed, uint64_t seed_len,
                                     uint64_t d, uint64_t mod);
    void field_hash_element(uint8_t *out, const uint64_t *a, uint64_t d);
    int field_ext_is_equal(const uint64_t *a, const uint64_t *b, uint64_t d);
    void field_base_conversion(uint64_t *out, const uint64_t *in, uint64_t source_component,
                               uint64_t target_component, uint64_t d, uint64_t poly_size,
                               const uint64_t *w_i, NTT_proc proc);

    // complex polynomial
    double **load_rous_CT(double *rous_real, double *rous_imag, uint64_t size);
    void CT_NR(double *x, double **ws, uint64_t n);
    double **load_rous_GS(double *rous_real, double *rous_imag, uint64_t size);
    void GS_RN(double *x, double **ws, uint64_t n);
    void bit_reverse_array(double *v, uint64_t N, uint32_t prec);
    void complex_poly_scale_double(double *v, double scale, uint64_t N);
    void complex_poly_round_to_RNS(RNS_Polynomial out, double *in, uint64_t N);
    void complex_polys_ifft_scale_round_to_RNS_batch(void **rows_in, void **outs_rns,
                                                     uint64_t count, uint64_t n_complex,
                                                     uint32_t log_prec, double **gs_ws,
                                                     double temp_delta);

    // polynomial
    IntPolynomial polynomial_new_int_polynomial(uint64_t N);
    IntPolynomial *polynomial_new_int_polynomial_array(uint64_t size, uint64_t N);
    RNS_Polynomial polynomial_new_RNS_polynomial(uint64_t N, uint64_t rns_mask, incNTT ntt);
    void polynomial_RNS_zero(RNS_Polynomial p);
    RNS_Polynomial *polynomial_new_array_of_RNS_polynomials(uint64_t N, uint64_t rns_mask,
                                                            uint64_t size, incNTT ntt);
    void polynomial_to_RNS(RNS_Polynomial out, IntPolynomial in);
    void polynomial_gen_random_RNSc_polynomial(RNSc_Polynomial out);
    void polynomial_mul_RNS_polynomial(RNS_Polynomial out, RNS_Polynomial in1, RNS_Polynomial in2);
    void polynomial_sub_RNS_polynomial(RNS_Polynomial out, RNS_Polynomial in1, RNS_Polynomial in2);
    void polynomial_sub_RNSc_polynomial(RNSc_Polynomial out, RNSc_Polynomial in1,
                                        RNSc_Polynomial in2);
    void polynomial_RNSc_to_RNS(RNS_Polynomial out, RNSc_Polynomial in);
    void polynomial_RNS_to_RNSc(RNSc_Polynomial out, RNS_Polynomial in);
    void polynomial_RNSc_add_noise(RNSc_Polynomial out, RNSc_Polynomial in, double sigma);
    void polynomial_floor_division_RNSc(RNSc_Polynomial out);
    void polynomial_round_division_RNSc(RNSc_Polynomial out);
    void polynomial_floor_division_RNSc_wo_free(RNSc_Polynomial out, uint64_t divide_mask);
    typedef struct _RNS_BaseConversionParams
    {
        uint64_t in_mask;
        uint64_t out_mask;
        uint32_t w;
        uint32_t v;
        uint32_t *D;
        uint32_t *P;
        uint64_t *Dhat;
        uint64_t **D_mod_p;
    } *RNS_BaseConversionParams;

    RNS_BaseConversionParams init_base_conversion_params(incNTT ntt, uint64_t in_mask,
                                                         uint64_t out_mask);
    void free_base_conversion_params(RNS_BaseConversionParams params);

    void rns_compute_scaling_factors(uint64_t *delta_out, incNTT ntt, uint64_t in_mask,
                                     uint64_t out_mask);
    void polynomial_RNSc_scaled_lift(RNSc_Polynomial out, RNSc_Polynomial in, uint64_t *delta);
    void polynomial_base_conversion_RNSc(RNSc_Polynomial out, RNSc_Polynomial in,
                                         RNS_BaseConversionParams params);
    void polynomial_RNSc_permute(RNSc_Polynomial out, RNSc_Polynomial in, uint64_t gen);
    void free_RNS_polynomial(void *p);
    void polynomial_RNSc_negate(RNSc_Polynomial out, RNSc_Polynomial in);
    void polynomial_add_RNSc_polynomial(RNSc_Polynomial out, RNSc_Polynomial in1,
                                        RNSc_Polynomial in2);
    void polynomial_add_RNS_polynomial(RNS_Polynomial out, RNS_Polynomial in1, RNS_Polynomial in2);
    void polynomial_int_permute_mod_Q(IntPolynomial out, IntPolynomial in, uint64_t gen);
    void polynomial_copy_RNS_polynomial(RNS_Polynomial out, RNS_Polynomial in);
    void polynomial_RNSc_mul_by_xai(RNSc_Polynomial out, RNSc_Polynomial in, uint64_t a);
    void polynomial_RNSc_mul_by_xai_minus1(RNSc_Polynomial out, RNSc_Polynomial in, uint64_t a);
    void polynomial_int_decompose_i(IntPolynomial out, IntPolynomial in, uint64_t Bg_bit,
                                    uint64_t l, uint64_t q, uint64_t bit_size, uint64_t i);
    void polynomial_mul_addto_RNS_polynomial(RNS_Polynomial out, RNS_Polynomial in1,
                                             RNS_Polynomial in2);

    void free_polynomial(void *p);
    void array_to_RNS(RNS_Polynomial out, uint64_t **in);
    void polynomial_RNS_get_hash(uint64_t *out, RNS_Polynomial p);
    uint64_t *polynomial_RNS_get_hash_p(RNS_Polynomial p);
    RNS_Polynomial *polynomial_new_RNS_polynomial_array(uint64_t size, uint64_t N,
                                                        uint64_t rns_mask, incNTT ntt);
    void free_RNS_polynomial_array(uint64_t size, RNS_Polynomial *p);
    void polynomial_scale_RNSc_polynomial(RNSc_Polynomial out, RNSc_Polynomial in1, uint64_t scale);
    void polynomial_scale_RNS_polynomial_RNS(RNS_Polynomial out, RNS_Polynomial in1,
                                             uint64_t *scale);
    void polynomial_round_division_RNSc_wo_free(RNSc_Polynomial out, uint64_t divide_mask);
    bool polynomial_eq(RNS_Polynomial a, RNS_Polynomial b);
    void polynomial_multo_RNS_polynomial(RNS_Polynomial out, RNS_Polynomial in);
    int polynomial_RNS_inverse(RNS_Polynomial out, RNS_Polynomial in);
    void polynomial_RNSc_mod_reduce_lifted(RNSc_Polynomial out, RNSc_Polynomial in, uint64_t idx);
    void polynomial_RNSc_mod_reduce(RNSc_Polynomial out, RNSc_Polynomial in);
    void polynomial_RNS_broadcast_slot(RNS_Polynomial out, RNS_Polynomial in, uint64_t slot_idx);
    void polynomial_RNS_rotate_slot(RNS_Polynomial out, RNS_Polynomial in, uint64_t rot);
    void polynomial_RNS_copy_slot(RNS_Polynomial out, uint64_t dst, RNS_Polynomial in,
                                  uint64_t src);
    void polynomial_gen_gaussian_RNSc_polynomial(RNSc_Polynomial out, double sigma);
    void int_array_to_RNS(RNS_Polynomial out, uint64_t *in);
    void polynomial_RNSc_add_integer(RNSc_Polynomial out, RNSc_Polynomial in1, uint64_t in2);
    void polynomial_scale_RNS_polynomial(RNS_Polynomial out, RNS_Polynomial in1, uint64_t scale);
    void polynomial_RNS_negate(RNS_Polynomial out, RNS_Polynomial in);
    void polynomial_RNS_add_integer(RNS_Polynomial out, RNS_Polynomial in1, uint64_t in2);
    void polynomial_scale_addto_RNSc_polynomial(RNSc_Polynomial out, RNSc_Polynomial in1,
                                                uint64_t scale);
    void polynomial_scale_addto_RNS_polynomial(RNS_Polynomial out, RNS_Polynomial in1,
                                               uint64_t scale);
    void polynomial_mul_subto_RNS_polynomial(RNS_Polynomial out, RNS_Polynomial in1,
                                             RNS_Polynomial in2);
    void polynomial_copy_RNSc_polynomial(RNSc_Polynomial out, RNSc_Polynomial in);

    // vector
    ZqVector alloc_ZqVector(uint64_t n, NTT_proc *ntt, uint64_t l);
    void ZqVector_add(ZqVector out, ZqVector in1, ZqVector in2);
    void ZqVector_sub(ZqVector out, ZqVector in1, ZqVector in2);
    void ZqVector_scale(ZqVector out, ZqVector in1, uint64_t scale);

    // Multi-precision Polynomial arithmetic
    typedef struct _MPPolynomial
    {
        uint64_t **coeffs;
        uint64_t N, d;
    } *MPPolynomial;

    typedef struct _MPScalar
    {
        mp_vector_t *digits;
        uint64_t d;
    } *MPScalar;

    MPPolynomial new_mp_polynomial(uint64_t N, uint64_t d);
    void free_mp_polynomial(MPPolynomial p);
    void mp_polynomial_mul_by_xai(MPPolynomial out, MPPolynomial in, uint64_t a);
    void mp_polynomial_negate(MPPolynomial out, MPPolynomial in);
    void mp_polynomial_add(MPPolynomial out, MPPolynomial a, MPPolynomial b);
    void mp_polynomial_drop_digits(MPPolynomial p, uint64_t num_digits);
    void mp_polynomial_rnd(MPPolynomial poly);
    void mp_polynomial_scale(MPPolynomial out, MPPolynomial in, uint64_t scale);
    void mp_polynomial_sp_scale_mp(MPPolynomial out, MPPolynomial in, mp_vector_t *scale);
    void mp_polynomial_scale_addto(MPPolynomial out, MPPolynomial in, uint64_t scale);
    void mp_polynomial_zero(MPPolynomial poly);
    void mp_polynomial_propagate_carry(MPPolynomial p);
    void mp_polynomial_mul_addto_sparse_MPPolynomial(MPPolynomial out, MPPolynomial a, uint64_t *b,
                                                     uint64_t size);
    uint64_t array32_bit_slice52(uint64_t *array, uint64_t start);
    void setup_mod_switch_delta(uint64_t d, uint64_t p);
    void mp_scale(MPScalar out, MPScalar in, mp_vector_t *m);
    void mp_sub(MPScalar out, MPScalar a, MPScalar b);
    void mp_polynomial_mod_reduce(MPPolynomial out, MPScalar q, mp_vector_t *m, uint64_t k);
    void mp_polynomial_from_RNS(MPPolynomial out, RNS_Polynomial in, MPScalar *PW, MPScalar q,
                                mp_vector_t *m, uint64_t k);
    void mp_polynomial_to_RNSc(RNSc_Polynomial out, MPPolynomial in);
    MPScalar mp_load(uint64_t *in, uint64_t d);
    mp_vector_t *load_m512(uint64_t in);
    int get_mp_vector_size(void);
    void mp_polynomial_int_sp_scale_mp(MPPolynomial out, uint64_t *in, MPScalar scale);
    void mp_polynomial_int_sp_scale_addto_mp(MPPolynomial out, uint64_t *in, MPScalar scale);

#ifdef __cplusplus
}
#endif

#endif // __NTT_H__
