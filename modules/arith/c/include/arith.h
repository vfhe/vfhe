// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#ifndef __NTT_H__
#define __NTT_H__

#include <engine.h>

#if VFHE_HAVE_AVX512IFMA
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

#if VFHE_HAVE_AVX512IFMA
typedef __m512i mp_vector_t;
#else
typedef uint64_t mp_vector_t;
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    /* A modulus q together with everything needed to reduce modulo it: the
       Barrett constants (k, m, m52), the IFMA split of m the 50-bit kernels
       multiply with, and the 2^52 / 2^104 residues the multiprecision path
       folds with. This is all the modular-arithmetic kernels need -- `modq`,
       `mul_modq` and every `mod_eltwise_*` take one of these and nothing else.

       The constants depend on the engine's `modq` (mod.c vs mod_portable.c),
       so `mod_new` is defined next to it in each and is the single place they
       are derived. */
    typedef struct _Modulus
    {
        uint64_t q;
        uint64_t k;
        uint64_t m;
        uint64_t m52;
        uint64_t ifma_barr_lo;
        uint64_t ifma_prod_right_shift;
        uint64_t mp_w1;
        uint64_t mp_w2;
    } *Modulus;

    Modulus mod_new(uint64_t q);
    void mod_free(Modulus mod);

    /* A negacyclic NTT of one length over one modulus: the twiddle tables
       (whose layout is engine-specific -- hence `void **`) and the roots they
       were built from.

       `mod` is **borrowed**: whoever created the modulus owns it, and
       `ntt_free_plan` leaves it alone. That lets several plans over the same
       prime at different lengths -- which is exactly what polycom's per-level
       codes are -- share one set of Barrett constants. */
    typedef struct _NTT_Plan
    {
        Modulus mod;
        uint64_t n;
        uint64_t root_of_unity;
        uint64_t inv_root_of_unity;
        void **ws_fwd;
        void **w_precon_fwd;
        void **ws_inv;
        void **w_precon_inv;
    } *NTT_Plan;

    /* The incomplete NTT of R_q = Z_q[X]/(X^N+1) split into `split_degree`
       blocks, over `l` RNS primes. Owns both arrays: `mods[i]` is the modulus
       every kernel over prime i uses, and `plans[i]` is its length-N/split_degree
       transform, borrowing `mods[i]`. Grows in place (rns_base_extend_with_primes)
       -- see Ring.rns_rows on the Python side before sizing anything by `l`. */
    typedef struct _RNS_Base
    {
        NTT_Plan *plans;
        Modulus *mods;
        uint64_t split_degree;
        uint64_t **w;
        uint64_t N, l;
    } *RNS_Base;

    // Builds the base for `l` primes over transforms of length N / split_degree,
    // with its moduli, plans, and root-of-unity rows.
    RNS_Base new_rns_base(uint64_t *primes, uint64_t split_degree, uint64_t N, uint64_t l);
    uint64_t **rns_base_get_rou_matrix(RNS_Base base);
    void rns_base_extend_with_primes(RNS_Base base, uint64_t *new_primes, uint64_t count);
    // (Z_q[i](Q/q[i]))**-1 for i in [0,l): RNS, so it needs inverse_mod.
    void compute_RNS_Qhat_array(uint64_t *out, uint64_t *p, uint64_t l);

    // Releases a base and everything it owns: its moduli, its plans, and the
    // root-of-unity rows. Any NTT_Plan built elsewhere against one of these
    // moduli (rs_new_plans borrows them) must already be gone, and so must any
    // RNS_Polynomial allocated on this base.
    void rns_base_free(RNS_Base base);

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
        RNS_Base base;
        uint64_t rns_mask;
        uint64_t allocated_l;
    } *RNS_Polynomial;

    /* RNS polynomial in coefficient representation*/
    typedef struct _RNSc_Polynomial
    {
        uint64_t **coeffs;
        RNS_Base base;
        uint64_t rns_mask;
        uint64_t allocated_l;
    } *RNSc_Polynomial;

    typedef struct _ZqVector
    {
        uint64_t **elements;
        uint64_t n, l;
        Modulus *mods;
    } *ZqVector;

    typedef struct _IntPolynomial
    {
        uint64_t *coeffs;
        uint64_t N;
    } *IntPolynomial;

#if VFHE_HAVE_AVX512IFMA
    void ntt_precompute_fwd(uint64_t n, Modulus mod, uint64_t root_of_unity, __m512i ***out_ws,
                            __m512i ***out_w_precon);
    void ntt_precompute_inv(uint64_t n, Modulus mod, uint64_t inv_root_of_unity, __m512i ***out_ws,
                            __m512i ***out_w_precon);
    void ntt_free_precompute(__m512i **ws, __m512i **w_precon, uint64_t n);
#else
void ntt_precompute_fwd(uint64_t n, Modulus mod, uint64_t root_of_unity, uint64_t ***out_ws,
                        uint64_t ***out_w_precon);
void ntt_precompute_inv(uint64_t n, Modulus mod, uint64_t inv_root_of_unity, uint64_t ***out_ws,
                        uint64_t ***out_w_precon);
void ntt_free_precompute(uint64_t **ws, uint64_t **w_precon, uint64_t n);
#endif

    void ntt_forward(uint64_t *out, uint64_t *in, NTT_Plan plan);
    void ntt_reverse(uint64_t *out, uint64_t *in, NTT_Plan plan);

    uint64_t add_modq(uint64_t a, uint64_t b, uint64_t q);
    uint64_t sub_modq(uint64_t a, uint64_t b, uint64_t q);
    uint64_t negate_modq(uint64_t a, uint64_t q);
    uint64_t mul_modq(uint64_t a, uint64_t b, Modulus mod);
    // One 64-bit word reduced mod q, and the 128-bit value hi * 2^64 + lo
    // reduced mod q. Both are Barrett reductions -- no division.
    uint64_t modq(uint64_t x, Modulus mod);
    uint64_t modq_wide(uint64_t hi, uint64_t lo, Modulus mod);

    void mod_eltwise_mul(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod);
    void mod_eltwise_mul_addto(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n,
                               Modulus mod);
    void mod_eltwise_mul_subto(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n,
                               Modulus mod);
    void mod_eltwise_scale(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, Modulus mod);
    void mod_eltwise_fma(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, Modulus mod);
    void mod_eltwise_add_scalar(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n,
                                Modulus mod);
    void mod_eltwise_sub_scalar(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n,
                                Modulus mod);
    void mod_eltwise_negate(uint64_t *out, uint64_t *in, uint64_t n, Modulus mod);
    void mod_eltwise_add(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod);
    void mod_eltwise_sub(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod);
    void mod_eltwise_reduce(uint64_t *out, uint64_t *in, uint64_t n, Modulus mod);
    void mod_eltwise_reduce_signed(uint64_t *out, int64_t *in, uint64_t n, Modulus mod);
    void mod_reduce_array_mp(uint64_t *out, uint64_t *in_high, uint64_t *in_low, uint64_t n,
                             Modulus mod);

    // Returns NULL, explaining on stderr, when 2n does not divide q - 1, i.e.
    // when the modulus has no primitive 2n-th root of unity.
    NTT_Plan ntt_new_plan(uint64_t n, Modulus mod);
    void ntt_free_plan(NTT_Plan plan);

    uint64_t power_mod(uint64_t base, uint64_t exp, uint64_t mod);
    uint64_t inverse_mod(uint64_t a, uint64_t m);
    uint64_t inverse_mod_eea(uint64_t a, uint64_t p);
    bool is_prime(uint64_t n);
    uint64_t generate_Nth_root_of_unity(uint64_t q, uint64_t n);
    uint64_t next_special_prime(uint64_t x, uint64_t n, bool primitive);

    // field arithmetic
    void field_ext_add(uint64_t *c, const uint64_t *a, const uint64_t *b, uint64_t d, uint64_t q);
    void field_ext_sub(uint64_t *c, const uint64_t *a, const uint64_t *b, uint64_t d, uint64_t q);
    void field_ext_neg(uint64_t *c, const uint64_t *a, uint64_t d, uint64_t q);
    void field_ext_mul(uint64_t *c, const uint64_t *a, const uint64_t *b, uint64_t d, uint64_t w,
                       Modulus mod);
    void field_ext_pow(uint64_t *res, const uint64_t *base, uint64_t exp_lo, uint64_t exp_hi,
                       uint64_t d, uint64_t w, Modulus mod);
    int field_ext_inv(uint64_t *ainv, const uint64_t *a, uint64_t d, uint64_t w, Modulus mod);
    void field_sample_random_element(uint64_t *a, const uint8_t *seed, uint64_t seed_len,
                                     uint64_t d, uint64_t mod);
    void field_hash_element(uint8_t *out, const uint64_t *a, uint64_t d);
    int field_ext_is_equal(const uint64_t *a, const uint64_t *b, uint64_t d);
    void field_base_conversion(uint64_t *out, const uint64_t *in, uint64_t source_component,
                               uint64_t target_component, uint64_t d, uint64_t poly_size,
                               const uint64_t *w_i, Modulus mod);

    // vectors of field elements
    //
    // n elements of F_p[x]/(x^d - w) held as d coefficient planes: element i is
    // (coeffs[0][i], ..., coeffs[d-1][i]). Splitting the coefficients apart is
    // what lets every arithmetic kernel below be a handful of mod_eltwise_*
    // calls over contiguous length-n runs, one per plane, rather than n calls
    // over a d-word element.
    //
    // The layout contract, which the caller allocating a FieldVector must meet:
    //
    // - Each plane is field_vec_padded_length(n) words and is 64-byte aligned,
    //   because the tuned eltwise kernels process whole SIMD vectors with no
    //   tail loop and load a plane as an __m512i.
    // - Words n..allocated_n-1 are padding. They are read and written by the
    //   arithmetic kernels, so they must hold values below q -- allocate zeroed
    //   and they stay reduced -- but they carry no meaning: everything that
    //   reads a value (sum, equality, hashing, element access) stops at n.
    // - Every coefficient of a live element is in [0, q).
    //
    // Arithmetic outputs may alias their inputs. The movement operations
    // (copy, split_even_odd, get/set) may not.
    typedef struct _FieldVector
    {
        uint64_t **coeffs; // d planes of allocated_n words
        uint64_t n;        // elements the caller gave meaning to
        uint64_t allocated_n;
        uint64_t d;
        uint64_t w;  // the extension is F_p[x]/(x^d - w)
        Modulus mod; // borrowed; must outlive the vector
    } *FieldVector;

    // Words a plane must hold for a vector of n elements: n rounded up to the
    // eltwise kernels' vector length. The single source of that number.
    uint64_t field_vec_padded_length(uint64_t n);

    void field_vec_add(FieldVector out, const FieldVector a, const FieldVector b);
    void field_vec_sub(FieldVector out, const FieldVector a, const FieldVector b);
    void field_vec_neg(FieldVector out, const FieldVector a);
    // Broadcast against one element `s`, which is d coefficients.
    void field_vec_add_scalar(FieldVector out, const FieldVector a, const uint64_t *s);
    void field_vec_sub_scalar(FieldVector out, const FieldVector a, const uint64_t *s);
    void field_vec_scalar_sub(FieldVector out, const uint64_t *s, const FieldVector a);
    // Elementwise (Hadamard) product, and the product with one element.
    void field_vec_mul(FieldVector out, const FieldVector a, const FieldVector b);
    void field_vec_scale(FieldVector out, const FieldVector a, const uint64_t *s);
    // Sum of the first n elements, into d coefficients.
    void field_vec_sum(uint64_t *out, const FieldVector a);
    // Elementwise inverse by Montgomery's trick: one inversion and three
    // multiplications per element. Returns 0 without writing `out` if any of the
    // first n elements is zero, 1 otherwise.
    int field_vec_inv(FieldVector out, const FieldVector a);
    // Transpose between the planes and one d-word element.
    void field_vec_get_element(uint64_t *out, const FieldVector a, uint64_t index);
    void field_vec_set_element(FieldVector out, uint64_t index, const uint64_t *value);
    // Bulk transpose of `count` consecutive elements laid out d coefficients at
    // a time, so building a vector costs one call rather than `count`.
    void field_vec_set_range(FieldVector out, uint64_t start, const uint64_t *values,
                             uint64_t count);
    void field_vec_get_range(uint64_t *out, const FieldVector a, uint64_t start, uint64_t count);
    void field_vec_copy(FieldVector out, const FieldVector a);
    // Deinterleave: even gets elements 0, 2, 4, ... and odd gets 1, 3, 5, ....
    // `a->n` must be even and each output must hold a->n / 2 elements.
    void field_vec_split_even_odd(FieldVector even, FieldVector odd, const FieldVector a);
    // Interleave, the inverse of split_even_odd: out gets even[0], odd[0],
    // even[1], odd[1], .... Both inputs hold out->n / 2 elements.
    void field_vec_interleave(FieldVector out, const FieldVector even, const FieldVector odd);
    // The `count` vectors one after another; out->n is the sum of their lengths.
    void field_vec_concat(FieldVector out, const FieldVector *parts, uint64_t count);
    // out[i] = a[indices[i]] for i < count. Every index must be below a->n.
    void field_vec_gather(FieldVector out, const FieldVector a, const uint64_t *indices,
                          uint64_t count);
    // out[i] = a[2i] + r * (a[2i+1] - a[2i]) for i < a->n / 2, in one pass: the
    // interpolation between adjacent pairs that binds one variable of a
    // multilinear table. `r` is one element. `a->n` must be even, out must hold
    // a->n / 2 elements with canonical padding, and may not alias a.
    void field_vec_fold(FieldVector out, const FieldVector a, const uint64_t *r);
    int field_vec_is_equal(const FieldVector a, const FieldVector b);
    // Uniform elements from a seed. One draw stream covers the whole vector, so
    // no two coefficients repeat by construction.
    void field_vec_sample_random(FieldVector out, const uint8_t *seed, uint64_t seed_len);
    // BLAKE3 over the elements in index order, 32 bytes.
    void field_vec_hash(uint8_t *out, const FieldVector a);
    // One digest per window of `group` elements starting every `stride` indices:
    // window k covers elements k * stride .. k * stride + group - 1, and the
    // count is the number of whole windows that fit. `out` takes 32 bytes each.
    uint64_t field_vec_hash_count(const FieldVector a, uint64_t group, uint64_t stride);
    void field_vec_hash_elements(uint8_t *out, const FieldVector a, uint64_t group,
                                 uint64_t stride);

    // pseudo-Mersenne prime field
    //
    // F_p for p = 2^n - c with small c (the Crandall/pseudo-Mersenne family). An
    // element is PMF_LANES uint64_t words -- one AVX-512 zmm register -- holding
    // L = ceil(n / 52) limbs of 52 bits in lanes 0..L-1, little-endian. Lanes
    // L..PMF_LANES-1 are ALWAYS zero, and buffers are 64-byte aligned, so the
    // tuned engine can load an element as one __m512i.
    //
    // Every public input and output is CANONICAL: value in [0, p), every limb
    // below 2^52, padding lanes zero. Lazy reduction is deliberately not offered:
    // IFMA truncates both multiplicands to 52 bits, so carrying a value back
    // below 2^52 after a multiply is a hardware requirement, not a policy choice.
    //
    // Reduction folds on 2^(52L) == e (mod p), where e = c << (52L - n). That e
    // must fit a single limb is what bounds c, and hence which n are usable --
    // see pmf_new_params.
    //
    // Self-contained: this field carries its own parameter block and does not go
    // through Modulus / the NTT plan machinery.
    //
    // Nothing here is constant-time: exponents and seeds are public values.
#define PMF_LIMB_BITS 52
#define PMF_LIMB_MASK ((1ULL << PMF_LIMB_BITS) - 1)
#define PMF_MAX_LIMBS 6 // L in {5, 6}; raise only alongside the kernels
#define PMF_LANES 8     // one zmm: L limbs plus zero padding

    // Immutable after construction, and so shareable across threads: no operation
    // writes to it, and none allocates.
    typedef struct _PMFParams
    {
        uint64_t n;        // p = 2^n - c
        uint64_t c;        // odd, 0 < c < 2^(n - 52 * (L - 1))
        uint64_t limbs;    // L = ceil(n / 52)
        uint64_t shift;    // s = 52 * L - n, in [0, 51]
        uint64_t fold;     // e = c << s; 2^(52L) == e (mod p)
        uint64_t top_bits; // 52 - s: live bits of limb L-1
        uint64_t top_mask; // (1 << top_bits) - 1
        uint64_t nbytes;   // (n + 7) / 8: length of the canonical encoding

        uint64_t p[PMF_MAX_LIMBS]; // the modulus, as limbs
    } *PMFParams;

    // Returns NULL, explaining on stderr, unless L is 5 or 6, c is odd and
    // nonzero, and e = c << s fits a 52-bit limb. Does NOT require p to be prime:
    // the arithmetic is correct for any modulus 2^n - c, and primality is the
    // caller's business.
    PMFParams pmf_new_params(uint64_t n, uint64_t c);
    void pmf_free_params(PMFParams params);
    uint64_t pmf_limbs(PMFParams params);
    uint64_t pmf_byte_length(PMFParams params);

    // out may alias any input, including out == a == b.
    void pmf_add(uint64_t *out, const uint64_t *a, const uint64_t *b, PMFParams params);
    void pmf_sub(uint64_t *out, const uint64_t *a, const uint64_t *b, PMFParams params);
    void pmf_neg(uint64_t *out, const uint64_t *a, PMFParams params);
    void pmf_mul(uint64_t *out, const uint64_t *a, const uint64_t *b, PMFParams params);
    int pmf_is_equal(const uint64_t *a, const uint64_t *b, PMFParams params);
    // in: L limbs each below 2^52, not necessarily below p. out: canonical.
    void pmf_canonicalize(uint64_t *out, const uint64_t *in, PMFParams params);

    // The canonical encoding: the value in fixed-width big-endian, exactly
    // pmf_byte_length(params) bytes. It is the form the field is compared and
    // hashed in, because it depends on neither the limb representation nor the
    // padding lanes. pmf_from_bytes accepts only an encoding below p -- a value
    // at or above it would have two encodings -- and returns 0 otherwise,
    // leaving out untouched.
    void pmf_to_bytes(uint8_t *out, const uint64_t *a, PMFParams params);
    int pmf_from_bytes(uint64_t *out, const uint8_t *in, PMFParams params);
    // BLAKE3 over the canonical encoding, 32 bytes.
    void pmf_hash(uint8_t *out, const uint64_t *a, PMFParams params);
    // Uniform in [0, p) from a seed, by rejection. Not constant-time: seeds are
    // public values.
    void pmf_sample_random(uint64_t *out, const uint8_t *seed, uint64_t seed_len, PMFParams params);

    // vectors of pseudo-Mersenne elements
    //
    // n elements held as L limb planes: limb j of element i is limbs[j][i]. A
    // single element fills one AVX-512 register with its own limbs, so every
    // carry runs across lanes; a vector turns that inside out, putting one limb
    // of PMF_LANES DIFFERENT elements in a register. Carries then travel between
    // planes at a fixed lane and nothing crosses lanes, which is what makes the
    // arithmetic worth more than a loop over the element kernels.
    //
    // The layout contract, which the caller allocating a PMFVector must meet:
    //
    // - Each plane is pmf_vec_padded_length(n) words and is 64-byte aligned, so
    //   a group of PMF_LANES elements' limbs loads as one aligned __m512i.
    // - Words n..allocated_n-1 are padding. The arithmetic reads and writes
    //   them, so they must hold canonical elements -- allocate zeroed and they
    //   stay canonical -- but they carry no meaning: everything that reads a
    //   value (sum, equality, hashing, sampling, element access) stops at n.
    // - Every live element is canonical, exactly as for a single element.
    //
    // Arithmetic outputs may alias their inputs. The movement operations (copy,
    // split_even_odd, get/set) may not.
    typedef struct _PMFVector
    {
        uint64_t **limbs; // L planes of allocated_n words
        uint64_t n;       // elements the caller gave meaning to
        uint64_t allocated_n;
        PMFParams params; // borrowed; must outlive the vector
    } *PMFVector;

    // Words a plane must hold for a vector of n elements: n rounded up to the
    // group width. The single source of that number.
    uint64_t pmf_vec_padded_length(uint64_t n);

    void pmf_vec_add(PMFVector out, const PMFVector a, const PMFVector b);
    void pmf_vec_sub(PMFVector out, const PMFVector a, const PMFVector b);
    void pmf_vec_neg(PMFVector out, const PMFVector a);
    void pmf_vec_mul(PMFVector out, const PMFVector a, const PMFVector b);
    // Broadcast against one element `s`, which is PMF_LANES words.
    void pmf_vec_add_scalar(PMFVector out, const PMFVector a, const uint64_t *s);
    void pmf_vec_sub_scalar(PMFVector out, const PMFVector a, const uint64_t *s);
    void pmf_vec_scalar_sub(PMFVector out, const uint64_t *s, const PMFVector a);
    void pmf_vec_scale(PMFVector out, const PMFVector a, const uint64_t *s);
    // Sum of the first n elements, into one PMF_LANES-word element.
    void pmf_vec_sum(uint64_t *out, const PMFVector a);
    // Transpose between the planes and one PMF_LANES-word element buffer; the
    // range forms move `count` consecutive elements laid out PMF_LANES words
    // apart, so building a vector costs one call rather than `count`.
    void pmf_vec_get_element(uint64_t *out, const PMFVector a, uint64_t index);
    void pmf_vec_set_element(PMFVector out, uint64_t index, const uint64_t *value);
    void pmf_vec_set_range(PMFVector out, uint64_t start, const uint64_t *values, uint64_t count);
    void pmf_vec_get_range(uint64_t *out, const PMFVector a, uint64_t start, uint64_t count);
    void pmf_vec_copy(PMFVector out, const PMFVector a);
    // Deinterleave: even gets elements 0, 2, 4, ... and odd gets 1, 3, 5, ....
    // `a->n` must be even and each output must hold a->n / 2 elements.
    void pmf_vec_split_even_odd(PMFVector even, PMFVector odd, const PMFVector a);
    // Interleave, the inverse of split_even_odd: out gets even[0], odd[0],
    // even[1], odd[1], .... Both inputs hold out->n / 2 elements.
    void pmf_vec_interleave(PMFVector out, const PMFVector even, const PMFVector odd);
    // The `count` vectors one after another; out->n is the sum of their lengths.
    void pmf_vec_concat(PMFVector out, const PMFVector *parts, uint64_t count);
    // out[i] = a[indices[i]] for i < count. Every index must be below a->n.
    void pmf_vec_gather(PMFVector out, const PMFVector a, const uint64_t *indices, uint64_t count);
    // out[i] = a[2i] + r * (a[2i+1] - a[2i]) for i < a->n / 2, in one pass: the
    // interpolation between adjacent pairs that binds one variable of a
    // multilinear table. `r` is one element. `a->n` must be even, out must hold
    // a->n / 2 elements with canonical padding, and may not alias a.
    void pmf_vec_fold(PMFVector out, const PMFVector a, const uint64_t *r);
    int pmf_vec_is_equal(const PMFVector a, const PMFVector b);
    // Uniform elements from a seed. One draw stream covers the whole vector.
    void pmf_vec_sample_random(PMFVector out, const uint8_t *seed, uint64_t seed_len);
    // BLAKE3 over the canonical encodings, in index order, 32 bytes.
    void pmf_vec_hash(uint8_t *out, const PMFVector a);
    // One digest per window of `group` elements starting every `stride` indices:
    // window k covers elements k * stride .. k * stride + group - 1, and the
    // count is the number of whole windows that fit. `out` takes 32 bytes each.
    uint64_t pmf_vec_hash_count(const PMFVector a, uint64_t group, uint64_t stride);
    void pmf_vec_hash_elements(uint8_t *out, const PMFVector a, uint64_t group, uint64_t stride);

    // the negacyclic NTT over a pseudo-Mersenne field, on PMFVector planes
    //
    // The transform of F_p[X]/(X^n + 1) for n a power of two, in the basis
    // arith's RNS transforms use: with psi the primitive 2n-th root of unity the
    // plan was built from, the forward transform of a vector holding the
    // coefficients of P in natural order leaves position j holding
    // P(psi^(2*brv(j)+1)), brv reversing the log2(n) index bits (Cooley-Tukey,
    // natural in, bit-reversed out). So positions 2i and 2i+1 hold P(x) and
    // P(-x) for x = psi^(2*brv(i)+1). The inverse takes that order back to
    // coefficients (Gentleman-Sande) and includes the 1/n scaling.
    //
    // The plan uses exactly the psi it is given and records it; which psi to
    // use across lengths -- so that psi_{n/2} = psi_n^2 -- is the caller's
    // convention (PseudoMersenneField.root_of_unity derives every root from the
    // least quadratic non-residue, as the RNS plans do).
    //
    // `params` is borrowed: the plan must not outlive it. Immutable once built,
    // so shareable across threads.
    typedef struct _PMFNTTPlan
    {
        PMFParams params;
        uint64_t n;
        uint64_t logn;
        uint64_t root_of_unity[PMF_LANES];     // psi, one element
        uint64_t inv_root_of_unity[PMF_LANES]; // psi^-1
        uint64_t inv_n[PMF_LANES];             // n^-1, applied by the inverse
        uint64_t **ws_fwd;                     // L planes of n words: limb k of
        uint64_t **ws_inv;                     // psi^i (psi^-i) at ws[k][brv(i)]
    } *PMFNTTPlan;

    // `root_of_unity` is one element (PMF_LANES words) that must be a primitive
    // 2n-th root of unity, i.e. satisfy psi^n == -1. Returns NULL, explaining on
    // stderr, if it is not or if n is not a power of two. Costs n scalar
    // multiplications and 2 * n * L words of tables.
    PMFNTTPlan pmf_ntt_new_plan(uint64_t n, const uint64_t *root_of_unity, PMFParams params);
    void pmf_ntt_free_plan(PMFNTTPlan plan);
    // In place on the vector's planes. `a->n` must equal `plan->n` and the
    // vector must be over the plan's `params`. Every element, padding included,
    // is canonical afterwards.
    void pmf_vec_ntt_forward(PMFVector a, PMFNTTPlan plan);
    void pmf_vec_ntt_inverse(PMFVector a, PMFNTTPlan plan);

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
    RNS_Polynomial polynomial_new_RNS_polynomial(uint64_t N, uint64_t rns_mask, RNS_Base base);
    void polynomial_RNS_zero(RNS_Polynomial p);
    RNS_Polynomial *polynomial_new_array_of_RNS_polynomials(uint64_t N, uint64_t rns_mask,
                                                            uint64_t size, RNS_Base base);
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

    RNS_BaseConversionParams init_base_conversion_params(RNS_Base base, uint64_t in_mask,
                                                         uint64_t out_mask);
    void free_base_conversion_params(RNS_BaseConversionParams params);

    void rns_compute_scaling_factors(uint64_t *delta_out, RNS_Base base, uint64_t in_mask,
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
                                                        uint64_t rns_mask, RNS_Base base);
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
    ZqVector alloc_ZqVector(uint64_t n, Modulus *mods, uint64_t l);
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
