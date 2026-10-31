// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
// Python-facing ABI of the arith engine (ported from the original ctypes
// prototypes). Handles are passed as `void *` (as the old ctypes code did);
// the structs below are cdef'd only so Python can cast a handle and read
// fields (rns_mask, coeffs, l, ...). Layout is filled by cffi from arith.h.

typedef struct _Modulus *Modulus;
typedef struct _NTT_Plan *NTT_Plan;

typedef struct _RNS_Base
{
    NTT_Plan *plans;
    Modulus *mods;
    uint64_t split_degree;
    uint64_t **w;
    uint64_t N;
    uint64_t l;
    uint64_t narrow_mask;
    uint32_t **w32;
} *RNS_Base;

// rows64[i] or rows32[i] holds row i, per base->narrow_mask; the other is NULL.
typedef struct _RNS_Polynomial
{
    uint64_t **rows64;
    uint32_t **rows32;
    RNS_Base base;
    uint64_t rns_mask;
    uint64_t allocated_l;
} *RNS_Polynomial;

// --- allocation ---
void *safe_aligned_malloc(uint64_t size);
// libc free, releasing safe_aligned_malloc'd (posix_memalign, 64B) buffers
// that Python owns and hands to the kernels.
void free(void *ptr);

// --- generic arithmetic ---
// A ring is opaque; only its address is handled here. An element is the
// implementation's handle plus the domain it is currently in.
typedef struct _ArithRing *ArithRing;
typedef struct
{
    void *handle;
    int domain;
} ArithElement;
int arith_copy(void *ring, ArithElement *out, const ArithElement *in);
int arith_zero(void *ring, ArithElement *out);
int arith_add(void *ring, ArithElement *out, const ArithElement *a, const ArithElement *b);
int arith_sub(void *ring, ArithElement *out, const ArithElement *a, const ArithElement *b);
int arith_mul(void *ring, ArithElement *out, const ArithElement *a, const ArithElement *b);
int arith_copy_handles(void *ring, void *out, void *in, int domain);
int arith_add_handles(void *ring, void *out, void *a, void *b, int domain);
int arith_sub_handles(void *ring, void *out, void *a, void *b, int domain);
int arith_mul_handles(void *ring, void *out, void *a, void *b, int domain);
int arith_scale_int(void *ring, ArithElement *out, const ArithElement *a, uint64_t scale);
typedef struct
{
    void *handle;
} ArithScalar;
int arith_scalar_new(void *ring, const uint64_t *per_component, ArithScalar *out);
void arith_scalar_free(void *ring, ArithScalar *scalar);
int arith_scale_by(void *ring, ArithElement *out, const ArithElement *a, ArithScalar scale);
int arith_permute(void *ring, ArithElement *out, const ArithElement *a, uint64_t gen);
int arith_sample_gaussian(void *ring, ArithElement *out, double sigma);
int arith_round_division(void *ring, ArithElement *element, void *to);
int arith_to_mul(void *ring, ArithElement *element);
int arith_to_canonical(void *ring, ArithElement *element);
int arith_round_division_approximate(void *ring, ArithElement *element, void *to);
int arith_decompose_digit(void *ring, ArithElement *out, ArithElement *a, uint64_t primes,
                          uint64_t log_base, uint64_t level, uint64_t dropped, int centered);
void arith_set_generic_only(int enabled);
int arith_generic_only(void);
void *arith_rns_ring_get(uint64_t N, uint64_t rns_mask, void *base);
void arith_rns_ring_cache_clear(void);

// --- RNS base / base conversion ---
void *new_rns_base(uint64_t *primes, uint64_t split_degree, uint64_t N, uint64_t count);
void rns_base_extend_with_primes(void *base, uint64_t *new_primes, uint64_t count);
uint64_t **rns_base_get_rou_matrix(void *base);
void *init_base_conversion_params(void *base, uint64_t in_mask, uint64_t out_mask);
void *init_base_conversion_params_exact(void *base, uint64_t in_mask, uint64_t out_mask);
void free_base_conversion_params(void *params);

// --- polynomial lifecycle / io ---
void *polynomial_new_RNS_polynomial(uint64_t N, uint64_t rns_mask, void *base);
void polynomial_copy_RNS_polynomial(void *out, void *in);
void free_RNS_polynomial(void *p);
void int_array_to_RNS(void *out, uint64_t *in);
void array_to_RNS(void *out, uint64_t **in);

// --- sampling ---
void polynomial_gen_random_RNSc_polynomial(void *out);
void polynomial_gen_gaussian_RNSc_polynomial(void *out, double sigma);

// --- arithmetic ---
void polynomial_mul_RNS_polynomial(void *out, void *in1, void *in2);
void polynomial_multo_RNS_polynomial(void *out, void *in);
void polynomial_sub_RNS_polynomial(void *out, void *in1, void *in2);
void polynomial_sub_RNSc_polynomial(void *out, void *in1, void *in2);
void polynomial_add_RNSc_polynomial(void *out, void *in1, void *in2);
void polynomial_add_RNS_polynomial(void *out, void *in1, void *in2);
void polynomial_scale_RNSc_polynomial(void *out, void *in1, uint64_t scale);
void polynomial_scale_RNS_polynomial_RNS(void *out, void *in1, uint64_t *scale);
void polynomial_RNSc_negate(void *out, void *in);
void polynomial_RNSc_add_integer(void *out, void *in1, uint64_t in2);
void polynomial_RNS_add_integer(void *out, void *in1, uint64_t in2);

// --- automorphism / domain conversion ---
void polynomial_RNSc_permute(void *out, void *in, uint64_t gen);
void polynomial_RNSc_to_centered_doubles(double *out, void *in, double scale);
void polynomial_RNS_automorphism_index(uint32_t *idx, uint64_t N, uint64_t gen);
void polynomial_RNS_permute(void *out, void *in, const uint32_t *idx);
void polynomial_RNSc_to_RNS(void *out, void *in);
void polynomial_RNS_to_RNSc(void *out, void *in);

// --- tower ---
void polynomial_base_conversion_RNSc(void *out, void *in, void *params);
void polynomial_RNSc_mod_reduce(void *out, void *in);
void polynomial_RNSc_mod_reduce_lifted(void *out, void *in, uint64_t idx);
void polynomial_RNSc_mod_reduce_lifted_centered(void *out, void *in, uint64_t idx);
void polynomial_RNSc_decompose_digit(void *out, void *in, uint64_t idx, uint64_t log_base,
                                     uint64_t level);
void polynomial_round_division_RNSc_wo_free(void *out, uint64_t divide_mask);
void polynomial_floor_division_RNSc_wo_free(void *out, uint64_t divide_mask);
void polynomial_RNSc_scaled_lift(void *out, void *in, uint64_t *delta);

// --- slots / inverse / eq ---
int polynomial_RNS_inverse(void *out, void *in);
bool polynomial_eq(void *a, void *b);
void polynomial_RNS_broadcast_slot(void *out, void *in, uint64_t slot_idx);
void polynomial_RNS_rotate_slot(void *out, void *in, uint64_t rot);
void polynomial_RNS_copy_slot(void *out, uint64_t dst, void *in, uint64_t src);

// --- hashing ---
void polynomial_RNS_get_hash(uint64_t *out, void *p);
void polynomial_RNS_hash_update(void *state, void *p);
void polynomial_RNS_expand_seeded(void *out, const uint8_t *seed, uint64_t seed_len,
                                  uint64_t stream);
bool polynomial_RNS_matches_seeded(void *p, bool canonical, uint64_t index, const uint8_t *seed,
                                   uint64_t seed_len, uint64_t stream);
uint64_t rns_row_bytes(uint64_t q, uint64_t N, bool tight);
void polynomial_RNS_write_rows(uint8_t *out, void *p, const uint64_t *rows, uint64_t count,
                               bool tight);
int64_t polynomial_RNS_read_rows(void *p, const uint8_t *in, const uint64_t *rows, uint64_t count,
                                 bool tight, bool validate);
uint64_t *polynomial_RNS_get_hash_p(void *p);
void polynomial_RNS_lift_mod_2_128(uint64_t *low, uint64_t *high, RNS_Polynomial *in,
                                   uint64_t count);

// --- negacyclic FFT ---
typedef struct _FFT_Plan *FFT_Plan;
FFT_Plan fft_new_plan(uint64_t n);
void fft_free_plan(FFT_Plan plan);
void fft_forward(double *x, FFT_Plan plan);
void fft_inverse(double *x, FFT_Plan plan);
void fft_pointwise_mul(double *out, const double *a, const double *b, FFT_Plan plan);

// --- complex (CKKS encoding) ---
void complex_slot_positions(uint32_t *out, uint64_t N);
void complex_poly_fft(double *x, FFT_Plan plan, const uint32_t *slots);
void complex_poly_ifft(double *x, FFT_Plan plan, const uint32_t *slots);
void complex_poly_scale_double(double *v, double scale, uint64_t N);
void complex_poly_to_interleaved(double *out, const double *in, uint64_t N);
bool complex_poly_round_to_RNS(void *out, double *in, uint64_t N);
bool complex_polys_ifft_scale_round_to_RNS_batch(void **rows_in, void **outs_rns, uint64_t count,
                                                 FFT_Plan plan, const uint32_t *slots,
                                                 double temp_delta);
bool complex_values_ifft_scale_round_to_RNS_batch(RNS_Polynomial *outs, const double *const *values,
                                                  const uint64_t *lengths, const uint64_t *shifts,
                                                  uint64_t count, FFT_Plan plan,
                                                  const uint32_t *slots, double scale,
                                                  uint64_t n_threads);

// --- multiprecision ---
// A digit vector is one uint64 in the portable build and an __m512i (8 lanes) in
// the tuned one; arith.h owns the real typedef. Keep it opaque here so the C that
// cffi generates matches the header in both builds. Python reads the lanes by
// casting: ffi.cast("uint64_t *", scalar.digits), get_mp_vector_size() lanes per
// digit.
typedef struct _mp_vector mp_vector_t;
typedef struct _MPScalar
{
    mp_vector_t *digits;
    uint64_t d;
} *MPScalar;
typedef struct _MPPolynomial
{
    uint64_t **coeffs;
    uint64_t N;
    uint64_t d;
} *MPPolynomial;
int get_mp_vector_size(void);
void mp_scale(void *out, void *in, void *m);
void mp_sub(void *out, void *a, void *b);
void *mp_load(uint64_t *in, uint64_t d);
mp_vector_t *load_m512(uint64_t in);
void *new_mp_polynomial(uint64_t N, uint64_t d);
void mp_polynomial_from_RNS(void *out, void *in, MPScalar *PW, void *q, mp_vector_t *m, uint64_t k);
void mp_polynomial_to_RNSc(void *out, void *in);
void free_mp_polynomial(void *p);
void mp_polynomial_scale_to_2k(uint64_t *out, void *in, const uint64_t *half, uint64_t q_inv_neg,
                               uint64_t k);

// --- field extensions ---
Modulus mod_new(uint64_t q);
void mod_free(void *mod);
void field_ext_add(uint64_t *c, const uint64_t *a, const uint64_t *b, uint64_t d, uint64_t q);
void field_ext_sub(uint64_t *c, const uint64_t *a, const uint64_t *b, uint64_t d, uint64_t q);
void field_ext_neg(uint64_t *c, const uint64_t *a, uint64_t d, uint64_t q);
void field_ext_mul(uint64_t *c, const uint64_t *a, const uint64_t *b, uint64_t d, uint64_t w,
                   void *mod);
void field_ext_pow(uint64_t *res, const uint64_t *base, const uint64_t *exp, uint64_t exp_words,
                   uint64_t d, uint64_t w, void *mod);
int field_ext_inv(uint64_t *ainv, const uint64_t *a, uint64_t d, uint64_t w, void *mod);
void field_ext_frobenius(uint64_t *res, const uint64_t *a, uint64_t k, uint64_t d, uint64_t w,
                         void *mod);
uint64_t field_ext_root_subfield_degree(uint64_t n, void *mod);
int field_ext_root_of_unity(uint64_t *out, uint64_t n, uint64_t d, uint64_t w, void *mod);
typedef struct _FieldNTTPlan
{
    uint64_t n;
    uint64_t logn;
    uint64_t d;
    uint64_t w;
    Modulus mod;
    uint64_t root[8];
    uint64_t inv_root[8];
    uint64_t inv_n[8];
    uint64_t **ws_fwd;
    uint64_t **ws_inv;
} *FieldNTTPlan;
FieldNTTPlan field_ntt_new_plan(uint64_t n, const uint64_t *root_of_unity, uint64_t d, uint64_t w,
                                void *mod);
void field_ntt_free_plan(FieldNTTPlan plan);
void field_ntt_forward(uint64_t *const *planes, uint64_t blocks, FieldNTTPlan plan);
void field_ntt_inverse(uint64_t *const *planes, uint64_t blocks, FieldNTTPlan plan);
uint64_t ntt_plan_root(NTT_Plan plan);
void field_ntt_forward_base(uint64_t *const *planes, uint64_t blocks, uint64_t d, NTT_Plan plan);
void field_ntt_inverse_base(uint64_t *const *planes, uint64_t blocks, uint64_t d, NTT_Plan plan);
void field_sample_random_element(uint64_t *a, const uint8_t *seed, uint64_t seed_len, uint64_t d,
                                 uint64_t mod);
void field_hash_element(uint8_t *out, const uint64_t *a, uint64_t d);
int field_ext_is_equal(const uint64_t *a, const uint64_t *b, uint64_t d);
void field_base_conversion(uint64_t *out, const uint64_t *in, uint64_t source_component,
                           uint64_t target_component, uint64_t d, uint64_t poly_size,
                           const uint64_t *w_i, void *mod);

// --- vectors of field elements ---
// n elements as d coefficient planes: element i is coeffs[j][i] over j. Python
// owns the buffers (aligned64) and fills this struct; the planes must be
// field_vec_padded_length(n) words with the padding reduced, which is what lets
// the kernels run the tuned eltwise path with no tail. arith.h states the full
// contract.
typedef struct _FieldVector
{
    uint64_t **coeffs;
    uint64_t n;
    uint64_t allocated_n;
    uint64_t d;
    uint64_t w;
    Modulus mod;
} *FieldVector;
uint64_t field_vec_padded_length(uint64_t n);
typedef struct _RNS_CrtMaps *RNS_CrtMaps;
RNS_CrtMaps rns_crt_maps_new(void *base, uint64_t row, bool to_slot_zero);
void polynomial_RNS_sample_exceptional(void *out, const uint8_t *seed, uint64_t seed_len,
                                       RNS_CrtMaps const *maps);
void rns_crt_maps_free(RNS_CrtMaps maps);
void polynomial_RNS_crt_components(FieldVector out, RNS_Polynomial *in, uint64_t count,
                                   RNS_CrtMaps maps, bool element_major);
void polynomial_RNS_crt_linear_combination(FieldVector out, RNS_Polynomial *const *parts,
                                           uint64_t part_count, uint64_t count,
                                           const FieldVector coefficients, RNS_CrtMaps maps);
void polynomial_RNS_from_crt_components(RNS_Polynomial *out, FieldVector in, uint64_t count,
                                        RNS_CrtMaps maps);
void field_vec_add(FieldVector out, const FieldVector a, const FieldVector b);
void field_vec_sub(FieldVector out, const FieldVector a, const FieldVector b);
void field_vec_neg(FieldVector out, const FieldVector a);
void field_vec_add_scalar(FieldVector out, const FieldVector a, const uint64_t *s);
void field_vec_sub_scalar(FieldVector out, const FieldVector a, const uint64_t *s);
void field_vec_scalar_sub(FieldVector out, const uint64_t *s, const FieldVector a);
void field_vec_mul(FieldVector out, const FieldVector a, const FieldVector b);
void field_vec_scale(FieldVector out, const FieldVector a, const uint64_t *s);
void field_ntt_to_interleaved(FieldVector out, const FieldVector in, uint64_t blocks);
void field_ntt_to_blocks(FieldVector out, const FieldVector in, uint64_t blocks);
void field_ntt_points(FieldVector out, uint64_t psi, uint64_t bits);
void field_vec_fma(FieldVector out, const FieldVector a, const FieldVector b, const FieldVector c);
void field_vec_fma_scalar(FieldVector out, const FieldVector a, const FieldVector b,
                          const uint64_t *s);
void field_vec_sum(uint64_t *out, const FieldVector a);
int field_vec_inv(FieldVector out, const FieldVector a);
void field_vec_get_element(uint64_t *out, const FieldVector a, uint64_t index);
void field_vec_set_element(FieldVector out, uint64_t index, const uint64_t *value);
void field_vec_set_range(FieldVector out, uint64_t start, const uint64_t *values, uint64_t count);
void field_vec_get_range(uint64_t *out, const FieldVector a, uint64_t start, uint64_t count);
void field_vec_copy(FieldVector out, const FieldVector a);
void field_vec_split_even_odd(FieldVector even, FieldVector odd, const FieldVector a);
void field_vec_fma_interleave(FieldVector out, const FieldVector a, const FieldVector b,
                              const FieldVector c_even, const uint64_t *s_even,
                              const FieldVector c_odd, const uint64_t *s_odd);
void field_vec_fold_twisted(FieldVector out, const FieldVector word, const FieldVector untwist,
                            const uint64_t *r);
void field_vec_fold_twisted_blocks(FieldVector out, const FieldVector word,
                                   const FieldVector untwist, const FieldVector r);
void field_vec_mul_plane(FieldVector out, const FieldVector a, const FieldVector t);
void field_vec_fma_plane(FieldVector out, const FieldVector a, const FieldVector b,
                         const FieldVector t);
void field_vec_lift_twisted(FieldVector out, const FieldVector a, const FieldVector b,
                            const FieldVector t, uint64_t period);
void field_vec_interleave(FieldVector out, const FieldVector even, const FieldVector odd);
void field_vec_concat(FieldVector out, const FieldVector *parts, uint64_t count);
void field_vec_clear_padding(FieldVector a);
bool field_vec_gather(FieldVector out, const FieldVector a, const uint64_t *indices,
                      uint64_t count);
void field_vec_fold(FieldVector out, const FieldVector a, const uint64_t *r);
void field_vec_fold_blocks(FieldVector out, const FieldVector a, uint64_t block, const uint64_t *r);
void field_vec_split_blocks(FieldVector lo, FieldVector hi, const FieldVector a, uint64_t block);
void field_vec_frobenius(FieldVector out, const FieldVector a, uint64_t k);
int field_vec_is_equal(const FieldVector a, const FieldVector b);
void field_vec_sample_random(FieldVector out, const uint8_t *seed, uint64_t seed_len,
                             uint64_t start);
void field_vec_sample_random_at(FieldVector out, const uint8_t *seed, uint64_t seed_len,
                                const uint64_t *indices, uint64_t count);
void field_vec_sample_random_element(uint64_t *out, const uint8_t *seed, uint64_t seed_len,
                                     uint64_t index, uint64_t d, uint64_t mod);
void field_vec_hash(uint8_t *out, const FieldVector a);
uint64_t field_vec_hash_fiber_count(const FieldVector a, uint64_t group, uint64_t stride);
void field_vec_hash_fibers(uint8_t *out, const FieldVector a, uint64_t group, uint64_t stride);
uint64_t field_vec_hash_count(const FieldVector a, uint64_t group, uint64_t stride);
void field_vec_hash_elements(uint8_t *out, const FieldVector a, uint64_t group, uint64_t stride);
NTT_Plan ntt_new_plan(uint64_t n, void *mod);
void ntt_free_plan(void *plan);
uint64_t inverse_mod_eea(uint64_t a, uint64_t p);
uint64_t generate_Nth_root_of_unity(uint64_t q, uint64_t n);

// --- pseudo-Mersenne prime field (p = 2^n - c) ---
// The parameter block stays opaque -- accessors, not a cdef'd struct -- so the
// AVX-512 phase can add members without moving the ABI. An element is
// pmf_limbs() uint64 words of 52-bit limbs, little-endian, in a 64-byte-aligned
// buffer of 8 lanes whose upper lanes are always zero.
typedef struct _PMFParams *PMFParams;
PMFParams pmf_new_params(uint64_t n, uint64_t c);
void pmf_free_params(PMFParams params);
uint64_t pmf_limbs(void *params);
uint64_t pmf_byte_length(void *params);
void pmf_add(uint64_t *out, const uint64_t *a, const uint64_t *b, void *params);
void pmf_sub(uint64_t *out, const uint64_t *a, const uint64_t *b, void *params);
void pmf_neg(uint64_t *out, const uint64_t *a, void *params);
void pmf_mul(uint64_t *out, const uint64_t *a, const uint64_t *b, void *params);
int pmf_is_equal(const uint64_t *a, const uint64_t *b, void *params);
void pmf_canonicalize(uint64_t *out, const uint64_t *in, void *params);
void pmf_to_bytes(uint8_t *out, const uint64_t *a, void *params);
int pmf_from_bytes(uint64_t *out, const uint8_t *in, void *params);
void pmf_hash(uint8_t *out, const uint64_t *a, void *params);
void pmf_sample_random(uint64_t *out, const uint8_t *seed, uint64_t seed_len, void *params);

// --- vectors of pseudo-Mersenne elements ---
// n elements as L limb planes: limb j of element i at limbs[j][i]. Python owns
// the buffers (aligned64) and fills this struct; the planes must be
// pmf_vec_padded_length(n) words with the padding canonical, which is what lets
// the kernels run whole groups with no tail. arith.h states the full contract.
typedef struct _PMFVector
{
    uint64_t **limbs;
    uint64_t n;
    uint64_t allocated_n;
    PMFParams params;
} *PMFVector;
uint64_t pmf_vec_padded_length(uint64_t n);
void pmf_vec_add(PMFVector out, const PMFVector a, const PMFVector b);
void pmf_vec_sub(PMFVector out, const PMFVector a, const PMFVector b);
void pmf_vec_neg(PMFVector out, const PMFVector a);
void pmf_vec_mul(PMFVector out, const PMFVector a, const PMFVector b);
void pmf_vec_add_scalar(PMFVector out, const PMFVector a, const uint64_t *s);
void pmf_vec_sub_scalar(PMFVector out, const PMFVector a, const uint64_t *s);
void pmf_vec_scalar_sub(PMFVector out, const uint64_t *s, const PMFVector a);
void pmf_vec_scale(PMFVector out, const PMFVector a, const uint64_t *s);
void pmf_vec_sum(uint64_t *out, const PMFVector a);
void pmf_vec_get_element(uint64_t *out, const PMFVector a, uint64_t index);
void pmf_vec_set_element(PMFVector out, uint64_t index, const uint64_t *value);
void pmf_vec_set_range(PMFVector out, uint64_t start, const uint64_t *values, uint64_t count);
void pmf_vec_get_range(uint64_t *out, const PMFVector a, uint64_t start, uint64_t count);
void pmf_vec_set_uint64(PMFVector out, uint64_t start, const uint64_t *low, const uint64_t *high,
                        uint64_t count, uint32_t shift, uint32_t width);
bool pmf_vec_get_uint64(uint64_t *out, const PMFVector a);
void pmf_vec_copy(PMFVector out, const PMFVector a);
void pmf_vec_split_even_odd(PMFVector even, PMFVector odd, const PMFVector a);
void pmf_vec_interleave(PMFVector out, const PMFVector even, const PMFVector odd);
void pmf_vec_concat(PMFVector out, const PMFVector *parts, uint64_t count);
void pmf_vec_clear_padding(PMFVector a);
bool pmf_vec_gather(PMFVector out, const PMFVector a, const uint64_t *indices, uint64_t count);
void pmf_vec_fold(PMFVector out, const PMFVector a, const uint64_t *r);
void pmf_vec_fold_inner_products(uint64_t *q0, uint64_t *q1, uint64_t *q2, const PMFVector a,
                                 const PMFVector b);
void pmf_vec_fold_and_inner_products(PMFVector a_out, PMFVector b_out, uint64_t *q0, uint64_t *q1,
                                     uint64_t *q2, const PMFVector a, const PMFVector b,
                                     const uint64_t *r);
#define PMF_FOLD_MAX_TABLES ...
#define PMF_FOLD_MAX_NODES ...
#define PMF_FOLD_MAX_SUMS ...
#define PMF_FOLD_MAX_PRODUCTS ...
#define PMF_FOLD_MAX_FACTORS ...
void pmf_vec_fold_product_sums(uint64_t *const *sums, const PMFVector *tables, uint64_t count,
                               const uint64_t *shape, uint64_t nodes, const PMFVector weights);
void pmf_vec_fold_and_product_sums(PMFVector *outs, uint64_t *const *sums, const PMFVector *tables,
                                   uint64_t count, const uint64_t *r, const uint64_t *shape,
                                   uint64_t nodes, const PMFVector weights);
void pmf_vec_lift_twisted(PMFVector out, const PMFVector a, const PMFVector b, const PMFVector t);
void pmf_vec_fold_twisted(PMFVector out, const PMFVector word, const PMFVector untwist,
                          const uint64_t *r);
int pmf_vec_is_equal(const PMFVector a, const PMFVector b);
void pmf_vec_sample_random(PMFVector out, const uint8_t *seed, uint64_t seed_len);
void pmf_vec_hash(uint8_t *out, const PMFVector a);
uint64_t pmf_vec_hash_count(const PMFVector a, uint64_t group, uint64_t stride);
void pmf_vec_hash_elements(uint8_t *out, const PMFVector a, uint64_t group, uint64_t stride);

// --- the negacyclic NTT over a pseudo-Mersenne field ---
// Opaque: Python supplies the root and keeps it, so nothing here is read back.
typedef struct _PMFNTTPlan *PMFNTTPlan;
PMFNTTPlan pmf_ntt_new_plan(uint64_t n, const uint64_t *root_of_unity, PMFParams params);
void pmf_ntt_free_plan(PMFNTTPlan plan);
void pmf_vec_ntt_forward(PMFVector a, PMFNTTPlan plan);
void pmf_vec_ntt_inverse(PMFVector a, PMFNTTPlan plan);
void pmf_vec_ntt_forward_blocks(PMFVector a, uint64_t blocks, PMFNTTPlan plan);
void pmf_vec_ntt_inverse_blocks(PMFVector a, uint64_t blocks, PMFNTTPlan plan);
void pmf_ntt_to_interleaved(PMFVector out, const PMFVector in, uint64_t blocks);
void pmf_ntt_to_blocks(PMFVector out, const PMFVector in, uint64_t blocks);
void pmf_vec_ntt_encode_blocks(PMFVector out, const PMFVector msg, uint64_t blocks,
                               PMFNTTPlan plan);

// --- the execution tape ---
typedef enum
{
    ARITH_TRACE_INPUT = 1,
    ARITH_TRACE_COPY,
    ARITH_TRACE_ZERO,
    ARITH_TRACE_ADD,
    ARITH_TRACE_SUB,
    ARITH_TRACE_MUL,
    ARITH_TRACE_MUL_ADDTO,
    ARITH_TRACE_MUL_SUBTO,
    ARITH_TRACE_SCALE_INT,
    ARITH_TRACE_SCALE_ADDTO,
    ARITH_TRACE_SCALE_BY,
    ARITH_TRACE_PERMUTE,
    ARITH_TRACE_MUL_BY_MONOMIAL,
    ARITH_TRACE_SAMPLE_UNIFORM,
    ARITH_TRACE_SAMPLE_GAUSSIAN,
    ARITH_TRACE_SAMPLE_UNIFORM_SEEDED,
    ARITH_TRACE_FROM_INT_ARRAY,
    ARITH_TRACE_ROUND_DIVISION,
    ARITH_TRACE_MOD_REDUCE_LIFTED,
    ARITH_TRACE_ROUND_DIVISION_APPROXIMATE,
    ARITH_TRACE_DECOMPOSE_DIGIT,
    ARITH_TRACE_MARK,
} ArithTraceKind;
typedef struct
{
    uint64_t sequence;
    int kind;
    uint64_t inputs[3];
    uint32_t n_inputs;
    uint64_t N;
    uint64_t mask;
    void *base;
    int domain;
    int secret;
    void *value;
    uint64_t param;
    uint64_t param2;
    double sigma;
    uint64_t *scalar;
    uint64_t scalar_len;
    uint64_t primes;
    uint64_t log_base;
    uint64_t level;
    uint64_t dropped;
    int centered;
    char *label;
    int enter;
    int untraced;
    int failed;
} ArithTraceEvent;
void arith_trace_enable(void);
void arith_trace_disable(void);
int arith_trace_is_enabled(void);
void arith_trace_clear(void);
uint64_t arith_trace_length(void);
int arith_trace_read(uint64_t index, ArithTraceEvent *out);
void *arith_trace_take_value(uint64_t index);
void arith_trace_release(uint64_t count);
uint64_t arith_trace_next_sequence(void);
void arith_trace_mark(const char *label, int enter);
uint64_t arith_trace_version(const void *handle);
void arith_trace_set_limit(uint64_t limit);
uint64_t arith_trace_limit(void);
int arith_trace_wait(uint64_t count, double seconds);
void arith_trace_set_verify(int enabled);
int arith_trace_verify(void);
uint64_t arith_trace_untraced(void);
