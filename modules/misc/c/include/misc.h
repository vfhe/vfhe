#pragma once
#include <arith.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // misc
    void gen_sparse_ternary_array_modq(uint64_t *out, uint64_t size, uint64_t h, uint64_t q);
    uint64_t next_power_of_2(uint64_t x);
    void array_reduce_mod_N(uint64_t *out, uint64_t *in, uint64_t size, uint64_t p);
    void array_mod_switch(uint64_t *out, uint64_t *in, uint64_t p, uint64_t q, uint64_t n);
    void array_mod_switch_from_2k(uint64_t *out, uint64_t *in, uint64_t p, uint64_t q, uint64_t n);
    uint64_t int_mod_switch(uint64_t in, uint64_t p, uint64_t q);
    NTT_proc *new_ntt_list(uint64_t *primes, uint64_t N, uint64_t l);
    incNTT new_incomplete_ntt_list(uint64_t *primes, uint64_t split_degree, uint64_t N, uint64_t l);
    uint64_t **incNTT_get_rou_matrix(incNTT ntt);
    uint64_t double2int(double x);
    void compute_RNS_Qhat_array(uint64_t *out, uint64_t *p, uint64_t l);
    void array_additive_inverse_mod_switch(uint64_t *out, uint64_t *in, uint64_t p, uint64_t q,
                                           uint64_t n);
    uint64_t mod_dist(uint64_t a, uint64_t b, uint64_t q);
    void print_array(const char *msg, uint64_t *v, size_t size);
    uint64_t mod_switch(uint64_t v, uint64_t p, uint64_t q);
    unsigned char char_rev(unsigned char b);
    uint32_t int_rev(uint32_t b);
    void bit_rev(uint64_t *out, uint64_t *in, uint64_t n, uint64_t log_n);

    // Misc from third party
    void generate_random_bytes(uint64_t amount, uint8_t *pointer);
    double generate_normal_random(double sigma);
    void *safe_malloc(size_t size);
    void *safe_aligned_malloc(size_t size);

    // Build/CPU introspection -- powers the runtime "you could be faster" hint.
    int vfhe_build_is_portable(void);  // 1 if compiled with PORTABLE_BUILD
    int vfhe_cpu_has_avx512ifma(void); // 1 if THIS CPU supports AVX-512 IFMA

    // Test-only: pin the PRNG to a reproducible stream so probabilistic FHE
    // tests are deterministic. Production uses hardware entropy (never calls these).
    void vfhe_prng_set_deterministic_seed(uint64_t seed);
    void vfhe_prng_clear_deterministic_seed(void);

#ifdef __cplusplus
}
#endif
