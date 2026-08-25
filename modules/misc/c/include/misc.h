// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <arith.h>
#include <vfhe_cpu.h>

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
    Modulus *new_modulus_list(uint64_t *primes, uint64_t l);
    NTT_Plan *new_ntt_plan_list(Modulus *mods, uint64_t N, uint64_t l);
    RNS_Base new_rns_base(uint64_t *primes, uint64_t split_degree, uint64_t N, uint64_t l);
    uint64_t **rns_base_get_rou_matrix(RNS_Base base);
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
    void *safe_malloc(size_t size);
    void *safe_realloc(void *ptr, size_t size);
    void *safe_aligned_malloc(size_t size);

    // Which engine this binary is (CPU capability lives in vfhe_cpu.h, which
    // this header includes).
    const char *vfhe_engine_active(void); // e.g. "portable", "avx512ifma"

    // --- Randomness (prng.c) ---------------------------------------------

    // Fills p[0..3] (32 bytes) with entropy from RDRAND where the build has it,
    // /dev/urandom otherwise. There is no error return: on failure it prints to
    // stdout and returns, leaving p's contents undefined. This is the seed
    // source the expanders below draw from -- callers wanting random data want
    // generate_random_bytes, not this.
    void generate_rnd_seed(uint64_t *p);

    // Writes `amount` bytes to `pointer`, expanded from one freshly drawn seed.
    // The expander is whichever PRF the build has: an AES-NI keystream on tuned
    // x86-64, SHAKE256 under USE_SHAKE, BLAKE3 otherwise -- so the byte stream
    // differs between engines even from an identical seed. Every call draws a
    // new seed, which dominates the cost for small amounts.
    void get_rnd_from_hash(uint64_t amount, uint8_t *pointer);

    // Writes `amount` bytes to `pointer` from a 1 KiB internal pool, refilling
    // it through get_rnd_from_hash when what remains will not cover the
    // request. Amortizes the seed draw across many small requests. `amount`
    // must not exceed 1024 -- a larger request reads past the pool. The pool is
    // shared mutable state, so this is not thread-safe.
    void get_rnd_from_buffer(uint64_t amount, uint8_t *pointer);

    // Writes `amount` unpredictable bytes to `pointer`. The general-purpose
    // entry point, and the one to reach for by default: it serves requests
    // under 512 bytes from the pool and expands a fresh seed for larger ones.
    // Not thread-safe.
    void generate_random_bytes(uint64_t amount, uint8_t *pointer);

    // Returns one sample from a zero-mean Gaussian with standard deviation
    // `sigma`, by Box-Muller over generate_random_bytes. Consumes 16 random
    // bytes per call, keeping one of the transform's two outputs and discarding
    // the other. The support is unbounded, so a caller needing a tail bound
    // must clamp or resample.
    double generate_normal_random(double sigma);

    // Writes `count` values uniform in [0, bound) to out[0..count).
    //
    // Seeded rather than entropy-backed: the result is a pure function of
    // (context, seed), so it is byte-for-byte reproducible across runs and
    // across engines, and the deterministic-seed override below has no effect
    // on it. Use it where a value must be recomputable from a transcript rather
    // than merely unpredictable; use generate_random_bytes where it must be
    // unpredictable.
    //
    // `context` is a NUL-terminated domain-separation tag: one seed under two
    // different tags gives two independent streams. Pass a fixed string literal
    // per call site, never anything caller-controlled. `seed` may be any length.
    //
    // out[i] depends on i alone and not on `count`, so raising `count` extends
    // the sequence instead of changing it. `bound` must be at least 1 (asserted;
    // 0 would make the rejection loop spin forever). Values are drawn by
    // rejection from a mask, so the number of hash blocks consumed depends on
    // how far `bound` sits below the next power of two -- never more than about
    // two draws per value on average, but not constant-time in `bound`.
    void prng_sample_below(uint64_t *out, uint64_t count, uint64_t bound, const char *context,
                           const uint8_t *seed, uint64_t seed_len);

    // Test-only: makes every generator above reproducible by replacing the
    // hardware seed source with a splitmix64 stream started from `seed`. Also
    // discards pooled bytes, so the next draw comes from `seed`. Reproducible
    // within one build only, since get_rnd_from_hash's expander is
    // engine-dependent. Does not affect prng_sample_below, which is already a
    // pure function of its arguments. Not thread-safe; production never calls
    // it.
    void vfhe_prng_set_deterministic_seed(uint64_t seed);

    // Returns the generators above to hardware entropy and discards pooled
    // bytes.
    void vfhe_prng_clear_deterministic_seed(void);

#ifdef __cplusplus
}
#endif
