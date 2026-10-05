// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // --- Hashing (hash.c) ------------------------------------------------

    // Whether hash_batch can take inputs of `len` bytes: a nonzero multiple
    // of 64, at most 1024.
    //
    // Both bounds are the lane-parallel core's, not this module's. It
    // compresses whole 64-byte blocks, so a length leaving a partial block
    // would have to carry that block's true length, which the many-input form
    // has no argument for; and it hashes a single BLAKE3 chunk, so past 1024
    // bytes an input is a tree of chunks rather than one compression chain.
    bool hash_batch_fits(uint64_t len);

    // BLAKE3 over `count` independent inputs of `len` bytes each, laid out end
    // to end in `in`; writes `count` 32-byte digests to `out`. Each digest is
    // exactly what hashing that input on its own gives.
    //
    // Many short independent inputs are what BLAKE3's SIMD lanes are for -- 8
    // at a time under AVX2, 16 under AVX-512 -- and hashing them one at a time
    // leaves the lanes idle and pays a hasher init and finalize each. At
    // 64-byte inputs the batch is ~9x, and reaches the same throughput BLAKE3
    // gets on one long buffer.
    //
    // A `len` that hash_batch_fits rejects is still hashed, one input at a
    // time: callers need not branch, they only lose the speedup. `in` and
    // `out` must not overlap.
    void hash_batch(uint8_t *out, const uint8_t *in, uint64_t count, uint64_t len);

    // Incremental BLAKE3 for callers that cannot embed the hasher type: the
    // caller allocates hash_stream_size() bytes of state. hash_stream_digest
    // writes the 32-byte digest of everything given so far and leaves the
    // state usable.
    uint64_t hash_stream_size(void);
    void hash_stream_init(void *state);
    void hash_stream_update(void *state, const uint8_t *in, uint64_t len);
    void hash_stream_digest(const void *state, uint8_t out[32]);

    // --- Randomness (prng.c) ---------------------------------------------
    //
    // The entropy-backed generators below may be called from any number of
    // threads at once: each thread draws from a pool of its own, and a forked
    // child discards the pool it inherits instead of repeating its parent.

    // Fills p[0..3] (32 bytes) with entropy from RDRAND where the build has it,
    // /dev/urandom otherwise. There is no error return: on failure it prints to
    // stdout and returns, leaving p's contents undefined. This is the seed
    // source the expanders below draw from -- callers wanting random data want
    // generate_random_bytes, not this.
    void generate_rnd_seed(uint64_t *p);

    // Writes `amount` bytes to `pointer`, expanded from one freshly drawn seed.
    // The expander is whichever PRF the build has: an AES-NI keystream on tuned
    // x86-64 (VAES where the engine has it), BLAKE3 otherwise -- so the byte
    // stream differs between engines even from an identical seed. Every call draws a
    // new seed, which dominates the cost for small amounts.
    void get_rnd_from_hash(uint64_t amount, uint8_t *pointer);

    // Writes `amount` bytes to `pointer` from the calling thread's 1 KiB pool,
    // refilling it through get_rnd_from_hash when what remains will not cover
    // the request. Amortizes the seed draw across many small requests. A
    // request larger than the pool is served straight from get_rnd_from_hash.
    void get_rnd_from_buffer(uint64_t amount, uint8_t *pointer);

    // Writes `amount` unpredictable bytes to `pointer`. The general-purpose
    // entry point, and the one to reach for by default: it serves requests
    // under 512 bytes from the pool and expands a fresh seed for larger ones.
    void generate_random_bytes(uint64_t amount, uint8_t *pointer);

    // Writes `count` values exactly uniform in [0, bound) to out[0..count),
    // from the entropy stream: each is a fresh 64-bit word masked to the bits
    // of bound - 1, redrawn while it is not below `bound`. The redraws reveal
    // only how many draws were discarded. `bound` must be at least 1.
    void generate_uniform_below(uint64_t *out, uint64_t count, uint64_t bound);

    // Returns one sample from a zero-mean Gaussian with standard deviation
    // `sigma`, by Box-Muller over generate_random_bytes (normal.c). Consumes 16
    // random bytes per call, keeping one of the transform's two outputs and
    // discarding the other. The support is unbounded, so a caller needing a
    // tail bound must clamp or resample.
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

    // The same sequence from position `start`: out[i] is value start + i of the
    // sequence prng_sample_below writes, which is that call with start = 0.
    //
    // Each position is located by its index rather than by the draws before it,
    // so any window costs what its own values cost -- a verifier that needs a
    // handful of positions of a long pseudorandom vector pays for those and not
    // for the vector. That is a property of the definition, not an
    // optimization: the two entry points cannot disagree.
    void prng_sample_below_from(uint64_t *out, uint64_t count, uint64_t start, uint64_t bound,
                                const char *context, const uint8_t *seed, uint64_t seed_len);

    // --- Seed expansion (expand.c) ---------------------------------------

    // The key a seed expands under: the first 16 bytes of BLAKE3 in
    // derive-key mode with `context`, over `seed` followed by each word of
    // `label` as 8 little-endian bytes. `context` is a fixed string literal
    // per use; `label` separates the streams drawn from one seed.
    void prng_expand_key(uint8_t key[16], const char *context, const uint8_t *seed,
                         uint64_t seed_len, const uint64_t *label, uint64_t label_len);

    // Writes values [start, start + count) of the sequence below `bound` that
    // `key` defines to out[0..count). Every engine computes the same sequence,
    // and stored seeds rely on it never changing:
    //
    //   keystream block (j, t) = AES-128_key(LE64(j) || LE64(t))
    //   w = 4 if bound <= 2^32 else 8;  m = the smallest 2^k - 1 >= bound - 1
    //   value i = the first, over attempts t = 0, 1, ..., of
    //             (the w-byte little-endian word at byte i*w of attempt t's
    //              keystream) & m  that is below bound.
    //
    // Values are exactly uniform in [0, bound), and value i depends on i
    // alone, so any window can be computed on its own. A rejected value costs
    // one more block, which is rare when bound is close to a power of two.
    // Uses AES-NI/VAES where the engine has them. Not constant time: the key
    // is public. `bound` must be at least 1.
    void prng_expand_below(uint64_t *out, uint64_t count, uint64_t start, uint64_t bound,
                           const uint8_t key[16]);
    // The same values at 32 bits, for `bound` <= 2^32.
    void prng_expand_below32(uint32_t *out, uint64_t count, uint64_t start, uint64_t bound,
                             const uint8_t key[16]);
    // AES-128 in counter mode: keystream blocks first .. first + count - 1 of
    // `attempt` under `key` (the blocks defined above), 16 bytes each.
    void prng_aes128_ctr(uint8_t *out, uint64_t count, uint64_t first, uint64_t attempt,
                         const uint8_t key[16]);

    // Test-only: makes every generator above reproducible by replacing the
    // hardware seed source with a splitmix64 stream started from `seed`. Also
    // discards every thread's pooled bytes, so the next draw comes from
    // `seed`. Reproducible within one build only, since get_rnd_from_hash's
    // expander is engine-dependent, and on one thread only: threads drawing
    // at once take their seeds from the stream in whatever order they reach
    // it, though never the same seed. Set or clear it while no other thread
    // draws. Does not affect prng_sample_below, which is already a pure
    // function of its arguments. Production never calls it.
    void vfhe_prng_set_deterministic_seed(uint64_t seed);

    // Returns the generators above to hardware entropy and discards every
    // thread's pooled bytes.
    void vfhe_prng_clear_deterministic_seed(void);

#ifdef __cplusplus
}
#endif
