// SPDX-License-Identifier: Apache-2.0
/**
 * @file base.h
 * @brief Foundational utilities shared by every C module.
 *
 * The dependency-free bottom of the module graph (`base <- rng <- arith`).
 * A handful of small concerns, one source file each under `src/`:
 *
 *  - **cpu**       runtime x86 instruction-set detection (cpu.c)
 *  - **alloc**     abort-on-failure allocation (alloc.c)
 *  - **bits**      powers of two and bit reversal (bits.c)
 *  - **modswitch** rescaling residues between moduli (modswitch.c)
 *  - **debug**     printing helpers for development (debug.c)
 *
 * Everything here is a plain function on machine words: no context objects,
 * no global state, no I/O besides the debug printer.
 */
#ifndef VFHE_BASE_H
#define VFHE_BASE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // --- cpu ---------------------------------------------------------------------

    /**
     * The x86 instruction-set features the engine can dispatch on. Each flag is
     * true only when both the CPU advertises the feature and (for the AVX
     * families) the OS has enabled the corresponding register state. Always all
     * false on non-x86 hosts.
     *
     * Building block for a future single "fat" wheel: one binary that selects the
     * AVX-512 IFMA arith kernels / AES-NI rng backend at run time instead of at
     * compile time. Realizing that also requires the SIMD backends to be built
     * with per-function target attributes so both code paths coexist in one
     * translation unit -- see modules/arith/README.md.
     */
    typedef struct cpu_features
    {
        bool avx2;       /**< AVX2 (256-bit integer). */
        bool avx512f;    /**< AVX-512 Foundation (512-bit), OS-enabled. */
        bool avx512ifma; /**< AVX-512 IFMA (52-bit madd); implies @c avx512f. */
        bool aes;        /**< AES-NI. */
    } cpu_features;

    /**
     * Detect the current CPU's features into @p out (a plain query -- no caching,
     * no global state). Cheap enough to call at context-init frequency.
     */
    void cpu_detect(cpu_features *out);

    // --- alloc -------------------------------------------------------------------

    /**
     * malloc that aborts the process on failure.
     *
     * Rationale: the engine's allocation failures are unrecoverable (mid-way
     * through a multi-limb operation there is no consistent state to roll back
     * to), so callers are freed from checking for NULL. Never returns NULL for
     * nonzero sizes.
     *
     * @param size bytes to allocate
     * @return the allocation; release with `free()`
     */
    void *safe_malloc(size_t size);

    /**
     * 64-byte-aligned malloc that aborts the process on failure.
     *
     * 64 bytes covers both a cache line and an AVX-512 vector, so buffers from
     * this allocator are valid operands for every SIMD kernel in the engine.
     *
     * @param size bytes to allocate
     * @return the allocation; release with `free()`
     */
    void *safe_aligned_malloc(size_t size);

    // --- bits --------------------------------------------------------------------

    /** Smallest power of two >= @p x (with `next_power_of_2(0) == 1`). */
    uint64_t next_power_of_2(uint64_t x);

    /** Reverse the bit order of a byte. */
    unsigned char char_rev(unsigned char b);

    /** Reverse the bit order of a 32-bit word. */
    uint32_t int_rev(uint32_t b);

    /**
     * Bit-reversal permutation: out[i] = in[bitrev_{log_n}(i)] for i in [0, n).
     *
     * @p in must hold at least `2^log_n` readable entries when @p log_n exceeds
     * log2(n) (the NTT twist tables use this to sample every other entry of a
     * double-length table). @p out and @p in must not overlap.
     *
     * @param out   n outputs
     * @param in    input table (see size note above)
     * @param n     number of outputs
     * @param log_n index width in bits used for the reversal
     */
    void bit_rev(uint64_t *out, const uint64_t *in, uint64_t n, uint64_t log_n);

    // --- modswitch ---------------------------------------------------------------

    /**
     * Rescale a residue from modulus @p p to modulus @p q:
     * `round(v * q / p) mod q`. A modulus of 0 is interpreted as 2^64.
     *
     * Computed in double precision: exact for the intended uses (moduli below
     * ~2^52, or power-of-two source moduli); do not use where bit-exact rounding
     * of near-2^63 products matters.
     *
     * @param v residue in [0, p)
     * @param p source modulus (0 = 2^64)
     * @param q target modulus (0 = 2^64)
     * @return the rescaled residue in [0, q)
     */
    uint64_t mod_switch(uint64_t v, uint64_t p, uint64_t q);

    /** Element-wise ::mod_switch: out[i] = mod_switch(in[i], p, q). May alias. */
    void array_mod_switch(uint64_t *out, const uint64_t *in, uint64_t p, uint64_t q, uint64_t n);

    /**
     * Reduce each element modulo the power of two covering @p p:
     * out[i] = in[i] & (next_power_of_2(p) - 1). May alias.
     */
    void array_reduce_mod_N(uint64_t *out, const uint64_t *in, uint64_t size, uint64_t p);

    /**
     * Map uniform 64-bit words onto [0, q): mask each element down to the power
     * of two covering @p p, then rescale from that 2^k to @p q.
     *
     * With `p == q` this is the engine's rejection-free uniform sampler modulo q
     * (the 2^k -> q rescaling has a negligible, bounded bias for the ~50-bit
     * primes in use). May alias.
     *
     * @param out output residues in [0, q)
     * @param in  n uniform 64-bit words
     * @param p   selects the masking window next_power_of_2(p)
     * @param q   target modulus
     * @param n   element count
     */
    void array_mod_switch_from_2k(uint64_t *out, const uint64_t *in, uint64_t p, uint64_t q,
                                  uint64_t n);

    /**
     * Centered mod-switch for small (noise) values: elements in the upper half
     * of [0, p) are treated as negative and mapped to the corresponding negative
     * residue mod @p q; the lower half is copied unchanged. May alias.
     */
    void array_additive_inverse_mod_switch(uint64_t *out, const uint64_t *in, uint64_t p,
                                           uint64_t q, uint64_t n);

    /**
     * Circular distance between two residues mod q:
     * `min((a - b) mod q, (b - a) mod q)`.
     */
    uint64_t mod_dist(uint64_t a, uint64_t b, uint64_t q);

    // --- debug -------------------------------------------------------------------

    /** Print `msg: v[0], v[1], ...` to stdout (development helper). */
    void print_array(const char *msg, const uint64_t *v, size_t size);

#ifdef __cplusplus
}
#endif

#endif // VFHE_BASE_H
