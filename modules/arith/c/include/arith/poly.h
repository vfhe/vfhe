// SPDX-License-Identifier: Apache-2.0
/**
 * @file arith/poly.h
 * @brief RNS polynomials: the engine's central data type.
 *
 * An ::rns_poly is an element of its ring's `R_Q = Z_Q[X]/(X^N + 1)`, stored
 * as one length-N residue row per selected limb. Two orthogonal pieces of
 * state travel with the data and are the source of truth for both C and
 * Python:
 *
 *  - **mask** -- which limbs of the ring's pool are active (bit i = limb i).
 *    Binary operations intersect masks; tower operations (arith/tower.h)
 *    shrink or grow them. Rows of inactive limbs hold garbage.
 *  - **domain** -- ::VFHE_COEFF (coefficient form) or ::VFHE_EVAL
 *    (evaluation/NTT form). Every operation declares the domain it needs and
 *    returns ::VFHE_ERR_DOMAIN instead of silently computing garbage when an
 *    operand is in the wrong one. The check is one integer compare per call.
 *
 * Domain requirements at a glance:
 *  - either domain: add, sub, negate, scale, add_scalar, copy, eq, digest
 *  - ::VFHE_EVAL: mul (all variants), inverse, slot operations
 *  - ::VFHE_COEFF: monomial shifts, Galois permutation, gadget decomposition,
 *    Gaussian noise, and everything in arith/tower.h
 *
 * Layout note (split_degree > 1): a row is a `split_degree x poly_size`
 * block matrix. In COEFF domain, coefficient j of the polynomial lives at
 * `row[(j % split_degree) * poly_size + j / split_degree]`; in EVAL domain
 * each block holds its own transform. ::poly_from_int_array performs this
 * interleaving for you.
 */
#ifndef VFHE_ARITH_POLY_H
#define VFHE_ARITH_POLY_H

#include <stdbool.h>
#include <stdint.h>

#include "error.h"
#include "ring.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /** Representation domain of an ::rns_poly. */
    typedef enum vfhe_domain
    {
        VFHE_COEFF = 0, /**< Coefficient form. */
        VFHE_EVAL = 1,  /**< Evaluation (NTT) form. */
    } vfhe_domain;

    /**
     * An RNS polynomial.
     *
     * Owns its residue rows; borrows the ring. All rows live in one aligned
     * backing allocation (`slab`); `limb[i]` is a 64-byte-aligned array of `N`
     * residues mod `ring->limbs[i]->q`, backed for every slot in `[0, alloc_l)`
     * regardless of the mask so the mask can grow later (base extension)
     * without reallocating. Rows carry a 64-byte-aligned stride, so they are
     * contiguous but not necessarily packed for `N < 8`.
     */
    typedef struct rns_poly
    {
        uint64_t **limb;      /**< Residue rows (point into @c slab), one per pool slot. */
        uint64_t *slab;       /**< Single backing allocation for all rows (owned). */
        const ring_ctx *ring; /**< Ambient ring (borrowed). */
        uint64_t mask;        /**< Active-limb bit mask. */
        uint64_t alloc_l;     /**< Number of rows allocated (pool size at creation). */
        vfhe_domain domain;   /**< Current representation domain. */
    } rns_poly;

    typedef rns_poly *rns_poly_t;

    /**
     * A plain integer polynomial (single row of N machine words, no modulus).
     * Used as an exchange format with gadget decomposition and the (future)
     * circuit-compiler bridge.
     */
    typedef struct int_poly
    {
        uint64_t *coeffs; /**< N coefficients. */
        uint64_t N;       /**< Degree bound. */
    } *int_poly_t;

    // --- Lifecycle ---------------------------------------------------------------

    /**
     * Allocate a polynomial over @p ring with the given limb mask.
     *
     * Rows are allocated for the ring's *entire current pool* (headroom for later
     * base extension) but not initialized; the domain starts as ::VFHE_COEFF.
     *
     * @param ring ambient ring
     * @param mask active-limb mask (subset of the pool)
     * @return new polynomial, or NULL on allocation failure
     */
    rns_poly_t poly_new(const ring_ctx *ring, uint64_t mask);

    /** Free a polynomial (rows + struct). NULL is a no-op. */
    void poly_free(void *p);

    /** Allocate @p size polynomials with identical parameters. */
    rns_poly_t *poly_array_new(uint64_t size, const ring_ctx *ring, uint64_t mask);

    /** Free an array from ::poly_array_new (frees each element, then the array). */
    void poly_array_free(uint64_t size, rns_poly_t *p);

    /** Zero every active row. Domain is preserved (zero is zero in both). */
    void poly_zero(rns_poly_t p);

    /** Copy active rows, mask, and domain from @p in to @p out (same ring). */
    void poly_copy(rns_poly_t out, const rns_poly_t in);

    /**
     * Structural equality on the union of the two masks: limbs active in both
     * must match exactly; limbs active in only one must be zero there.
     * Domains are not compared.
     */
    bool poly_eq(const rns_poly_t a, const rns_poly_t b);

    // --- State accessors (also the Python ABI for reading state) ----------------

    /** Active-limb mask. */
    uint64_t poly_mask(const rns_poly_t p);

    /** Current domain (::VFHE_COEFF or ::VFHE_EVAL). */
    int poly_domain(const rns_poly_t p);

    /**
     * Force the domain tag without touching the data.
     *
     * Expert API for raw loads (e.g. writing residues directly through
     * ::poly_limb_data): the caller asserts what the bytes mean. Everything else
     * should rely on operations maintaining the tag.
     */
    void poly_assume_domain(rns_poly_t p, vfhe_domain domain);

    /** Borrowed pointer to limb row @p idx (N residues). No mask check. */
    uint64_t *poly_limb_data(const rns_poly_t p, uint64_t idx);

    // --- Domain conversion -------------------------------------------------------

    /**
     * Convert to evaluation form: out = NTT(in), per active limb and block.
     * `out == in` converts in place; if @p in is already ::VFHE_EVAL this is a
     * copy (or no-op in place).
     * @return ::VFHE_OK
     */
    int poly_to_eval(rns_poly_t out, const rns_poly_t in);

    /**
     * Convert to coefficient form: out = NTT^-1(in). Counterpart of
     * ::poly_to_eval with the same aliasing rules.
     * @return ::VFHE_OK
     */
    int poly_to_coeff(rns_poly_t out, const rns_poly_t in);

    // --- Loading -----------------------------------------------------------------

    /**
     * Load signed coefficients and move to evaluation form.
     *
     * @p in holds N int64 coefficients (as uint64 bit patterns); each is reduced
     * into every active limb, interleaved into the split layout, and transformed.
     * Result is ::VFHE_EVAL.
     * @return ::VFHE_OK
     */
    int poly_from_int_array(rns_poly_t out, const uint64_t *in);

    /**
     * Load per-limb residue rows (values already reduced, natural coefficient
     * order, N per row; `in[i]` indexed by pool slot) and move to evaluation
     * form. Result is ::VFHE_EVAL.
     * @return ::VFHE_OK
     */
    int poly_from_residues(rns_poly_t out, uint64_t *const *in);

    /** ::poly_from_int_array reading from an ::int_poly_t. */
    int poly_from_int_poly(rns_poly_t out, const int_poly_t in);

    // --- Arithmetic (domain-generic) ----------------------------------------------
    // Binary ops require both inputs in the same domain, intersect the masks, and
    // give `out` the inputs' domain. `out` may alias inputs (same ring assumed).

    /** out = a + b. @return ::VFHE_OK or ::VFHE_ERR_DOMAIN. */
    int poly_add(rns_poly_t out, const rns_poly_t a, const rns_poly_t b);

    /** out = a - b. @return ::VFHE_OK or ::VFHE_ERR_DOMAIN. */
    int poly_sub(rns_poly_t out, const rns_poly_t a, const rns_poly_t b);

    /** out = -a (mask and domain copied from a). */
    void poly_negate(rns_poly_t out, const rns_poly_t a);

    /** out = a * s for a machine-word scalar s (reduced per limb). */
    void poly_scale(rns_poly_t out, const rns_poly_t a, uint64_t s);

    /**
     * out += a * s. Requires `out` and `a` in the same domain.
     * @return ::VFHE_OK or ::VFHE_ERR_DOMAIN.
     */
    int poly_scale_addto(rns_poly_t out, const rns_poly_t a, uint64_t s);

    /**
     * out = a * s where s is given per pool slot (`s[idx]` for limb idx) --
     * i.e. multiplication by an RNS-decomposed big scalar.
     */
    void poly_scale_vec(rns_poly_t out, const rns_poly_t a, const uint64_t *s);

    /**
     * out = a + c for a signed integer constant c (two's-complement in @p c).
     * Adds c to the constant coefficient in COEFF domain, or to every evaluation
     * slot in EVAL domain -- both mean "add the constant polynomial c".
     * @return ::VFHE_OK
     */
    int poly_add_scalar(rns_poly_t out, const rns_poly_t a, uint64_t c);

    /**
     * Slot-wise inverse: out[i] = in[i]^-1 in every evaluation slot, via batched
     * Montgomery inversion (one modular inverse per limb).
     *
     * @return ::VFHE_OK; ::VFHE_ERR_DOMAIN if @p in is not ::VFHE_EVAL;
     *         ::VFHE_ERR_UNSUPPORTED if `split_degree != 1`;
     *         ::VFHE_ERR_NOT_INVERTIBLE if any slot is zero (out is unspecified).
     */
    int poly_inverse(rns_poly_t out, const rns_poly_t in);

    // --- Multiplication (::VFHE_EVAL only) ----------------------------------------
    // One kernel, three accumulation modes; see ::vfhe_acc and the ring's
    // multiplication strategy. `out` must not alias `a` or `b`.

    /** out = a * b. @return ::VFHE_OK, ::VFHE_ERR_DOMAIN, or ::VFHE_ERR_ARG on aliasing. */
    int poly_mul(rns_poly_t out, const rns_poly_t a, const rns_poly_t b);

    /** out += a * b (fused; `out` must be ::VFHE_EVAL too). */
    int poly_mul_addto(rns_poly_t out, const rns_poly_t a, const rns_poly_t b);

    /** out -= a * b (fused). */
    int poly_mul_subto(rns_poly_t out, const rns_poly_t a, const rns_poly_t b);

    /** out = out * a, in place. */
    int poly_mul_into(rns_poly_t out, const rns_poly_t a);

    // --- Sampling ----------------------------------------------------------------

    /**
     * Fill active rows with uniform residues and tag the result as @p domain.
     * (A uniform element is uniform in either representation; the caller decides
     * which one this sample *is*.)
     */
    void poly_sample_uniform(rns_poly_t p, vfhe_domain domain);

    /**
     * Sample a discrete-Gaussian-ish polynomial: one rounded normal per
     * coefficient, identical across limbs. Result is ::VFHE_COEFF.
     */
    void poly_sample_gaussian(rns_poly_t p, double sigma);

    /**
     * out = in + e with e Gaussian per coefficient (same noise across limbs).
     * @return ::VFHE_OK or ::VFHE_ERR_DOMAIN if @p in is not ::VFHE_COEFF.
     */
    int poly_add_noise(rns_poly_t out, const rns_poly_t in, double sigma);

    // --- Monomial shifts and Galois action (::VFHE_COEFF only) --------------------

    /**
     * out = in * X^a in the negacyclic ring (a taken mod 2N; wrap negates).
     * Requires `split_degree == 1` and `out != in`.
     * @return ::VFHE_OK, ::VFHE_ERR_DOMAIN, ::VFHE_ERR_UNSUPPORTED, or ::VFHE_ERR_ARG.
     */
    int poly_mul_xai(rns_poly_t out, const rns_poly_t in, uint64_t a);

    /** out = in * (X^a - 1). Same requirements as ::poly_mul_xai. */
    int poly_mul_xai_minus1(rns_poly_t out, const rns_poly_t in, uint64_t a);

    /**
     * Galois automorphism X -> X^gen: out(X) = in(X^gen) in the negacyclic ring.
     * @p gen must be odd, in (0, 2N). Requires `out != in`.
     * @return ::VFHE_OK, ::VFHE_ERR_DOMAIN, or ::VFHE_ERR_ARG.
     */
    int poly_permute(rns_poly_t out, const rns_poly_t in, uint64_t gen);

    // --- Gadget decomposition (::VFHE_COEFF only) ---------------------------------

    /**
     * Extract digit @p level of the base-2^log_base decomposition of the *last
     * active limb's* row, replicated into every active limb of @p out.
     * (Digit-decomposition step for gadget products where the top limb carries
     * the composite value.)
     * @return ::VFHE_OK or ::VFHE_ERR_DOMAIN.
     */
    int poly_decompose_digit(rns_poly_t out, const rns_poly_t in, uint64_t log_base,
                             uint64_t level);

    // --- Evaluation-slot operations (::VFHE_EVAL only) -----------------------------

    /** Broadcast slot @p slot_idx of each block to the whole block. */
    int poly_broadcast_slot(rns_poly_t out, const rns_poly_t in, uint64_t slot_idx);

    /** Broadcast limb row @p src_limb of @p in to every active row of @p out. */
    int poly_broadcast_limb(rns_poly_t out, const rns_poly_t in, uint64_t src_limb);

    /** Cyclically rotate the slots of each block left by @p rot. `out != in`. */
    int poly_rotate_slots(rns_poly_t out, const rns_poly_t in, uint64_t rot);

    /** Copy slot @p src to slot @p dst within each block. */
    int poly_copy_slot(rns_poly_t out, uint64_t dst, const rns_poly_t in, uint64_t src);

// --- Digest ------------------------------------------------------------------

/** BLAKE3 digest size of ::poly_digest, in 64-bit words. */
#define VFHE_POLY_DIGEST_WORDS 4

    /**
     * BLAKE3 hash of the active rows (32 bytes into @p out, as 4 words).
     * Hashes the current representation verbatim; canonicalize the domain first
     * if you need cross-representation stability.
     */
    void poly_digest(uint64_t *out, const rns_poly_t p);

    /** ::poly_digest into a freshly allocated 4-word buffer (caller frees). */
    uint64_t *poly_digest_alloc(const rns_poly_t p);

    // --- Integer polynomials -------------------------------------------------------

    /** Allocate an integer polynomial of degree bound @p N (uninitialized). */
    int_poly_t int_poly_new(uint64_t N);

    /** Free an integer polynomial. */
    void int_poly_free(void *p);

    /** Allocate an array of @p size integer polynomials. */
    int_poly_t *int_poly_array_new(uint64_t size, uint64_t N);

    /** Free an array from ::int_poly_array_new. */
    void int_poly_array_free(uint64_t size, int_poly_t *p);

    /** Cyclic index permutation out[i * gen mod N] = in[i] (no sign handling). */
    void int_poly_permute(int_poly_t out, const int_poly_t in, uint64_t gen);

    /**
     * Signed gadget digit extraction: out[c] = digit @p i of
     * `(in[c] + offset) / 2^(bit_size - (i+1) * Bg_bit)` in base `2^Bg_bit`,
     * where offset centers the rounding (`2^(bit_size - l*Bg_bit - 1)`).
     */
    void int_poly_decompose_digit(int_poly_t out, const int_poly_t in, uint64_t Bg_bit, uint64_t l,
                                  uint64_t bit_size, uint64_t i);

#ifdef __cplusplus
}
#endif

#endif // VFHE_ARITH_POLY_H
