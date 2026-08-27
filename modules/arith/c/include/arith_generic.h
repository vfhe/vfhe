// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#ifndef VFHE_ARITH_GENERIC_H
#define VFHE_ARITH_GENERIC_H

#include <arith.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // Arithmetic over a ring whose representation the caller does not name.
    //
    // A module that only needs ring operations takes an ArithRing and calls
    // the arith_* functions below; which implementation runs is a property of
    // the ring it was handed. A module that needs the representation itself --
    // per-coefficient access, per-prime kernels -- does not get a finer
    // interface here, it keeps that code in its own per-implementation file.
    // So every entry point below acts on whole elements: one dispatch per
    // O(N) or O(N log N) of work, and none inside a loop over coefficients.

    // Which representation an element currently holds.
    //
    // ARITH_DOMAIN_CANONICAL is a canonical form: one representative per ring
    // element, so equality, hashing and serialization are meaningful only
    // there. ARITH_DOMAIN_MUL is the image under an injective ring
    // homomorphism chosen to make multiplication cheap -- the NTT/CRT
    // decomposition for RNS, an evaluation or Montgomery form elsewhere. It
    // need not be canonical, and for some implementations the two coincide
    // (ARITH_CAP_DOMAINS_COINCIDE), in which case conversion is a no-op and
    // every element reports ARITH_DOMAIN_CANONICAL.
    typedef enum
    {
        ARITH_DOMAIN_EMPTY = 0,
        ARITH_DOMAIN_CANONICAL = 1,
        ARITH_DOMAIN_MUL = 2,
    } ArithDomain;

    // Groups of operations an implementation carries. The bits match the
    // Python Capability flag, so a value crosses the boundary unchanged.
    typedef enum
    {
        ARITH_CAP_CORE = 1u << 0,
        ARITH_CAP_QUOTIENT_POLY_RING = 1u << 1,
        ARITH_CAP_TOWER = 1u << 2,
        ARITH_CAP_SAMPLING = 1u << 3,
        ARITH_CAP_EXACT = 1u << 4,
        ARITH_CAP_DOMAINS_COINCIDE = 1u << 5,
    } ArithCapability;

    // What every slot returns. A caller that ignores the result of an
    // operation an implementation does not provide computes nothing and is
    // not told, so check it wherever the ring is not statically known.
    typedef enum
    {
        ARITH_OK = 0,
        ARITH_UNIMPLEMENTED = 1, // no such slot for this implementation
        ARITH_BAD_DOMAIN = 2,    // operands are not in the domain required
    } ArithStatus;

    typedef struct _ArithRing *ArithRing;

    // Which implementation a ring is, as a tag the dispatcher switches on so
    // the call to it is direct. DYNAMIC is the zero value and means the
    // switch has no arm for this ring: it dispatches through the method table
    // instead, which is how a backend loaded at runtime plugs in. A ring must
    // carry the tag matching the table it was built with.
    typedef enum
    {
        ARITH_IMPL_DYNAMIC = 0,
        ARITH_IMPL_RNS = 1,
    } ArithImpl;

    // An element handle. The implementation owns the storage behind `handle`;
    // `domain` is the caller-visible half, because generic code must know
    // which domain an element is in to route an operation.
    //
    // A domain conversion may replace the storage: RNS converts in place, an
    // FFT backend's mul domain is a different type entirely, and a device
    // backend's may not be host memory at all. Never hold a pointer obtained
    // from an element across a conversion.
    typedef struct
    {
        void *handle;
        ArithDomain domain;
    } ArithElement;

    // A ring element used as a multiplier, in the implementation's own form:
    // one residue per prime for RNS, a single value where the ring is not a
    // product. Built and released through arith_scalar_*, never by hand.
    typedef struct
    {
        void *handle;
    } ArithScalar;

    // One table per (implementation, backend), built once and shared by every
    // ring of that kind. A slot left NULL answers ARITH_UNIMPLEMENTED.
    typedef struct
    {
        const char *implementation;
        const char *backend;
        uint32_t capabilities;

        // lifecycle
        ArithStatus (*new_element)(ArithRing ring, ArithElement *out);
        ArithStatus (*new_like)(ArithRing ring, const ArithElement *model, ArithElement *out);
        void (*free_element)(ArithRing ring, ArithElement *element);
        ArithStatus (*copy)(ArithRing ring, ArithElement *out, const ArithElement *in);
        ArithStatus (*zero)(ArithRing ring, ArithElement *out);

        // domain movement
        ArithStatus (*to_mul)(ArithRing ring, ArithElement *element);
        ArithStatus (*to_canonical)(ArithRing ring, ArithElement *element);

        // arithmetic; operands share a domain, and mul needs the mul domain
        ArithStatus (*add)(ArithRing ring, ArithElement *out, const ArithElement *a,
                           const ArithElement *b);
        ArithStatus (*sub)(ArithRing ring, ArithElement *out, const ArithElement *a,
                           const ArithElement *b);
        ArithStatus (*mul)(ArithRing ring, ArithElement *out, const ArithElement *a,
                           const ArithElement *b);
        ArithStatus (*mul_addto)(ArithRing ring, ArithElement *out, const ArithElement *a,
                                 const ArithElement *b);
        ArithStatus (*scale_int)(ArithRing ring, ArithElement *out, const ArithElement *a,
                                 uint64_t scale);
        ArithStatus (*mul_subto)(ArithRing ring, ArithElement *out, const ArithElement *a,
                                 const ArithElement *b);
        ArithStatus (*scale_addto)(ArithRing ring, ArithElement *out, const ArithElement *a,
                                   uint64_t scale);
        ArithStatus (*scale_by)(ArithRing ring, ArithElement *out, const ArithElement *a,
                                ArithScalar scale);

        // quotient-polynomial-ring group (ARITH_CAP_QUOTIENT_POLY_RING)
        ArithStatus (*permute)(ArithRing ring, ArithElement *out, const ArithElement *a,
                               uint64_t gen);
        // out = a * X^power; `minus_one` subtracts a as well, which is the
        // fused form the gadget decomposition wants.
        ArithStatus (*mul_by_monomial)(ArithRing ring, ArithElement *out, const ArithElement *a,
                                       uint64_t power, int minus_one);

        // sampling group (ARITH_CAP_SAMPLING)
        ArithStatus (*sample_uniform)(ArithRing ring, ArithElement *out);
        ArithStatus (*sample_gaussian)(ArithRing ring, ArithElement *out, double sigma);
        // Load `count` integers into an element -- the implementation-neutral
        // way in, since an integer array means the same thing to every
        // representation. The resulting domain is whichever one the
        // implementation loads into (RNS fuses the forward transform, so it
        // lands in the mul domain); it is recorded in the element, and a
        // caller wanting a particular domain converts, which costs nothing
        // when it is already there.
        ArithStatus (*from_int_array)(ArithRing ring, ArithElement *out, const uint64_t *values,
                                      uint64_t count);

        // tower group (ARITH_CAP_TOWER); `to` is the destination ring, never a
        // set of primes: which primes leave is the implementation's business.
        ArithStatus (*round_division)(ArithRing ring, ArithElement *element, ArithRing to);
        ArithStatus (*mod_reduce_lifted)(ArithRing ring, ArithElement *out, const ArithElement *a,
                                         ArithRing from);

        // scalars
        ArithStatus (*scalar_new)(ArithRing ring, const uint64_t *per_component, ArithScalar *out);
        void (*scalar_free)(ArithRing ring, ArithScalar *scalar);
    } ArithMethods;

    // A ring instance: its implementation tag, its method table, and the
    // implementation's own parameters behind `params`. The tag is the hot
    // dispatch path; the table answers introspection (names, capabilities)
    // and serves rings the switch does not know.
    struct _ArithRing
    {
        const ArithMethods *vt;
        void *params;
        uint64_t N;
        ArithImpl impl;
    };

    // --- the generic interface modules call ---------------------------------

    const char *arith_implementation(ArithRing ring);
    const char *arith_backend(ArithRing ring);
    uint32_t arith_capabilities(ArithRing ring);
    // Whether every group in `capability` is present. The check a caller makes
    // before an operation the implementation may not define at all.
    int arith_supports(ArithRing ring, uint32_t capability);
    // The domain multiplication needs its operands in: ARITH_DOMAIN_MUL, or
    // ARITH_DOMAIN_CANONICAL where the two coincide.
    ArithDomain arith_mul_domain(ArithRing ring);

    ArithStatus arith_new(ArithRing ring, ArithElement *out);
    ArithStatus arith_new_like(ArithRing ring, const ArithElement *model, ArithElement *out);
    void arith_free(ArithRing ring, ArithElement *element);
    ArithStatus arith_copy(ArithRing ring, ArithElement *out, const ArithElement *in);
    ArithStatus arith_zero(ArithRing ring, ArithElement *out);
    // Zero, labelled as being in `domain`. A ring homomorphism sends 0 to 0,
    // so the zero element is the same in both domains and may be claimed as
    // either without converting.
    ArithStatus arith_zero_in(ArithRing ring, ArithElement *out, ArithDomain domain);

    ArithStatus arith_to_mul(ArithRing ring, ArithElement *element);
    ArithStatus arith_to_canonical(ArithRing ring, ArithElement *element);

    ArithStatus arith_add(ArithRing ring, ArithElement *out, const ArithElement *a,
                          const ArithElement *b);
    ArithStatus arith_sub(ArithRing ring, ArithElement *out, const ArithElement *a,
                          const ArithElement *b);
    ArithStatus arith_mul(ArithRing ring, ArithElement *out, const ArithElement *a,
                          const ArithElement *b);
    ArithStatus arith_mul_addto(ArithRing ring, ArithElement *out, const ArithElement *a,
                                const ArithElement *b);
    ArithStatus arith_scale_int(ArithRing ring, ArithElement *out, const ArithElement *a,
                                uint64_t scale);
    ArithStatus arith_mul_subto(ArithRing ring, ArithElement *out, const ArithElement *a,
                                const ArithElement *b);
    ArithStatus arith_scale_addto(ArithRing ring, ArithElement *out, const ArithElement *a,
                                  uint64_t scale);
    ArithStatus arith_scale_by(ArithRing ring, ArithElement *out, const ArithElement *a,
                               ArithScalar scale);

    ArithStatus arith_permute(ArithRing ring, ArithElement *out, const ArithElement *a,
                              uint64_t gen);
    ArithStatus arith_mul_by_monomial(ArithRing ring, ArithElement *out, const ArithElement *a,
                                      uint64_t power, int minus_one);

    ArithStatus arith_sample_uniform(ArithRing ring, ArithElement *out);
    ArithStatus arith_sample_gaussian(ArithRing ring, ArithElement *out, double sigma);
    ArithStatus arith_from_int_array(ArithRing ring, ArithElement *out, const uint64_t *values,
                                     uint64_t count);

    ArithStatus arith_round_division(ArithRing ring, ArithElement *element, ArithRing to);
    ArithStatus arith_mod_reduce_lifted(ArithRing ring, ArithElement *out, const ArithElement *a,
                                        ArithRing from);

    ArithStatus arith_scalar_new(ArithRing ring, const uint64_t *per_component, ArithScalar *out);
    void arith_scalar_free(ArithRing ring, ArithScalar *scalar);

    // --- the RNS implementation ---------------------------------------------

    // A ring over `base` holding the primes `rns_mask` selects. The returned
    // ring borrows the base, which outlives every ring built on it, and must
    // be released with arith_ring_free.
    ArithRing arith_rns_ring_new(uint64_t N, uint64_t rns_mask, RNS_Base base);
    // The shared ring for these parameters, created on first ask and kept for
    // the life of the process. Prefer this to arith_rns_ring_new wherever the
    // ring outlives the call, so no caller has to own it.
    ArithRing arith_rns_ring_get(uint64_t N, uint64_t rns_mask, RNS_Base base);
    void arith_rns_ring_cache_clear(void);
    void arith_ring_free(ArithRing ring);
    // The RNS_Polynomial behind an element of an RNS ring, for the
    // per-implementation code that needs the representation itself. Inline
    // because it is a cast: a backend file uses it in loops and should not
    // depend on the linker inlining a call away.
    static inline RNS_Polynomial arith_rns_polynomial(const ArithElement *element)
    {
        return (RNS_Polynomial)element->handle;
    }

#ifdef __cplusplus
}
#endif

#endif // VFHE_ARITH_GENERIC_H
