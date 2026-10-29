// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
#include <arith_generic.h>
#include <arith_trace.h>

#include "arith_dispatch.h"

// The generic half of the interface: capability answers, and the dispatch
// that routes an operation to the ring's implementation after checking the
// domain rules every implementation shares.
//
// Dispatch is a switch on the ring's tag, with the method table as the
// default arm. Both routes must compute the same thing: a ring reached
// either way is the same ring. One dispatch per whole-element operation;
// nothing here runs per coefficient.
//
// The execution tape (arith_trace.h) is told of each operation: opened before
// it runs, so an operand an in-place operation overwrites is copied first, and
// closed with the status after. Off, that is one load and one branch here.

#define TRACE_BEGIN(kind, out, a, b, param, param2, sigma, scalar)                                 \
    const uint64_t traced_ =                                                                       \
        ARITH_TRACE_ON() ? arith_trace_begin(kind, ring, out, a, b, param, param2, sigma, scalar)  \
                         : 0
#define TRACE_END(status, out)                                                                     \
    do                                                                                             \
    {                                                                                              \
        if (traced_ != 0)                                                                          \
            arith_trace_end(traced_, out, status);                                                 \
    } while (0)

const char *arith_implementation(ArithRing ring) { return ring->vt->implementation; }

const char *arith_backend(ArithRing ring) { return ring->vt->backend; }

uint32_t arith_capabilities(ArithRing ring) { return ring->vt->capabilities; }

int arith_supports(ArithRing ring, uint32_t capability)
{
    return (ring->vt->capabilities & capability) == capability;
}

ArithDomain arith_mul_domain(ArithRing ring)
{
    if (arith_supports(ring, ARITH_CAP_DOMAINS_COINCIDE))
    {
        return ARITH_DOMAIN_CANONICAL;
    }
    return ARITH_DOMAIN_MUL;
}

// Route one operation: a tagged implementation by direct call, anything else
// through the method table, where a NULL slot answers ARITH_UNIMPLEMENTED
// rather than being dereferenced. A statement expression (gnu11) so that
// wrappers which stamp the result's domain afterwards can read the status.
#define ARITH_DISPATCH(ring, slot, ...)                                                            \
    __extension__({                                                                                \
        ArithStatus dispatched_;                                                                   \
        switch ((ring)->impl)                                                                      \
        {                                                                                          \
        case ARITH_IMPL_RNS:                                                                       \
            dispatched_ = arith_rns_##slot(__VA_ARGS__);                                           \
            break;                                                                                 \
        default:                                                                                   \
            dispatched_ =                                                                          \
                (ring)->vt->slot == NULL ? ARITH_UNIMPLEMENTED : (ring)->vt->slot(__VA_ARGS__);    \
            break;                                                                                 \
        }                                                                                          \
        dispatched_;                                                                               \
    })

ArithStatus arith_new(ArithRing ring, ArithElement *out)
{
    return ARITH_DISPATCH(ring, new_element, ring, out);
}

ArithStatus arith_new_like(ArithRing ring, const ArithElement *model, ArithElement *out)
{
    return ARITH_DISPATCH(ring, new_like, ring, model, out);
}

void arith_free(ArithRing ring, ArithElement *element)
{
    switch (ring->impl)
    {
    case ARITH_IMPL_RNS:
        arith_rns_free_element(ring, element);
        break;
    default:
        if (ring->vt->free_element != NULL)
        {
            ring->vt->free_element(ring, element);
        }
        break;
    }
}

ArithStatus arith_copy(ArithRing ring, ArithElement *out, const ArithElement *in)
{
    TRACE_BEGIN(ARITH_TRACE_COPY, out, in, NULL, 0, 0, 0.0, NULL);
    ArithStatus status = ARITH_DISPATCH(ring, copy, ring, out, in);
    TRACE_END(status, out);
    return status;
}

ArithStatus arith_zero(ArithRing ring, ArithElement *out)
{
    TRACE_BEGIN(ARITH_TRACE_ZERO, out, NULL, NULL, 0, 0, 0.0, NULL);
    ArithStatus status = ARITH_DISPATCH(ring, zero, ring, out);
    TRACE_END(status, out);
    return status;
}

ArithStatus arith_zero_in(ArithRing ring, ArithElement *out, ArithDomain domain)
{
    ArithStatus status = arith_zero(ring, out);
    if (status == ARITH_OK)
    {
        out->domain = domain;
    }
    return status;
}

// A conversion is not an event, but it rewrites the storage, which the
// tape must know when it verifies values against their versions.
ArithStatus arith_to_mul(ArithRing ring, ArithElement *element)
{
    if (element->domain == arith_mul_domain(ring))
    {
        return ARITH_OK;
    }
    ArithStatus status = ARITH_DISPATCH(ring, to_mul, ring, element);
    if (ARITH_TRACE_ON() && status == ARITH_OK)
        arith_trace_converted(element);
    return status;
}

ArithStatus arith_to_canonical(ArithRing ring, ArithElement *element)
{
    if (element->domain == ARITH_DOMAIN_CANONICAL)
    {
        return ARITH_OK;
    }
    ArithStatus status = ARITH_DISPATCH(ring, to_canonical, ring, element);
    if (ARITH_TRACE_ON() && status == ARITH_OK)
        arith_trace_converted(element);
    return status;
}

// Addition and subtraction are defined in either domain, but only between
// operands that are in the same one: the two representations of a value are
// different bit patterns, and adding across them is meaningless.
static ArithStatus check_same_domain(const ArithElement *a, const ArithElement *b)
{
    if (a->domain != b->domain || a->domain == ARITH_DOMAIN_EMPTY)
    {
        return ARITH_BAD_DOMAIN;
    }
    return ARITH_OK;
}

ArithStatus arith_add(ArithRing ring, ArithElement *out, const ArithElement *a,
                      const ArithElement *b)
{
    ArithStatus status = check_same_domain(a, b);
    if (status != ARITH_OK)
    {
        return status;
    }
    TRACE_BEGIN(ARITH_TRACE_ADD, out, a, b, 0, 0, 0.0, NULL);
    status = ARITH_DISPATCH(ring, add, ring, out, a, b);
    out->domain = a->domain;
    TRACE_END(status, out);
    return status;
}

ArithStatus arith_sub(ArithRing ring, ArithElement *out, const ArithElement *a,
                      const ArithElement *b)
{
    ArithStatus status = check_same_domain(a, b);
    if (status != ARITH_OK)
    {
        return status;
    }
    TRACE_BEGIN(ARITH_TRACE_SUB, out, a, b, 0, 0, 0.0, NULL);
    status = ARITH_DISPATCH(ring, sub, ring, out, a, b);
    out->domain = a->domain;
    TRACE_END(status, out);
    return status;
}

// Multiplication needs both operands in the mul domain. Converting them here
// would hide the cost of a transform inside an operator and, worse, leave the
// caller's operands in a domain it did not choose; the caller converts.
static ArithStatus check_mul_domain(ArithRing ring, const ArithElement *a, const ArithElement *b)
{
    ArithDomain required = arith_mul_domain(ring);
    if (a->domain != required || b->domain != required)
    {
        return ARITH_BAD_DOMAIN;
    }
    return ARITH_OK;
}

ArithStatus arith_mul(ArithRing ring, ArithElement *out, const ArithElement *a,
                      const ArithElement *b)
{
    ArithStatus status = check_mul_domain(ring, a, b);
    if (status != ARITH_OK)
    {
        return status;
    }
    TRACE_BEGIN(ARITH_TRACE_MUL, out, a, b, 0, 0, 0.0, NULL);
    status = ARITH_DISPATCH(ring, mul, ring, out, a, b);
    out->domain = arith_mul_domain(ring);
    TRACE_END(status, out);
    return status;
}

ArithStatus arith_mul_addto(ArithRing ring, ArithElement *out, const ArithElement *a,
                            const ArithElement *b)
{
    ArithStatus status = check_mul_domain(ring, a, b);
    if (status != ARITH_OK)
    {
        return status;
    }
    if (out->domain != arith_mul_domain(ring))
    {
        return ARITH_BAD_DOMAIN;
    }
    TRACE_BEGIN(ARITH_TRACE_MUL_ADDTO, out, a, b, 0, 0, 0.0, NULL);
    status = ARITH_DISPATCH(ring, mul_addto, ring, out, a, b);
    TRACE_END(status, out);
    return status;
}

ArithStatus arith_scale_int(ArithRing ring, ArithElement *out, const ArithElement *a,
                            uint64_t scale)
{
    if (a->domain == ARITH_DOMAIN_EMPTY)
    {
        return ARITH_BAD_DOMAIN;
    }
    TRACE_BEGIN(ARITH_TRACE_SCALE_INT, out, a, NULL, scale, 0, 0.0, NULL);
    ArithStatus status = ARITH_DISPATCH(ring, scale_int, ring, out, a, scale);
    out->domain = a->domain;
    TRACE_END(status, out);
    return status;
}

// --- the slots added for the consumer modules ---------------------------

ArithStatus arith_mul_subto(ArithRing ring, ArithElement *out, const ArithElement *a,
                            const ArithElement *b)
{
    ArithStatus status = check_mul_domain(ring, a, b);
    if (status != ARITH_OK)
    {
        return status;
    }
    if (out->domain != arith_mul_domain(ring))
    {
        return ARITH_BAD_DOMAIN;
    }
    TRACE_BEGIN(ARITH_TRACE_MUL_SUBTO, out, a, b, 0, 0, 0.0, NULL);
    status = ARITH_DISPATCH(ring, mul_subto, ring, out, a, b);
    TRACE_END(status, out);
    return status;
}

ArithStatus arith_scale_addto(ArithRing ring, ArithElement *out, const ArithElement *a,
                              uint64_t scale)
{
    ArithStatus status = check_same_domain(out, a);
    if (status != ARITH_OK)
    {
        return status;
    }
    TRACE_BEGIN(ARITH_TRACE_SCALE_ADDTO, out, a, NULL, scale, 0, 0.0, NULL);
    status = ARITH_DISPATCH(ring, scale_addto, ring, out, a, scale);
    TRACE_END(status, out);
    return status;
}

ArithStatus arith_scale_by(ArithRing ring, ArithElement *out, const ArithElement *a,
                           ArithScalar scale)
{
    if (a->domain == ARITH_DOMAIN_EMPTY)
    {
        return ARITH_BAD_DOMAIN;
    }
    TRACE_BEGIN(ARITH_TRACE_SCALE_BY, out, a, NULL, 0, 0, 0.0, &scale);
    ArithStatus status = ARITH_DISPATCH(ring, scale_by, ring, out, a, scale);
    out->domain = a->domain;
    TRACE_END(status, out);
    return status;
}

// A Galois automorphism permutes coefficients, so it is defined on the
// canonical form; in the mul domain the same map is a permutation of a
// different index set and is not this operation.
ArithStatus arith_permute(ArithRing ring, ArithElement *out, const ArithElement *a, uint64_t gen)
{
    if (a->domain != ARITH_DOMAIN_CANONICAL)
    {
        return ARITH_BAD_DOMAIN;
    }
    TRACE_BEGIN(ARITH_TRACE_PERMUTE, out, a, NULL, gen, 0, 0.0, NULL);
    ArithStatus status = ARITH_DISPATCH(ring, permute, ring, out, a, gen);
    out->domain = ARITH_DOMAIN_CANONICAL;
    TRACE_END(status, out);
    return status;
}

ArithStatus arith_mul_by_monomial(ArithRing ring, ArithElement *out, const ArithElement *a,
                                  uint64_t power, int minus_one)
{
    if (a->domain != ARITH_DOMAIN_CANONICAL)
    {
        return ARITH_BAD_DOMAIN;
    }
    TRACE_BEGIN(ARITH_TRACE_MUL_BY_MONOMIAL, out, a, NULL, power, (uint64_t)minus_one, 0.0, NULL);
    ArithStatus status = ARITH_DISPATCH(ring, mul_by_monomial, ring, out, a, power, minus_one);
    out->domain = ARITH_DOMAIN_CANONICAL;
    TRACE_END(status, out);
    return status;
}

ArithStatus arith_sample_uniform(ArithRing ring, ArithElement *out)
{
    TRACE_BEGIN(ARITH_TRACE_SAMPLE_UNIFORM, out, NULL, NULL, 0, 0, 0.0, NULL);
    ArithStatus status = ARITH_DISPATCH(ring, sample_uniform, ring, out);
    out->domain = ARITH_DOMAIN_CANONICAL;
    TRACE_END(status, out);
    return status;
}

ArithStatus arith_sample_gaussian(ArithRing ring, ArithElement *out, double sigma)
{
    TRACE_BEGIN(ARITH_TRACE_SAMPLE_GAUSSIAN, out, NULL, NULL, 0, 0, sigma, NULL);
    ArithStatus status = ARITH_DISPATCH(ring, sample_gaussian, ring, out, sigma);
    out->domain = ARITH_DOMAIN_CANONICAL;
    TRACE_END(status, out);
    return status;
}

ArithStatus arith_sample_uniform_seeded(ArithRing ring, ArithElement *out, const uint8_t *seed,
                                        uint64_t seed_len, uint64_t stream)
{
    TRACE_BEGIN(ARITH_TRACE_SAMPLE_UNIFORM_SEEDED, out, NULL, NULL, stream, 0, 0.0, NULL);
    ArithStatus status =
        ARITH_DISPATCH(ring, sample_uniform_seeded, ring, out, seed, seed_len, stream);
    out->domain = arith_mul_domain(ring);
    TRACE_END(status, out);
    return status;
}

ArithStatus arith_from_int_array(ArithRing ring, ArithElement *out, const uint64_t *values,
                                 uint64_t count)
{
    // The implementation reports the domain it loaded into; overriding it here
    // would label the element with a representation it is not in.
    TRACE_BEGIN(ARITH_TRACE_FROM_INT_ARRAY, out, NULL, NULL, count, 0, 0.0, NULL);
    ArithStatus status = ARITH_DISPATCH(ring, from_int_array, ring, out, values, count);
    TRACE_END(status, out);
    return status;
}

// Tower moves are defined on the canonical form: they divide and round the
// value, which the mul domain does not represent coefficient-wise.
ArithStatus arith_round_division(ArithRing ring, ArithElement *element, ArithRing to)
{
    if (element->domain != ARITH_DOMAIN_CANONICAL)
    {
        return ARITH_BAD_DOMAIN;
    }
    TRACE_BEGIN(ARITH_TRACE_ROUND_DIVISION, element, NULL, NULL, 0, 0, 0.0, NULL);
    ArithStatus status = ARITH_DISPATCH(ring, round_division, ring, element, to);
    TRACE_END(status, element);
    return status;
}

ArithStatus arith_round_division_approximate(ArithRing ring, ArithElement *element, ArithRing to)
{
    if (element->domain != ARITH_DOMAIN_CANONICAL)
    {
        return ARITH_BAD_DOMAIN;
    }
    TRACE_BEGIN(ARITH_TRACE_ROUND_DIVISION_APPROXIMATE, element, NULL, NULL, 0, 0, 0.0, NULL);
    ArithStatus status = ARITH_DISPATCH(ring, round_division_approximate, ring, element, to);
    TRACE_END(status, element);
    return status;
}

ArithStatus arith_decompose_digit(ArithRing ring, ArithElement *out, const ArithElement *a,
                                  uint64_t primes, uint64_t log_base, uint64_t level,
                                  uint64_t dropped, int centered)
{
    if (a->domain != ARITH_DOMAIN_CANONICAL)
    {
        return ARITH_BAD_DOMAIN;
    }
    const uint64_t traced_ =
        ARITH_TRACE_ON()
            ? arith_trace_begin_digit(ring, out, a, primes, log_base, level, dropped, centered)
            : 0;
    ArithStatus status = ARITH_DISPATCH(ring, decompose_digit, ring, out, a, primes, log_base,
                                        level, dropped, centered);
    out->domain = ARITH_DOMAIN_CANONICAL;
    TRACE_END(status, out);
    return status;
}

ArithStatus arith_mod_reduce_lifted(ArithRing ring, ArithElement *out, const ArithElement *a,
                                    ArithRing from)
{
    if (a->domain != ARITH_DOMAIN_CANONICAL)
    {
        return ARITH_BAD_DOMAIN;
    }
    TRACE_BEGIN(ARITH_TRACE_MOD_REDUCE_LIFTED, out, a, NULL, 0, 0, 0.0, NULL);
    ArithStatus status = ARITH_DISPATCH(ring, mod_reduce_lifted, ring, out, a, from);
    out->domain = ARITH_DOMAIN_CANONICAL;
    TRACE_END(status, out);
    return status;
}

int arith_generic_only_state = 0;

void arith_set_generic_only(int enabled) { arith_generic_only_state = enabled != 0; }

int arith_generic_only(void) { return arith_generic_only_state; }

ArithStatus arith_scalar_new(ArithRing ring, const uint64_t *per_component, ArithScalar *out)
{
    return ARITH_DISPATCH(ring, scalar_new, ring, per_component, out);
}

void arith_scalar_free(ArithRing ring, ArithScalar *scalar)
{
    switch (ring->impl)
    {
    case ARITH_IMPL_RNS:
        arith_rns_scalar_free(ring, scalar);
        break;
    default:
        if (ring->vt->scalar_free != NULL)
        {
            ring->vt->scalar_free(ring, scalar);
        }
        break;
    }
}
