// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include <arith_generic.h>

#include "arith_dispatch.h"

// The generic half of the interface: capability answers, and the dispatch
// that routes an operation to the ring's implementation after checking the
// domain rules every implementation shares.
//
// Dispatch is a switch on the ring's tag with the method table as its
// default. The switch arms are direct calls, which LTO inlines across
// translation units at this build's -O3 -- a function pointer it cannot,
// once a second table exists -- so compiled-in implementations cost nothing
// at the boundary, and a ring the switch does not know (a backend loaded at
// runtime) still works through its table. Either way it is one dispatch per
// whole-element operation; nothing here runs per coefficient.

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

// Route one operation: the tagged implementations by direct call, everything
// else through the method table, where a NULL slot answers ARITH_UNIMPLEMENTED
// rather than being dereferenced. A statement expression (gnu11), so wrappers
// that stamp the result's domain afterwards can use the status.
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
    return ARITH_DISPATCH(ring, copy, ring, out, in);
}

ArithStatus arith_zero(ArithRing ring, ArithElement *out)
{
    return ARITH_DISPATCH(ring, zero, ring, out);
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

ArithStatus arith_to_mul(ArithRing ring, ArithElement *element)
{
    if (element->domain == arith_mul_domain(ring))
    {
        return ARITH_OK;
    }
    return ARITH_DISPATCH(ring, to_mul, ring, element);
}

ArithStatus arith_to_canonical(ArithRing ring, ArithElement *element)
{
    if (element->domain == ARITH_DOMAIN_CANONICAL)
    {
        return ARITH_OK;
    }
    return ARITH_DISPATCH(ring, to_canonical, ring, element);
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
    status = ARITH_DISPATCH(ring, add, ring, out, a, b);
    out->domain = a->domain;
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
    status = ARITH_DISPATCH(ring, sub, ring, out, a, b);
    out->domain = a->domain;
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
    status = ARITH_DISPATCH(ring, mul, ring, out, a, b);
    out->domain = arith_mul_domain(ring);
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
    return ARITH_DISPATCH(ring, mul_addto, ring, out, a, b);
}

ArithStatus arith_scale_int(ArithRing ring, ArithElement *out, const ArithElement *a,
                            uint64_t scale)
{
    if (a->domain == ARITH_DOMAIN_EMPTY)
    {
        return ARITH_BAD_DOMAIN;
    }
    ArithStatus status = ARITH_DISPATCH(ring, scale_int, ring, out, a, scale);
    out->domain = a->domain;
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
    return ARITH_DISPATCH(ring, mul_subto, ring, out, a, b);
}

ArithStatus arith_scale_addto(ArithRing ring, ArithElement *out, const ArithElement *a,
                              uint64_t scale)
{
    ArithStatus status = check_same_domain(out, a);
    if (status != ARITH_OK)
    {
        return status;
    }
    return ARITH_DISPATCH(ring, scale_addto, ring, out, a, scale);
}

ArithStatus arith_scale_by(ArithRing ring, ArithElement *out, const ArithElement *a,
                           ArithScalar scale)
{
    if (a->domain == ARITH_DOMAIN_EMPTY)
    {
        return ARITH_BAD_DOMAIN;
    }
    ArithStatus status = ARITH_DISPATCH(ring, scale_by, ring, out, a, scale);
    out->domain = a->domain;
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
    ArithStatus status = ARITH_DISPATCH(ring, permute, ring, out, a, gen);
    out->domain = ARITH_DOMAIN_CANONICAL;
    return status;
}

ArithStatus arith_mul_by_monomial(ArithRing ring, ArithElement *out, const ArithElement *a,
                                  uint64_t power, int minus_one)
{
    if (a->domain != ARITH_DOMAIN_CANONICAL)
    {
        return ARITH_BAD_DOMAIN;
    }
    ArithStatus status = ARITH_DISPATCH(ring, mul_by_monomial, ring, out, a, power, minus_one);
    out->domain = ARITH_DOMAIN_CANONICAL;
    return status;
}

ArithStatus arith_sample_uniform(ArithRing ring, ArithElement *out)
{
    ArithStatus status = ARITH_DISPATCH(ring, sample_uniform, ring, out);
    out->domain = ARITH_DOMAIN_CANONICAL;
    return status;
}

ArithStatus arith_sample_gaussian(ArithRing ring, ArithElement *out, double sigma)
{
    ArithStatus status = ARITH_DISPATCH(ring, sample_gaussian, ring, out, sigma);
    out->domain = ARITH_DOMAIN_CANONICAL;
    return status;
}

ArithStatus arith_from_int_array(ArithRing ring, ArithElement *out, const uint64_t *values,
                                 uint64_t count)
{
    // The implementation reports the domain it loaded into; overriding it here
    // would label the element with a representation it is not in.
    return ARITH_DISPATCH(ring, from_int_array, ring, out, values, count);
}

// Tower moves are defined on the canonical form: they divide and round the
// value, which the mul domain does not represent coefficient-wise.
ArithStatus arith_round_division(ArithRing ring, ArithElement *element, ArithRing to)
{
    if (element->domain != ARITH_DOMAIN_CANONICAL)
    {
        return ARITH_BAD_DOMAIN;
    }
    return ARITH_DISPATCH(ring, round_division, ring, element, to);
}

ArithStatus arith_mod_reduce_lifted(ArithRing ring, ArithElement *out, const ArithElement *a,
                                    ArithRing from)
{
    if (a->domain != ARITH_DOMAIN_CANONICAL)
    {
        return ARITH_BAD_DOMAIN;
    }
    ArithStatus status = ARITH_DISPATCH(ring, mod_reduce_lifted, ring, out, a, from);
    out->domain = ARITH_DOMAIN_CANONICAL;
    return status;
}

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
