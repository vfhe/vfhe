// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include <arith_generic.h>

// The generic half of the interface: capability answers, and the dispatch
// that routes an operation to the ring's method table after checking the
// domain rules every implementation shares.
//
// One indirect call per operation, each of which is at least O(N) of work in
// the implementation. Nothing here runs per coefficient.

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

// A slot the implementation left out computes nothing; say so rather than
// dereferencing NULL, so a caller that checks gets a usable answer and one
// that does not gets a crash at the call rather than corrupt data later.
#define ARITH_REQUIRE_SLOT(ring, slot)                                                             \
    do                                                                                             \
    {                                                                                              \
        if ((ring)->vt->slot == NULL)                                                              \
        {                                                                                          \
            return ARITH_UNIMPLEMENTED;                                                            \
        }                                                                                          \
    } while (0)

ArithStatus arith_new(ArithRing ring, ArithElement *out)
{
    ARITH_REQUIRE_SLOT(ring, new_element);
    return ring->vt->new_element(ring, out);
}

ArithStatus arith_new_like(ArithRing ring, const ArithElement *model, ArithElement *out)
{
    ARITH_REQUIRE_SLOT(ring, new_like);
    return ring->vt->new_like(ring, model, out);
}

void arith_free(ArithRing ring, ArithElement *element)
{
    if (ring->vt->free_element != NULL)
    {
        ring->vt->free_element(ring, element);
    }
}

ArithStatus arith_copy(ArithRing ring, ArithElement *out, const ArithElement *in)
{
    ARITH_REQUIRE_SLOT(ring, copy);
    return ring->vt->copy(ring, out, in);
}

ArithStatus arith_zero(ArithRing ring, ArithElement *out)
{
    ARITH_REQUIRE_SLOT(ring, zero);
    return ring->vt->zero(ring, out);
}

ArithStatus arith_to_mul(ArithRing ring, ArithElement *element)
{
    if (element->domain == arith_mul_domain(ring))
    {
        return ARITH_OK;
    }
    ARITH_REQUIRE_SLOT(ring, to_mul);
    return ring->vt->to_mul(ring, element);
}

ArithStatus arith_to_canonical(ArithRing ring, ArithElement *element)
{
    if (element->domain == ARITH_DOMAIN_CANONICAL)
    {
        return ARITH_OK;
    }
    ARITH_REQUIRE_SLOT(ring, to_canonical);
    return ring->vt->to_canonical(ring, element);
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
    ARITH_REQUIRE_SLOT(ring, add);
    status = ring->vt->add(ring, out, a, b);
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
    ARITH_REQUIRE_SLOT(ring, sub);
    status = ring->vt->sub(ring, out, a, b);
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
    ARITH_REQUIRE_SLOT(ring, mul);
    status = ring->vt->mul(ring, out, a, b);
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
    ARITH_REQUIRE_SLOT(ring, mul_addto);
    return ring->vt->mul_addto(ring, out, a, b);
}

ArithStatus arith_scale_int(ArithRing ring, ArithElement *out, const ArithElement *a,
                            uint64_t scale)
{
    if (a->domain == ARITH_DOMAIN_EMPTY)
    {
        return ARITH_BAD_DOMAIN;
    }
    ARITH_REQUIRE_SLOT(ring, scale_int);
    ArithStatus status = ring->vt->scale_int(ring, out, a, scale);
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
    ARITH_REQUIRE_SLOT(ring, mul_subto);
    return ring->vt->mul_subto(ring, out, a, b);
}

ArithStatus arith_scale_addto(ArithRing ring, ArithElement *out, const ArithElement *a,
                              uint64_t scale)
{
    ArithStatus status = check_same_domain(out, a);
    if (status != ARITH_OK)
    {
        return status;
    }
    ARITH_REQUIRE_SLOT(ring, scale_addto);
    return ring->vt->scale_addto(ring, out, a, scale);
}

ArithStatus arith_scale_by(ArithRing ring, ArithElement *out, const ArithElement *a,
                           ArithScalar scale)
{
    if (a->domain == ARITH_DOMAIN_EMPTY)
    {
        return ARITH_BAD_DOMAIN;
    }
    ARITH_REQUIRE_SLOT(ring, scale_by);
    ArithStatus status = ring->vt->scale_by(ring, out, a, scale);
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
    ARITH_REQUIRE_SLOT(ring, permute);
    ArithStatus status = ring->vt->permute(ring, out, a, gen);
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
    ARITH_REQUIRE_SLOT(ring, mul_by_monomial);
    ArithStatus status = ring->vt->mul_by_monomial(ring, out, a, power, minus_one);
    out->domain = ARITH_DOMAIN_CANONICAL;
    return status;
}

ArithStatus arith_sample_uniform(ArithRing ring, ArithElement *out)
{
    ARITH_REQUIRE_SLOT(ring, sample_uniform);
    ArithStatus status = ring->vt->sample_uniform(ring, out);
    out->domain = ARITH_DOMAIN_CANONICAL;
    return status;
}

ArithStatus arith_sample_gaussian(ArithRing ring, ArithElement *out, double sigma)
{
    ARITH_REQUIRE_SLOT(ring, sample_gaussian);
    ArithStatus status = ring->vt->sample_gaussian(ring, out, sigma);
    out->domain = ARITH_DOMAIN_CANONICAL;
    return status;
}

ArithStatus arith_from_int_array(ArithRing ring, ArithElement *out, const uint64_t *values,
                                 uint64_t count)
{
    ARITH_REQUIRE_SLOT(ring, from_int_array);
    // The implementation reports the domain it loaded into; overriding it here
    // would label the element with a representation it is not in.
    return ring->vt->from_int_array(ring, out, values, count);
}

// Tower moves are defined on the canonical form: they divide and round the
// value, which the mul domain does not represent coefficient-wise.
ArithStatus arith_round_division(ArithRing ring, ArithElement *element, ArithRing to)
{
    if (element->domain != ARITH_DOMAIN_CANONICAL)
    {
        return ARITH_BAD_DOMAIN;
    }
    ARITH_REQUIRE_SLOT(ring, round_division);
    return ring->vt->round_division(ring, element, to);
}

ArithStatus arith_mod_reduce_lifted(ArithRing ring, ArithElement *out, const ArithElement *a,
                                    ArithRing from)
{
    if (a->domain != ARITH_DOMAIN_CANONICAL)
    {
        return ARITH_BAD_DOMAIN;
    }
    ARITH_REQUIRE_SLOT(ring, mod_reduce_lifted);
    ArithStatus status = ring->vt->mod_reduce_lifted(ring, out, a, from);
    out->domain = ARITH_DOMAIN_CANONICAL;
    return status;
}

ArithStatus arith_scalar_new(ArithRing ring, const uint64_t *per_component, ArithScalar *out)
{
    ARITH_REQUIRE_SLOT(ring, scalar_new);
    return ring->vt->scalar_new(ring, per_component, out);
}

void arith_scalar_free(ArithRing ring, ArithScalar *scalar)
{
    if (ring->vt->scalar_free != NULL)
    {
        ring->vt->scalar_free(ring, scalar);
    }
}
