// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include <arith_generic.h>
#include <misc.h>

// RNS as an implementation of the generic arithmetic interface.
//
// Each slot forwards to the rns_polynomial.c kernel for its domain: the
// canonical domain is the coefficient representation (the _RNSc_ family) and
// the mul domain is the NTT representation (the _RNS_ family), which is what
// the two symbol families have always meant. Nothing here computes; the whole
// file is the mapping between the domain flag and the symbol to call.

typedef struct
{
    uint64_t rns_mask;
    RNS_Base base;
} RNSParams;

static RNSParams *params_of(ArithRing ring) { return (RNSParams *)ring->params; }

RNS_Polynomial arith_rns_polynomial(const ArithElement *element)
{
    return (RNS_Polynomial)element->handle;
}

static ArithStatus rns_new(ArithRing ring, ArithElement *out)
{
    RNSParams *params = params_of(ring);
    out->handle = polynomial_new_RNS_polynomial(ring->N, params->rns_mask, params->base);
    out->domain = ARITH_DOMAIN_EMPTY;
    return out->handle == NULL ? ARITH_UNIMPLEMENTED : ARITH_OK;
}

// The model's own mask, not the ring's: an element may hold a subset of the
// ring's primes, and a temporary is only useful if it matches the operand it
// will be combined with.
static ArithStatus rns_new_like(ArithRing ring, const ArithElement *model, ArithElement *out)
{
    RNS_Polynomial source = arith_rns_polynomial(model);
    out->handle = polynomial_new_RNS_polynomial(source->base->N, source->rns_mask, source->base);
    out->domain = ARITH_DOMAIN_EMPTY;
    (void)ring;
    return out->handle == NULL ? ARITH_UNIMPLEMENTED : ARITH_OK;
}

static void rns_free(ArithRing ring, ArithElement *element)
{
    (void)ring;
    if (element->handle != NULL)
    {
        free_RNS_polynomial(element->handle);
        element->handle = NULL;
        element->domain = ARITH_DOMAIN_EMPTY;
    }
}

static ArithStatus rns_copy(ArithRing ring, ArithElement *out, const ArithElement *in)
{
    (void)ring;
    polynomial_copy_RNS_polynomial(arith_rns_polynomial(out), arith_rns_polynomial(in));
    out->domain = in->domain;
    return ARITH_OK;
}

static ArithStatus rns_zero(ArithRing ring, ArithElement *out)
{
    (void)ring;
    polynomial_RNS_zero(arith_rns_polynomial(out));
    // All-zero is the same word pattern in both representations, and zero is
    // its own transform, so the canonical domain is the honest label.
    out->domain = ARITH_DOMAIN_CANONICAL;
    return ARITH_OK;
}

// Both conversions are in place: for RNS the two domains share one buffer.
// That is an RNS property, not a promise of the interface -- see the note on
// ArithElement in arith_generic.h.
static ArithStatus rns_to_mul(ArithRing ring, ArithElement *element)
{
    (void)ring;
    RNS_Polynomial p = arith_rns_polynomial(element);
    polynomial_RNSc_to_RNS(p, (RNSc_Polynomial)p);
    element->domain = ARITH_DOMAIN_MUL;
    return ARITH_OK;
}

static ArithStatus rns_to_canonical(ArithRing ring, ArithElement *element)
{
    (void)ring;
    RNS_Polynomial p = arith_rns_polynomial(element);
    polynomial_RNS_to_RNSc((RNSc_Polynomial)p, p);
    element->domain = ARITH_DOMAIN_CANONICAL;
    return ARITH_OK;
}

// Addition is linear, so it commutes with the transform between the two
// domains: one coefficient-wise kernel is correct in both, and the _RNSc_
// entry points are casts onto these. An implementation whose mul domain has a
// different element type -- complex doubles under an FFT -- needs two kernels
// here; RNS does not.
static ArithStatus rns_add(ArithRing ring, ArithElement *out, const ArithElement *a,
                           const ArithElement *b)
{
    (void)ring;
    polynomial_add_RNS_polynomial(arith_rns_polynomial(out), arith_rns_polynomial(a),
                                  arith_rns_polynomial(b));
    return ARITH_OK;
}

static ArithStatus rns_sub(ArithRing ring, ArithElement *out, const ArithElement *a,
                           const ArithElement *b)
{
    (void)ring;
    polynomial_sub_RNS_polynomial(arith_rns_polynomial(out), arith_rns_polynomial(a),
                                  arith_rns_polynomial(b));
    return ARITH_OK;
}

static ArithStatus rns_mul(ArithRing ring, ArithElement *out, const ArithElement *a,
                           const ArithElement *b)
{
    (void)ring;
    polynomial_mul_RNS_polynomial(arith_rns_polynomial(out), arith_rns_polynomial(a),
                                  arith_rns_polynomial(b));
    return ARITH_OK;
}

static ArithStatus rns_mul_addto(ArithRing ring, ArithElement *out, const ArithElement *a,
                                 const ArithElement *b)
{
    (void)ring;
    polynomial_mul_addto_RNS_polynomial(arith_rns_polynomial(out), arith_rns_polynomial(a),
                                        arith_rns_polynomial(b));
    return ARITH_OK;
}

static ArithStatus rns_scale_int(ArithRing ring, ArithElement *out, const ArithElement *a,
                                 uint64_t scale)
{
    (void)ring;
    // Scaling by an integer is linear too, so one kernel serves both domains.
    polynomial_scale_RNS_polynomial(arith_rns_polynomial(out), arith_rns_polynomial(a), scale);
    return ARITH_OK;
}

static const ArithMethods RNS_NTT_METHODS = {
    .implementation = "rns",
    .backend = "ntt",
    .capabilities = ARITH_CAP_CORE | ARITH_CAP_QUOTIENT_POLY_RING | ARITH_CAP_TOWER |
                    ARITH_CAP_SAMPLING | ARITH_CAP_EXACT,
    .new_element = rns_new,
    .new_like = rns_new_like,
    .free_element = rns_free,
    .copy = rns_copy,
    .zero = rns_zero,
    .to_mul = rns_to_mul,
    .to_canonical = rns_to_canonical,
    .add = rns_add,
    .sub = rns_sub,
    .mul = rns_mul,
    .mul_addto = rns_mul_addto,
    .scale_int = rns_scale_int,
};

ArithRing arith_rns_ring_new(uint64_t N, uint64_t rns_mask, RNS_Base base)
{
    ArithRing ring = (ArithRing)safe_malloc(sizeof(*ring));
    RNSParams *params = (RNSParams *)safe_malloc(sizeof(*params));
    params->rns_mask = rns_mask;
    params->base = base;
    ring->vt = &RNS_NTT_METHODS;
    ring->params = params;
    ring->N = N;
    return ring;
}

// The base is borrowed: it is process-lifetime and shared by every ring built
// on it, so only the ring's own two allocations are released.
void arith_ring_free(ArithRing ring)
{
    if (ring != NULL)
    {
        free(ring->params);
        free(ring);
    }
}
