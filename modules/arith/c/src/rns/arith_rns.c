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

static ArithStatus rns_mul_subto(ArithRing ring, ArithElement *out, const ArithElement *a,
                                 const ArithElement *b)
{
    (void)ring;
    polynomial_mul_subto_RNS_polynomial(arith_rns_polynomial(out), arith_rns_polynomial(a),
                                        arith_rns_polynomial(b));
    return ARITH_OK;
}

static ArithStatus rns_scale_addto(ArithRing ring, ArithElement *out, const ArithElement *a,
                                   uint64_t scale)
{
    (void)ring;
    polynomial_scale_addto_RNS_polynomial(arith_rns_polynomial(out), arith_rns_polynomial(a),
                                          scale);
    return ARITH_OK;
}

static ArithStatus rns_scale_by(ArithRing ring, ArithElement *out, const ArithElement *a,
                                ArithScalar scale)
{
    (void)ring;
    polynomial_scale_RNS_polynomial_RNS(arith_rns_polynomial(out), arith_rns_polynomial(a),
                                        (uint64_t *)scale.handle);
    return ARITH_OK;
}

static ArithStatus rns_permute(ArithRing ring, ArithElement *out, const ArithElement *a,
                               uint64_t gen)
{
    (void)ring;
    polynomial_RNSc_permute((RNSc_Polynomial)arith_rns_polynomial(out),
                            (RNSc_Polynomial)arith_rns_polynomial(a), gen);
    return ARITH_OK;
}

static ArithStatus rns_mul_by_monomial(ArithRing ring, ArithElement *out, const ArithElement *a,
                                       uint64_t power, int minus_one)
{
    (void)ring;
    RNSc_Polynomial z = (RNSc_Polynomial)arith_rns_polynomial(out);
    RNSc_Polynomial x = (RNSc_Polynomial)arith_rns_polynomial(a);
    if (minus_one)
    {
        polynomial_RNSc_mul_by_xai_minus1(z, x, power);
    }
    else
    {
        polynomial_RNSc_mul_by_xai(z, x, power);
    }
    return ARITH_OK;
}

static ArithStatus rns_sample_uniform(ArithRing ring, ArithElement *out)
{
    (void)ring;
    polynomial_gen_random_RNSc_polynomial((RNSc_Polynomial)arith_rns_polynomial(out));
    return ARITH_OK;
}

static ArithStatus rns_sample_gaussian(ArithRing ring, ArithElement *out, double sigma)
{
    (void)ring;
    polynomial_gen_gaussian_RNSc_polynomial((RNSc_Polynomial)arith_rns_polynomial(out), sigma);
    return ARITH_OK;
}

// The integer polynomial is the implementation-neutral carrier; RNS reduces it
// per prime on the way in.
static ArithStatus rns_from_int_array(ArithRing ring, ArithElement *out, const uint64_t *values,
                                      uint64_t count)
{
    IntPolynomial tmp = polynomial_new_int_polynomial(ring->N);
    memcpy(tmp->coeffs, values, count * sizeof(uint64_t));
    // polynomial_to_RNS reduces per prime and then runs the forward
    // transform, so the element lands in the mul domain.
    polynomial_to_RNS(arith_rns_polynomial(out), tmp);
    free_polynomial(tmp);
    out->domain = ARITH_DOMAIN_MUL;
    return ARITH_OK;
}

// Which primes leave is derived from the two rings, not asked of the caller:
// the destination's mask says what stays.
static ArithStatus rns_round_division(ArithRing ring, ArithElement *element, ArithRing to)
{
    const uint64_t divide_mask = params_of(ring)->rns_mask & ~params_of(to)->rns_mask;
    polynomial_round_division_RNSc_wo_free((RNSc_Polynomial)arith_rns_polynomial(element),
                                           divide_mask);
    return ARITH_OK;
}

// `from` is a single-prime ring: its residue is lifted to every prime of this
// one. rns_mask_get_active_index recovers which prime that is.
static ArithStatus rns_mod_reduce_lifted(ArithRing ring, ArithElement *out, const ArithElement *a,
                                         ArithRing from)
{
    (void)ring;
    const int idx = rns_mask_get_active_index(params_of(from)->rns_mask, 0);
    if (idx < 0)
    {
        return ARITH_BAD_DOMAIN;
    }
    polynomial_RNSc_mod_reduce_lifted((RNSc_Polynomial)arith_rns_polynomial(out),
                                      (RNSc_Polynomial)arith_rns_polynomial(a), (uint64_t)idx);
    return ARITH_OK;
}

// One residue per prime the ring holds, in the base's row order, which is what
// the per-prime kernels index by.
static ArithStatus rns_scalar_new(ArithRing ring, const uint64_t *per_component, ArithScalar *out)
{
    RNSParams *params = params_of(ring);
    const uint64_t rows = rns_mask_to_l(params->rns_mask);
    uint64_t *values = (uint64_t *)safe_malloc(rows * sizeof(uint64_t));
    memcpy(values, per_component, rows * sizeof(uint64_t));
    out->handle = values;
    return ARITH_OK;
}

static void rns_scalar_free(ArithRing ring, ArithScalar *scalar)
{
    (void)ring;
    free(scalar->handle);
    scalar->handle = NULL;
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
    .mul_subto = rns_mul_subto,
    .scale_addto = rns_scale_addto,
    .scale_by = rns_scale_by,
    .permute = rns_permute,
    .mul_by_monomial = rns_mul_by_monomial,
    .sample_uniform = rns_sample_uniform,
    .sample_gaussian = rns_sample_gaussian,
    .from_int_array = rns_from_int_array,
    .round_division = rns_round_division,
    .mod_reduce_lifted = rns_mod_reduce_lifted,
    .scalar_new = rns_scalar_new,
    .scalar_free = rns_scalar_free,
};

// Rings are shared and never freed, the same contract the RNS base they borrow
// already has: an element does not point at its ring (arith_* takes it as an
// argument), but a structure built over one may hold it, and nothing can prove
// the last such structure is gone. The table is tiny -- one entry per distinct
// (N, mask, base) a process uses.
#define ARITH_RNS_RING_CACHE_MAX 256

static struct
{
    uint64_t N, mask;
    RNS_Base base;
    ArithRing ring;
} ring_cache[ARITH_RNS_RING_CACHE_MAX];
static size_t ring_cache_len = 0;

ArithRing arith_rns_ring_get(uint64_t N, uint64_t rns_mask, RNS_Base base)
{
    for (size_t i = 0; i < ring_cache_len; i++)
    {
        if (ring_cache[i].N == N && ring_cache[i].mask == rns_mask && ring_cache[i].base == base)
        {
            return ring_cache[i].ring;
        }
    }
    ArithRing ring = arith_rns_ring_new(N, rns_mask, base);
    if (ring_cache_len < ARITH_RNS_RING_CACHE_MAX)
    {
        ring_cache[ring_cache_len].N = N;
        ring_cache[ring_cache_len].mask = rns_mask;
        ring_cache[ring_cache_len].base = base;
        ring_cache[ring_cache_len].ring = ring;
        ring_cache_len++;
    }
    return ring;
}

// Drop the cache. Only the dynamic-extension reload needs this: the rings
// point at a method table in the retired library.
void arith_rns_ring_cache_clear(void)
{
    for (size_t i = 0; i < ring_cache_len; i++)
    {
        arith_ring_free(ring_cache[i].ring);
    }
    ring_cache_len = 0;
}

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
