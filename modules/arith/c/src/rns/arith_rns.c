// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
#include <arith_generic.h>

#include "arith_dispatch.h"
#include <alloc.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

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

ArithStatus arith_rns_new_element(ArithRing ring, ArithElement *out)
{
    RNSParams *params = params_of(ring);
    out->handle = polynomial_new_RNS_polynomial(ring->N, params->rns_mask, params->base);
    out->domain = ARITH_DOMAIN_EMPTY;
    return out->handle == NULL ? ARITH_UNIMPLEMENTED : ARITH_OK;
}

// The model's own mask, not the ring's: an element may hold a subset of the
// ring's primes, and a temporary is only useful if it matches the operand it
// will be combined with.
ArithStatus arith_rns_new_like(ArithRing ring, const ArithElement *model, ArithElement *out)
{
    RNS_Polynomial source = arith_rns_polynomial(model);
    out->handle = polynomial_new_RNS_polynomial(source->base->N, source->rns_mask, source->base);
    out->domain = ARITH_DOMAIN_EMPTY;
    (void)ring;
    return out->handle == NULL ? ARITH_UNIMPLEMENTED : ARITH_OK;
}

void arith_rns_free_element(ArithRing ring, ArithElement *element)
{
    (void)ring;
    if (element->handle != NULL)
    {
        free_RNS_polynomial(element->handle);
        element->handle = NULL;
        element->domain = ARITH_DOMAIN_EMPTY;
    }
}

ArithStatus arith_rns_copy(ArithRing ring, ArithElement *out, const ArithElement *in)
{
    (void)ring;
    polynomial_copy_RNS_polynomial(arith_rns_polynomial(out), arith_rns_polynomial(in));
    out->domain = in->domain;
    return ARITH_OK;
}

ArithStatus arith_rns_zero(ArithRing ring, ArithElement *out)
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
ArithStatus arith_rns_to_mul(ArithRing ring, ArithElement *element)
{
    (void)ring;
    RNS_Polynomial p = arith_rns_polynomial(element);
    polynomial_RNSc_to_RNS(p, (RNSc_Polynomial)p);
    element->domain = ARITH_DOMAIN_MUL;
    return ARITH_OK;
}

ArithStatus arith_rns_to_canonical(ArithRing ring, ArithElement *element)
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
ArithStatus arith_rns_add(ArithRing ring, ArithElement *out, const ArithElement *a,
                          const ArithElement *b)
{
    (void)ring;
    polynomial_add_RNS_polynomial(arith_rns_polynomial(out), arith_rns_polynomial(a),
                                  arith_rns_polynomial(b));
    return ARITH_OK;
}

ArithStatus arith_rns_sub(ArithRing ring, ArithElement *out, const ArithElement *a,
                          const ArithElement *b)
{
    (void)ring;
    polynomial_sub_RNS_polynomial(arith_rns_polynomial(out), arith_rns_polynomial(a),
                                  arith_rns_polynomial(b));
    return ARITH_OK;
}

ArithStatus arith_rns_mul(ArithRing ring, ArithElement *out, const ArithElement *a,
                          const ArithElement *b)
{
    (void)ring;
    polynomial_mul_RNS_polynomial(arith_rns_polynomial(out), arith_rns_polynomial(a),
                                  arith_rns_polynomial(b));
    return ARITH_OK;
}

ArithStatus arith_rns_mul_addto(ArithRing ring, ArithElement *out, const ArithElement *a,
                                const ArithElement *b)
{
    (void)ring;
    polynomial_mul_addto_RNS_polynomial(arith_rns_polynomial(out), arith_rns_polynomial(a),
                                        arith_rns_polynomial(b));
    return ARITH_OK;
}

ArithStatus arith_rns_scale_int(ArithRing ring, ArithElement *out, const ArithElement *a,
                                uint64_t scale)
{
    (void)ring;
    // Scaling by an integer is linear too, so one kernel serves both domains.
    polynomial_scale_RNS_polynomial(arith_rns_polynomial(out), arith_rns_polynomial(a), scale);
    return ARITH_OK;
}

ArithStatus arith_rns_mul_subto(ArithRing ring, ArithElement *out, const ArithElement *a,
                                const ArithElement *b)
{
    (void)ring;
    polynomial_mul_subto_RNS_polynomial(arith_rns_polynomial(out), arith_rns_polynomial(a),
                                        arith_rns_polynomial(b));
    return ARITH_OK;
}

ArithStatus arith_rns_scale_addto(ArithRing ring, ArithElement *out, const ArithElement *a,
                                  uint64_t scale)
{
    (void)ring;
    polynomial_scale_addto_RNS_polynomial(arith_rns_polynomial(out), arith_rns_polynomial(a),
                                          scale);
    return ARITH_OK;
}

ArithStatus arith_rns_scale_by(ArithRing ring, ArithElement *out, const ArithElement *a,
                               ArithScalar scale)
{
    (void)ring;
    polynomial_scale_RNS_polynomial_RNS(arith_rns_polynomial(out), arith_rns_polynomial(a),
                                        (uint64_t *)scale.handle);
    return ARITH_OK;
}

ArithStatus arith_rns_permute(ArithRing ring, ArithElement *out, const ArithElement *a,
                              uint64_t gen)
{
    (void)ring;
    polynomial_RNSc_permute((RNSc_Polynomial)arith_rns_polynomial(out),
                            (RNSc_Polynomial)arith_rns_polynomial(a), gen);
    return ARITH_OK;
}

ArithStatus arith_rns_mul_by_monomial(ArithRing ring, ArithElement *out, const ArithElement *a,
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

ArithStatus arith_rns_sample_uniform(ArithRing ring, ArithElement *out)
{
    (void)ring;
    polynomial_gen_random_RNSc_polynomial((RNSc_Polynomial)arith_rns_polynomial(out));
    return ARITH_OK;
}

ArithStatus arith_rns_sample_gaussian(ArithRing ring, ArithElement *out, double sigma)
{
    (void)ring;
    polynomial_gen_gaussian_RNSc_polynomial((RNSc_Polynomial)arith_rns_polynomial(out), sigma);
    return ARITH_OK;
}

ArithStatus arith_rns_sample_uniform_seeded(ArithRing ring, ArithElement *out, const uint8_t *seed,
                                            uint64_t seed_len, uint64_t stream)
{
    (void)ring;
    polynomial_RNS_expand_seeded(arith_rns_polynomial(out), seed, seed_len, stream);
    return ARITH_OK;
}

// The integer polynomial is the implementation-neutral carrier; RNS reduces it
// per prime on the way in.
ArithStatus arith_rns_from_int_array(ArithRing ring, ArithElement *out, const uint64_t *values,
                                     uint64_t count)
{
    IntPolynomial tmp = polynomial_new_int_polynomial(ring->N);
    const uint64_t loaded = count < ring->N ? count : ring->N;
    memcpy(tmp->coeffs, values, loaded * sizeof(uint64_t));
    memset(tmp->coeffs + loaded, 0, (ring->N - loaded) * sizeof(uint64_t));
    // polynomial_to_RNS reduces per prime and then runs the forward
    // transform, so the element lands in the mul domain.
    polynomial_to_RNS(arith_rns_polynomial(out), tmp);
    free_polynomial(tmp);
    out->domain = ARITH_DOMAIN_MUL;
    return ARITH_OK;
}

// Which primes leave is derived from the two rings, not asked of the caller:
// the destination's mask says what stays.
ArithStatus arith_rns_round_division(ArithRing ring, ArithElement *element, ArithRing to)
{
    const uint64_t divide_mask = params_of(ring)->rns_mask & ~params_of(to)->rns_mask;
    polynomial_round_division_RNSc_wo_free((RNSc_Polynomial)arith_rns_polynomial(element),
                                           divide_mask);
    return ARITH_OK;
}

// The row functions of the key switch's ModDown: the rows that leave are
// prepared in place, every row that stays divided, and the rows that left
// zeroed and dropped from the mask, as the exact division drops them.
ArithStatus arith_rns_round_division_approximate(ArithRing ring, ArithElement *element,
                                                 ArithRing to)
{
    const uint64_t divide_mask = params_of(ring)->rns_mask & ~params_of(to)->rns_mask;
    RNSc_Polynomial x = (RNSc_Polynomial)arith_rns_polynomial(element);
    polynomial_mod_down_RNSc_prepare(x, divide_mask);
    polynomial_mod_down_RNSc_rows(x, x, NULL, divide_mask, x->rns_mask & ~divide_mask);
    polynomial_round_division_RNSc_finish(x, divide_mask);
    return ARITH_OK;
}

// The constants of a group's digit: for the group J of primes `primes`,
// (Q_J / p_k)^-1 mod p_k at k for k in J, and (Q_J / p_k) mod p_i at
// k * l + i for every row i in `rows`, l the base's prime count. Kept per
// (base, group, rows, l) for the life of the process, as the shared rings
// are; a base that grew since has another l, and another entry.
typedef struct
{
    RNS_Base base;
    uint64_t primes, rows, l;
    uint64_t *inverse, *factors;
} GroupConstants;

static GroupConstants *group_cache = NULL;
static size_t group_cache_len = 0, group_cache_cap = 0;
static pthread_mutex_t group_cache_lock = PTHREAD_MUTEX_INITIALIZER;

static void group_constants(RNS_Base base, uint64_t primes, uint64_t rows, const uint64_t **inverse,
                            const uint64_t **factors)
{
    const uint64_t l = base->l;
    pthread_mutex_lock(&group_cache_lock);
    for (size_t c = 0; c < group_cache_len; c++)
    {
        const GroupConstants *g = &group_cache[c];
        if (g->base == base && g->primes == primes && g->rows == rows && g->l == l)
        {
            *inverse = g->inverse;
            *factors = g->factors;
            pthread_mutex_unlock(&group_cache_lock);
            return;
        }
    }
    GroupConstants g = {base,
                        primes,
                        rows,
                        l,
                        (uint64_t *)safe_malloc(l * sizeof(uint64_t)),
                        (uint64_t *)safe_malloc(l * l * sizeof(uint64_t))};
    memset(g.inverse, 0, l * sizeof(uint64_t));
    memset(g.factors, 0, l * l * sizeof(uint64_t));
    for (uint64_t k = 0; k < l; k++)
    {
        if (!(primes & (1ULL << k)))
            continue;
        for (uint64_t i = 0; i < l; i++)
        {
            if (!((rows | (1ULL << k)) & (1ULL << i)))
                continue;
            uint64_t f = 1;
            for (uint64_t m = 0; m < l; m++)
                if (m != k && (primes & (1ULL << m)))
                    f = mul_modq(f, modq(base->mods[m]->q, base->mods[i]), base->mods[i]);
            if (i == k)
                g.inverse[k] = inverse_mod(f, base->mods[k]->q);
            if (rows & (1ULL << i))
                g.factors[k * l + i] = f;
        }
    }
    if (group_cache_len == group_cache_cap)
    {
        group_cache_cap = group_cache_cap ? 2 * group_cache_cap : 8;
        group_cache =
            (GroupConstants *)safe_realloc(group_cache, group_cache_cap * sizeof(GroupConstants));
    }
    group_cache[group_cache_len++] = g;
    *inverse = g.inverse;
    *factors = g.factors;
    pthread_mutex_unlock(&group_cache_lock);
}

static void group_cache_clear(void)
{
    pthread_mutex_lock(&group_cache_lock);
    for (size_t c = 0; c < group_cache_len; c++)
    {
        free(group_cache[c].inverse);
        free(group_cache[c].factors);
    }
    free(group_cache);
    group_cache = NULL;
    group_cache_len = group_cache_cap = 0;
    pthread_mutex_unlock(&group_cache_lock);
}

// A group of primes J: the residues modulo its members, scaled by the
// inverses, combined on every row of the output with the factors -- the
// fast base conversion of the residue modulo Q_J.
static ArithStatus group_digit(RNSc_Polynomial out, RNSc_Polynomial in, uint64_t primes,
                               int centered)
{
    RNS_Base base = in->base;
    const uint64_t *inverse, *factors;
    group_constants(base, primes, out->rns_mask, &inverse, &factors);
    struct _RNS_Polynomial group_storage;
    RNS_Polynomial group = polynomial_RNS_view(&group_storage, (RNS_Polynomial)in, primes);
    RNS_Polynomial scaled = polynomial_new_RNS_polynomial(base->N, primes, base);
    polynomial_scale_RNS_polynomial_RNS(scaled, group, (uint64_t *)inverse);
    polynomial_RNSc_lifted_combination(out, (RNSc_Polynomial)scaled, primes, factors, centered);
    free_RNS_polynomial(scaled);
    return ARITH_OK;
}

ArithStatus arith_rns_decompose_digit(ArithRing ring, ArithElement *out, const ArithElement *a,
                                      uint64_t primes, uint64_t log_base, uint64_t level,
                                      uint64_t dropped, int centered)
{
    (void)ring;
    RNSc_Polynomial in = (RNSc_Polynomial)arith_rns_polynomial(a);
    RNSc_Polynomial digit = (RNSc_Polynomial)arith_rns_polynomial(out);
    const int grouped = primes & (primes - 1);
    if (primes == 0 || (primes & ~in->rns_mask) || (log_base && grouped))
        return ARITH_BAD_DOMAIN;
    const uint64_t j = (uint64_t)__builtin_ctzll(primes);
    if (log_base && centered && log_base > 1)
        polynomial_RNSc_decompose_digit_balanced(digit, in, j, log_base, level, dropped);
    else if (log_base)
        polynomial_RNSc_decompose_digit_rounded(digit, in, j, log_base, level, dropped);
    else if (grouped)
        return group_digit(digit, in, primes, centered);
    else if (centered)
        polynomial_RNSc_mod_reduce_lifted_centered(digit, in, j);
    else
        polynomial_RNSc_mod_reduce_lifted(digit, in, j);
    return ARITH_OK;
}

// `from` is a single-prime ring: its residue is lifted to every prime of this
// one. rns_mask_get_active_index recovers which prime that is.
ArithStatus arith_rns_mod_reduce_lifted(ArithRing ring, ArithElement *out, const ArithElement *a,
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
ArithStatus arith_rns_scalar_new(ArithRing ring, const uint64_t *per_component, ArithScalar *out)
{
    RNSParams *params = params_of(ring);
    // Indexed by base row, as the scaling kernels read it: up to the ring's
    // highest prime, past its prime count when its primes do not start the
    // base.
    const uint64_t rows = 64 - (uint64_t)__builtin_clzll(params->rns_mask);
    uint64_t *values = (uint64_t *)safe_malloc(rows * sizeof(uint64_t));
    memcpy(values, per_component, rows * sizeof(uint64_t));
    out->handle = values;
    return ARITH_OK;
}

void arith_rns_scalar_free(ArithRing ring, ArithScalar *scalar)
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
    .new_element = arith_rns_new_element,
    .new_like = arith_rns_new_like,
    .free_element = arith_rns_free_element,
    .copy = arith_rns_copy,
    .zero = arith_rns_zero,
    .to_mul = arith_rns_to_mul,
    .to_canonical = arith_rns_to_canonical,
    .add = arith_rns_add,
    .sub = arith_rns_sub,
    .mul = arith_rns_mul,
    .mul_addto = arith_rns_mul_addto,
    .scale_int = arith_rns_scale_int,
    .mul_subto = arith_rns_mul_subto,
    .scale_addto = arith_rns_scale_addto,
    .scale_by = arith_rns_scale_by,
    .permute = arith_rns_permute,
    .mul_by_monomial = arith_rns_mul_by_monomial,
    .sample_uniform = arith_rns_sample_uniform,
    .sample_gaussian = arith_rns_sample_gaussian,
    .sample_uniform_seeded = arith_rns_sample_uniform_seeded,
    .from_int_array = arith_rns_from_int_array,
    .round_division = arith_rns_round_division,
    .round_division_approximate = arith_rns_round_division_approximate,
    .mod_reduce_lifted = arith_rns_mod_reduce_lifted,
    .decompose_digit = arith_rns_decompose_digit,
    .scalar_new = arith_rns_scalar_new,
    .scalar_free = arith_rns_scalar_free,
};

// Rings are shared and never freed, the same contract the RNS base they borrow
// already has: an element does not point at its ring (arith_* takes it as an
// argument), but a structure built over one may hold it, and nothing can prove
// the last such structure is gone. Each (N, mask, base) therefore has one
// handle for the life of the process. The table grows as needed, under a lock
// so that no lookup scans it while it is reallocated.
typedef struct
{
    uint64_t N, mask;
    RNS_Base base;
    ArithRing ring;
} RingCacheEntry;

static RingCacheEntry *ring_cache = NULL;
static size_t ring_cache_len = 0, ring_cache_cap = 0;
static pthread_mutex_t ring_cache_lock = PTHREAD_MUTEX_INITIALIZER;

ArithRing arith_rns_ring_get(uint64_t N, uint64_t rns_mask, RNS_Base base)
{
    pthread_mutex_lock(&ring_cache_lock);
    for (size_t i = 0; i < ring_cache_len; i++)
    {
        if (ring_cache[i].N == N && ring_cache[i].mask == rns_mask && ring_cache[i].base == base)
        {
            ArithRing ring = ring_cache[i].ring;
            pthread_mutex_unlock(&ring_cache_lock);
            return ring;
        }
    }
    if (ring_cache_len == ring_cache_cap)
    {
        ring_cache_cap = ring_cache_cap ? 2 * ring_cache_cap : 64;
        ring_cache =
            (RingCacheEntry *)safe_realloc(ring_cache, ring_cache_cap * sizeof(*ring_cache));
    }
    ArithRing ring = arith_rns_ring_new(N, rns_mask, base);
    ring_cache[ring_cache_len++] = (RingCacheEntry){N, rns_mask, base, ring};
    pthread_mutex_unlock(&ring_cache_lock);
    return ring;
}

uint64_t arith_rns_ring_mask(ArithRing ring) { return params_of(ring)->rns_mask; }

// Drop the cache. Only the dynamic-extension reload needs this: the rings
// point at a method table in the retired library.
void arith_rns_ring_cache_clear(void)
{
    group_cache_clear();
    pthread_mutex_lock(&ring_cache_lock);
    for (size_t i = 0; i < ring_cache_len; i++)
    {
        arith_ring_free(ring_cache[i].ring);
    }
    ring_cache_len = 0;
    pthread_mutex_unlock(&ring_cache_lock);
}

// A fork copies only the forking thread. Both caches are locked across it, so
// the child never inherits a lock held by a thread that does not exist there,
// or a cache caught halfway through growing.
static void lock_caches(void)
{
    pthread_mutex_lock(&group_cache_lock);
    pthread_mutex_lock(&ring_cache_lock);
}

static void unlock_caches(void)
{
    pthread_mutex_unlock(&ring_cache_lock);
    pthread_mutex_unlock(&group_cache_lock);
}

__attribute__((constructor)) static void caches_watch_forks(void)
{
    if (pthread_atfork(lock_caches, unlock_caches, unlock_caches) != 0)
    {
        perror("vfhe: pthread_atfork"); /* GCOVR_EXCL_LINE */
        abort();                        /* GCOVR_EXCL_LINE */
    }
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
    ring->impl = ARITH_IMPL_RNS;
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
