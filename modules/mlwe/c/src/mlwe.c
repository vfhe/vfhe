// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "mlwe.h"
#include "util.h"
#include <crypto.h>

// MLWE RNS functions

RNS_MLWE_Key mlwe_alloc_key(ArithRing ring, uint64_t r, uint64_t l, double sigma)
{
    RNS_MLWE_Key res = (RNS_MLWE_Key)safe_malloc(sizeof(*res));
    res->sigma = sigma;
    res->N = ring->N;
    res->l = l;
    res->r = r;
    res->ring = ring;
    res->s = (ArithElement *)safe_malloc(r * sizeof(ArithElement));
    for (size_t i = 0; i < r; i++)
    {
        arith_new(ring, &res->s[i]);
    }
    return res;
}

void free_mlwe_RNS_key(RNS_MLWE_Key key)
{
    for (size_t i = 0; i < key->r; i++)
    {
        arith_free(key->ring, &key->s[i]);
    }
    free(key->s);
    free(key);
}

MLWE mlwe_alloc_sample(ArithRing ring, uint64_t r)
{
    MLWE res = (MLWE)safe_malloc(sizeof(*res));
    res->a = (ArithElement *)safe_malloc(r * sizeof(ArithElement));
    for (size_t i = 0; i < r; i++)
    {
        arith_new(ring, &res->a[i]);
    }
    arith_new(ring, &res->b);
    res->r = r;
    res->ring = ring;
    return res;
}

ArithDomain mlwe_domain(MLWE c) { return c->b.domain; }

void mlwe_copy_array(RNS_MLWE *out, RNS_MLWE *in, uint64_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        mlwe_copy_RNS_sample(out[i], in[i]);
    }
}

RNS_MLWE *mlwe_create_copy_array(RNS_MLWE *in, uint64_t size)
{
    RNS_MLWE *res = (RNS_MLWE *)safe_malloc(size * sizeof(*res));
    for (size_t i = 0; i < size; i++)
    {
        res[i] = mlwe_alloc_sample(in[0]->ring, in[0]->r);
    }
    mlwe_copy_array(res, in, size);
    return res;
}

RNS_MLWE *mlwe_alloc_RNS_sample_array2(uint64_t size, RNS_MLWE c)
{
    RNS_MLWE *res;
    res = (RNS_MLWE *)safe_malloc(size * sizeof(*res));
    for (size_t i = 0; i < size; i++)
    {
        res[i] = mlwe_alloc_sample(c->ring, c->r);
    }
    return res;
}

void free_RNS_mlwe_array(uint64_t size, RNS_MLWE *v)
{
    for (size_t i = 0; i < size; i++)
    {
        free_mlwe_RNS_sample(v[i]);
    }
    free(v);
}

void free_RNS_mlwe_sample(RNS_MLWE c)
{
    for (size_t i = 0; i < c->r; i++)
    {
        arith_free(c->ring, &c->a[i]);
    }
    arith_free(c->ring, &c->b);
    free(c->a);
    free(c);
}

void mlwe_copy_RNS_sample(RNS_MLWE out, RNS_MLWE in)
{
    for (size_t i = 0; i < out->r; i++)
    {
        arith_copy(out->ring, &out->a[i], &in->a[i]);
    }
    arith_copy(out->ring, &out->b, &in->b);
}

void mlwe_copy_RNSc_sample(RNSc_MLWE out, RNSc_MLWE in) { mlwe_copy_RNS_sample(out, in); }

void free_mlwe_RNS_sample(void *p) { free_RNS_mlwe_sample((RNS_MLWE)p); }

void mlwe_RNS_sample_of_zero(RNS_MLWE out, RNS_MLWE_Key key)
{
    arith_sample_gaussian(out->ring, &out->b, key->sigma);
    arith_to_mul(out->ring, &out->b);
    for (size_t i = 0; i < out->r; i++)
    {
        arith_sample_uniform(out->ring, &out->a[i]);
        arith_to_mul(out->ring, &out->a[i]);
        arith_mul_addto(out->ring, &out->b, &key->s[i], &out->a[i]);
    }
}

void mlwe_RNS_sample_of_zero_seeded(RNS_MLWE out, RNS_MLWE_Key key, const uint8_t *seed,
                                    uint64_t seed_len)
{
    arith_sample_gaussian(out->ring, &out->b, key->sigma);
    arith_to_mul(out->ring, &out->b);
    for (size_t i = 0; i < out->r; i++)
    {
        arith_sample_uniform_seeded(out->ring, &out->a[i], seed, seed_len, i);
        arith_mul_addto(out->ring, &out->b, &key->s[i], &out->a[i]);
    }
}

void mlwe_RNSc_sample_seeded(RNSc_MLWE out, RNS_MLWE_Key key, const ArithElement *m,
                             const uint8_t *seed, uint64_t seed_len)
{
    mlwe_RNS_sample_of_zero_seeded(out, key, seed, seed_len);
    mlwe_RNS_to_RNSc(out, out);
    arith_add(out->ring, &out->b, &out->b, m);
}

// --- Batched encryption ----------------------------------------------------
//
// Key generation is many independent encryptions -- a few messages, each times
// every gadget element -- so it runs them in parallel. Every sample is a
// function of two seeds, its mask's and its noise's, all drawn on the calling
// thread in order, so the samples do not depend on the number of threads.

#define SAMPLE_SEED_BYTES 32

typedef struct
{
    RNS_MLWE *out;
    ArithRing ring;
    RNS_MLWE_Key key;
    const ArithElement *msgs;
    const ArithScalar *scales;
    uint64_t n_scales;
    const uint8_t *mask_seeds, *noise_seeds;
} ScaledSamples;

static void sample_scaled_job(void *ctx, uint64_t i)
{
    const ScaledSamples *job = (const ScaledSamples *)ctx;
    const ArithRing ring = job->ring;
    const RNS_MLWE_Key key = job->key;
    RNS_MLWE out = job->out[i] = mlwe_alloc_sample(ring, key->r);

    int64_t *noise = (int64_t *)safe_malloc(ring->N * sizeof(int64_t));
    prng_normal_seeded(noise, ring->N, key->sigma, &job->noise_seeds[i * SAMPLE_SEED_BYTES],
                       SAMPLE_SEED_BYTES);
    arith_from_int_array(ring, &out->b, (const uint64_t *)noise, ring->N);
    free(noise);

    // a_0 holds the scaled message until its mask is drawn.
    arith_scale_by(ring, &out->a[0], &job->msgs[i / job->n_scales], job->scales[i % job->n_scales]);
    arith_add(ring, &out->b, &out->b, &out->a[0]);
    for (size_t j = 0; j < key->r; j++)
    {
        arith_sample_uniform_seeded(ring, &out->a[j], &job->mask_seeds[i * SAMPLE_SEED_BYTES],
                                    SAMPLE_SEED_BYTES, j);
        arith_mul_addto(ring, &out->b, &key->s[j], &out->a[j]);
    }
}

void mlwe_RNS_sample_scaled_batch(RNS_MLWE *out, ArithRing ring, RNS_MLWE_Key key,
                                  const ArithElement *msgs, uint64_t n_msgs,
                                  uint64_t *const *scales, uint64_t n_scales, uint8_t *mask_seeds,
                                  uint64_t n_threads)
{
    const uint64_t count = n_msgs * n_scales;
    if (count == 0)
        return;
    uint8_t *noise_seeds = (uint8_t *)safe_malloc(count * SAMPLE_SEED_BYTES);
    generate_random_bytes(count * SAMPLE_SEED_BYTES, mask_seeds);
    generate_random_bytes(count * SAMPLE_SEED_BYTES, noise_seeds);
    ArithScalar *scalars = (ArithScalar *)safe_malloc(n_scales * sizeof(ArithScalar));
    for (uint64_t k = 0; k < n_scales; k++)
        arith_scalar_new(ring, scales[k], &scalars[k]);

    ScaledSamples job = {out, ring, key, msgs, scalars, n_scales, mask_seeds, noise_seeds};
    vfhe_parallel_for(count, n_threads, sample_scaled_job, &job);

    for (uint64_t k = 0; k < n_scales; k++)
        arith_scalar_free(ring, &scalars[k]);
    free(scalars);
    free(noise_seeds);
}

void mlwe_RNSc_sample_of_zero(RNSc_MLWE out, RNS_MLWE_Key key)
{
    mlwe_RNS_sample_of_zero(out, key);
    mlwe_RNS_to_RNSc(out, out);
}

void mlwe_scale_RNSc_mlwe(RNSc_MLWE c, uint64_t scale)
{
    for (size_t i = 0; i < c->r; i++)
    {
        arith_scale_int(c->ring, &c->a[i], &c->a[i], scale);
    }
    arith_scale_int(c->ring, &c->b, &c->b, scale);
}

// One value per component is the implementation-neutral way to name a
// scalar; the ring turns it into whatever it multiplies by.
void mlwe_scale_RNS_mlwe_RNS(RNS_MLWE c, const uint64_t *per_component)
{
    ArithScalar scale;
    if (arith_scalar_new(c->ring, per_component, &scale) != ARITH_OK)
    {
        return;
    }
    for (size_t i = 0; i < c->r; i++)
    {
        arith_scale_by(c->ring, &c->a[i], &c->a[i], scale);
    }
    arith_scale_by(c->ring, &c->b, &c->b, scale);
    arith_scalar_free(c->ring, &scale);
}

// out += in*scale
void mlwe_scale_RNS_mlwe_addto(RNS_MLWE out, RNS_MLWE in, uint64_t scale)
{
    for (size_t i = 0; i < out->r; i++)
    {
        arith_scale_addto(out->ring, &out->a[i], &in->a[i], scale);
    }
    arith_scale_addto(out->ring, &out->b, &in->b, scale);
}

void mlwe_RNSc_sample(RNSc_MLWE out, RNS_MLWE_Key key, const ArithElement *m)
{
    mlwe_RNSc_sample_of_zero(out, key);
    arith_add(out->ring, &out->b, &out->b, m);
}

void mlwe_RNS_linear_decrypt(ArithElement *out, RNS_MLWE in, RNS_MLWE_Key key)
{
    arith_mul(in->ring, out, &in->a[0], &key->s[0]);
    for (size_t i = 1; i < in->r; i++)
    {
        arith_mul_addto(in->ring, out, &in->a[i], &key->s[i]);
    }

    arith_sub(in->ring, out, &in->b, out);
}

void mlwe_RNS_mul_by_poly(RNS_MLWE out, RNS_MLWE in, const ArithElement *poly)
{
    for (size_t i = 0; i < out->r; i++)
    {
        arith_mul(out->ring, &out->a[i], &in->a[i], poly);
    }
    arith_mul(out->ring, &out->b, &in->b, poly);
}

void mlwe_RNS_mul_addto_by_poly(RNS_MLWE out, RNS_MLWE in, const ArithElement *poly)
{
    for (size_t i = 0; i < out->r; i++)
    {
        arith_mul_addto(out->ring, &out->a[i], &in->a[i], poly);
    }
    arith_mul_addto(out->ring, &out->b, &in->b, poly);
}

void mlwe_RNS_mul_subto_by_poly(RNS_MLWE out, RNS_MLWE in, const ArithElement *poly)
{
    for (size_t i = 0; i < out->r; i++)
    {
        arith_mul_subto(out->ring, &out->a[i], &in->a[i], poly);
    }
    arith_mul_subto(out->ring, &out->b, &in->b, poly);
}

void mlwe_RNS_linear_combination(RNS_MLWE out, RNS_MLWE *in, const ArithElement *coeff, uint64_t n)
{
    int started = 0;
    for (uint64_t i = 0; i < n; i++)
    {
        if (coeff[i].handle == NULL)
            continue;
        if (started)
            mlwe_RNS_mul_addto_by_poly(out, in[i], &coeff[i]);
        else
            mlwe_RNS_mul_by_poly(out, in[i], &coeff[i]);
        started = 1;
    }
    if (!started)
        mlwe_RNS_trivial_sample_of_zero(out);
}

typedef struct
{
    RNS_MLWE *out;
    RNS_MLWE *in;
    const ArithElement *coeff;
    uint64_t n_in;
} LinearCombinations;

static void linear_combination_job(void *ctx, uint64_t j)
{
    LinearCombinations *lc = (LinearCombinations *)ctx;
    mlwe_RNS_linear_combination(lc->out[j], lc->in, &lc->coeff[j * lc->n_in], lc->n_in);
}

void mlwe_RNS_linear_combinations(RNS_MLWE *out, RNS_MLWE *in, const ArithElement *coeff,
                                  uint64_t n_out, uint64_t n_in, uint64_t n_threads)
{
    LinearCombinations lc = {out, in, coeff, n_in};
    vfhe_parallel_for(n_out, n_threads, linear_combination_job, &lc);
}

RNSc_MLWE mlwe_new_RNSc_sample_of_zero(RNS_MLWE_Key key)
{
    RNSc_MLWE res = mlwe_alloc_sample(key->ring, key->r);
    mlwe_RNSc_sample_of_zero(res, key);
    return res;
}

RNS_MLWE mlwe_new_RNS_sample_of_zero(RNS_MLWE_Key key)
{
    RNS_MLWE res = mlwe_alloc_sample(key->ring, key->r);
    mlwe_RNS_sample_of_zero(res, key);
    return res;
}

void mlwe_RNS_trivial_sample_of_zero(RNS_MLWE out)
{
    const ArithDomain d = arith_mul_domain(out->ring);
    for (size_t j = 0; j < out->r; j++)
    {
        arith_zero_in(out->ring, &out->a[j], d);
    }
    arith_zero_in(out->ring, &out->b, d);
}

void mlwe_automorphism_RNSc_GHS(RNSc_MLWE out, RNSc_MLWE in, uint64_t gen, RNS_MLWE_KS_Key ksk,
                                uint64_t lvl)
{
    RNSc_MLWE tmp = mlwe_alloc_sample(out->ring, out->r);
    for (size_t i = 0; i < out->r; i++)
    {
        arith_permute(out->ring, &tmp->a[i], &in->a[i], gen);
    }
    arith_permute(out->ring, &tmp->b, &in->b, gen);
    mlwe_RNSc_GHS_hybrid_keyswitch(out, tmp, ksk, lvl);
    free_mlwe_RNS_sample(tmp);
}

struct _MLWE_Hoisted
{
    RNSc_MLWE in;
    // One per component; empty for a component the key passes through.
    GadgetDigits *digits;
    ArithRing key_ring;
    uint64_t log_base;
};

MLWE_Hoisted mlwe_hoist(RNSc_MLWE in, RNS_MLWE_KS_Key ksk)
{
    assert(ksk->count == in->r);
    MLWE_Hoisted h = (MLWE_Hoisted)safe_malloc(sizeof(*h));
    h->in = mlwe_alloc_sample(in->ring, in->r);
    mlwe_copy_RNSc_sample(h->in, in);
    h->key_ring = ksk->ring;
    h->log_base = ksk->log_base;
    h->digits = (GadgetDigits *)safe_malloc(in->r * sizeof(GadgetDigits));
    for (size_t i = 0; i < in->r; i++)
    {
        if (ksk->s[i] == NULL)
            h->digits[i] = (GadgetDigits){NULL, 0};
        else
            gadget_decompose(&h->digits[i], ksk->s[i], &in->a[i], ksk->log_base, ksk->balanced);
    }
    return h;
}

void free_mlwe_hoisted(MLWE_Hoisted h)
{
    for (size_t i = 0; i < h->in->r; i++)
        gadget_digits_free(&h->digits[i]);
    free(h->digits);
    free_mlwe_RNS_sample(h->in);
    free(h);
}

static int mlwe_hoisted_fits(MLWE_Hoisted h, RNS_MLWE_KS_Key ksk)
{
    if (ksk->ring != h->key_ring || ksk->log_base != h->log_base || ksk->count != h->in->r)
        return 0;
    for (size_t i = 0; i < h->in->r; i++)
    {
        if ((ksk->s[i] == NULL) != (h->digits[i].n == 0))
            return 0;
    }
    return 1;
}

// Like mlwe_automorphism_RNSc_GHS, but reusing the decomposition in `h`.
int mlwe_automorphism_RNSc_GHS_hoisted(RNSc_MLWE out, MLWE_Hoisted h, uint64_t gen,
                                       RNS_MLWE_KS_Key ksk, uint64_t lvl)
{
    (void)lvl;
    if (!mlwe_hoisted_fits(h, ksk))
        return -1;
    RNSc_MLWE in = h->in;
    RNSc_MLWE acc = mlwe_alloc_sample(ksk->ring, out->r);
    mlwe_RNS_trivial_sample_of_zero(acc);
    for (size_t i = 0; i < in->r; i++)
    {
        if (ksk->s[i] != NULL)
            gadget_mul_subto_automorphism(acc, ksk->s[i], &h->digits[i], gen);
    }
    mlwe_RNS_to_RNSc(acc, acc);
    mlwe_round_division(acc, in->ring);

    // Add the pass-through components and b after the rescale, as the key
    // switch does.
    ArithElement permuted;
    arith_new(in->ring, &permuted);
    size_t keep_idx = 0;
    for (size_t i = 0; i < in->r; i++)
    {
        if (ksk->s[i] == NULL)
        {
            arith_permute(in->ring, &permuted, &in->a[i], gen);
            arith_add(acc->ring, &acc->a[keep_idx], &acc->a[keep_idx], &permuted);
            keep_idx++;
        }
    }
    arith_permute(in->ring, &permuted, &in->b, gen);
    arith_add(acc->ring, &acc->b, &acc->b, &permuted);
    arith_free(in->ring, &permuted);

    mlwe_copy_RNSc_sample(out, acc);
    free_mlwe_RNS_sample(acc);
    return 0;
}

typedef struct
{
    RNSc_MLWE *out;
    MLWE_Hoisted h;
    RNSc_MLWE *in;
    const uint64_t *gens;
    RNS_MLWE_KS_Key *ksks;
    uint64_t lvl;
} AutomorphismJobs;

static void hoisted_automorphism_job(void *ctx, uint64_t i)
{
    AutomorphismJobs *jobs = (AutomorphismJobs *)ctx;
    mlwe_automorphism_RNSc_GHS_hoisted(jobs->out[i], jobs->h, jobs->gens[i], jobs->ksks[i],
                                       jobs->lvl);
}

int mlwe_automorphisms_RNSc_GHS_hoisted(RNSc_MLWE *out, MLWE_Hoisted h, const uint64_t *gens,
                                        RNS_MLWE_KS_Key *ksks, uint64_t n, uint64_t lvl,
                                        uint64_t n_threads)
{
    for (uint64_t i = 0; i < n; i++)
    {
        if (!mlwe_hoisted_fits(h, ksks[i]))
            return -1;
    }
    AutomorphismJobs jobs = {out, h, NULL, gens, ksks, lvl};
    vfhe_parallel_for(n, n_threads, hoisted_automorphism_job, &jobs);
    return 0;
}

static void automorphism_job(void *ctx, uint64_t i)
{
    AutomorphismJobs *jobs = (AutomorphismJobs *)ctx;
    if (jobs->gens[i] == 1)
        mlwe_copy_RNSc_sample(jobs->out[i], jobs->in[i]);
    else
        mlwe_automorphism_RNSc_GHS(jobs->out[i], jobs->in[i], jobs->gens[i], jobs->ksks[i],
                                   jobs->lvl);
}

void mlwe_automorphism_RNSc_GHS_batch(RNSc_MLWE *out, RNSc_MLWE *in, const uint64_t *gens,
                                      RNS_MLWE_KS_Key *ksks, uint64_t n, uint64_t lvl,
                                      uint64_t n_threads)
{
    AutomorphismJobs jobs = {out, NULL, in, gens, ksks, lvl};
    vfhe_parallel_for(n, n_threads, automorphism_job, &jobs);
}

void mlwe_partial_trace(RNSc_MLWE out, RNSc_MLWE in, uint64_t *gens, RNS_MLWE_KS_Key *ksks,
                        uint64_t size, uint64_t lvl)
{
    RNSc_MLWE tmp = mlwe_alloc_sample(out->ring, out->r);
    mlwe_copy_RNSc_sample(tmp, in);
    for (size_t i = 0; i < size; i++)
    {
        mlwe_automorphism_RNSc_GHS(out, tmp, gens[i], ksks[i], lvl);
        mlwe_addto_RNSc_sample(tmp, out);
    }
    mlwe_copy_RNSc_sample(out, tmp);
    free_mlwe_RNS_sample(tmp);
}

void mlwe_trace(RNSc_MLWE out, RNSc_MLWE in, RNS_MLWE_KS_Key *ksks, uint64_t lvl)
{
    const uint64_t log_N = (uint64_t)log2(in->ring->N);
    uint64_t *gens = (uint64_t *)malloc(log_N * sizeof(uint64_t));
    for (size_t i = 1; i <= log_N; i++)
        gens[i - 1] = (1ULL << (log_N - i + 1)) + 1;
    mlwe_partial_trace(out, in, gens, ksks, log_N, lvl);
    free(gens);
}

void mlwe_add_RNSc_sample(RNSc_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2)
{
    for (size_t i = 0; i < out->r; i++)
    {
        arith_add(out->ring, &out->a[i], &in1->a[i], &in2->a[i]);
    }
    arith_add(out->ring, &out->b, &in1->b, &in2->b);
}

void mlwe_add_RNS_sample(RNS_MLWE out, RNS_MLWE in1, RNS_MLWE in2)
{
    for (size_t i = 0; i < out->r; i++)
    {
        arith_add(out->ring, &out->a[i], &in1->a[i], &in2->a[i]);
    }
    arith_add(out->ring, &out->b, &in1->b, &in2->b);
}

void mlwe_add_RNSc_polynomial(RNSc_MLWE out, RNSc_MLWE in1, const ArithElement *in2)
{
    arith_add(out->ring, &out->b, &in1->b, in2);
}

void mlwe_sub_RNSc_polynomial(RNSc_MLWE out, RNSc_MLWE in1, const ArithElement *in2)
{
    arith_sub(out->ring, &out->b, &in1->b, in2);
}

void mlwe_RNS_add_polynomial(RNS_MLWE out, RNS_MLWE in1, const ArithElement *in2)
{
    arith_add(out->ring, &out->b, &in1->b, in2);
}

void mlwe_RNS_sub_polynomial(RNS_MLWE out, RNS_MLWE in1, const ArithElement *in2)
{
    arith_sub(out->ring, &out->b, &in1->b, in2);
}

void mlwe_sub_RNSc_sample(RNSc_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2)
{
    for (size_t i = 0; i < out->r; i++)
    {
        arith_sub(out->ring, &out->a[i], &in1->a[i], &in2->a[i]);
    }
    arith_sub(out->ring, &out->b, &in1->b, &in2->b);
}

void mlwe_RNSc_mul_by_xai(RNSc_MLWE out, RNSc_MLWE in, uint64_t a)
{
    for (size_t i = 0; i < out->r; i++)
    {
        arith_mul_by_monomial(out->ring, &out->a[i], &in->a[i], a, 0);
    }
    arith_mul_by_monomial(out->ring, &out->b, &in->b, a, 0);
}

void mlwe_RNSc_mul_by_xai_minus1(RNSc_MLWE out, RNSc_MLWE in, uint64_t a)
{
    for (size_t i = 0; i < out->r; i++)
    {
        arith_mul_by_monomial(out->ring, &out->a[i], &in->a[i], a, 1);
    }
    arith_mul_by_monomial(out->ring, &out->b, &in->b, a, 1);
}

void mlwe_addto_RNSc_sample(RNSc_MLWE out, RNSc_MLWE in) { mlwe_add_RNSc_sample(out, out, in); }

void mlwe_RNSc_to_RNS(RNS_MLWE out, RNSc_MLWE in)
{
    for (size_t i = 0; i < out->r; i++)
    {
        arith_copy(out->ring, &out->a[i], &in->a[i]);
        arith_to_mul(out->ring, &out->a[i]);
    }
    arith_copy(out->ring, &out->b, &in->b);
    arith_to_mul(out->ring, &out->b);
}

void mlwe_RNS_to_RNSc(RNSc_MLWE out, RNS_MLWE in)
{
    for (size_t i = 0; i < out->r; i++)
    {
        arith_copy(out->ring, &out->a[i], &in->a[i]);
        arith_to_canonical(out->ring, &out->a[i]);
    }
    arith_copy(out->ring, &out->b, &in->b);
    arith_to_canonical(out->ring, &out->b);
}

// GHS hybrid key switch. The product must accumulate in the ring the key
// lives in: `out` is only guaranteed to be allocated for its own, narrower
// ring, so the key carries an accumulator that is, and only the finished
// result -- already back in `in`'s ring -- reaches `out`.
void mlwe_RNSc_GHS_hybrid_keyswitch(RNSc_MLWE out, RNSc_MLWE in, RNS_MLWE_KS_Key ksk, uint64_t lvl)
{
    (void)lvl;
    assert(in != out);
    // Scratch of its own: the key is shared across threads, so nothing
    // hanging off it may be written.
    RNSc_MLWE acc = mlwe_alloc_sample(ksk->ring, out->r);
    ArithRing target = in->ring;

    // compute -a_i^T * ksk_i. A NULL ksk->s[i] marks a component that keeps
    // the target key (e.g. the linear part during relinearization); it is
    // copied through below instead of being key-switched.
    mlwe_RNS_trivial_sample_of_zero(acc);
    for (size_t i = 0; i < in->r; i++)
    {
        if (ksk->s[i] != NULL)
        {
            gadget_mul_subto_polynomial(acc, ksk->s[i], &in->a[i], ksk->log_base, ksk->balanced);
        }
    }
    // convert to RNSc and rescale to in's ring
    mlwe_RNS_to_RNSc(acc, acc);
    mlwe_round_division(acc, target);

    // Fold in the components that keep the target key. These stay in the base
    // ring, so they are added *after* the rescale (never divided out). The k-th
    // pass-through a-component lands in acc->a[k]; in->b always passes through.
    size_t keep_idx = 0;
    for (size_t i = 0; i < in->r; i++)
    {
        if (ksk->s[i] == NULL)
        {
            arith_add(acc->ring, &acc->a[keep_idx], &acc->a[keep_idx], &in->a[i]);
            keep_idx++;
        }
    }
    arith_add(acc->ring, &acc->b, &acc->b, &in->b);
    mlwe_copy_RNSc_sample(out, acc);
    free_mlwe_RNS_sample(acc);
}

RNS_MLWE_KS_Key mlwe_new_RNS_automorphism_key(RNS_MLWE_Key key, uint64_t gen)
{
    (void)key;
    (void)gen;
    assert(false); // todo: reimplement to consider new ring
    return NULL;
}

void mlwe_full_packing_keyswitch_scaled_rec(RNSc_MLWE *vec, uint64_t ell, RNS_MLWE_KS_Key *ksks,
                                            uint64_t lvl)
{
    if (ell == 0)
    {
        return;
    }
    const uint64_t half = 1ULL << (ell - 1);
    RNSc_MLWE *even = (RNSc_MLWE *)malloc(half * sizeof(RNSc_MLWE));
    RNSc_MLWE *odd = (RNSc_MLWE *)malloc(half * sizeof(RNSc_MLWE));
    for (size_t i = 0; i < half; i++)
    {
        even[i] = vec[2 * i];
        odd[i] = vec[2 * i + 1];
    }

    mlwe_full_packing_keyswitch_scaled_rec(even, ell - 1, ksks, lvl);
    mlwe_full_packing_keyswitch_scaled_rec(odd, ell - 1, ksks, lvl);

    RNSc_MLWE C_tilde = even[0];
    const uint64_t N = vec[0]->ring->N;
    const uint64_t r = vec[0]->r;

    RNSc_MLWE tmp = mlwe_alloc_sample(vec[0]->ring, r);
    RNSc_MLWE tmp2 = mlwe_alloc_sample(ksks[ell - 1]->ring, r);

    // tmp = odd[0] * X^(N>>ell)
    mlwe_RNSc_mul_by_xai(tmp, odd[0], N >> ell);

    // C_tilde = even[0] - tmp
    mlwe_sub_RNSc_sample(C_tilde, even[0], tmp);

    // tmp2 = autom(C_tilde, (1<<ell) + 1)
    uint64_t gen = (1ULL << ell) + 1;
    mlwe_automorphism_RNSc_GHS(tmp2, C_tilde, gen, ksks[ell - 1], lvl);

    // C_tilde = C_tilde + tmp2 + 2 * tmp
    mlwe_scale_RNSc_mlwe(tmp, 2);
    mlwe_addto_RNSc_sample(C_tilde, tmp2);
    mlwe_addto_RNSc_sample(C_tilde, tmp);

    free_mlwe_RNS_sample(tmp);
    free_mlwe_RNS_sample(tmp2);
    free(even);
    free(odd);
}

void mlwe_full_packing_keyswitch_scaled(RNSc_MLWE *vec, uint64_t ell, RNS_MLWE_KS_Key *ksks,
                                        uint64_t lvl)
{
    if (ell == 0)
    {
        return;
    }
    mlwe_full_packing_keyswitch_scaled_rec(vec, ell, ksks, lvl);
}

uint64_t mlwe_extended_rank(uint64_t r)
{
    // r quadratic pairs (i <= j) is r*(r+1)/2, plus the r linear components.
    return r * (r + 3) / 2;
}

void mlwe_tensor_product(ArithElement *out, RNS_MLWE in1, RNS_MLWE in2)
{
    // Symmetric tensor product of the two ciphertext vectors. With the linear
    // decryption linear_decrypt(c) = b - sum_i a_i * s_i, the product of the
    // two linear decryptions is
    //
    //   b1*b2 - sum_i (a1_i*b2 + b1*a2_i) * s_i + sum_{i<=j} q_ij * s_i*s_j,
    //   where q_ij = a1_i*a2_j + a1_j*a2_i (i < j) and q_ii = a1_i*a2_i,
    //
    // so the product decrypts under the extended key made of the r*(r+1)/2
    // quadratic terms -(s_i*s_j) followed by the r linear terms s_i. The output
    // slots follow that same order: the quadratic pairs in lexicographic (i,j)
    // order with i <= j, then the linear components, then the constant term in
    // out[R] (R = mlwe_extended_rank(r)). At r = 1 this is O[0] = a1*a2,
    // O[1] = a1*b2 + b1*a2, O[2] = b1*b2, matching a plain convolution.
    const uint64_t r = in1->r;
    assert(in2->r == r);
    const uint64_t R = mlwe_extended_rank(r);
    size_t k = 0;

    // Quadratic slots: q_ij for i <= j.
    for (size_t i = 0; i < r; i++)
    {
        for (size_t j = i; j < r; j++)
        {
            arith_mul(in1->ring, &out[k], &in1->a[i], &in2->a[j]);
            if (i != j)
            {
                arith_mul_addto(in1->ring, &out[k], &in1->a[j], &in2->a[i]);
            }
            k++;
        }
    }

    // Linear slots: a1_i*b2 + b1*a2_i.
    for (size_t i = 0; i < r; i++)
    {
        arith_mul(in1->ring, &out[k], &in1->a[i], &in2->b);
        arith_mul_addto(in1->ring, &out[k], &in1->b, &in2->a[i]);
        k++;
    }
    assert(k == R);

    // Constant term.
    arith_mul(in1->ring, &out[R], &in1->b, &in2->b);
}

void mlwe_multiply(RNS_MLWE out, RNS_MLWE in1, RNS_MLWE in2, RNS_MLWE_KS_Key ksk)
{
    const uint64_t r = in1->r;

    // The tensor product produces a rank-R ciphertext (R = r*(r+3)/2): R "a"
    // components O[0..R-1] plus the constant term O[R]. Lay it out over an MLWE
    // so the components map to a-slots and b directly.
    const uint64_t R = mlwe_extended_rank(r);
    if (ksk == NULL)
    {
        // No relinearization key: hand back the extended (rank-R) product.
        assert(out->r == R);
        ArithElement *tensor = (ArithElement *)malloc((R + 1) * sizeof(ArithElement));
        for (size_t j = 0; j < R; j++)
        {
            tensor[j] = out->a[j];
        }
        tensor[R] = out->b;
        mlwe_tensor_product(tensor, in1, in2);
        free(tensor);
        return;
    }

    // Relinearize down to rank r by reusing the GHS hybrid key-switch. The rlk
    // carries a real key-switch key for each of the R-r quadratic components
    // (O[0..R-r-1]) and NULL for each of the r linear components (O[R-r..R-1]),
    // which keep the target key and are copied through by the key-switch.
    assert(out->r == r);
    RNS_MLWE ext = mlwe_alloc_sample(in1->ring, R);
    ArithElement *tensor = (ArithElement *)malloc((R + 1) * sizeof(ArithElement));
    for (size_t j = 0; j < R; j++)
    {
        tensor[j] = ext->a[j];
    }
    tensor[R] = ext->b;
    mlwe_tensor_product(tensor, in1, in2);
    free(tensor);

    RNSc_MLWE ext_c = mlwe_alloc_sample(in1->ring, R);
    mlwe_RNS_to_RNSc(ext_c, ext);
    mlwe_RNSc_GHS_hybrid_keyswitch(out, ext_c, ksk, 0);
    // Restore the NTT representation callers expect from a product.
    mlwe_RNSc_to_RNS(out, out);

    free_mlwe_RNS_sample(ext);
    free_mlwe_RNS_sample(ext_c);
}

typedef struct
{
    RNS_MLWE *out;
    RNS_MLWE *in1;
    RNS_MLWE *in2;
    RNS_MLWE_KS_Key ksk;
    ArithRing to;
} SampleJobs;

static void multiply_job(void *ctx, uint64_t i)
{
    SampleJobs *jobs = (SampleJobs *)ctx;
    mlwe_multiply(jobs->out[i], jobs->in1[i], jobs->in2[i], jobs->ksk);
}

void mlwe_multiply_batch(RNS_MLWE *out, RNS_MLWE *in1, RNS_MLWE *in2, RNS_MLWE_KS_Key ksk,
                         uint64_t n, uint64_t n_threads)
{
    SampleJobs jobs = {out, in1, in2, ksk, NULL};
    vfhe_parallel_for(n, n_threads, multiply_job, &jobs);
}

static void round_division_job(void *ctx, uint64_t i)
{
    SampleJobs *jobs = (SampleJobs *)ctx;
    RNS_MLWE c = jobs->out[i];
    if (mlwe_domain(c) == ARITH_DOMAIN_MUL)
        mlwe_RNS_to_RNSc(c, c);
    mlwe_round_division(c, jobs->to);
}

void mlwe_round_division_batch(RNSc_MLWE *io, ArithRing to, uint64_t n, uint64_t n_threads)
{
    SampleJobs jobs = {io, NULL, NULL, NULL, to};
    vfhe_parallel_for(n, n_threads, round_division_job, &jobs);
}

static void to_mul_domain_job(void *ctx, uint64_t i)
{
    SampleJobs *jobs = (SampleJobs *)ctx;
    RNS_MLWE c = jobs->out[i];
    if (mlwe_domain(c) != ARITH_DOMAIN_MUL)
        mlwe_RNSc_to_RNS(c, c);
}

void mlwe_RNSc_to_RNS_batch(RNS_MLWE *io, uint64_t n, uint64_t n_threads)
{
    SampleJobs jobs = {io, NULL, NULL, NULL, NULL};
    vfhe_parallel_for(n, n_threads, to_mul_domain_job, &jobs);
}

// Rescale every component down to `to`. Which primes leave is the ring's
// business, not the caller's.
void mlwe_round_division(RNSc_MLWE out, ArithRing to)
{
    if (out->ring == to)
    {
        return;
    }
    for (size_t j = 0; j < out->r; j++)
    {
        arith_round_division(out->ring, &out->a[j], to);
    }
    arith_round_division(out->ring, &out->b, to);
    out->ring = to;
}
