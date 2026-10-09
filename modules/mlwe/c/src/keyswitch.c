// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "mlwe.h"
#include "rns_rows.h"
#include "util.h"

// Key switching and the operations built on it: the GHS hybrid key switch
// (a BV one when the key lives in the sample's own ring), automorphisms --
// single, hoisted, batched and summed --, the trace, packing through the
// trace, and ring switching. Everything here works one whole ring element at
// a time through arith_*; the gadget products are gadget_rns.c's, and what
// needs residues is in keyswitch_rns.c.

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

void mlwe_automorphism_RNSc_GHS(RNSc_MLWE out, RNSc_MLWE in, uint64_t gen, RNS_MLWE_KS_Key ksk,
                                uint64_t lvl)
{
    RNSc_MLWE tmp = mlwe_alloc_sample(out->ring, out->r);
    if (out->ring->impl == ARITH_IMPL_RNS)
        mlwe_rns_permute(tmp, in, gen);
    else
    {
        for (size_t i = 0; i < out->r; i++)
        {
            arith_permute(out->ring, &tmp->a[i], &in->a[i], gen);
        }
        arith_permute(out->ring, &tmp->b, &in->b, gen);
    }
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
    RNSc_MLWE out = jobs->out[i];
    RNSc_MLWE in = jobs->in[i];
    // A mul-domain input is converted into `out`, which then serves as the
    // input: the automorphism reads its input before writing.
    if (mlwe_domain(in) == ARITH_DOMAIN_MUL)
    {
        mlwe_RNS_to_RNSc(out, in);
        in = out;
    }
    if (jobs->gens[i] == 1)
    {
        if (in != out)
            mlwe_copy_RNSc_sample(out, in);
    }
    else
        mlwe_automorphism_RNSc_GHS(out, in, jobs->gens[i], jobs->ksks[i], jobs->lvl);
}

void mlwe_automorphism_RNSc_GHS_batch(RNSc_MLWE *out, RNSc_MLWE *in, const uint64_t *gens,
                                      RNS_MLWE_KS_Key *ksks, uint64_t n, uint64_t lvl,
                                      uint64_t n_threads)
{
    AutomorphismJobs jobs = {out, NULL, in, gens, ksks, lvl};
    vfhe_parallel_for(n, n_threads, automorphism_job, &jobs);
}

// Each worker sums a contiguous range of terms: key-switch products in the
// keys' ring (`acc`, mul domain) and pass-through parts in the output ring
// (`kept`, canonical). The ranges are added afterwards and `acc` is divided
// down once.
typedef struct
{
    RNS_MLWE *in;
    const uint64_t *gens;
    RNS_MLWE_KS_Key *ksks;
    uint64_t n, n_ranges;
    RNS_MLWE *acc;
    RNSc_MLWE *kept;
    ArithRing ring;
} AutomorphismSum;

static void automorphism_sum_job(void *ctx, uint64_t k)
{
    AutomorphismSum *sum = (AutomorphismSum *)ctx;
    RNS_MLWE acc = sum->acc[k];
    RNSc_MLWE kept = sum->kept[k];
    const uint64_t r = kept->r;
    RNSc_MLWE canonical = NULL;
    RNSc_MLWE permuted = mlwe_alloc_sample(sum->ring, r);
    mlwe_RNS_trivial_sample_of_zero(acc);
    for (size_t j = 0; j < r; j++)
        arith_zero_in(sum->ring, &kept->a[j], ARITH_DOMAIN_CANONICAL);
    arith_zero_in(sum->ring, &kept->b, ARITH_DOMAIN_CANONICAL);

    for (uint64_t i = k * sum->n / sum->n_ranges; i < (k + 1) * sum->n / sum->n_ranges; i++)
    {
        RNSc_MLWE in = sum->in[i];
        if (mlwe_domain(in) == ARITH_DOMAIN_MUL)
        {
            if (canonical == NULL)
                canonical = mlwe_alloc_sample(sum->ring, r);
            mlwe_RNS_to_RNSc(canonical, in);
            in = canonical;
        }
        if (sum->gens[i] == 1)
        {
            mlwe_addto_RNSc_sample(kept, in);
            continue;
        }
        for (size_t j = 0; j < r; j++)
            arith_permute(sum->ring, &permuted->a[j], &in->a[j], sum->gens[i]);
        arith_permute(sum->ring, &permuted->b, &in->b, sum->gens[i]);

        RNS_MLWE_KS_Key ksk = sum->ksks[i];
        size_t keep_idx = 0;
        for (size_t j = 0; j < r; j++)
        {
            if (ksk->s[j] != NULL)
                gadget_mul_subto_polynomial(acc, ksk->s[j], &permuted->a[j], ksk->log_base,
                                            ksk->balanced);
            else
            {
                arith_add(sum->ring, &kept->a[keep_idx], &kept->a[keep_idx], &permuted->a[j]);
                keep_idx++;
            }
        }
        arith_add(sum->ring, &kept->b, &kept->b, &permuted->b);
    }
    free_mlwe_RNS_sample(permuted);
    if (canonical != NULL)
        free_mlwe_RNS_sample(canonical);
}

int mlwe_automorphism_sum_RNSc_GHS(RNSc_MLWE out, RNS_MLWE *in, const uint64_t *gens,
                                   RNS_MLWE_KS_Key *ksks, uint64_t n, uint64_t n_threads)
{
    // All key products share one accumulator, so the keys must agree on its
    // ring and on which components pass through.
    RNS_MLWE_KS_Key first = NULL;
    for (uint64_t i = 0; i < n; i++)
    {
        if (gens[i] == 1)
            continue;
        RNS_MLWE_KS_Key ksk = ksks[i];
        if (ksk == NULL || ksk->count != out->r)
            return -1;
        if (first == NULL)
            first = ksk;
        else if (ksk->ring != first->ring)
            return -1;
        for (size_t j = 0; j < out->r; j++)
        {
            if ((ksk->s[j] == NULL) != (first->s[j] == NULL))
                return -1;
        }
    }

    const uint64_t n_ranges = n == 0 ? 1 : vfhe_threads_for(n_threads, n);
    AutomorphismSum sum = {in, gens, ksks, n, n_ranges, NULL, NULL, out->ring};
    sum.acc = (RNS_MLWE *)safe_malloc(n_ranges * sizeof(RNS_MLWE));
    sum.kept = (RNSc_MLWE *)safe_malloc(n_ranges * sizeof(RNSc_MLWE));
    for (uint64_t k = 0; k < n_ranges; k++)
    {
        // With no keyed term, `acc` is zero over out's ring.
        sum.acc[k] = mlwe_alloc_sample(first != NULL ? first->ring : out->ring, out->r);
        sum.kept[k] = mlwe_alloc_sample(out->ring, out->r);
    }
    vfhe_parallel_for(n_ranges, n_threads, automorphism_sum_job, &sum);

    RNS_MLWE acc = sum.acc[0];
    RNSc_MLWE kept = sum.kept[0];
    for (uint64_t k = 1; k < n_ranges; k++)
    {
        mlwe_add_RNS_sample(acc, acc, sum.acc[k]);
        mlwe_addto_RNSc_sample(kept, sum.kept[k]);
    }
    mlwe_RNS_to_RNSc(acc, acc);
    mlwe_round_division(acc, out->ring);
    mlwe_add_RNSc_sample(out, acc, kept);

    for (uint64_t k = 0; k < n_ranges; k++)
    {
        free_mlwe_RNS_sample(sum.acc[k]);
        free_mlwe_RNS_sample(sum.kept[k]);
    }
    free(sum.acc);
    free(sum.kept);
    return 0;
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

void mlwe_full_packing_keyswitch_scaled(RNSc_MLWE *vec, uint64_t ell, RNS_MLWE_KS_Key *ksks,
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

    mlwe_full_packing_keyswitch_scaled(even, ell - 1, ksks, lvl);
    mlwe_full_packing_keyswitch_scaled(odd, ell - 1, ksks, lvl);

    RNSc_MLWE C_tilde = even[0];
    const uint64_t N = vec[0]->ring->N;
    const uint64_t r = vec[0]->r;

    RNSc_MLWE tmp = mlwe_alloc_sample(vec[0]->ring, r);
    RNSc_MLWE tmp2 = mlwe_alloc_sample(vec[0]->ring, r);

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

// Ring switching: the subring map into the key ring's dimension, then a key
// switch, which may change the rank, from the key the map leaves the sample
// under.
void mlwe_ring_switch(RNSc_MLWE out, RNSc_MLWE *in, uint64_t count, RNS_MLWE_KS_Key ksk)
{
    const uint64_t N = in[0]->ring->N, n = out->ring->N;
    const uint64_t rank = N >= n ? in[0]->r * (N / n) : in[0]->r;
    assert(ksk->count == rank);
    assert(N < n || count == 1);
    RNSc_MLWE tmp = mlwe_alloc_sample(out->ring, rank);
    if (N >= n)
        mlwe_project_subring(tmp, in[0]);
    else
        mlwe_embed_subring(tmp, in, count);
    mlwe_RNSc_GHS_hybrid_keyswitch(out, tmp, ksk, 0);
    free_mlwe_RNS_sample(tmp);
}
