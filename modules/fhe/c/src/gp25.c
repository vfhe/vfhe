// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "fhe.h"

#include <assert.h>
#include <stdlib.h>
#include <util.h>

/* ------------------------------------------------------------------------------------------------
 * The blind rotation of [GP25] runs on a vector of n accumulators. Per input component, a sparse
 * key s = sum_t s_t Y^j_t (j_1 > ... > j_h) is consumed as h + 1 gaps -- n - j_1, j_1 - j_2, ...,
 * j_h, which sum to n -- each an oblivious rotation of the vector, and after each of the first h
 * a multiplication of every accumulator k by X^(+-a_k), the sign that of s_t. A rotation by a
 * gap is one layer per bit of it, and a layer is n independent CMUXes, one per accumulator, so
 * each layer is one parallel loop. The signed monomial of accumulator k needs only accumulator k,
 * so it runs inside the last layer of its gap, on the same item.
 *
 * A layer can also take two bits, (v_0, v_1) at power p = 2^b: accumulator k takes accumulator
 * k - d p, d = v_0 + 2 v_1, given as three MGSW keys of [d = 1], [d = 2], [d = 3]. Along each chain
 * k, k + p, k + 2p, ... that is a shift by d, so the chains, cut into segments, are the sequences
 * of mgsw_shift_sequences, each preceded by the three accumulators before it; before a chain's
 * start, those are the chain's last ones through sigma. Each accumulator is then decomposed once
 * for two bits instead of twice, plus three per segment. There are at least p segments, so bits
 * pair only while 4 p <= n, and the remaining bits take a layer each.
 *
 * Layers read one array of accumulators and write the other. The arrays alternate over the whole
 * rotation; if the result ends in the scratch array, each accumulator exchanges its storage with
 * the caller's, so nothing is copied.
 * ------------------------------------------------------------------------------------------------
 */

typedef struct
{
    RNSc_MLWE *in, *out; // out == NULL: no rotation, the monomial is applied to `in` in place
    uint64_t n, power;
    RNS_MLWE *bit; // MGSW of this layer's bit
    RNS_MLWE_KS_Key aut;
    const uint64_t *a; // NULL: no monomial
    RNS_MLWE *sign;    // MGSW of the sign bit; NULL: X^a_k whatever the sign
    uint64_t ell;
    const GadgetParams *gadget_params;
} Layer;

// acc <- acc * X^a if the sign is 0, acc * X^-a if it is 1: a CMUX between the two.
static void signed_monomial(RNSc_MLWE acc, uint64_t a, RNS_MLWE *sign, uint64_t ell,
                            const GadgetParams *gadget_params)
{
    const uint64_t two_n = 2 * acc->ring->N;
    RNSc_MLWE positive = mlwe_alloc_sample(acc->ring, acc->r);
    mlwe_RNSc_mul_by_xai(positive, acc, a % two_n);
    if (sign == NULL)
    {
        mlwe_copy_RNSc_sample(acc, positive);
        free_mlwe_RNS_sample(positive);
        return;
    }
    RNSc_MLWE negative = mlwe_alloc_sample(acc->ring, acc->r);
    mlwe_RNSc_mul_by_xai(negative, acc, (two_n - a % two_n) % two_n);
    mgsw_CMUX_to_coeff(acc, positive, negative, sign, ell, 0, gadget_params);
    free_mlwe_RNS_sample(positive);
    free_mlwe_RNS_sample(negative);
}

static void layer_item(void *ctx, uint64_t k)
{
    const Layer *L = (const Layer *)ctx;
    RNSc_MLWE acc = L->in[k];
    if (L->out != NULL)
    {
        acc = L->out[k];
        if (k < L->power)
            mgsw_NCMUX_to_coeff(acc, L->in[k], L->in[L->n - L->power + k], L->bit, L->aut, L->ell,
                                0, L->gadget_params);
        else
            mgsw_CMUX_to_coeff(acc, L->in[k], L->in[k - L->power], L->bit, L->ell, 0,
                               L->gadget_params);
    }
    if (L->a != NULL)
        signed_monomial(acc, L->a[k], L->sign, L->ell, L->gadget_params);
}

// The accumulators as a pair of arrays, `cur` holding the current values.
typedef struct
{
    RNSc_MLWE *caller, *scratch, *cur, *next;
    uint64_t n;
} Accumulators;

static void alloc_item(void *ctx, uint64_t k)
{
    const Accumulators *A = (const Accumulators *)ctx;
    A->scratch[k] = mlwe_alloc_sample(A->caller[0]->ring, A->caller[0]->r);
}

static void accumulators_init(Accumulators *A, RNSc_MLWE *acc, uint64_t n, uint64_t n_threads)
{
    A->caller = acc;
    A->n = n;
    A->scratch = (RNSc_MLWE *)safe_malloc(n * sizeof(RNSc_MLWE));
    vfhe_parallel_for(n, n_threads, alloc_item, A);
    A->cur = acc;
    A->next = A->scratch;
}

// Accumulator k ends in the caller's sample: if it is in the scratch one, the two samples
// exchange their components (they share a ring and a rank), and the scratch sample, which then
// holds the caller's old components, is freed either way.
static void finish_item(void *ctx, uint64_t k)
{
    const Accumulators *A = (const Accumulators *)ctx;
    RNSc_MLWE caller = A->caller[k], scratch = A->scratch[k];
    if (A->cur != A->caller)
    {
        assert(caller->r == scratch->r);
        ArithElement *a = caller->a;
        caller->a = scratch->a;
        scratch->a = a;
        const ArithElement b = caller->b;
        caller->b = scratch->b;
        scratch->b = b;
    }
    free_mlwe_RNS_sample(scratch);
}

static void accumulators_finish(Accumulators *A, uint64_t n_threads)
{
    vfhe_parallel_for(A->n, n_threads, finish_item, A);
    free(A->scratch);
}

// The number of bits of the layer that starts at bit b.
static uint64_t layer_width(uint64_t n, uint64_t n_bits, uint64_t bits_per_layer, uint64_t b)
{
    return bits_per_layer == 2 && b + 1 < n_bits && (4ULL << b) <= n ? 2 : 1;
}

uint64_t gp25_rotation_layers(uint64_t n, uint64_t n_bits, uint64_t bits_per_layer, uint64_t *bits)
{
    uint64_t layers = 0;
    for (uint64_t b = 0; b < n_bits; layers++)
    {
        const uint64_t width = layer_width(n, n_bits, bits_per_layer, b);
        if (bits != NULL)
            bits[layers] = width;
        b += width;
    }
    return layers;
}

// The number of MGSW keys of a rotation: one per one-bit layer, three per two-bit layer.
static uint64_t rotation_keys(uint64_t n, uint64_t n_bits, uint64_t bits_per_layer)
{
    const uint64_t layers = gp25_rotation_layers(n, n_bits, bits_per_layer, NULL);
    return layers + 2 * (n_bits - layers);
}

// A two-bit layer cuts its chains into segments for about two per thread, but of at least
// MIN_SEGMENT accumulators, so the three inputs before each stay a small share.
enum
{
    MIN_SEGMENT = 16
};

// The two-bit layer at power p: out[k] = in[k - d p], or sigma(in[k - d p + n]) when k < d p.
static void two_bit_layer(RNSc_MLWE *out, RNSc_MLWE *in, uint64_t n, uint64_t p,
                          RNS_MLWE *const *keys, RNS_MLWE_KS_Key aut, uint64_t ell,
                          const GadgetParams *gadget_params, uint64_t n_threads)
{
    const uint64_t sigma = 2 * in[0]->ring->N - 1;
    const uint64_t tasks = 2 * vfhe_threads_for(n_threads, n);
    uint64_t segment = (n + tasks - 1) / tasks;
    segment = segment < MIN_SEGMENT ? MIN_SEGMENT : segment;

    uint64_t n_sequences = 0;
    for (uint64_t chain = 0; chain < p && chain < n; chain++)
        n_sequences += ((n - chain + p - 1) / p + segment - 1) / segment;
    uint64_t *lengths = (uint64_t *)safe_malloc(n_sequences * sizeof(uint64_t));
    RNSc_MLWE *inputs = (RNSc_MLWE *)safe_malloc((n + 3 * n_sequences) * sizeof(RNSc_MLWE));
    uint64_t *gens = (uint64_t *)safe_malloc((n + 3 * n_sequences) * sizeof(uint64_t));
    RNSc_MLWE *outputs = (RNSc_MLWE *)safe_malloc(n * sizeof(RNSc_MLWE));

    uint64_t seq = 0, n_inputs = 0, n_outputs = 0;
    for (uint64_t chain = 0; chain < p && chain < n; chain++)
    {
        const uint64_t length = (n - chain + p - 1) / p;
        for (uint64_t start = 0; start < length; start += segment, seq++)
        {
            lengths[seq] = length - start < segment ? length - start : segment;
            // Input u of the segment is accumulator chain + p (start + u), u >= -3, through sigma
            // once per wrap below 0; the inputs from u = 0 are the segment's own.
            for (int64_t u = -3; u < (int64_t)lengths[seq]; u++)
            {
                int64_t k = (int64_t)(chain + p * start) + u * (int64_t)p;
                uint64_t wraps = 0;
                for (; k < 0; k += (int64_t)n)
                    wraps++;
                inputs[n_inputs] = in[k];
                gens[n_inputs++] = wraps & 1 ? sigma : 1;
                if (u >= 0)
                    outputs[n_outputs++] = out[k];
            }
        }
    }
    mgsw_shift_sequences(outputs, inputs, gens, aut, lengths, n_sequences, keys, 3, ell,
                         gadget_params, n_threads);
    free(lengths);
    free(inputs);
    free(gens);
    free(outputs);
}

// Rotates by the value `keys` encrypt, in the layers of gp25_rotation_layers; with `a`, then
// multiplies by the signed monomials.
static void rotate_then_multiply(Accumulators *A, RNS_MLWE *const *keys, uint64_t n_bits,
                                 uint64_t bits_per_layer, RNS_MLWE_KS_Key aut, const uint64_t *a,
                                 RNS_MLWE *sign, uint64_t ell, const GadgetParams *gadget_params,
                                 uint64_t n_threads)
{
    Layer L = {A->cur, NULL, A->n, 0, NULL, aut, NULL, sign, ell, gadget_params};
    int pending = a != NULL; // the monomials, until the last layer, if one-bit, takes them
    for (uint64_t b = 0; b < n_bits;)
    {
        const uint64_t width = layer_width(A->n, n_bits, bits_per_layer, b);
        if (width == 2)
        {
            two_bit_layer(A->next, A->cur, A->n, 1ULL << b, keys, aut, ell, gadget_params,
                          n_threads);
            keys += 3;
        }
        else
        {
            L.in = A->cur;
            L.out = A->next;
            L.power = 1ULL << b;
            L.bit = keys[0];
            L.a = b + 1 == n_bits ? a : NULL;
            pending = pending && L.a == NULL;
            vfhe_parallel_for(A->n, n_threads, layer_item, &L);
            keys += 1;
        }
        RNSc_MLWE *result = A->next;
        A->next = A->cur;
        A->cur = result;
        b += width;
    }
    if (pending)
    {
        L.in = A->cur;
        L.out = NULL;
        L.a = a;
        vfhe_parallel_for(A->n, n_threads, layer_item, &L);
    }
}

void gp25_rotate(RNSc_MLWE *acc, uint64_t n, RNS_MLWE *const *keys, uint64_t n_bits,
                 uint64_t bits_per_layer, RNS_MLWE_KS_Key aut, uint64_t ell,
                 const GadgetParams *gadget_params, uint64_t n_threads)
{
    Accumulators A;
    accumulators_init(&A, acc, n, n_threads);
    rotate_then_multiply(&A, keys, n_bits, bits_per_layer, aut, NULL, NULL, ell, gadget_params,
                         n_threads);
    accumulators_finish(&A, n_threads);
}

void gp25_multiply_by_signed_monomials(RNSc_MLWE *acc, uint64_t n, const uint64_t *a,
                                       RNS_MLWE *sign, uint64_t ell,
                                       const GadgetParams *gadget_params, uint64_t n_threads)
{
    Layer L = {acc, NULL, n, 0, NULL, NULL, a, sign, ell, gadget_params};
    vfhe_parallel_for(n, n_threads, layer_item, &L);
}

void gp25_blind_rotate(RNSc_MLWE *acc, uint64_t n, const uint64_t *a, uint64_t rank, uint64_t h,
                       uint64_t gap_bits, uint64_t bits_per_layer, RNS_MLWE *const *gap_keys,
                       RNS_MLWE *const *sign_keys, RNS_MLWE_KS_Key aut, uint64_t ell,
                       const GadgetParams *gadget_params, uint64_t n_threads)
{
    const uint64_t two_n = 2 * acc[0]->ring->N;
    const uint64_t keys_per_gap = rotation_keys(n, gap_bits, bits_per_layer);
    Accumulators A;
    accumulators_init(&A, acc, n, n_threads);
    uint64_t *exponent = (uint64_t *)safe_malloc(n * sizeof(uint64_t));
    for (uint64_t i = 0; i < rank; i++)
    {
        // Each component's rotations wrap every accumulator once, which applies X -> X^-1 to
        // what it holds; the odd components are negated so the wraps do not cancel them.
        for (uint64_t k = 0; k < n; k++)
            exponent[k] = i & 1 ? (two_n - a[i * n + k] % two_n) % two_n : a[i * n + k] % two_n;
        for (uint64_t t = 0; t <= h; t++)
        {
            rotate_then_multiply(&A, gap_keys, gap_bits, bits_per_layer, aut,
                                 t < h ? exponent : NULL,
                                 t < h && sign_keys != NULL ? sign_keys[i * h + t] : NULL, ell,
                                 gadget_params, n_threads);
            gap_keys += keys_per_gap;
        }
    }
    free(exponent);
    accumulators_finish(&A, n_threads);
}
