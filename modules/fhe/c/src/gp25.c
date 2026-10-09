// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "fhe.h"

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
 * Layers read one array of accumulators and write the other. The arrays alternate over the whole
 * rotation, and the result is copied back once at the end if it is not in the caller's.
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
    uint64_t ell, log_base;
    bool balanced;
} Layer;

// acc <- acc * X^a if the sign is 0, acc * X^-a if it is 1: a CMUX between the two.
static void signed_monomial(RNSc_MLWE acc, uint64_t a, RNS_MLWE *sign, uint64_t ell,
                            uint64_t log_base, bool balanced)
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
    mgsw_CMUX_to_coeff(acc, positive, negative, sign, ell, 0, log_base, balanced);
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
                                0, L->log_base, L->balanced);
        else
            mgsw_CMUX_to_coeff(acc, L->in[k], L->in[k - L->power], L->bit, L->ell, 0, L->log_base,
                               L->balanced);
    }
    if (L->a != NULL)
        signed_monomial(acc, L->a[k], L->sign, L->ell, L->log_base, L->balanced);
}

// The accumulators as a pair of arrays, `cur` holding the current values.
typedef struct
{
    RNSc_MLWE *caller, *scratch, *cur, *next;
    uint64_t n;
} Accumulators;

static void accumulators_init(Accumulators *A, RNSc_MLWE *acc, uint64_t n)
{
    A->caller = acc;
    A->n = n;
    A->scratch = (RNSc_MLWE *)safe_malloc(n * sizeof(RNSc_MLWE));
    for (uint64_t k = 0; k < n; k++)
        A->scratch[k] = mlwe_alloc_sample(acc[0]->ring, acc[0]->r);
    A->cur = acc;
    A->next = A->scratch;
}

typedef struct
{
    RNSc_MLWE *out, *in;
} Copy;

static void copy_item(void *ctx, uint64_t k)
{
    const Copy *c = (const Copy *)ctx;
    mlwe_copy_RNSc_sample(c->out[k], c->in[k]);
}

static void accumulators_finish(Accumulators *A, uint64_t n_threads)
{
    if (A->cur != A->caller)
    {
        Copy c = {A->caller, A->cur};
        vfhe_parallel_for(A->n, n_threads, copy_item, &c);
    }
    for (uint64_t k = 0; k < A->n; k++)
        free_mlwe_RNS_sample(A->scratch[k]);
    free(A->scratch);
}

// Rotates by the value of `bits` (LSB first); with `a`, multiplies by the signed monomials after
// the last bit.
static void rotate_then_multiply(Accumulators *A, RNS_MLWE *const *bits, uint64_t n_bits,
                                 RNS_MLWE_KS_Key aut, const uint64_t *a, RNS_MLWE *sign,
                                 uint64_t ell, uint64_t log_base, bool balanced, uint64_t n_threads)
{
    Layer L = {A->cur, NULL, A->n, 0, NULL, aut, NULL, sign, ell, log_base, balanced};
    for (uint64_t b = 0; b < n_bits; b++)
    {
        L.in = A->cur;
        L.out = A->next;
        L.power = 1ULL << b;
        L.bit = bits[b];
        L.a = b + 1 == n_bits ? a : NULL;
        vfhe_parallel_for(A->n, n_threads, layer_item, &L);
        A->next = A->cur;
        A->cur = L.out;
    }
    if (n_bits == 0 && a != NULL)
    {
        L.in = A->cur;
        L.out = NULL;
        L.a = a;
        vfhe_parallel_for(A->n, n_threads, layer_item, &L);
    }
}

void gp25_rotate(RNSc_MLWE *acc, uint64_t n, RNS_MLWE *const *bits, uint64_t n_bits,
                 RNS_MLWE_KS_Key aut, uint64_t ell, uint64_t log_base, bool balanced,
                 uint64_t n_threads)
{
    Accumulators A;
    accumulators_init(&A, acc, n);
    rotate_then_multiply(&A, bits, n_bits, aut, NULL, NULL, ell, log_base, balanced, n_threads);
    accumulators_finish(&A, n_threads);
}

void gp25_multiply_by_signed_monomials(RNSc_MLWE *acc, uint64_t n, const uint64_t *a,
                                       RNS_MLWE *sign, uint64_t ell, uint64_t log_base,
                                       bool balanced, uint64_t n_threads)
{
    Layer L = {acc, NULL, n, 0, NULL, NULL, a, sign, ell, log_base, balanced};
    vfhe_parallel_for(n, n_threads, layer_item, &L);
}

void gp25_blind_rotate(RNSc_MLWE *acc, uint64_t n, const uint64_t *a, uint64_t rank, uint64_t h,
                       uint64_t gap_bits, RNS_MLWE *const *gap_keys, RNS_MLWE *const *sign_keys,
                       RNS_MLWE_KS_Key aut, uint64_t ell, uint64_t log_base, bool balanced,
                       uint64_t n_threads)
{
    const uint64_t two_n = 2 * acc[0]->ring->N;
    Accumulators A;
    accumulators_init(&A, acc, n);
    uint64_t *exponent = (uint64_t *)safe_malloc(n * sizeof(uint64_t));
    for (uint64_t i = 0; i < rank; i++)
    {
        // Each component's rotations wrap every accumulator once, which applies X -> X^-1 to
        // what it holds; the odd components are negated so the wraps do not cancel them.
        for (uint64_t k = 0; k < n; k++)
            exponent[k] = i & 1 ? (two_n - a[i * n + k] % two_n) % two_n : a[i * n + k] % two_n;
        for (uint64_t t = 0; t <= h; t++)
        {
            rotate_then_multiply(&A, gap_keys, gap_bits, aut, t < h ? exponent : NULL,
                                 t < h && sign_keys != NULL ? sign_keys[i * h + t] : NULL, ell,
                                 log_base, balanced, n_threads);
            gap_keys += gap_bits;
        }
    }
    free(exponent);
    accumulators_finish(&A, n_threads);
}
