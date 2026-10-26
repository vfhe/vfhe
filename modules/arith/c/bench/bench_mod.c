// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file bench_mod.c
 * @brief The element-wise modular product for each prime family: the step between the
 *        transforms of a polynomial product.
 *
 * Each prime's operands are held in the word width an RNS row over it is stored in: 32-bit
 * words below 2^30 (`mod_eltwise_mul_w32`), 64-bit words otherwise (`mod_eltwise_mul`).
 */
#include <stdbool.h>
#include <stdlib.h>

#include <alloc.h>
#include <arith.h>

#include "arith_internal.h" /* mod_eltwise_mul_w32, rns_prime_is_narrow */
#include "bench.h"

typedef struct
{
    Modulus mod;
    uint64_t q, n;
    bool narrow; /* 32-bit words, as a row over this prime is stored */
    void *a, *b, *out;
} Operands;

static uint64_t word(const Operands *o, const void *row, uint64_t i)
{
    return o->narrow ? ((const uint32_t *)row)[i] : ((const uint64_t *)row)[i];
}

/* arg[0] is the length, arg[1] the prime's bit length. */
static void *setup(const BenchCase *c)
{
    const uint64_t n = c->arg[0], bits = c->arg[1];
    const uint64_t q = next_special_prime(UINT64_C(1) << (bits - 1), n, true);
    if (64 - (uint64_t)__builtin_clzll(q) != bits)
        return NULL;
    Operands *o = malloc(sizeof *o);
    o->q = q;
    o->n = n;
    o->mod = mod_new(q);
    o->narrow = rns_prime_is_narrow(q);
    const size_t size = n * (o->narrow ? sizeof(uint32_t) : sizeof(uint64_t));
    o->a = safe_aligned_malloc(size);
    o->b = safe_aligned_malloc(size);
    o->out = safe_aligned_malloc(size);
    for (uint64_t i = 0; i < n; i++)
    {
        const uint64_t a = (UINT64_C(0x9E3779B97F4A7C15) * (i + 1)) % q;
        const uint64_t b = (UINT64_C(0xC2B2AE3D27D4EB4F) * (i + 7)) % q;
        if (o->narrow)
        {
            ((uint32_t *)o->a)[i] = (uint32_t)a;
            ((uint32_t *)o->b)[i] = (uint32_t)b;
        }
        else
        {
            ((uint64_t *)o->a)[i] = a;
            ((uint64_t *)o->b)[i] = b;
        }
    }
    return o;
}

static void product(void *state)
{
    Operands *o = state;
    if (o->narrow)
        mod_eltwise_mul_w32(o->out, o->a, o->b, o->n, o->mod);
    else
        mod_eltwise_mul(o->out, o->a, o->b, o->n, o->mod);
    BENCH_KEEP(o->out);
}

/* Compares every product with a 128-bit reduction. */
static int check_product(void *state)
{
    Operands *o = state;
    for (uint64_t i = 0; i < o->n; i++)
    {
        const unsigned __int128 expected = (unsigned __int128)word(o, o->a, i) * word(o, o->b, i);
        if (word(o, o->out, i) != (uint64_t)(expected % o->q))
            return 1;
    }
    return 0;
}

static void teardown(void *state)
{
    Operands *o = state;
    free(o->a);
    free(o->b);
    free(o->out);
    mod_free(o->mod);
    free(o);
}

#define PRODUCT(n, bits)                                                                           \
    {"eltwise_product", "n" #n "-q" #bits, 0, {n, bits}, 1, setup, product, check_product, teardown}

static const BenchCase cases[] = {
    PRODUCT(16384, 30),
    PRODUCT(16384, 50),
    PRODUCT(16384, 61),
};

int main(int argc, char **argv)
{
    return bench_main(argc, argv, cases, sizeof cases / sizeof *cases);
}
