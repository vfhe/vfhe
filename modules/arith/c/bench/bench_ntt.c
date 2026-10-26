// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file bench_ntt.c
 * @brief The negacyclic NTT, forward and inverse, for each prime family and size.
 *
 * One prime per family: 30 bits, 50 bits (the IFMA family's q < 2^50) and 61 bits (the
 * 64-bit family). Each is transformed in the word width an RNS row over it is stored in:
 * 32-bit words below 2^30 (`ntt_forward_w32`, `ntt_reverse_w32`), 64-bit words otherwise.
 * The cost of a polynomial operation over an RNS ring is close to the sum of these
 * per-prime costs, so the table also serves parameter selection.
 */
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include <alloc.h>
#include <arith.h>

#include "arith_internal.h" /* rns_prime_is_narrow */
#include "bench.h"

typedef struct
{
    Modulus mod;
    NTT_Plan plan;
    uint64_t n;
    bool narrow;        /* 32-bit words, as a row over this prime is stored */
    void *coefficients; /* the input of the forward transform */
    void *evaluations;  /* the forward transform of `coefficients` */
    void *out;
} Transform;

/* arg[0] is the length, arg[1] the prime's bit length. */
static void *setup(const BenchCase *c)
{
    const uint64_t n = c->arg[0], bits = c->arg[1];
    const uint64_t q = next_special_prime(UINT64_C(1) << (bits - 1), n, true);
    if (64 - (uint64_t)__builtin_clzll(q) != bits)
        return NULL;
    /* As for a ring's rows, the 32-bit entry points take every narrow prime and decide
       themselves how to transform it. */
    const bool narrow = rns_prime_is_narrow(q);
    Transform *t = malloc(sizeof *t);
    t->n = n;
    t->mod = mod_new(q);
    t->plan = ntt_new_plan(n, t->mod);
    t->narrow = narrow;
    const size_t word = t->narrow ? sizeof(uint32_t) : sizeof(uint64_t);
    t->coefficients = safe_aligned_malloc(n * word);
    t->evaluations = safe_aligned_malloc(n * word);
    t->out = safe_aligned_malloc(n * word);
    for (uint64_t i = 0; i < n; i++)
    {
        const uint64_t value = (UINT64_C(0x9E3779B97F4A7C15) * (i + 1)) % q;
        if (t->narrow)
            ((uint32_t *)t->coefficients)[i] = (uint32_t)value;
        else
            ((uint64_t *)t->coefficients)[i] = value;
    }
    if (t->narrow)
        ntt_forward_w32(t->evaluations, t->coefficients, t->plan);
    else
        ntt_forward(t->evaluations, t->coefficients, t->plan);
    return t;
}

static void forward(void *state)
{
    Transform *t = state;
    if (t->narrow)
        ntt_forward_w32(t->out, t->coefficients, t->plan);
    else
        ntt_forward(t->out, t->coefficients, t->plan);
    BENCH_KEEP(t->out);
}

static void inverse(void *state)
{
    Transform *t = state;
    if (t->narrow)
        ntt_reverse_w32(t->out, t->evaluations, t->plan);
    else
        ntt_reverse(t->out, t->evaluations, t->plan);
    BENCH_KEEP(t->out);
}

static size_t bytes(const Transform *t)
{
    return t->n * (t->narrow ? sizeof(uint32_t) : sizeof(uint64_t));
}

/* After `forward`, out holds the evaluations. */
static int check_forward(void *state)
{
    Transform *t = state;
    return memcmp(t->out, t->evaluations, bytes(t)) != 0;
}

/* After `inverse`, out holds the coefficients again. */
static int check_inverse(void *state)
{
    Transform *t = state;
    return memcmp(t->out, t->coefficients, bytes(t)) != 0;
}

static void teardown(void *state)
{
    Transform *t = state;
    free(t->coefficients);
    free(t->evaluations);
    free(t->out);
    ntt_free_plan(t->plan);
    mod_free(t->mod);
    free(t);
}

#define FORWARD(n, bits, headline)                                                                 \
    {"ntt_forward", "n" #n "-q" #bits, headline,      {n, bits}, 1,                                \
     setup,         forward,           check_forward, teardown}
#define INVERSE(n, bits)                                                                           \
    {"ntt_inverse", "n" #n "-q" #bits, 0, {n, bits}, 1, setup, inverse, check_inverse, teardown}

static const BenchCase cases[] = {
    FORWARD(2048, 30, 0),  FORWARD(2048, 50, 0),  FORWARD(2048, 61, 0),  FORWARD(4096, 30, 0),
    FORWARD(4096, 50, 0),  FORWARD(4096, 61, 0),  FORWARD(8192, 30, 0),  FORWARD(8192, 50, 0),
    FORWARD(8192, 61, 0),  FORWARD(16384, 30, 1), FORWARD(16384, 50, 1), FORWARD(16384, 61, 1),
    FORWARD(65536, 30, 0), FORWARD(65536, 50, 0), FORWARD(65536, 61, 0), INVERSE(2048, 30),
    INVERSE(2048, 50),     INVERSE(2048, 61),     INVERSE(4096, 30),     INVERSE(4096, 50),
    INVERSE(4096, 61),     INVERSE(8192, 30),     INVERSE(8192, 50),     INVERSE(8192, 61),
    INVERSE(16384, 30),    INVERSE(16384, 50),    INVERSE(16384, 61),    INVERSE(65536, 30),
    INVERSE(65536, 50),    INVERSE(65536, 61),
};

int main(int argc, char **argv)
{
    return bench_main(argc, argv, cases, sizeof cases / sizeof *cases);
}
