// SPDX-FileCopyrightText: 2026 The vFHE Authors
/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file fuzz_field_ext.c
 * @brief Fuzzes extension-field arithmetic.
 *
 * Each input is decoded into a prime q, an extension degree d, the parameter w
 * of x^d - w, and three elements. The laws asserted are the ones that hold in
 * F_q[x]/(x^d - w) for EVERY w, so every input is a legal ring and no search for
 * an irreducible polynomial is needed:
 *
 *   1. (a + b) - b == a;
 *   2. a + (-a) == 0;
 *   3. a * b == b * a;
 *   4. a * (b + c) == a * b + a * c;
 *   5. a * a^-1 == 1, whenever field_ext_inv reports success.
 *
 * Deterministic per input (no RNG), so a saved crash reproduces exactly.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <arith.h>

typedef struct
{
    const uint8_t *p;
    size_t n, i;
} cursor;

static uint64_t take_u64(cursor *c)
{
    uint64_t v = 0;
    for (int b = 0; b < 8; b++)
        v = (v << 8) | (uint64_t)(c->i < c->n ? c->p[c->i++] : 0);
    return v;
}

static uint8_t take_u8(cursor *c) { return (uint8_t)(c->i < c->n ? c->p[c->i++] : 0); }

static void take_element(cursor *c, uint64_t *out, uint64_t d, uint64_t q)
{
    for (uint64_t i = 0; i < d; i++)
        out[i] = take_u64(c) % q;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    cursor c = {data, size, 0};

    /* Degree 1 is the base field, which is worth covering: it is the path the
     * prime-subfield kernels take. */
    const uint64_t d = 1 + (take_u8(&c) & 3);       /* 1..4 */
    const uint64_t qbits = 20 + (take_u8(&c) % 42); /* 20..61, as in fuzz_ntt */
    const uint64_t q = next_special_prime((uint64_t)1 << qbits, 2, true);

    Modulus mod = mod_new(q);
    if (!mod)
        return 0;
    const uint64_t w = take_u64(&c) % q;

    uint64_t *a = calloc(d, sizeof(uint64_t));
    uint64_t *b = calloc(d, sizeof(uint64_t));
    uint64_t *e = calloc(d, sizeof(uint64_t));
    uint64_t *x = calloc(d, sizeof(uint64_t));
    uint64_t *y = calloc(d, sizeof(uint64_t));
    uint64_t *z = calloc(d, sizeof(uint64_t));
    take_element(&c, a, d, q);
    take_element(&c, b, d, q);
    take_element(&c, e, d, q);

    /* 1. (a + b) - b == a */
    field_ext_add(x, a, b, d, q);
    field_ext_sub(y, x, b, d, q);
    if (!field_ext_is_equal(y, a, d))
        abort();

    /* 2. a + (-a) == 0 */
    field_ext_neg(x, a, d, q);
    field_ext_add(y, a, x, d, q);
    memset(z, 0, d * sizeof(uint64_t));
    if (!field_ext_is_equal(y, z, d))
        abort();

    /* 3. a * b == b * a */
    field_ext_mul(x, a, b, d, w, mod);
    field_ext_mul(y, b, a, d, w, mod);
    if (!field_ext_is_equal(x, y, d))
        abort();

    /* 4. a * (b + c) == a * b + a * c */
    field_ext_add(z, b, e, d, q);
    field_ext_mul(x, a, z, d, w, mod);
    field_ext_mul(y, a, b, d, w, mod);
    field_ext_mul(z, a, e, d, w, mod);
    field_ext_add(y, y, z, d, q);
    if (!field_ext_is_equal(x, y, d))
        abort();

    /* 5. a * a^-1 == 1 where the inverse exists */
    if (field_ext_inv(x, a, d, w, mod))
    {
        field_ext_mul(y, a, x, d, w, mod);
        memset(z, 0, d * sizeof(uint64_t));
        z[0] = 1;
        if (!field_ext_is_equal(y, z, d))
            abort();
    }

    free(a);
    free(b);
    free(e);
    free(x);
    free(y);
    free(z);
    mod_free(mod);
    return 0;
}
