// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
//
// Vectors of extension-field elements, stored as d coefficient planes.
//
// The plane layout is the whole point: with the coefficients of position j of
// every element contiguous, the arithmetic here is a fixed number of
// mod_eltwise_* calls -- the engine-tuned kernels -- over length-allocated_n
// runs, independent of how many elements the vector holds. The contract those
// planes satisfy is stated at the declaration in arith.h.
#include <arith.h>
#include <blake3.h>
#include "arith_internal.h"
#include "misc.h"

uint64_t field_vec_padded_length(uint64_t n)
{
    const uint64_t unit = MOD_MIN_VECTOR_LEN;
    return ((n + unit - 1) / unit) * unit;
}

void field_vec_add(FieldVector out, const FieldVector a, const FieldVector b)
{
    for (uint64_t j = 0; j < a->d; j++)
        mod_eltwise_add(out->coeffs[j], a->coeffs[j], b->coeffs[j], a->allocated_n, a->mod);
}

void field_vec_sub(FieldVector out, const FieldVector a, const FieldVector b)
{
    for (uint64_t j = 0; j < a->d; j++)
        mod_eltwise_sub(out->coeffs[j], a->coeffs[j], b->coeffs[j], a->allocated_n, a->mod);
}

void field_vec_neg(FieldVector out, const FieldVector a)
{
    for (uint64_t j = 0; j < a->d; j++)
        mod_eltwise_negate(out->coeffs[j], a->coeffs[j], a->allocated_n, a->mod);
}

void field_vec_add_scalar(FieldVector out, const FieldVector a, const uint64_t *s)
{
    for (uint64_t j = 0; j < a->d; j++)
        mod_eltwise_add_scalar(out->coeffs[j], a->coeffs[j], s[j], a->allocated_n, a->mod);
}

void field_vec_sub_scalar(FieldVector out, const FieldVector a, const uint64_t *s)
{
    for (uint64_t j = 0; j < a->d; j++)
        mod_eltwise_sub_scalar(out->coeffs[j], a->coeffs[j], s[j], a->allocated_n, a->mod);
}

void field_vec_scalar_sub(FieldVector out, const uint64_t *s, const FieldVector a)
{
    // s - a = -(a - s): the eltwise layer has no reversed-operand kernel.
    for (uint64_t j = 0; j < a->d; j++)
    {
        mod_eltwise_sub_scalar(out->coeffs[j], a->coeffs[j], s[j], a->allocated_n, a->mod);
        mod_eltwise_negate(out->coeffs[j], out->coeffs[j], a->allocated_n, a->mod);
    }
}

// The schoolbook product of two degree-(d-1) polynomials, one plane at a time,
// followed by the fold on x^d == w. Same shape as field_ext_mul, with each
// coefficient-times-coefficient replaced by a whole-plane kernel call, and the
// 2d-1 intermediate planes held in one scratch allocation.
static void field_vec_mul_generic(FieldVector out, const FieldVector a, const void *b_or_scalar,
                                  int b_is_vector)
{
    const uint64_t d = a->d, len = a->allocated_n;
    const uint64_t wide = 2 * d - 1;
    uint64_t *scratch = (uint64_t *)safe_aligned_malloc(wide * len * sizeof(uint64_t));
    memset(scratch, 0, wide * len * sizeof(uint64_t));

    for (uint64_t i = 0; i < d; i++)
    {
        for (uint64_t j = 0; j < d; j++)
        {
            uint64_t *acc = scratch + (i + j) * len;
            if (b_is_vector)
            {
                const FieldVector b = (const FieldVector)b_or_scalar;
                mod_eltwise_mul_addto(acc, a->coeffs[i], b->coeffs[j], len, a->mod);
            }
            else
            {
                const uint64_t *s = (const uint64_t *)b_or_scalar;
                mod_eltwise_fma(acc, a->coeffs[i], s[j], len, a->mod);
            }
        }
    }

    for (uint64_t i = wide; i-- > d;)
        mod_eltwise_fma(scratch + (i - d) * len, scratch + i * len, a->w, len, a->mod);

    // Copied only now, so `out` may alias either input.
    for (uint64_t j = 0; j < d; j++)
        memcpy(out->coeffs[j], scratch + j * len, len * sizeof(uint64_t));
    free(scratch);
}

void field_vec_mul(FieldVector out, const FieldVector a, const FieldVector b)
{
    field_vec_mul_generic(out, a, (const void *)b, 1);
}

void field_vec_scale(FieldVector out, const FieldVector a, const uint64_t *s)
{
    field_vec_mul_generic(out, a, (const void *)s, 0);
}

void field_vec_sum(uint64_t *out, const FieldVector a)
{
    for (uint64_t j = 0; j < a->d; j++)
    {
        const uint64_t q = a->mod->q;
        uint64_t acc = 0;
        for (uint64_t i = 0; i < a->n; i++)
            acc = add_modq(acc, a->coeffs[j][i], q);
        out[j] = acc;
    }
}

void field_vec_get_element(uint64_t *out, const FieldVector a, uint64_t index)
{
    for (uint64_t j = 0; j < a->d; j++)
        out[j] = a->coeffs[j][index];
}

void field_vec_set_element(FieldVector out, uint64_t index, const uint64_t *value)
{
    for (uint64_t j = 0; j < out->d; j++)
        out->coeffs[j][index] = value[j];
}

void field_vec_set_range(FieldVector out, uint64_t start, const uint64_t *values, uint64_t count)
{
    const uint64_t d = out->d;
    for (uint64_t j = 0; j < d; j++)
    {
        uint64_t *plane = out->coeffs[j] + start;
        for (uint64_t i = 0; i < count; i++)
            plane[i] = values[i * d + j];
    }
}

void field_vec_get_range(uint64_t *out, const FieldVector a, uint64_t start, uint64_t count)
{
    const uint64_t d = a->d;
    for (uint64_t j = 0; j < d; j++)
    {
        const uint64_t *plane = a->coeffs[j] + start;
        for (uint64_t i = 0; i < count; i++)
            out[i * d + j] = plane[i];
    }
}

void field_vec_copy(FieldVector out, const FieldVector a)
{
    for (uint64_t j = 0; j < a->d; j++)
        memcpy(out->coeffs[j], a->coeffs[j], a->allocated_n * sizeof(uint64_t));
}

void field_vec_split_even_odd(FieldVector even, FieldVector odd, const FieldVector a)
{
    const uint64_t half = a->n / 2;
    for (uint64_t j = 0; j < a->d; j++)
    {
        const uint64_t *plane = a->coeffs[j];
        for (uint64_t i = 0; i < half; i++)
        {
            even->coeffs[j][i] = plane[2 * i];
            odd->coeffs[j][i] = plane[2 * i + 1];
        }
    }
}

int field_vec_is_equal(const FieldVector a, const FieldVector b)
{
    if (a->n != b->n || a->d != b->d)
        return 0;
    for (uint64_t j = 0; j < a->d; j++)
    {
        if (memcmp(a->coeffs[j], b->coeffs[j], a->n * sizeof(uint64_t)) != 0)
            return 0;
    }
    return 1;
}

int field_vec_inv(FieldVector out, const FieldVector a)
{
    const uint64_t d = a->d, n = a->n;
    if (n == 0)
        return 1;

    // Prefix products, one inversion of the last, then a reverse sweep peeling
    // each factor back off: 3(n-1) multiplications and one inversion.
    uint64_t *prefix = (uint64_t *)malloc(n * d * sizeof(uint64_t));
    uint64_t *element = (uint64_t *)malloc(d * sizeof(uint64_t));
    uint64_t *running = (uint64_t *)malloc(d * sizeof(uint64_t));
    uint64_t *tmp = (uint64_t *)malloc(d * sizeof(uint64_t));

    field_vec_get_element(prefix, a, 0);
    for (uint64_t i = 1; i < n; i++)
    {
        field_vec_get_element(element, a, i);
        field_ext_mul(prefix + i * d, prefix + (i - 1) * d, element, d, a->w, a->mod);
    }

    int status = field_ext_inv(running, prefix + (n - 1) * d, d, a->w, a->mod);
    if (status)
    {
        for (uint64_t i = n; i-- > 1;)
        {
            // Read a[i] before writing out[i]: the two may be the same vector.
            field_vec_get_element(element, a, i);
            field_ext_mul(tmp, running, prefix + (i - 1) * d, d, a->w, a->mod);
            field_vec_set_element(out, i, tmp);
            field_ext_mul(running, running, element, d, a->w, a->mod);
        }
        field_vec_set_element(out, 0, running);
    }

    free(prefix);
    free(element);
    free(running);
    free(tmp);
    return status;
}

void field_vec_sample_random(FieldVector out, const uint8_t *seed, uint64_t seed_len)
{
    // One draw stream over all n * d coefficients, scattered into the planes.
    // Drawing per plane from the same seed would give every plane the same
    // values, and hence every element the same coefficient repeated.
    const uint64_t d = out->d, n = out->n;
    if (n == 0)
        return;
    uint64_t *flat = (uint64_t *)malloc(n * d * sizeof(uint64_t));
    prng_sample_below(flat, n * d, out->mod->q, "field_vec_sample", seed, seed_len);
    field_vec_set_range(out, 0, flat, n);
    free(flat);
}

// The elements in index order, d words each, as the bytes a digest covers.
static void hash_span(uint8_t *out, const FieldVector a, uint64_t start, uint64_t count)
{
    const uint64_t d = a->d;
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    uint64_t *element = (uint64_t *)malloc(d * sizeof(uint64_t));
    for (uint64_t i = 0; i < count; i++)
    {
        field_vec_get_element(element, a, start + i);
        blake3_hasher_update(&hasher, (const uint8_t *)element, d * sizeof(uint64_t));
    }
    free(element);
    blake3_hasher_finalize(&hasher, out, BLAKE3_OUT_LEN);
}

void field_vec_hash(uint8_t *out, const FieldVector a) { hash_span(out, a, 0, a->n); }

uint64_t field_vec_hash_count(const FieldVector a, uint64_t group, uint64_t stride)
{
    if (group == 0 || stride == 0 || a->n < group)
        return 0;
    return (a->n - group) / stride + 1;
}

void field_vec_hash_elements(uint8_t *out, const FieldVector a, uint64_t group, uint64_t stride)
{
    const uint64_t count = field_vec_hash_count(a, group, stride);
    for (uint64_t k = 0; k < count; k++)
        hash_span(out + k * BLAKE3_OUT_LEN, a, k * stride, group);
}
