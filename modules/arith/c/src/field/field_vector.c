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
#include "util.h"
#include <crypto.h>

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
// coefficient-times-coefficient replaced by a whole-plane kernel call.
//
// Done over the whole vector at once the 2d^2 + d - 1 passes would each sweep a
// (2d-1)-plane accumulator, so beyond the last private cache every pass reloads
// it from memory and the rate is set by bandwidth rather than by the kernels.
// The product is therefore tiled: the same passes run over FIELD_VEC_MUL_TILE
// elements at a time out of one scratch that stays resident, which holds the
// in-cache rate at any length. The tile is a multiple of the eltwise vector
// width and the planes are 64-byte aligned, so every tile meets the kernels'
// length and alignment preconditions -- including the last, since allocated_n
// is itself a whole number of vector widths.
#define FIELD_VEC_MUL_TILE 4096

static void field_vec_mul_generic(FieldVector out, const FieldVector a, const void *b_or_scalar,
                                  int b_is_vector)
{
    const uint64_t d = a->d, len = a->allocated_n;
    const uint64_t wide = 2 * d - 1;
    const uint64_t tile = len < FIELD_VEC_MUL_TILE ? len : FIELD_VEC_MUL_TILE;
    uint64_t *scratch = (uint64_t *)safe_aligned_malloc(wide * tile * sizeof(uint64_t));

    for (uint64_t base = 0; base < len; base += tile)
    {
        const uint64_t run = len - base < tile ? len - base : tile;
        for (uint64_t k = 0; k < wide; k++)
            memset(scratch + k * tile, 0, run * sizeof(uint64_t));

        for (uint64_t i = 0; i < d; i++)
        {
            for (uint64_t j = 0; j < d; j++)
            {
                uint64_t *acc = scratch + (i + j) * tile;
                if (b_is_vector)
                {
                    const FieldVector b = (const FieldVector)b_or_scalar;
                    mod_eltwise_mul_addto(acc, a->coeffs[i] + base, b->coeffs[j] + base, run,
                                          a->mod);
                }
                else
                {
                    const uint64_t *s = (const uint64_t *)b_or_scalar;
                    mod_eltwise_fma(acc, a->coeffs[i] + base, s[j], run, a->mod);
                }
            }
        }

        for (uint64_t i = wide; i-- > d;)
            mod_eltwise_fma(scratch + (i - d) * tile, scratch + i * tile, a->w, run, a->mod);

        // Copied only now, so `out` may alias either input: this tile's inputs
        // are read before it is overwritten, and later tiles are untouched.
        for (uint64_t j = 0; j < d; j++)
            memcpy(out->coeffs[j] + base, scratch + j * tile, run * sizeof(uint64_t));
    }
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

void field_vec_interleave(FieldVector out, const FieldVector even, const FieldVector odd)
{
    for (uint64_t j = 0; j < out->d; j++)
    {
        uint64_t *plane = out->coeffs[j];
        for (uint64_t i = 0; i < even->n; i++)
        {
            plane[2 * i] = even->coeffs[j][i];
            plane[2 * i + 1] = odd->coeffs[j][i];
        }
    }
}

void field_vec_concat(FieldVector out, const FieldVector *parts, uint64_t count)
{
    uint64_t at = 0;
    for (uint64_t p = 0; p < count; p++)
    {
        for (uint64_t j = 0; j < out->d; j++)
            memcpy(out->coeffs[j] + at, parts[p]->coeffs[j], parts[p]->n * sizeof(uint64_t));
        at += parts[p]->n;
    }
}

void field_vec_gather(FieldVector out, const FieldVector a, const uint64_t *indices, uint64_t count)
{
    for (uint64_t j = 0; j < a->d; j++)
        for (uint64_t i = 0; i < count; i++)
            out->coeffs[j][i] = a->coeffs[j][indices[i]];
}

// Deinterleave into two scratch vectors of out's shape, then three whole-plane
// operations: one allocation and no Python-visible intermediates. The scratch
// planes are `half` words each, a multiple of the eltwise vector width, so each
// stays 64-byte aligned inside the one block.
void field_vec_fold(FieldVector out, const FieldVector a, const uint64_t *r)
{
    const uint64_t d = a->d, half = out->allocated_n;
    // Pairs that exist in a's allocation; the rest of out's padding stays zero.
    const uint64_t pairs = a->allocated_n / 2 < half ? a->allocated_n / 2 : half;
    uint64_t *scratch = (uint64_t *)safe_aligned_malloc(2 * d * half * sizeof(uint64_t));
    uint64_t **planes = (uint64_t **)malloc(2 * d * sizeof(uint64_t *));
    struct _FieldVector even = *out, odd = *out;

    memset(scratch, 0, 2 * d * half * sizeof(uint64_t));
    even.coeffs = planes;
    odd.coeffs = planes + d;
    for (uint64_t j = 0; j < d; j++)
    {
        even.coeffs[j] = scratch + (2 * j) * half;
        odd.coeffs[j] = scratch + (2 * j + 1) * half;
        for (uint64_t i = 0; i < pairs; i++)
        {
            even.coeffs[j][i] = a->coeffs[j][2 * i];
            odd.coeffs[j][i] = a->coeffs[j][2 * i + 1];
        }
    }

    field_vec_sub(&odd, &odd, &even);
    field_vec_scale(&odd, &odd, r);
    field_vec_add(out, &even, &odd);

    free(planes);
    free(scratch);
}

// The runs of `block` elements that alternate between the two outputs: `lo`
// collects runs 0, 2, 4, ... and `hi` runs 1, 3, 5, .... With block == 1 that
// is field_vec_split_even_odd, which is what runs for it.
void field_vec_split_blocks(FieldVector lo, FieldVector hi, const FieldVector a, uint64_t block)
{
    if (block == 1)
    {
        field_vec_split_even_odd(lo, hi, a);
        return;
    }
    const uint64_t bytes = block * sizeof(uint64_t);
    for (uint64_t j = 0; j < a->d; j++)
    {
        const uint64_t *plane = a->coeffs[j];
        for (uint64_t c = 0, at = 0; at + 2 * block <= a->n; c++, at += 2 * block)
        {
            memcpy(lo->coeffs[j] + c * block, plane + at, bytes);
            memcpy(hi->coeffs[j] + c * block, plane + at + block, bytes);
        }
    }
}

// field_vec_fold over pairs `block` apart rather than adjacent: out[i] is
// lo[i] + r * (hi[i] - lo[i]) on the split field_vec_split_blocks makes, which
// is the interpolation binding the variable that `block` indexes.
//
// Three whole-plane passes either way; what `block` decides is whether the two
// operands have to be gathered first. At block == a->n / 2 -- the last variable
// of a multilinear table -- the halves are already contiguous, so they are used
// where they lie and the only allocation is the one temporary the passes need.
void field_vec_fold_blocks(FieldVector out, const FieldVector a, uint64_t block, const uint64_t *r)
{
    if (block == 1)
    {
        field_vec_fold(out, a, r);
        return;
    }

    const uint64_t d = a->d, half = out->allocated_n;
    // The halves are contiguous runs of `half` words only when nothing was
    // rounded up to reach it; otherwise they are gathered like any other block.
    const int in_place = (block == a->n / 2) && (block == half);
    const uint64_t planes_held = in_place ? 1 : 3;
    uint64_t *scratch = (uint64_t *)safe_aligned_malloc(planes_held * d * half * sizeof(uint64_t));
    uint64_t **planes = (uint64_t **)malloc(3 * d * sizeof(uint64_t *));
    struct _FieldVector lo = *out, hi = *out, acc = *out;

    memset(scratch, 0, planes_held * d * half * sizeof(uint64_t));
    lo.coeffs = planes;
    hi.coeffs = planes + d;
    acc.coeffs = planes + 2 * d;
    for (uint64_t j = 0; j < d; j++)
        acc.coeffs[j] = scratch + j * half;

    if (in_place)
    {
        for (uint64_t j = 0; j < d; j++)
        {
            lo.coeffs[j] = a->coeffs[j];
            hi.coeffs[j] = a->coeffs[j] + block;
        }
    }
    else
    {
        for (uint64_t j = 0; j < d; j++)
        {
            lo.coeffs[j] = scratch + (d + j) * half;
            hi.coeffs[j] = scratch + (2 * d + j) * half;
        }
        field_vec_split_blocks(&lo, &hi, a, block);
    }

    field_vec_sub(&acc, &hi, &lo);
    field_vec_scale(&acc, &acc, r);
    field_vec_add(out, &lo, &acc);

    free(planes);
    free(scratch);
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

// The Frobenius applied to every element: the same permutation of coefficient
// positions and the same constant per position as field_ext_frobenius, which is
// where the map is derived -- so here it is d whole-plane scalings, independent
// of how many elements the vector holds, rather than one exponentiation each.
//
// Written through a scratch plane set because the map permutes the planes and
// `out` may be `a`.
void field_vec_frobenius(FieldVector out, const FieldVector a, uint64_t k)
{
    const uint64_t d = a->d, len = a->allocated_n;
    if (d <= 1 || k % d == 0)
    {
        field_vec_copy(out, a);
        return;
    }
    uint64_t *constants = (uint64_t *)malloc(d * sizeof(uint64_t));
    uint64_t *to = (uint64_t *)malloc(d * sizeof(uint64_t));
    uint64_t *scratch = (uint64_t *)safe_aligned_malloc(d * len * sizeof(uint64_t));

    frobenius_map(constants, to, k, d, a->w, a->mod);
    for (uint64_t j = 0; j < d; j++)
        mod_eltwise_scale(scratch + to[j] * len, a->coeffs[j], constants[j], len, a->mod);
    for (uint64_t j = 0; j < d; j++)
        memcpy(out->coeffs[j], scratch + j * len, len * sizeof(uint64_t));

    free(constants);
    free(to);
    free(scratch);
}

void field_vec_sample_random(FieldVector out, const uint8_t *seed, uint64_t seed_len,
                             uint64_t start)
{
    // One draw stream over all the coefficients, scattered into the planes:
    // element i takes draws i*d .. i*d + d - 1, so `start` is a position in
    // that stream and not a second stream. Drawing per plane from the same seed
    // would give every plane the same values, and hence every element the same
    // coefficient repeated.
    const uint64_t d = out->d, n = out->n;
    if (n == 0)
        return;
    uint64_t *flat = (uint64_t *)malloc(n * d * sizeof(uint64_t));
    prng_sample_below_from(flat, n * d, start * d, out->mod->q, "field_vec_sample", seed, seed_len);
    field_vec_set_range(out, 0, flat, n);
    free(flat);
}

void field_vec_sample_random_element(uint64_t *out, const uint8_t *seed, uint64_t seed_len,
                                     uint64_t index, uint64_t d, uint64_t mod)
{
    // The same stream, at the d draws element `index` takes -- so this and the
    // fill above agree by construction rather than by two definitions that have
    // to be kept in step.
    prng_sample_below_from(out, d, index * d, mod, "field_vec_sample", seed, seed_len);
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
