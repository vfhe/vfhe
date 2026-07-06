// SPDX-License-Identifier: Apache-2.0
/**
 * @file int_poly.c
 * @brief Plain integer polynomials (exchange format; see arith/poly.h).
 */
#include <stdlib.h>

#include <arith/poly.h>
#include <base.h>

int_poly_t int_poly_new(uint64_t N)
{
    int_poly_t res = (int_poly_t)safe_malloc(sizeof(*res));
    res->coeffs = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * N);
    res->N = N;
    return res;
}

void int_poly_free(void *p)
{
    free(((int_poly_t)p)->coeffs);
    free(p);
}

int_poly_t *int_poly_array_new(uint64_t size, uint64_t N)
{
    int_poly_t *res = (int_poly_t *)safe_malloc(sizeof(int_poly_t) * size);
    for (uint64_t i = 0; i < size; i++)
    {
        res[i] = int_poly_new(N);
    }
    return res;
}

void int_poly_array_free(uint64_t size, int_poly_t *p)
{
    for (uint64_t i = 0; i < size; i++)
    {
        int_poly_free(p[i]);
    }
    free(p);
}

void int_poly_permute(int_poly_t out, const int_poly_t in, uint64_t gen)
{
    const uint64_t N = in->N;
    uint64_t idx = 0;
    for (uint64_t i = 0; i < N; i++)
    {
        out->coeffs[idx] = in->coeffs[i];
        idx = (idx + gen) % N;
    }
}

void int_poly_decompose_digit(int_poly_t out, const int_poly_t in, uint64_t Bg_bit, uint64_t l,
                              uint64_t bit_size, uint64_t i)
{
    const uint64_t N = in->N;
    const uint64_t h_mask = (1UL << Bg_bit) - 1;
    const uint64_t h_bit = bit_size - (i + 1) * Bg_bit;
    // Center the truncation: adding half of the discarded range turns the
    // floor into a round, keeping digit magnitudes balanced around 0.
    uint64_t offset = 1ULL << (bit_size - l * Bg_bit - 1);
    for (uint64_t c = 0; c < N; c++)
    {
        const uint64_t coeff_off = in->coeffs[c] + offset;
        out->coeffs[c] = (coeff_off >> h_bit) & h_mask;
    }
}
