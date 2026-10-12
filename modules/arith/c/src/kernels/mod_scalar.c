// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
//
// Size-generic scalar element-wise kernels, compiled into every engine. They
// are the whole implementation on the portable engine, and on the vectorized
// ones they serve lengths below MOD_MIN_VECTOR_LEN, which the AVX512 kernels
// cannot handle: those step a full lane group per iteration.
//
// All arithmetic goes through the Modulus primitives: mul_modq for products,
// modq to reduce a wider value, add_modq / sub_modq / negate_modq otherwise.
// None of them divides -- modq is a Barrett reduction and the add/sub pair is a
// conditional subtract.
//
// Preconditions, matching the vectorized kernels these stand in for:
//   - add, sub, negate, add_scalar, sub_scalar, fma, mul_addto, mul_subto take
//     their array operands already reduced to [0, q);
//   - reduce, reduce_signed and reduce_array_mp accept any input, which is
//     what they are for, and reduce_centered any residue in [0, q_in);
//   - the scalar operand of scale, fma, add_scalar and sub_scalar is reduced on
//     entry and may be arbitrary.
#include <arith.h>
#include "arith_internal.h"

void mod_eltwise_mul_gen(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod)
{
    for (size_t i = 0; i < n; i++)
        out[i] = mul_modq(in1[i], in2[i], mod);
}

void mod_eltwise_mul_addto_gen(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod)
{
    const uint64_t q = mod->q;
    for (size_t i = 0; i < n; i++)
        out[i] = add_modq(out[i], mul_modq(in1[i], in2[i], mod), q);
}

// out += sum_t a[t] * b[t] (mod q): products of values below q < 2^62 are below 2^124, so 15 of
// them are summed in 128 bits before each reduction.
void mod_eltwise_inner_product_addto_gen(uint64_t *out, uint64_t *const *a, uint64_t *const *b,
                                         uint64_t terms, uint64_t n, Modulus mod)
{
    const uint64_t q = mod->q;
    for (size_t i = 0; i < n; i++)
    {
        uint64_t acc = out[i];
        for (uint64_t t0 = 0; t0 < terms; t0 += 15)
        {
            const uint64_t t1 = terms - t0 < 15 ? terms : t0 + 15;
            unsigned __int128 sum = 0;
            for (uint64_t t = t0; t < t1; t++)
                sum += (unsigned __int128)a[t][i] * b[t][i];
            acc = add_modq(acc, modq_wide((uint64_t)(sum >> 64), (uint64_t)sum, mod), q);
        }
        out[i] = acc;
    }
}

void mod_eltwise_mul_subto_gen(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod)
{
    const uint64_t q = mod->q;
    for (size_t i = 0; i < n; i++)
        out[i] = sub_modq(out[i], mul_modq(in1[i], in2[i], mod), q);
}

void mod_eltwise_scale_gen(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, Modulus mod)
{
    const uint64_t s = modq(scale, mod);
    for (size_t i = 0; i < n; i++)
        out[i] = mul_modq(in[i], s, mod);
}

void mod_eltwise_fma_gen(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, Modulus mod)
{
    const uint64_t q = mod->q, s = modq(scale, mod);
    for (size_t i = 0; i < n; i++)
        out[i] = add_modq(out[i], mul_modq(in[i], s, mod), q);
}

void mod_eltwise_add_gen(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod)
{
    const uint64_t q = mod->q;
    for (size_t i = 0; i < n; i++)
        out[i] = add_modq(in1[i], in2[i], q);
}

void mod_eltwise_sub_gen(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod)
{
    const uint64_t q = mod->q;
    for (size_t i = 0; i < n; i++)
        out[i] = sub_modq(in1[i], in2[i], q);
}

void mod_eltwise_negate_gen(uint64_t *out, uint64_t *in, uint64_t n, Modulus mod)
{
    const uint64_t q = mod->q;
    for (size_t i = 0; i < n; i++)
        out[i] = negate_modq(in[i], q);
}

void mod_eltwise_reduce_gen(uint64_t *out, uint64_t *in, uint64_t n, Modulus mod)
{
    for (size_t i = 0; i < n; i++)
        out[i] = modq(in[i], mod);
}

void mod_eltwise_reduce_signed_gen(uint64_t *out, int64_t *in, uint64_t n, Modulus mod)
{
    const uint64_t q = mod->q;
    for (size_t i = 0; i < n; i++)
    {
        const int64_t val = in[i];
        // Reduce the magnitude, then apply the sign: reducing the
        // two's-complement pattern instead would give a different residue.
        const uint64_t r = modq((val < 0) ? -(uint64_t)val : (uint64_t)val, mod);
        out[i] = (val < 0) ? negate_modq(r, q) : r;
    }
}

void mod_eltwise_reduce_centered_gen(uint64_t *out, uint64_t *in, uint64_t q_in, uint64_t n,
                                     Modulus mod)
{
    const uint64_t half = q_in >> 1, neg_q_in = negate_modq(modq(q_in, mod), mod->q);
    for (size_t i = 0; i < n; i++)
        out[i] = reduce_centered_modq(in[i], half, neg_q_in, mod);
}

void mod_eltwise_add_scalar_gen(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n,
                                Modulus mod)
{
    const uint64_t q = mod->q, s = modq(scalar, mod);
    for (size_t i = 0; i < n; i++)
        out[i] = add_modq(in[i], s, q);
}

void mod_eltwise_sub_scalar_gen(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n,
                                Modulus mod)
{
    const uint64_t q = mod->q, s = modq(scalar, mod);
    for (size_t i = 0; i < n; i++)
        out[i] = sub_modq(in[i], s, q);
}

void mod_reduce_array_mp_gen(uint64_t *out, uint64_t *in_high, uint64_t *in_low, uint64_t n,
                             Modulus mod)
{
    for (size_t i = 0; i < n; i++)
        out[i] = modq_wide(in_high[i], in_low[i], mod);
}

void mod_eltwise_lifted_combination_gen(uint64_t *out, const uint64_t *const *in,
                                        const uint64_t *q_in, const uint64_t *scale, uint64_t m,
                                        bool centered, uint64_t n, Modulus mod)
{
    const uint64_t p = mod->q, two_p = 2 * p;
    uint64_t s[MOD_COMBINATION_MAX_TERMS], sp[MOD_COMBINATION_MAX_TERMS];
    uint64_t half[MOD_COMBINATION_MAX_TERMS], h[MOD_COMBINATION_MAX_TERMS];
    assert(m <= MOD_COMBINATION_MAX_TERMS);
    for (uint64_t t = 0; t < m; t++)
    {
        s[t] = modq(scale[t], mod);
        sp[t] = (uint64_t)(((unsigned __int128)s[t] << 64) / p);
        half[t] = centered ? q_in[t] >> 1 : UINT64_MAX;
        h[t] = negate_modq(mul_modq(s[t], modq(q_in[t], mod), mod), p);
    }
    for (uint64_t j = 0; j < n; j++)
    {
        // Every term is a Shoup product in [0, 2p) plus, for a value above
        // half its modulus, the correction that centers it; the sum is kept
        // in [0, 2p) and reduced once.
        uint64_t acc = 0;
        for (uint64_t t = 0; t < m; t++)
        {
            const uint64_t v = in[t][j];
            acc += s[t] * v - (uint64_t)(((unsigned __int128)v * sp[t]) >> 64) * p;
            acc -= two_p & (0 - (uint64_t)(acc >= two_p));
            acc += h[t] & (0 - (uint64_t)(v > half[t]));
            acc -= two_p & (0 - (uint64_t)(acc >= two_p));
        }
        out[j] = acc - (p & (0 - (uint64_t)(acc >= p)));
    }
}
