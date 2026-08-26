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
//     what they are for;
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
