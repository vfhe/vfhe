// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
//
// Size-generic scalar element-wise kernels, compiled into *every* engine.
//
// The vectorized kernels (mod32/mod50/mod64.c) step one AVX512 lane group at a
// time -- `for (i = 0; i < n / 8; i++)` -- so they compute nothing at all when
// n < 8. These are the fallback the dispatchers in mod.c route such lengths to,
// and the whole implementation the portable engine uses (mod_portable.c
// forwards to them). `modq` and friends are resolved per engine, so the same
// source is correct in each build.
#include <arith.h>
#include "arith_internal.h"

void mod_eltwise_mul_gen(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc)
{
    for (size_t i = 0; i < n; i++)
        out[i] = modq((unsigned __int128)in1[i] * in2[i], proc);
}

void mod_eltwise_mul_addto_gen(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n,
                               NTT_proc proc)
{
    for (size_t i = 0; i < n; i++)
    {
        uint64_t prod = modq((unsigned __int128)in1[i] * in2[i], proc);
        out[i] = add_modq(out[i], prod, proc->q);
    }
}

void mod_eltwise_mul_subto_gen(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n,
                               NTT_proc proc)
{
    for (size_t i = 0; i < n; i++)
    {
        uint64_t prod = modq((unsigned __int128)in1[i] * in2[i], proc);
        out[i] = sub_modq(out[i], prod, proc->q);
    }
}

void mod_eltwise_scale_gen(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, NTT_proc proc)
{
    uint64_t s = scale % proc->q;
    for (size_t i = 0; i < n; i++)
        out[i] = modq((unsigned __int128)in[i] * s, proc);
}

void mod_eltwise_fma_gen(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, NTT_proc proc)
{
    uint64_t s = scale % proc->q;
    for (size_t i = 0; i < n; i++)
    {
        uint64_t prod = modq((unsigned __int128)in[i] * s, proc);
        out[i] = (out[i] + prod) % proc->q;
    }
}

void mod_eltwise_add_gen(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc)
{
    for (size_t i = 0; i < n; i++)
        out[i] = (in1[i] + in2[i]) % proc->q;
}

void mod_eltwise_sub_gen(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc)
{
    for (size_t i = 0; i < n; i++)
        out[i] = (in1[i] + proc->q - in2[i]) % proc->q;
}

void mod_eltwise_negate_gen(uint64_t *out, uint64_t *in, uint64_t n, NTT_proc proc)
{
    for (size_t i = 0; i < n; i++)
        out[i] = (proc->q - (in[i] % proc->q)) % proc->q;
}

void mod_eltwise_reduce_gen(uint64_t *out, uint64_t *in, uint64_t n, NTT_proc proc)
{
    for (size_t i = 0; i < n; i++)
        out[i] = in[i] % proc->q;
}

void mod_eltwise_reduce_signed_gen(uint64_t *out, int64_t *in, uint64_t n, NTT_proc proc)
{
    uint64_t q = proc->q;
    for (size_t i = 0; i < n; i++)
    {
        int64_t val = in[i];
        uint64_t abs_val = (val < 0) ? -(uint64_t)val : (uint64_t)val;
        uint64_t r = modq(abs_val, proc);
        if (val < 0)
        {
            out[i] = (r == 0) ? 0 : q - r;
        }
        else
        {
            out[i] = r;
        }
    }
}

void mod_eltwise_add_scalar_gen(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n,
                                NTT_proc proc)
{
    uint64_t s = scalar % proc->q;
    for (size_t i = 0; i < n; i++)
        out[i] = (in[i] + s) % proc->q;
}

void mod_eltwise_sub_scalar_gen(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n,
                                NTT_proc proc)
{
    uint64_t s = scalar % proc->q;
    for (size_t i = 0; i < n; i++)
        out[i] = (in[i] + proc->q - s) % proc->q;
}

void mod_reduce_array_mp_gen(uint64_t *out, uint64_t *in_high, uint64_t *in_low, uint64_t n,
                             NTT_proc proc)
{
    for (size_t i = 0; i < n; i++)
    {
        unsigned __int128 val = ((unsigned __int128)in_high[i] << 64) | in_low[i];
        out[i] = modq(val, proc);
    }
}
