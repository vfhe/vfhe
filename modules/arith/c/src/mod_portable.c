#include <arith.h>

#if !defined(__AVX512IFMA__) || defined(PORTABLE_BUILD) || defined(PORTABLE)

uint64_t modq(unsigned __int128 x, NTT_proc proc)
{
    if (x < (1ULL << 52))
    {
        uint64_t q_hat = (uint64_t)((x * proc->m52) >> 52);
        uint64_t res = (uint64_t)x - q_hat * proc->q;
        if (res >= proc->q)
            res -= proc->q;
        return res;
    }

    uint64_t x0 = (uint64_t)x & ((1ULL << 52) - 1);
    uint64_t x1 = (uint64_t)(x >> 52) & ((1ULL << 52) - 1);
    uint64_t x2 = (uint64_t)(x >> 104);

    unsigned __int128 reduced =
        x0 + (unsigned __int128)x1 * proc->mp_w1 + (unsigned __int128)x2 * proc->mp_w2;

    uint64_t r_lo = (uint64_t)reduced;
    uint64_t r_hi = (uint64_t)(reduced >> 64);
    unsigned __int128 r_lo_m = (unsigned __int128)r_lo * proc->m;
    unsigned __int128 r_hi_m = (unsigned __int128)r_hi * proc->m;
    unsigned __int128 prod_hi = r_hi_m + (r_lo_m >> 64);

    uint64_t q_hat = (uint64_t)(prod_hi >> (proc->k - 64));
    uint64_t res = (uint64_t)reduced - q_hat * proc->q;

    while (res >= proc->q)
        res -= proc->q;

    return res;
}

uint64_t add_modq(uint64_t a, uint64_t b, uint64_t q)
{
    uint64_t sum = a + b;
    return sum >= q ? sum - q : sum;
}

uint64_t sub_modq(uint64_t a, uint64_t b, uint64_t q) { return a >= b ? a - b : a + q - b; }

uint64_t negate_modq(uint64_t a, uint64_t q) { return a == 0 ? 0 : q - a; }

uint64_t mul_modq(uint64_t a, uint64_t b, NTT_proc proc)
{
    return modq((unsigned __int128)a * b, proc);
}

void mod_eltwise_mul(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc)
{
    for (size_t i = 0; i < n; i++)
        out[i] = modq((unsigned __int128)in1[i] * in2[i], proc);
}

void mod_eltwise_mul_addto(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc)
{
    for (size_t i = 0; i < n; i++)
    {
        uint64_t prod = modq((unsigned __int128)in1[i] * in2[i], proc);
        out[i] = add_modq(out[i], prod, proc->q);
    }
}

void mod_eltwise_mul_subto(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc)
{
    for (size_t i = 0; i < n; i++)
    {
        uint64_t prod = modq((unsigned __int128)in1[i] * in2[i], proc);
        out[i] = sub_modq(out[i], prod, proc->q);
    }
}

void mod_eltwise_scale(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, NTT_proc proc)
{
    uint64_t s = scale % proc->q;
    for (size_t i = 0; i < n; i++)
        out[i] = modq((unsigned __int128)in[i] * s, proc);
}

void mod_eltwise_fma(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, NTT_proc proc)
{
    uint64_t s = scale % proc->q;
    for (size_t i = 0; i < n; i++)
    {
        uint64_t prod = modq((unsigned __int128)in[i] * s, proc);
        out[i] = (out[i] + prod) % proc->q;
    }
}

void mod_eltwise_add(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc)
{
    for (size_t i = 0; i < n; i++)
        out[i] = (in1[i] + in2[i]) % proc->q;
}

void mod_eltwise_sub(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc)
{
    for (size_t i = 0; i < n; i++)
        out[i] = (in1[i] + proc->q - in2[i]) % proc->q;
}

void mod_eltwise_negate(uint64_t *out, uint64_t *in, uint64_t n, NTT_proc proc)
{
    for (size_t i = 0; i < n; i++)
        out[i] = (proc->q - (in[i] % proc->q)) % proc->q;
}

void mod_eltwise_reduce(uint64_t *out, uint64_t *in, uint64_t n, NTT_proc proc)
{
    for (size_t i = 0; i < n; i++)
        out[i] = in[i] % proc->q;
}

void mod_eltwise_reduce_signed(uint64_t *out, int64_t *in, uint64_t n, NTT_proc proc)
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

void mod_eltwise_add_scalar(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n, NTT_proc proc)
{
    uint64_t s = scalar % proc->q;
    for (size_t i = 0; i < n; i++)
        out[i] = (in[i] + s) % proc->q;
}

void mod_eltwise_sub_scalar(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n, NTT_proc proc)
{
    uint64_t s = scalar % proc->q;
    for (size_t i = 0; i < n; i++)
        out[i] = (in[i] + proc->q - s) % proc->q;
}

void mod_reduce_array_mp(uint64_t *out, uint64_t *in_high, uint64_t *in_low, uint64_t n,
                         NTT_proc proc)
{
    for (size_t i = 0; i < n; i++)
    {
        unsigned __int128 val = ((unsigned __int128)in_high[i] << 64) | in_low[i];
        out[i] = modq(val, proc);
    }
}

#endif
