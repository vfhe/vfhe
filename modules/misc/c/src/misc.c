#include "misc.h"
#include <inttypes.h>

static inline uint64_t Log2(uint64_t x)
{
    if (x == 0)
        return 0;
    return 63 - __builtin_clzll(x);
}

uint64_t next_power_of_2(uint64_t x)
{
    if (x == 0)
        return 1;
    if ((x & (x - 1)) == 0)
        return x;
    return 1ULL << (Log2(x) + 1);
}

uint64_t double2int(double x) { return ((uint64_t)((int64_t)x)); }

uint64_t mod_switch(uint64_t v, uint64_t p, uint64_t q)
{
    const double double_q = q == 0 ? pow(2, 64) : ((double)q);
    const double double_p = p == 0 ? pow(2, 64) : ((double)p);
    uint64_t val = (uint64_t)round((((double)v) * double_q) / double_p);
    return val < q ? val : val - q;
}

// Generates a sparse ternary array with Hamming Weight h, balanced (h/2 ones and h/2 negative ones)
void gen_sparse_ternary_array_modq(uint64_t *out, uint64_t size, uint64_t h, uint64_t q)
{
    memset(out, 0, sizeof(uint64_t) * size);
    uint64_t hw = 0, val = 1, *rnd_buffer;
    const uint64_t buffer_size = h * 10;
    rnd_buffer = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * buffer_size);
    while (hw < h)
    {
        generate_random_bytes(sizeof(uint64_t) * buffer_size, (uint8_t *)rnd_buffer);
        array_mod_switch_from_2k(rnd_buffer, rnd_buffer, size, size, buffer_size);
        uint64_t i = 0;
        while (i < buffer_size && hw < h)
        {
            const uint64_t idx = rnd_buffer[i++];
            if (out[idx])
                continue;
            out[idx] = (uint64_t)((q + (int64_t)val) % q);
            val = -val;
            hw++;
        }
    }
    free(rnd_buffer);
#ifndef NDEBUG
    uint64_t hw_check = 0, sum_check = 0;
    for (size_t i = 0; i < size; i++)
    {
        sum_check += out[i];
        hw_check += (out[i] != 0);
    }
    assert(hw_check == h);
    assert((sum_check % q) == 0);
#endif
}

void array_reduce_mod_N(uint64_t *out, uint64_t *in, uint64_t size, uint64_t p)
{
    const uint64_t mask = next_power_of_2(p) - 1;
    for (size_t i = 0; i < size; i++)
    {
        out[i] = in[i] & mask;
    }
}

/* Mod switch the additive inverse of negative values */
/* Used to adjust negative values in Gaussian keys when represented by the inverse */
void array_additive_inverse_mod_switch(uint64_t *out, uint64_t *in, uint64_t p, uint64_t q,
                                       uint64_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        if (in[i] > p / 2)
            out[i] = q - (p - in[i]);
        else
            out[i] = in[i];
    }
}

/* Switch each element mod p to mod q*/
void array_mod_switch(uint64_t *out, uint64_t *in, uint64_t p, uint64_t q, uint64_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        // out[i] = (round((((double) in[i])/((double) p))*q));
        out[i] = mod_switch(in[i], q, p);
    }
}

/* Switch each element mod next_power_of_two(p) to mod q */
void array_mod_switch_from_2k(uint64_t *out, uint64_t *in, uint64_t p, uint64_t q, uint64_t n)
{
    array_reduce_mod_N(out, in, n, p);
    uint64_t p2 = next_power_of_2(p);
    for (size_t i = 0; i < n; i++)
    {
        // out[i] = (round((((double) in[i])/((double) p))*q));
        out[i] = mod_switch(in[i], p2, q);
    }
}

uint64_t int_mod_switch(uint64_t in, uint64_t p, uint64_t q)
{
    // return (round((((double) in)/((double) p))*q));
    return mod_switch(in, p, q);
}

NTT_proc *new_ntt_list(uint64_t *primes, uint64_t N, uint64_t l)
{
    NTT_proc *ntt = (NTT_proc *)safe_malloc(sizeof(NTT_proc) * l);
    for (size_t i = 0; i < l; i++)
    {
        ntt[i] = ntt_new_proc(N, primes[i]);
    }
    return ntt;
}

// Computes (Z_q[i](Q/q[i]))**-1, for i in [0,l)
void compute_RNS_Qhat_array(uint64_t *out, uint64_t *p, uint64_t l)
{
    for (size_t i = 0; i < l; i++)
    {
        out[i] = 1;
        for (size_t j = 0; j < l; j++)
        {
            if (i != j)
            {
                const uint64_t inv = inverse_mod(p[j], p[i]);
                out[i] = (uint64_t)(((unsigned __int128)out[i] * inv) % p[i]);
            }
        }
    }
}

uint64_t mod_dist(uint64_t a, uint64_t b, uint64_t q)
{
    const uint64_t dist = (uint64_t)((q + (int64_t)a - (int64_t)b) % q);
    if (dist > q / 2)
        return q - dist;
    return dist;
}

void print_array(const char *msg, uint64_t *v, size_t size)
{
    printf("%s: ", msg);
    for (size_t i = 0; i < size; i++)
    {
        printf("%" PRIu64 ", ", v[i]);
    }
    printf("\n");
}

unsigned char char_rev(unsigned char b)
{
    b = (unsigned char)(((b & 0xF0) >> 4) | ((b & 0x0F) << 4));
    b = (unsigned char)(((b & 0xCC) >> 2) | ((b & 0x33) << 2));
    b = (unsigned char)(((b & 0xAA) >> 1) | ((b & 0x55) << 1));
    return b;
}

uint32_t int_rev(uint32_t b)
{
#if defined(__GNUC__) || defined(__clang__)
    uint32_t a = __builtin_bswap32(b);
#else
    uint32_t a = ((b >> 24) & 0xffu) | ((b >> 8) & 0xff00u) | ((b << 8) & 0xff0000u) | (b << 24);
#endif
    unsigned char *a_vec = (unsigned char *)&a;
    a_vec[0] = char_rev(a_vec[0]);
    a_vec[1] = char_rev(a_vec[1]);
    a_vec[2] = char_rev(a_vec[2]);
    a_vec[3] = char_rev(a_vec[3]);
    return a;
}

void bit_rev(uint64_t *out, uint64_t *in, uint64_t n, uint64_t log_n)
{
    for (size_t i = 0; i < n; i++)
    {
        out[i] = in[int_rev((uint32_t)i) >> (32 - log_n)];
    }
}
