// SPDX-License-Identifier: Apache-2.0
/**
 * @file aes_ctr.c
 * @brief AES-128 counter-mode expansion backend (x86 with AES-NI).
 *
 * Adapted from MOSFHET (https://github.com/antoniocgj/MOSFHET). Produces the
 * keystream in 256-byte blocks: 16 AES states advanced four counters at a
 * time (VAES processes four 128-bit lanes per 512-bit register when
 * available). The key is a process-global expanded once, on first use, from
 * the platform entropy source.
 *
 * Compiled only when the compiler advertises AES-NI (`__AES__`); portable
 * builds use the BLAKE3 backend in stream.c instead.
 */
#if defined(__AES__)

#include <assert.h>
#include <immintrin.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "rng_internal.h"

// AES-128 key expansion, from the Intel AES-NI white paper.
static inline __m128i AES_128_ASSIST(__m128i temp1, __m128i temp2)
{
    __m128i temp3;
    temp2 = _mm_shuffle_epi32(temp2, 0xff);
    temp3 = _mm_slli_si128(temp1, 0x4);
    temp1 = _mm_xor_si128(temp1, temp3);
    temp3 = _mm_slli_si128(temp3, 0x4);
    temp1 = _mm_xor_si128(temp1, temp3);
    temp3 = _mm_slli_si128(temp3, 0x4);
    temp1 = _mm_xor_si128(temp1, temp3);
    temp1 = _mm_xor_si128(temp1, temp2);
    return temp1;
}

#ifdef __VAES__
static __m512i aes_round_keys[11];
#else
static __m128i aes_round_keys[11];
#define _mm512_broadcast_i64x2(X) X
#endif

/** Expand @p seed into the 11 AES-128 round keys (broadcast on VAES). */
static void aes_key_setup(const __m128i *seed)
{
    __m128i tmp1, tmp2;
    tmp1 = _mm_loadu_si128((const __m128i *)seed);
    aes_round_keys[0] = _mm512_broadcast_i64x2(tmp1);

    // Round constants 0x01..0x36 of the AES-128 key schedule.
    static const int rcon[10] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36};
    // _mm_aeskeygenassist_si128 requires an immediate; unroll via switch.
    for (int r = 0; r < 10; r++)
    {
        switch (rcon[r])
        {
        case 0x01:
            tmp2 = _mm_aeskeygenassist_si128(tmp1, 0x01);
            break;
        case 0x02:
            tmp2 = _mm_aeskeygenassist_si128(tmp1, 0x02);
            break;
        case 0x04:
            tmp2 = _mm_aeskeygenassist_si128(tmp1, 0x04);
            break;
        case 0x08:
            tmp2 = _mm_aeskeygenassist_si128(tmp1, 0x08);
            break;
        case 0x10:
            tmp2 = _mm_aeskeygenassist_si128(tmp1, 0x10);
            break;
        case 0x20:
            tmp2 = _mm_aeskeygenassist_si128(tmp1, 0x20);
            break;
        case 0x40:
            tmp2 = _mm_aeskeygenassist_si128(tmp1, 0x40);
            break;
        case 0x80:
            tmp2 = _mm_aeskeygenassist_si128(tmp1, 0x80);
            break;
        case 0x1B:
            tmp2 = _mm_aeskeygenassist_si128(tmp1, 0x1B);
            break;
        default:
            tmp2 = _mm_aeskeygenassist_si128(tmp1, 0x36);
            break;
        }
        tmp1 = AES_128_ASSIST(tmp1, tmp2);
        aes_round_keys[r + 1] = _mm512_broadcast_i64x2(tmp1);
    }
}

/** One-time key initialization from the entropy source. */
static void aes_key_init_once(void)
{
    static bool initialized = false;
    if (!initialized)
    {
        __m128i s[2];
        rng_seed_bytes((uint64_t *)s);
        aes_key_setup(s);
        initialized = true;
    }
}

#ifdef __VAES__

/** Produce 256 keystream bytes (4 x 512-bit states), advancing the counter. */
static void aes_ctr_next_256(__m512i *out, __m512i *cnt)
{
    const __m512i incv = {0, 4, 0, 4, 0, 4, 0, 4};
    for (size_t i = 0; i < 4; i++)
    {
        out[i] = _mm512_xor_si512(*cnt, aes_round_keys[0]);
        *cnt = _mm512_add_epi64(*cnt, incv);
    }

    for (size_t i = 1; i < 10; i++)
    {
        for (size_t j = 0; j < 4; j++)
        {
            out[j] = _mm512_aesenc_epi128(out[j], aes_round_keys[i]);
        }
    }

    for (size_t i = 0; i < 4; i++)
    {
        out[i] = _mm512_aesenclast_epi128(out[i], aes_round_keys[10]);
    }
}

void aes_prng(uint8_t *output, uint64_t outlen, const uint8_t *input, uint64_t inlen)
{
    assert(outlen >= 256);
    assert(inlen >= 16);
    (void)inlen;
    aes_key_init_once();
    __m128i cnt = _mm_loadu_si128((const __m128i *)input);
    __m512i cntv = _mm512_broadcast_i64x2(cnt);
    __m512i cntv2 = {0, 0, 0, 1, 0, 2, 0, 3};
    cntv = _mm512_add_epi64(cntv, cntv2);
    size_t i;
    for (i = 0; i < outlen - 255; i += 256)
    {
        aes_ctr_next_256((__m512i *)&(output[i]), &cntv);
    }
    if (outlen > i)
    {
        __m512i tmp[4];
        aes_ctr_next_256(tmp, &cntv);
        memcpy(&output[i], tmp, outlen - i);
    }
}

#else

/** Produce 256 keystream bytes (16 x 128-bit states), advancing the counter. */
static void aes_ctr_next_256(__m128i *out, __m128i *cnt)
{
    const __m128i incv = {0, 4};
    for (size_t i = 0; i < 16; i++)
    {
        out[i] = _mm_xor_si128(*cnt, aes_round_keys[0]);
        *cnt = _mm_add_epi64(*cnt, incv);
    }

    for (size_t i = 1; i < 10; i++)
    {
        for (size_t j = 0; j < 16; j++)
        {
            out[j] = _mm_aesenc_si128(out[j], aes_round_keys[i]);
        }
    }

    for (size_t i = 0; i < 16; i++)
    {
        out[i] = _mm_aesenclast_si128(out[i], aes_round_keys[10]);
    }
}

void aes_prng(uint8_t *output, uint64_t outlen, const uint8_t *input, uint64_t inlen)
{
    assert(outlen >= 256);
    assert(inlen >= 16);
    (void)inlen;
    aes_key_init_once();
    __m128i cnt = _mm_loadu_si128((const __m128i *)input);
    size_t i;
    for (i = 0; i < outlen - 255; i += 256)
    {
        aes_ctr_next_256((__m128i *)&(output[i]), &cnt);
    }
    if (outlen > i)
    {
        __m128i tmp[16];
        aes_ctr_next_256(tmp, &cnt);
        memcpy(&output[i], tmp, outlen - i);
    }
}

#endif // __VAES__

#endif // __AES__
