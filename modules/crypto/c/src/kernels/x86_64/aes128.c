// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
// AES-128 on AES-NI, sixteen blocks in flight on VAES: the same round keys
// and bytes as portable/aes128.c, in time independent of the key.
#include "../gates.h"

#if VFHE_HAVE_AESNI
#include <immintrin.h>
#include <stdint.h>

#include "../aes128.h"

// One step of the key schedule, from the previous round key and its
// aeskeygenassist (Intel's AES-NI white paper).
static inline __m128i key_step(__m128i key, __m128i assist)
{
    assist = _mm_shuffle_epi32(assist, 0xff);
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    return _mm_xor_si128(key, assist);
}

void aes128_expand_key(uint8_t round_keys[176], const uint8_t key[16])
{
    __m128i k[11];
    k[0] = _mm_loadu_si128((const __m128i *)key);
    // aeskeygenassist takes the round constant as an immediate.
    k[1] = key_step(k[0], _mm_aeskeygenassist_si128(k[0], 0x01));
    k[2] = key_step(k[1], _mm_aeskeygenassist_si128(k[1], 0x02));
    k[3] = key_step(k[2], _mm_aeskeygenassist_si128(k[2], 0x04));
    k[4] = key_step(k[3], _mm_aeskeygenassist_si128(k[3], 0x08));
    k[5] = key_step(k[4], _mm_aeskeygenassist_si128(k[4], 0x10));
    k[6] = key_step(k[5], _mm_aeskeygenassist_si128(k[5], 0x20));
    k[7] = key_step(k[6], _mm_aeskeygenassist_si128(k[6], 0x40));
    k[8] = key_step(k[7], _mm_aeskeygenassist_si128(k[7], 0x80));
    k[9] = key_step(k[8], _mm_aeskeygenassist_si128(k[8], 0x1b));
    k[10] = key_step(k[9], _mm_aeskeygenassist_si128(k[9], 0x36));
    for (size_t r = 0; r < 11; r++)
        _mm_storeu_si128((__m128i *)&round_keys[16 * r], k[r]);
}

void aes128_ctr_keystream(uint8_t *out, const uint8_t rk[176], uint64_t j0, uint64_t t,
                          uint64_t count)
{
    __m128i k128[11];
    for (size_t r = 0; r < 11; r++)
        k128[r] = _mm_loadu_si128((const __m128i *)&rk[16 * r]);
    uint64_t done = 0;
#if VFHE_HAVE_VAES
    __m512i k512[11];
    for (size_t r = 0; r < 11; r++)
        k512[r] = _mm512_broadcast_i64x2(k128[r]);
    // Four counters per vector, sixteen blocks in flight: enough independent
    // work to cover the instruction's latency.
    const __m512i step = _mm512_set_epi64(0, 4, 0, 4, 0, 4, 0, 4);
    for (; done + 16 <= count; done += 16)
    {
        const uint64_t j = j0 + done;
        __m512i ctr =
            _mm512_set_epi64((long long)t, (long long)(j + 3), (long long)t, (long long)(j + 2),
                             (long long)t, (long long)(j + 1), (long long)t, (long long)j);
        __m512i b[4];
        for (size_t v = 0; v < 4; v++)
        {
            b[v] = _mm512_xor_si512(ctr, k512[0]);
            ctr = _mm512_add_epi64(ctr, step);
        }
        for (size_t r = 1; r < 10; r++)
            for (size_t v = 0; v < 4; v++)
                b[v] = _mm512_aesenc_epi128(b[v], k512[r]);
        for (size_t v = 0; v < 4; v++)
            _mm512_storeu_si512(&out[16 * (done + 4 * v)],
                                _mm512_aesenclast_epi128(b[v], k512[10]));
    }
#endif
    for (; done < count; done++)
    {
        __m128i b = _mm_xor_si128(_mm_set_epi64x((long long)t, (long long)(j0 + done)), k128[0]);
        for (size_t r = 1; r < 10; r++)
            b = _mm_aesenc_si128(b, k128[r]);
        _mm_storeu_si128((__m128i *)&out[16 * done], _mm_aesenclast_si128(b, k128[10]));
    }
}
#endif // VFHE_HAVE_AESNI
