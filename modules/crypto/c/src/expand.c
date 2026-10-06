// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
//
// Seed expansion: AES-128 in counter mode, filtered to values below a bound.
//
// crypto.h defines the sequence. It is computed here two ways that must agree
// byte for byte: with AES-NI (VAES where available), adapted from MOSFHET's
// counter-mode PRNG, and with a table-based AES on the portable engine. The
// AES-NI path, key schedule included, runs in time independent of the key,
// which the entropy stream (prng.c) relies on. The table-based path does not,
// so it only ever sees public keys: the portable entropy stream uses BLAKE3.
#include <x86_64.h>

#include "x86_64_crypto.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include <blake3.h>

#include "crypto.h"

#if VFHE_HAVE_AESNI
#include <immintrin.h>
#endif

// Keystream blocks drawn per pass of the filter: 4 KiB, which stays in L1
// between the cipher writing it and the filter reading it.
#define EXPAND_BLOCKS 256

// --- AES-128 (FIPS-197) ------------------------------------------------

#if !VFHE_HAVE_AESNI
static const uint8_t aes_sbox[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
};

// The 11 round keys, 16 bytes each, in the byte order XORed into the state.
static void aes128_expand_key(uint8_t round_keys[176], const uint8_t key[16])
{
    static const uint8_t rcon[10] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36};
    memcpy(round_keys, key, 16);
    for (size_t i = 4; i < 44; i++)
    {
        uint8_t t[4];
        memcpy(t, &round_keys[4 * (i - 1)], 4);
        if (i % 4 == 0)
        {
            const uint8_t first = t[0];
            t[0] = (uint8_t)(aes_sbox[t[1]] ^ rcon[i / 4 - 1]);
            t[1] = aes_sbox[t[2]];
            t[2] = aes_sbox[t[3]];
            t[3] = aes_sbox[first];
        }
        for (size_t k = 0; k < 4; k++)
            round_keys[4 * i + k] = (uint8_t)(round_keys[4 * (i - 4) + k] ^ t[k]);
    }
}

// MixColumns of SubBytes, as one column word (byte r = row r): what an
// input byte in row 0 contributes to its output column. A byte in row k
// contributes the same word rotated left by 8k bits.
static const uint32_t aes_te0[256] = {
    0xa56363c6, 0x847c7cf8, 0x997777ee, 0x8d7b7bf6, 0x0df2f2ff, 0xbd6b6bd6, 0xb16f6fde, 0x54c5c591,
    0x50303060, 0x03010102, 0xa96767ce, 0x7d2b2b56, 0x19fefee7, 0x62d7d7b5, 0xe6abab4d, 0x9a7676ec,
    0x45caca8f, 0x9d82821f, 0x40c9c989, 0x877d7dfa, 0x15fafaef, 0xeb5959b2, 0xc947478e, 0x0bf0f0fb,
    0xecadad41, 0x67d4d4b3, 0xfda2a25f, 0xeaafaf45, 0xbf9c9c23, 0xf7a4a453, 0x967272e4, 0x5bc0c09b,
    0xc2b7b775, 0x1cfdfde1, 0xae93933d, 0x6a26264c, 0x5a36366c, 0x413f3f7e, 0x02f7f7f5, 0x4fcccc83,
    0x5c343468, 0xf4a5a551, 0x34e5e5d1, 0x08f1f1f9, 0x937171e2, 0x73d8d8ab, 0x53313162, 0x3f15152a,
    0x0c040408, 0x52c7c795, 0x65232346, 0x5ec3c39d, 0x28181830, 0xa1969637, 0x0f05050a, 0xb59a9a2f,
    0x0907070e, 0x36121224, 0x9b80801b, 0x3de2e2df, 0x26ebebcd, 0x6927274e, 0xcdb2b27f, 0x9f7575ea,
    0x1b090912, 0x9e83831d, 0x742c2c58, 0x2e1a1a34, 0x2d1b1b36, 0xb26e6edc, 0xee5a5ab4, 0xfba0a05b,
    0xf65252a4, 0x4d3b3b76, 0x61d6d6b7, 0xceb3b37d, 0x7b292952, 0x3ee3e3dd, 0x712f2f5e, 0x97848413,
    0xf55353a6, 0x68d1d1b9, 0x00000000, 0x2cededc1, 0x60202040, 0x1ffcfce3, 0xc8b1b179, 0xed5b5bb6,
    0xbe6a6ad4, 0x46cbcb8d, 0xd9bebe67, 0x4b393972, 0xde4a4a94, 0xd44c4c98, 0xe85858b0, 0x4acfcf85,
    0x6bd0d0bb, 0x2aefefc5, 0xe5aaaa4f, 0x16fbfbed, 0xc5434386, 0xd74d4d9a, 0x55333366, 0x94858511,
    0xcf45458a, 0x10f9f9e9, 0x06020204, 0x817f7ffe, 0xf05050a0, 0x443c3c78, 0xba9f9f25, 0xe3a8a84b,
    0xf35151a2, 0xfea3a35d, 0xc0404080, 0x8a8f8f05, 0xad92923f, 0xbc9d9d21, 0x48383870, 0x04f5f5f1,
    0xdfbcbc63, 0xc1b6b677, 0x75dadaaf, 0x63212142, 0x30101020, 0x1affffe5, 0x0ef3f3fd, 0x6dd2d2bf,
    0x4ccdcd81, 0x140c0c18, 0x35131326, 0x2fececc3, 0xe15f5fbe, 0xa2979735, 0xcc444488, 0x3917172e,
    0x57c4c493, 0xf2a7a755, 0x827e7efc, 0x473d3d7a, 0xac6464c8, 0xe75d5dba, 0x2b191932, 0x957373e6,
    0xa06060c0, 0x98818119, 0xd14f4f9e, 0x7fdcdca3, 0x66222244, 0x7e2a2a54, 0xab90903b, 0x8388880b,
    0xca46468c, 0x29eeeec7, 0xd3b8b86b, 0x3c141428, 0x79dedea7, 0xe25e5ebc, 0x1d0b0b16, 0x76dbdbad,
    0x3be0e0db, 0x56323264, 0x4e3a3a74, 0x1e0a0a14, 0xdb494992, 0x0a06060c, 0x6c242448, 0xe45c5cb8,
    0x5dc2c29f, 0x6ed3d3bd, 0xefacac43, 0xa66262c4, 0xa8919139, 0xa4959531, 0x37e4e4d3, 0x8b7979f2,
    0x32e7e7d5, 0x43c8c88b, 0x5937376e, 0xb76d6dda, 0x8c8d8d01, 0x64d5d5b1, 0xd24e4e9c, 0xe0a9a949,
    0xb46c6cd8, 0xfa5656ac, 0x07f4f4f3, 0x25eaeacf, 0xaf6565ca, 0x8e7a7af4, 0xe9aeae47, 0x18080810,
    0xd5baba6f, 0x887878f0, 0x6f25254a, 0x722e2e5c, 0x241c1c38, 0xf1a6a657, 0xc7b4b473, 0x51c6c697,
    0x23e8e8cb, 0x7cdddda1, 0x9c7474e8, 0x211f1f3e, 0xdd4b4b96, 0xdcbdbd61, 0x868b8b0d, 0x858a8a0f,
    0x907070e0, 0x423e3e7c, 0xc4b5b571, 0xaa6666cc, 0xd8484890, 0x05030306, 0x01f6f6f7, 0x120e0e1c,
    0xa36161c2, 0x5f35356a, 0xf95757ae, 0xd0b9b969, 0x91868617, 0x58c1c199, 0x271d1d3a, 0xb99e9e27,
    0x38e1e1d9, 0x13f8f8eb, 0xb398982b, 0x33111122, 0xbb6969d2, 0x70d9d9a9, 0x898e8e07, 0xa7949433,
    0xb69b9b2d, 0x221e1e3c, 0x92878715, 0x20e9e9c9, 0x49cece87, 0xff5555aa, 0x78282850, 0x7adfdfa5,
    0x8f8c8c03, 0xf8a1a159, 0x80898909, 0x170d0d1a, 0xdabfbf65, 0x31e6e6d7, 0xc6424284, 0xb86868d0,
    0xc3414182, 0xb0999929, 0x772d2d5a, 0x110f0f1e, 0xcbb0b07b, 0xfc5454a8, 0xd6bbbb6d, 0x3a16162c,
};

static inline uint32_t rotl32(uint32_t x, unsigned k) { return (x << k) | (x >> ((32 - k) & 31)); }

static inline uint32_t load_column(const uint8_t *b)
{
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

// One block. The state is the FIPS-197 column-major layout: byte 4c + r is
// row r of column c, which is the order the block's bytes arrive in; held
// here as four column words.
static void aes128_encrypt_block(uint8_t out[16], const uint8_t in[16], const uint8_t rk[176])
{
    uint32_t s[4], t[4];
    for (size_t c = 0; c < 4; c++)
        s[c] = load_column(&in[4 * c]) ^ load_column(&rk[4 * c]);
    for (size_t round = 1; round < 10; round++)
    {
        // ShiftRows: row r of column c comes from column c + r.
        for (size_t c = 0; c < 4; c++)
            t[c] = aes_te0[s[c] & 0xff] ^ rotl32(aes_te0[(s[(c + 1) % 4] >> 8) & 0xff], 8) ^
                   rotl32(aes_te0[(s[(c + 2) % 4] >> 16) & 0xff], 16) ^
                   rotl32(aes_te0[s[(c + 3) % 4] >> 24], 24) ^ load_column(&rk[16 * round + 4 * c]);
        memcpy(s, t, sizeof s);
    }
    for (size_t c = 0; c < 4; c++)
    {
        const uint32_t w = (uint32_t)aes_sbox[s[c] & 0xff] |
                           ((uint32_t)aes_sbox[(s[(c + 1) % 4] >> 8) & 0xff] << 8) |
                           ((uint32_t)aes_sbox[(s[(c + 2) % 4] >> 16) & 0xff] << 16) |
                           ((uint32_t)aes_sbox[s[(c + 3) % 4] >> 24] << 24);
        t[c] = w ^ load_column(&rk[160 + 4 * c]);
        for (size_t r = 0; r < 4; r++)
            out[4 * c + r] = (uint8_t)(t[c] >> (8 * r));
    }
}

// The counter block for keystream block `j` of attempt `t`: two
// little-endian u64s, `j` first.
static void ctr_block(uint8_t block[16], uint64_t j, uint64_t t)
{
    for (size_t k = 0; k < 8; k++)
    {
        block[k] = (uint8_t)(j >> (8 * k));
        block[8 + k] = (uint8_t)(t >> (8 * k));
    }
}
#endif

// Keystream blocks [j0, j0 + count) of attempt `t`, 16 bytes each.
#if VFHE_HAVE_AESNI
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

// The same round keys as the table-based schedule, without table lookups.
static void aes128_expand_key(uint8_t round_keys[176], const uint8_t key[16])
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

static void ctr_keystream(uint8_t *out, const uint8_t rk[176], uint64_t j0, uint64_t t,
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
#else
static void ctr_keystream(uint8_t *out, const uint8_t rk[176], uint64_t j0, uint64_t t,
                          uint64_t count)
{
    uint8_t block[16];
    for (uint64_t done = 0; done < count; done++)
    {
        ctr_block(block, j0 + done, t);
        aes128_encrypt_block(&out[16 * done], block, rk);
    }
}
#endif

void prng_aes128_ctr(uint8_t *out, uint64_t count, uint64_t first, uint64_t attempt,
                     const uint8_t key[16])
{
    uint8_t rk[176];
    aes128_expand_key(rk, key);
    ctr_keystream(out, rk, first, attempt, count);
}

// --- Expansion -----------------------------------------------------------

void prng_expand_key(uint8_t key[16], const char *context, const uint8_t *seed, uint64_t seed_len,
                     const uint64_t *label, uint64_t label_len)
{
    blake3_hasher hasher;
    blake3_hasher_init_derive_key(&hasher, context);
    blake3_hasher_update(&hasher, seed, seed_len);
    for (uint64_t i = 0; i < label_len; i++)
    {
        uint8_t word[8];
        for (size_t k = 0; k < 8; k++)
            word[k] = (uint8_t)(label[i] >> (8 * k));
        blake3_hasher_update(&hasher, word, sizeof(word));
    }
    blake3_hasher_finalize(&hasher, key, 16);
}

// Smallest 2^k - 1 covering every value below `bound`.
static uint64_t below_mask(uint64_t bound)
{
    uint64_t mask = bound - 1;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    mask |= mask >> 32;
    return mask;
}

static inline uint64_t load_word(const uint8_t *bytes, size_t width)
{
    uint64_t w = 0;
    for (size_t k = 0; k < width; k++)
        w |= (uint64_t)bytes[k] << (8 * k);
    return w;
}

// Value `i` from attempt 1 on: the rare path, one block per try.
static uint64_t expand_retry(const uint8_t rk[176], uint64_t i, size_t width, uint64_t mask,
                             uint64_t bound)
{
    const uint64_t byte = i * width;
    uint8_t block[16];
    for (uint64_t t = 1;; t++)
    {
        ctr_keystream(block, rk, byte / 16, t, 1);
        const uint64_t v = load_word(&block[byte % 16], width) & mask;
        if (v < bound)
            return v;
    }
}

// Values [start, start + count) into `out`, `out_width` (4 or 8) bytes each.
// The output width does not change which keystream words are read.
static void expand_below(void *out, size_t out_width, uint64_t count, uint64_t start,
                         uint64_t bound, const uint8_t key[16])
{
    assert(bound >= 1);
    if (count == 0)
        return;
    uint8_t rk[176];
    aes128_expand_key(rk, key);
    const size_t width = bound <= (1ULL << 32) ? 4 : 8;
    const uint64_t mask = below_mask(bound);
    const uint64_t per_block = 16 / width;
    uint8_t stream[16 * EXPAND_BLOCKS] __attribute__((aligned(64)));
    uint64_t *out64 = (uint64_t *)out;
    uint32_t *out32 = (uint32_t *)out;

    uint64_t i = start;
    const uint64_t end = start + count;
    while (i < end)
    {
        // The blocks holding values [i, i + run), from the block value i is in.
        const uint64_t j = i / per_block;
        const uint64_t skip = i - j * per_block;
        uint64_t run = EXPAND_BLOCKS * per_block - skip;
        if (run > end - i)
            run = end - i;
        const uint64_t blocks = (skip + run + per_block - 1) / per_block;
        ctr_keystream(stream, rk, j, 0, blocks);
        const uint8_t *words = &stream[skip * width];
        uint64_t k = 0;
#if VFHE_HAVE_AVX512F
        // Mask and compare a vector at a time; the rare rejected lane is
        // recomputed by expand_retry.
        if (width == 8)
        {
            const __m512i m = _mm512_set1_epi64((long long)mask);
            const __m512i b = _mm512_set1_epi64((long long)bound);
            for (; k + 8 <= run; k += 8)
            {
                const __m512i v = _mm512_and_si512(_mm512_loadu_si512(&words[8 * k]), m);
                __mmask8 bad = (__mmask8)~_mm512_cmplt_epu64_mask(v, b);
                const uint64_t at = i - start + k;
                if (out_width == 8)
                    _mm512_storeu_si512(&out64[at], v);
                else
                    _mm256_storeu_si256((__m256i *)&out32[at], _mm512_cvtepi64_epi32(v));
                while (bad)
                {
                    const unsigned lane = (unsigned)__builtin_ctz(bad);
                    bad = (__mmask8)(bad & (bad - 1));
                    const uint64_t v2 = expand_retry(rk, i + k + lane, width, mask, bound);
                    if (out_width == 8)
                        out64[at + lane] = v2;
                    else
                        out32[at + lane] = (uint32_t)v2;
                }
            }
        }
        else
        {
            const __m512i m = _mm512_set1_epi32((int)(uint32_t)mask);
            const __m512i b = _mm512_set1_epi32((int)(uint32_t)(bound - 1));
            for (; k + 16 <= run; k += 16)
            {
                const __m512i v = _mm512_and_si512(_mm512_loadu_si512(&words[4 * k]), m);
                // bound may be 2^32, so compare against bound - 1 inclusively.
                __mmask16 bad = (__mmask16)~_mm512_cmple_epu32_mask(v, b);
                const uint64_t at = i - start + k;
                if (out_width == 4)
                {
                    _mm512_storeu_si512(&out32[at], v);
                }
                else
                {
                    _mm512_storeu_si512(&out64[at],
                                        _mm512_cvtepu32_epi64(_mm512_castsi512_si256(v)));
                    _mm512_storeu_si512(&out64[at + 8],
                                        _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(v, 1)));
                }
                while (bad)
                {
                    const unsigned lane = (unsigned)__builtin_ctz(bad);
                    bad = (__mmask16)(bad & (bad - 1));
                    const uint64_t v2 = expand_retry(rk, i + k + lane, width, mask, bound);
                    if (out_width == 8)
                        out64[at + lane] = v2;
                    else
                        out32[at + lane] = (uint32_t)v2;
                }
            }
        }
#endif
        for (; k < run; k++)
        {
            uint64_t v = load_word(&words[width * k], width) & mask;
            if (v >= bound)
                v = expand_retry(rk, i + k, width, mask, bound);
            if (out_width == 8)
                out64[i - start + k] = v;
            else
                out32[i - start + k] = (uint32_t)v;
        }
        i += run;
    }
}

void prng_expand_below(uint64_t *out, uint64_t count, uint64_t start, uint64_t bound,
                       const uint8_t key[16])
{
    expand_below(out, 8, count, start, bound, key);
}

void prng_expand_below32(uint32_t *out, uint64_t count, uint64_t start, uint64_t bound,
                         const uint8_t key[16])
{
    assert(bound <= (1ULL << 32));
    expand_below(out, 4, count, start, bound, key);
}
