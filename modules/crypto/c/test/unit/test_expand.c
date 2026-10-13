// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_expand.c
 * @brief The seed expander against AES's published vectors and against the
 * definition in crypto.h, read straight off the keystream.
 */
#include <crypto.h>
#include <stdlib.h>
#include <string.h>

#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static uint64_t from_le(const uint8_t *in, int width)
{
    uint64_t v = 0;
    for (int k = 0; k < width; k++)
        v |= (uint64_t)in[k] << (8 * k);
    return v;
}

/* FIPS-197 Appendix C.1, and the all-zero key and block. The counter block
 * is LE64(j) || LE64(t), so the plaintext 00 11 .. ff is j = 0x7766..00,
 * t = 0xffee..88. */
void test_aes_matches_fips197(void)
{
    uint8_t key[16], out[16];
    for (int k = 0; k < 16; k++)
        key[k] = (uint8_t)k;
    const uint8_t expected[16] = {0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30,
                                  0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4, 0xc5, 0x5a};
    prng_aes128_ctr(out, 1, 0x7766554433221100ULL, 0xffeeddccbbaa9988ULL, key);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expected, out, 16);

    memset(key, 0, sizeof key);
    const uint8_t zero[16] = {0x66, 0xe9, 0x4b, 0xd4, 0xef, 0x8a, 0x2c, 0x3b,
                              0x88, 0x4c, 0xfa, 0x59, 0xca, 0x34, 0x2b, 0x2e};
    prng_aes128_ctr(out, 1, 0, 0, key);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(zero, out, 16);
}

/* A long run goes through the wide path and its tail through the single
 * block one; both must be the blocks asked for one at a time. */
void test_bulk_keystream_is_blockwise(void)
{
    uint8_t key[16];
    for (int k = 0; k < 16; k++)
        key[k] = (uint8_t)(17 * k + 3);
    const uint64_t count = 53;
    uint8_t bulk[53 * 16], one[16];
    prng_aes128_ctr(bulk, count, 1000, 7, key);
    for (uint64_t j = 0; j < count; j++)
    {
        prng_aes128_ctr(one, 1, 1000 + j, 7, key);
        TEST_ASSERT_EQUAL_UINT8_ARRAY(one, &bulk[16 * j], 16);
    }
}

/* Value i of the definition, read off the keystream one block at a time. */
static uint64_t reference_value(const uint8_t key[16], uint64_t i, uint64_t bound)
{
    const int width = bound <= (1ULL << 32) ? 4 : 8;
    uint64_t mask = bound - 1;
    for (int s = 1; s < 64; s <<= 1)
        mask |= mask >> s;
    const uint64_t byte = i * (uint64_t)width;
    uint8_t block[16];
    for (uint64_t t = 0;; t++)
    {
        prng_aes128_ctr(block, 1, byte / 16, t, key);
        const uint64_t v = from_le(&block[byte % 16], width) & mask;
        if (v < bound)
            return v;
    }
}

static void assert_matches_definition(uint64_t bound, uint64_t start, uint64_t count)
{
    uint8_t key[16];
    for (int k = 0; k < 16; k++)
        key[k] = (uint8_t)(5 * k + (uint8_t)bound);
    uint64_t *out = (uint64_t *)malloc(count * sizeof(uint64_t));
    prng_expand_below(out, count, start, bound, key);
    for (uint64_t i = 0; i < count; i++)
        TEST_ASSERT_EQUAL_UINT64(reference_value(key, start + i, bound), out[i]);
    if (bound <= (1ULL << 32))
    {
        uint32_t *out32 = (uint32_t *)malloc(count * sizeof(uint32_t));
        prng_expand_below32(out32, count, start, bound, key);
        for (uint64_t i = 0; i < count; i++)
            TEST_ASSERT_EQUAL_UINT64(out[i], out32[i]);
        free(out32);
    }
    free(out);
}

/* Bounds just under a power of two (vfhe's primes, almost never rejecting),
 * just over one (rejecting about half), both word widths and the edges of
 * each; windows at unaligned starts and of lengths around the vector and
 * pass sizes. */
void test_expansion_matches_definition(void)
{
    const uint64_t bounds[] = {1,
                               2,
                               3,
                               (1ULL << 30) - 35,
                               (1ULL << 31) + 1,
                               1ULL << 32,
                               (1ULL << 32) + 1,
                               562949953408001ULL,
                               (1ULL << 48) + 1,
                               1152921504606815233ULL,
                               (1ULL << 62) - 57};
    const uint64_t windows[][2] = {{0, 1}, {0, 37}, {3, 64}, {5, 1029}, {1021, 4099}};
    for (size_t b = 0; b < sizeof bounds / sizeof bounds[0]; b++)
        for (size_t w = 0; w < sizeof windows / sizeof windows[0]; w++)
            assert_matches_definition(bounds[b], windows[w][0], windows[w][1]);
}

/* Fixed outputs, so a change to the definition -- which stored seeds depend on --
 * fails here on every engine rather than passing everywhere consistently. */
void test_expansion_is_frozen(void)
{
    uint8_t key[16];
    const uint64_t label[2] = {1, 562949953408001ULL};
    const uint8_t seed[4] = {1, 2, 3, 4};
    prng_expand_key(key, "vfhe test_expand 2026-10-04", seed, sizeof seed, label, 2);
    uint64_t out[4];
    prng_expand_below(out, 4, 0, 562949953408001ULL, key);
    const uint64_t expected[4] = {163687556019873ULL, 559985323326312ULL, 167986888455385ULL,
                                  339841244940742ULL};
    TEST_ASSERT_EQUAL_UINT64_ARRAY(expected, out, 4);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_aes_matches_fips197);
    RUN_TEST(test_bulk_keystream_is_blockwise);
    RUN_TEST(test_expansion_matches_definition);
    RUN_TEST(test_expansion_is_frozen);
    return UNITY_END();
}
