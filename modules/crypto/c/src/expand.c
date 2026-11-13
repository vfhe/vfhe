// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
// Seed expansion: AES-128 in counter mode, filtered to values below a bound.
// expand.h defines the sequence; the kernels produce the keystream and keep
// the words below the bound, each as the portable code or an engine's own.
#include "expand.h"

#include <assert.h>
#include <stdint.h>

#include <blake3.h>

#include "below.h"
#include "kernels/aes128.h"
#include "kernels/filter.h"

// Keystream blocks drawn per pass of the filter: 4 KiB, which stays in L1
// between the cipher writing it and the filter reading it.
#define EXPAND_BLOCKS 256

void expand_key(uint8_t key[16], const char *context, const uint8_t *seed, uint64_t seed_len,
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

void expand_aes128_ctr(uint8_t *out, uint64_t count, uint64_t first, uint64_t attempt,
                       const uint8_t key[16])
{
    uint8_t rk[176];
    aes128_expand_key(rk, key);
    aes128_ctr_keystream(out, rk, first, attempt, count);
}

// What the filter's retry needs: value `i` from attempt 1 on, one block per
// try. The rare path.
struct retry_ctx
{
    const uint8_t *rk;
    size_t width;
    uint64_t mask, bound;
};

static uint64_t retry(void *ctx, uint64_t i)
{
    const struct retry_ctx *c = (const struct retry_ctx *)ctx;
    const uint64_t byte = i * c->width;
    uint8_t block[16];
    for (uint64_t t = 1;; t++)
    {
        aes128_ctr_keystream(block, c->rk, byte / 16, t, 1);
        const uint64_t v = load_word(&block[byte % 16], c->width) & c->mask;
        if (v < c->bound)
            return v;
    }
}

// Values [start, start + count) into `out`, `out_width` (4 or 8) bytes each.
// The output width does not change which keystream words are read.
static void expand_values(void *out, size_t out_width, uint64_t count, uint64_t start,
                          uint64_t bound, const uint8_t key[16])
{
    assert(bound >= 1);
    assert(start <= UINT64_MAX - count); // the range [start, start + count) exists
    if (count == 0)
        return;
    uint8_t rk[176];
    aes128_expand_key(rk, key);
    const size_t width = bound <= (1ULL << 32) ? 4 : 8;
    const uint64_t per_block = 16 / width;
    struct retry_ctx ctx = {rk, width, below_mask(bound), bound};
    uint8_t stream[16 * EXPAND_BLOCKS] __attribute__((aligned(64)));

    const uint64_t end = start + count;
    for (uint64_t i = start; i < end;)
    {
        // The blocks holding values [i, i + run), from the block value i is in.
        const uint64_t j = i / per_block;
        const uint64_t skip = i - j * per_block;
        uint64_t run = EXPAND_BLOCKS * per_block - skip;
        if (run > end - i)
            run = end - i;
        const uint64_t blocks = (skip + run + per_block - 1) / per_block;
        aes128_ctr_keystream(stream, rk, j, 0, blocks);
        filter_below((uint8_t *)out + (i - start) * out_width, out_width, &stream[skip * width],
                     run, i, width, ctx.mask, bound, retry, &ctx);
        i += run;
    }
}

void expand_below(uint64_t *out, uint64_t count, uint64_t start, uint64_t bound,
                  const uint8_t key[16])
{
    expand_values(out, 8, count, start, bound, key);
}

void expand_below32(uint32_t *out, uint64_t count, uint64_t start, uint64_t bound,
                    const uint8_t key[16])
{
    assert(bound <= (1ULL << 32));
    expand_values(out, 4, count, start, bound, key);
}
