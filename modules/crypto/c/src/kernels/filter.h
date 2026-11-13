// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
// The filter of the seed expander: a run of keystream words masked and kept
// when below `bound`, the rest replaced by what `retry` gives for their index.
// portable/filter.c is one loop; x86_64/filter.c compares 8 (64-bit) or 16
// (32-bit) words at a time and leaves the tail to the loop below.
#ifndef VFHE_CRYPTO_KERNELS_FILTER_H
#define VFHE_CRYPTO_KERNELS_FILTER_H

#include <stddef.h>
#include <stdint.h>

// Writes `count` values to `out` (`out_width` 4 or 8 bytes each): word `k` of
// `words` (`width` bytes each) masked by `mask`, or `retry(ctx, index + k)`
// when the masked word is not below `bound`.
void filter_below(void *out, size_t out_width, const uint8_t *words, uint64_t count, uint64_t index,
                  size_t width, uint64_t mask, uint64_t bound,
                  uint64_t (*retry)(void *ctx, uint64_t index), void *ctx);

// Word `width` bytes wide, little-endian, from unaligned bytes.
static inline uint64_t load_word(const uint8_t *bytes, size_t width)
{
    uint64_t w = 0;
    for (size_t k = 0; k < width; k++)
        w |= (uint64_t)bytes[k] << (8 * k);
    return w;
}

// Value `k` of `out`, `out_width` bytes each.
static inline void store_value(void *out, size_t out_width, uint64_t k, uint64_t v)
{
    if (out_width == 8)
        ((uint64_t *)out)[k] = v;
    else
        ((uint32_t *)out)[k] = (uint32_t)v;
}

// Values [from, count) of filter_below, one at a time.
static inline void filter_below_tail(void *out, size_t out_width, const uint8_t *words,
                                     uint64_t count, uint64_t from, uint64_t index, size_t width,
                                     uint64_t mask, uint64_t bound,
                                     uint64_t (*retry)(void *ctx, uint64_t index), void *ctx)
{
    for (uint64_t k = from; k < count; k++)
    {
        uint64_t v = load_word(&words[width * k], width) & mask;
        if (v >= bound)
            v = retry(ctx, index + k);
        store_value(out, out_width, k, v);
    }
}

#endif // VFHE_CRYPTO_KERNELS_FILTER_H
