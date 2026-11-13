// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
#include "../gates.h"

#if !VFHE_HAVE_AVX512F
#include "../filter.h"

void filter_below(void *out, size_t out_width, const uint8_t *words, uint64_t count, uint64_t index,
                  size_t width, uint64_t mask, uint64_t bound,
                  uint64_t (*retry)(void *ctx, uint64_t index), void *ctx)
{
    filter_below_tail(out, out_width, words, count, 0, index, width, mask, bound, retry, ctx);
}
#endif // !VFHE_HAVE_AVX512F
