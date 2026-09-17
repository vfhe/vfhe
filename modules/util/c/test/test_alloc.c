// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
// The aligned allocator's contract: 64-byte alignment, which the SIMD kernels
// require, on both sides of the huge-page threshold.

#include <stdint.h>
#include <string.h>
#include <unity.h>
#include <util.h>

// Either side of the huge-page threshold.
#define SMALL (64u * 1024u)
#define LARGE (12u * 1024u * 1024u)

void setUp(void) {}
void tearDown(void) {}

static void every_size_is_64_byte_aligned(void)
{
    static const size_t sizes[] = {1, 8, 63, 64, 65, SMALL, LARGE};

    for (size_t i = 0; i < sizeof sizes / sizeof *sizes; i++)
    {
        void *ptr = safe_aligned_malloc(sizes[i]);
        TEST_ASSERT_NOT_NULL(ptr);
        TEST_ASSERT_EQUAL_UINT64(0, (uint64_t)((uintptr_t)ptr % 64));
        free(ptr);
    }
}

// First and last byte included: the huge-page advice is computed over an
// interior sub-range, and an off-by-one there would reach outside the
// allocation.
static void a_large_buffer_is_writable_end_to_end(void)
{
    uint8_t *buf = (uint8_t *)safe_aligned_malloc(LARGE);
    TEST_ASSERT_NOT_NULL(buf);

    memset(buf, 0xA5, LARGE);
    for (size_t i = 0; i < LARGE; i += 4093) // an odd stride, so the reads land at varying page offsets
        TEST_ASSERT_EQUAL_UINT8(0xA5, buf[i]);
    TEST_ASSERT_EQUAL_UINT8(0xA5, buf[0]);
    TEST_ASSERT_EQUAL_UINT8(0xA5, buf[LARGE - 1]);

    free(buf);
}

// Zero bytes may return NULL or a free-able pointer. Neither is a failure, so
// the allocator must not abort on the NULL.
static void a_zero_byte_request_does_not_abort(void)
{
    free(safe_aligned_malloc(0));
    TEST_PASS();
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(every_size_is_64_byte_aligned);
    RUN_TEST(a_large_buffer_is_writable_end_to_end);
    RUN_TEST(a_zero_byte_request_does_not_abort);
    return UNITY_END();
}
