// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
// alloc.h's contract: 64-byte alignment, what realloc keeps, and a refused
// allocation exiting with the call named.

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <sys/wait.h>
#include <unistd.h>

#include <unity.h>

#include <alloc.h>

#define SMALL (64u * 1024u)
#define LARGE (12u * 1024u * 1024u)

#if defined(__SANITIZE_ADDRESS__)
#define UNDER_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define UNDER_ASAN 1
#endif
#endif
#ifndef UNDER_ASAN
#define UNDER_ASAN 0
#endif

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

static void a_large_buffer_is_writable_end_to_end(void)
{
    uint8_t *buf = (uint8_t *)safe_aligned_malloc(LARGE);
    TEST_ASSERT_NOT_NULL(buf);

    memset(buf, 0xA5, LARGE);
    for (size_t i = 0; i < LARGE; i += 4093)
        TEST_ASSERT_EQUAL_UINT8(0xA5, buf[i]);
    TEST_ASSERT_EQUAL_UINT8(0xA5, buf[0]);
    TEST_ASSERT_EQUAL_UINT8(0xA5, buf[LARGE - 1]);

    free(buf);
}

static void a_zero_byte_request_is_served(void)
{
    free(safe_aligned_malloc(0));
    free(safe_malloc(0));
    TEST_PASS();
}

static void growing_keeps_what_was_there(void)
{
    uint8_t *buf = (uint8_t *)safe_malloc(64);
    TEST_ASSERT_NOT_NULL(buf);
    memset(buf, 0x5A, 64);

    buf = (uint8_t *)safe_realloc(buf, SMALL);
    TEST_ASSERT_NOT_NULL(buf);
    for (size_t i = 0; i < 64; i++)
        TEST_ASSERT_EQUAL_UINT8(0x5A, buf[i]);

    free(buf);
}

static void reallocating_nothing_is_an_allocation(void)
{
    void *ptr = safe_realloc(NULL, 128);
    TEST_ASSERT_NOT_NULL(ptr);
    free(ptr);
}

static void shrinking_returns_memory(void)
{
    void *ptr = safe_realloc(safe_malloc(SMALL), 16);
    TEST_ASSERT_NOT_NULL(ptr);
    free(ptr);
}

#if !UNDER_ASAN
static volatile size_t more_than_memory = SIZE_MAX;

static void out_of_memory_malloc(void) { (void)safe_malloc(more_than_memory); }
static void out_of_memory_realloc(void) { (void)safe_realloc(NULL, more_than_memory); }
static void out_of_memory_aligned(void) { (void)safe_aligned_malloc(more_than_memory); }

static int exit_code_of(void (*call)(void), char *said, size_t room)
{
    int pipes[2];
    pid_t child;
    ssize_t written;
    int status = 0;

    said[0] = '\0';
    if (pipe(pipes) != 0)
        return -1;

    child = fork();
    if (child == 0)
    {
        close(pipes[0]);
        dup2(pipes[1], STDERR_FILENO);
        call();
        _exit(0);
    }

    close(pipes[1]);
    written = read(pipes[0], said, room - 1);
    said[written > 0 ? (size_t)written : 0] = '\0';
    close(pipes[0]);
    if (waitpid(child, &status, 0) < 0 || !WIFEXITED(status))
        return -1;
    return WEXITSTATUS(status);
}
#endif

static void a_refused_allocation_exits_and_names_the_call(void)
{
#if UNDER_ASAN
    TEST_IGNORE_MESSAGE("ASan refuses the oversized request before the allocator sees it");
#else
    char said[256];

    TEST_ASSERT_EQUAL_INT(EXIT_FAILURE, exit_code_of(out_of_memory_malloc, said, sizeof said));
    TEST_ASSERT_NOT_NULL(strstr(said, "malloc failed"));

    TEST_ASSERT_EQUAL_INT(EXIT_FAILURE, exit_code_of(out_of_memory_realloc, said, sizeof said));
    TEST_ASSERT_NOT_NULL(strstr(said, "realloc failed"));

    TEST_ASSERT_EQUAL_INT(EXIT_FAILURE, exit_code_of(out_of_memory_aligned, said, sizeof said));
    TEST_ASSERT_NOT_NULL(strstr(said, "posix_memalign failed"));
    TEST_ASSERT_NOT_NULL(strstr(said, strerror(ENOMEM)));
#endif
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(every_size_is_64_byte_aligned);
    RUN_TEST(a_large_buffer_is_writable_end_to_end);
    RUN_TEST(a_zero_byte_request_is_served);
    RUN_TEST(growing_keeps_what_was_there);
    RUN_TEST(reallocating_nothing_is_an_allocation);
    RUN_TEST(shrinking_returns_memory);
    RUN_TEST(a_refused_allocation_exits_and_names_the_call);
    return UNITY_END();
}
