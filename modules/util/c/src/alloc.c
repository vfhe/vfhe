// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
// madvise and sysconf are POSIX extensions under glibc.
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif

#include "alloc.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

// Allocations from this size ask for huge pages: below it the page faults of
// a buffer cost less than aligning it to a huge page would.
#define HUGEPAGE_MIN_BYTES (8u << 20)

void *safe_malloc(size_t size)
{
    void *ptr = malloc(size);
    if (ptr == NULL && size > 0)
    {
        perror("malloc failed!");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void *safe_realloc(void *ptr, size_t size)
{
    void *grown = realloc(ptr, size);
    if (grown == NULL && size > 0)
    {
        perror("realloc failed!");
        exit(EXIT_FAILURE);
    }
    return grown;
}

// Asks for huge pages over the whole pages inside [ptr, ptr + size). Advice
// only: a refusal changes nothing.
static void advise_hugepages(void *ptr, size_t size)
{
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    if (size < HUGEPAGE_MIN_BYTES)
        return;
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0)
        return;
    const uintptr_t start = ((uintptr_t)ptr + (uintptr_t)page - 1) & ~((uintptr_t)page - 1);
    const uintptr_t end = ((uintptr_t)ptr + size) & ~((uintptr_t)page - 1);
    if (end > start)
        (void)madvise((void *)start, (size_t)(end - start), MADV_HUGEPAGE);
#else
    (void)ptr;
    (void)size;
#endif
}

void *safe_aligned_malloc(size_t size)
{
    void *ptr;
    const int err = posix_memalign(&ptr, 64, size);
    if (err != 0)
    {
        fprintf(stderr, "posix_memalign failed: %s\n", strerror(err));
        exit(EXIT_FAILURE);
    }
    advise_hugepages(ptr, size);
    return ptr;
}
