// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0

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

#define HUGEPAGE_MIN_BYTES (8u << 20)

static void advise_hugepages(void *ptr, size_t size);

void *safe_malloc(size_t size)
{
    void *ptr = malloc(size);
    if (!ptr && (size > 0))
    {
        perror("malloc failed!");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void *safe_realloc(void *ptr, size_t size)
{
    void *grown = realloc(ptr, size);
    if (!grown && (size > 0))
    {
        perror("realloc failed!");
        free(ptr);
        exit(EXIT_FAILURE);
    }
    return grown;
}

void *safe_aligned_malloc(size_t size)
{
    void *ptr;
    int err = posix_memalign(&ptr, 64, size);
    if (err != 0)
    {
        fprintf(stderr, "posix_memalign failed: %s\n", strerror(err));
        exit(EXIT_FAILURE);
    }
    advise_hugepages(ptr, size);
    return ptr;
}

static void advise_hugepages(void *ptr, size_t size)
{
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    long page;
    uintptr_t start, end;

    if (size < HUGEPAGE_MIN_BYTES)
        return;

    page = sysconf(_SC_PAGESIZE);
    if (page <= 0)
        return;

    start = ((uintptr_t)ptr + (uintptr_t)page - 1) & ~((uintptr_t)page - 1);
    end = ((uintptr_t)ptr + size) & ~((uintptr_t)page - 1);
    if (end > start)
        (void)madvise((void *)start, (size_t)(end - start), MADV_HUGEPAGE);
#else
    (void)ptr;
    (void)size;
#endif
}
