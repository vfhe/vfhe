// SPDX-License-Identifier: Apache-2.0
/**
 * @file alloc.c
 * @brief Abort-on-failure allocation (see base.h).
 */
#include <stdio.h>
#include <stdlib.h>

#include "base.h"

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

void *safe_aligned_malloc(size_t size)
{
    void *ptr;
    int err = posix_memalign(&ptr, 64, size);
    if (err != 0)
    {
        perror("posix_memalign failed!");
        exit(EXIT_FAILURE);
    }
    return ptr;
}
