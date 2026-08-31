// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "util.h"

#include <stdio.h>
#include <stdlib.h>

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
        perror("posix_memalign failed!");
        exit(EXIT_FAILURE);
    }
    return ptr;
}
