// SPDX-License-Identifier: Apache-2.0
/**
 * @file debug.c
 * @brief Development printing helpers (see base.h).
 */
#include <inttypes.h>
#include <stdio.h>

#include "base.h"

void print_array(const char *msg, const uint64_t *v, size_t size)
{
    printf("%s: ", msg);
    for (size_t i = 0; i < size; i++)
    {
        printf("%" PRIu64 ", ", v[i]);
    }
    printf("\n");
}
