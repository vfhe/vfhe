// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
// Reads a numeric setting from the environment: decimal digits and nothing
// else, within 64 bits. strtoull alone takes a sign, blanks and overflow.
#ifndef VFHE_UTIL_ENV_H
#define VFHE_UTIL_ENV_H

#include <ctype.h>
#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

// Returns true and sets `out` when `name` is set to an unsigned decimal
// integer that fits; false when it is unset, empty or anything else.
static inline bool env_uint64(const char *name, uint64_t *out)
{
    const char *text = getenv(name);
    if (text == NULL || !isdigit((unsigned char)*text))
        return false;
    char *end;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (*end != '\0' || errno != 0)
        return false;
    *out = (uint64_t)value;
    return true;
}

#endif // VFHE_UTIL_ENV_H
