// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
// CPU capability probes. Their own translation unit with no engine
// dependencies, so the engine picker's extension can carry just this.

#include <string.h>

#include "vfhe_cpu.h"

/* Whether this CPU can run an engine that requires `capability` — the name
 * an engine declares in tools/_engines.py. An unknown name answers no: a
 * missing case must fall back to a slower engine, never to illegal
 * instructions. An empty name means "no requirement". */
int vfhe_cpu_supports(const char *capability)
{
    if (capability == NULL || capability[0] == '\0')
        return 1;

#if defined(__x86_64__) || defined(_M_X64)
    __builtin_cpu_init();
    if (strcmp(capability, "avx512ifma") == 0)
        return __builtin_cpu_supports("avx512ifma") ? 1 : 0;
    if (strcmp(capability, "avx512f") == 0)
        return __builtin_cpu_supports("avx512f") ? 1 : 0;
    if (strcmp(capability, "avx2") == 0)
        return __builtin_cpu_supports("avx2") ? 1 : 0;
#elif defined(__aarch64__) || defined(_M_ARM64)
    if (strcmp(capability, "neon") == 0)
        return 1; /* Advanced SIMD is baseline on arm64 */
#endif

    return 0;
}
