// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
// CPU capability probes. Their own translation unit with no engine
// dependencies, so the engine picker's extension can carry just this.

#include <string.h>

#include "vfhe_cpu.h"

/* -1 unknown, 0 absent, 1 present. Private, because no public answer may be
 * true for a name this build cannot judge. */
static int probe(const char *capability)
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

    return -1;
}

/* Whether this CPU can run an engine that requires `capability` — the name
 * an engine declares in meson.build's engine facts. An unknown name answers no: a
 * missing case must fall back to a slower engine, never to illegal
 * instructions. An empty name means "no requirement". */
int vfhe_cpu_supports(const char *capability) { return probe(capability) > 0; }

/* Whether this build's probe can judge `capability` at all. Falling back is
 * right at runtime and wrong while developing: it makes a mistyped or
 * not-yet-taught name look like a CPU that merely lacks the feature. The
 * tools ask this and refuse; the picker does not, so an install degrades. */
int vfhe_cpu_knows(const char *capability) { return probe(capability) >= 0; }

#ifdef VFHE_CPU_MAIN
#include <stdio.h>

/* The probe as a command, for callers without an interpreter (the Makefile's
 * test recipe): exit 0 supported, 1 absent, 2 unjudgeable or misused. */
int main(int argc, char **argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "usage: vfhe-cpu <capability>\n");
        return 2;
    }
    if (!vfhe_cpu_knows(argv[1]))
    {
        fprintf(stderr,
                "cpu_probe.c cannot judge '%s' on this architecture: a typo "
                "in meson.build's engine facts, or a name it has yet to "
                "learn\n",
                argv[1]);
        return 2;
    }
    return vfhe_cpu_supports(argv[1]) ? 0 : 1;
}
#endif
