// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
// CPU capability probes. Their own translation unit with no engine
// dependencies, so the engine picker's extension can carry just this.

#include <stdio.h>
#include <string.h>
#include <unistd.h>

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

/* The last level of cache before memory, in bytes, or 0 if unknown.
 *
 * Asked of `sysconf` first, which answers on glibc; the sysfs walk below
 * covers a C library without those names. L3 is reported where there is one
 * and L2 otherwise, since what a caller wants is the boundary at which a
 * working set starts going to memory, whatever that level is numbered. */
static uint64_t cache_from_sysfs(void)
{
    uint64_t best = 0;
    for (int index = 0; index < 10; index++)
    {
        char path[128];
        int level = 0;
        unsigned long size = 0;
        char unit = 0;
        FILE *f;

        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu0/cache/index%d/level", index);
        f = fopen(path, "r");
        if (f == NULL)
            break;
        if (fscanf(f, "%d", &level) != 1)
            level = 0;
        fclose(f);

        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu0/cache/index%d/size", index);
        f = fopen(path, "r");
        if (f == NULL)
            continue;
        if (fscanf(f, "%lu%c", &size, &unit) >= 1 && level >= 2)
        {
            uint64_t bytes = size;
            if (unit == 'K' || unit == 'k')
                bytes *= 1024;
            else if (unit == 'M' || unit == 'm')
                bytes *= 1024 * 1024;
            if (bytes > best)
                best = bytes;
        }
        fclose(f);
    }
    return best;
}

uint64_t vfhe_cpu_last_level_cache_bytes(void)
{
#if defined(_SC_LEVEL3_CACHE_SIZE)
    long l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
    if (l3 > 0)
        return (uint64_t)l3;
#endif
#if defined(_SC_LEVEL2_CACHE_SIZE)
    long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
    if (l2 > 0)
        return (uint64_t)l2;
#endif
    return cache_from_sysfs();
}

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
