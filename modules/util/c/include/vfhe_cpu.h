// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
// Reports what this CPU can run. No engine dependencies: the picker
// (_vfhe_native.py) imports this in its own few-kilobyte extension, so
// choosing an engine never loads one.
#ifndef __VFHE_CPU_H__
#define __VFHE_CPU_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // 1 if this CPU satisfies the named capability ("avx512ifma", "avx2",
    // "neon", ...); an unknown name answers 0.
    int vfhe_cpu_supports(const char *capability);

    // 1 if this build can judge the named capability on this architecture, so
    // a caller can tell "this CPU lacks it" from "the probe cannot say".
    int vfhe_cpu_knows(const char *capability);

    // The size in bytes of the last level of cache before memory: L3 where
    // there is one, L2 otherwise. 0 means this platform does not report it,
    // which a caller should read as "unknown" rather than "none" -- the
    // number is only useful for deciding whether a working set has outgrown
    // the cache, and a guess answers that wrongly in both directions.
    uint64_t vfhe_cpu_last_level_cache_bytes(void);

#ifdef __cplusplus
}
#endif

#endif /* __VFHE_CPU_H__ */
