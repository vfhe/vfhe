// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
// What THIS CPU can run. Engine-independent by construction: the engine
// picker (_vfhe_native.py) imports this on its own, in a few-kilobyte
// extension, so choosing an engine never loads one.
#ifndef __VFHE_CPU_H__
#define __VFHE_CPU_H__

#ifdef __cplusplus
extern "C"
{
#endif

    // 1 if this CPU satisfies the named capability ("avx512ifma", "avx2",
    // "neon", ...); an unknown name answers 0.
    int vfhe_cpu_supports(const char *capability);

#ifdef __cplusplus
}
#endif

#endif /* __VFHE_CPU_H__ */
