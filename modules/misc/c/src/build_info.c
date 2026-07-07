#include "misc.h"

/* Whether this extension was compiled as the portable (scalar) baseline. */
int vfhe_build_is_portable(void)
{
#if defined(PORTABLE_BUILD)
    return 1;
#else
    return 0;
#endif
}

/* Whether the CPU running right now supports AVX-512 IFMA (the feature the
 * tuned build's kernels are gated on). Runtime CPUID via the compiler builtin;
 * always 0 off x86. */
int vfhe_cpu_has_avx512ifma(void)
{
#if defined(__x86_64__) || defined(_M_X64)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512ifma") ? 1 : 0;
#else
    return 0;
#endif
}
