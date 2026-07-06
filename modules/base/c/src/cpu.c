// SPDX-License-Identifier: Apache-2.0
/**
 * @file cpu.c
 * @brief Runtime x86 instruction-set detection (see base.h).
 *
 * A feature is reported only when the CPUID feature bit is set AND, for the AVX
 * families, the OS has enabled the register state (checked via XCR0). On any
 * non-x86 host every flag is false, so callers transparently fall back to the
 * portable kernels.
 */
#include "base.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#include <cpuid.h>

// Read the extended control register 0 (needs OSXSAVE, checked by the caller).
static uint64_t read_xcr0(void)
{
    uint32_t lo, hi;
    __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((uint64_t)hi << 32) | lo;
}

void cpu_detect(cpu_features *out)
{
    *out = (cpu_features){0};

    uint32_t a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d))
        return;
    const bool aes = (c & bit_AES) != 0;
    const bool osxsave = (c & bit_OSXSAVE) != 0;
    const bool avx = (c & bit_AVX) != 0;
    out->aes = aes;

    // AVX / AVX-512 also need the OS to have enabled the register state.
    const uint64_t xcr0 = osxsave ? read_xcr0() : 0;
    const bool ymm_ok = avx && (xcr0 & 0x6) == 0x6;      // XMM + YMM
    const bool zmm_ok = ymm_ok && (xcr0 & 0xE0) == 0xE0; // + OPMASK/ZMM_hi/HI16

    uint32_t a7, b7, c7, d7;
    if (!__get_cpuid_count(7, 0, &a7, &b7, &c7, &d7))
        return;
    out->avx2 = ymm_ok && (b7 & bit_AVX2) != 0;
    out->avx512f = zmm_ok && (b7 & bit_AVX512F) != 0;
    out->avx512ifma = out->avx512f && (b7 & bit_AVX512IFMA) != 0;
}

#else

void cpu_detect(cpu_features *out) { *out = (cpu_features){0}; }

#endif
