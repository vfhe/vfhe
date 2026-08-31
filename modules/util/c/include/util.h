// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stddef.h>
#include <stdint.h>

#include <vfhe_cpu.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // Index and modulus helpers
    void array_reduce_mod_N(uint64_t *out, uint64_t *in, uint64_t size, uint64_t p);
    void array_mod_switch_from_2k(uint64_t *out, uint64_t *in, uint64_t p, uint64_t q, uint64_t n);
    uint64_t double2int(double x);
    uint32_t int_rev(uint32_t b);
    void bit_rev(uint64_t *out, uint64_t *in, uint64_t n, uint64_t log_n);

    // Allocation that aborts rather than returning NULL
    void *safe_malloc(size_t size);
    void *safe_realloc(void *ptr, size_t size);
    void *safe_aligned_malloc(size_t size);

    // Which engine this binary is (CPU capability lives in vfhe_cpu.h, which
    // this header includes).
    const char *vfhe_engine_active(void); // e.g. "portable", "avx512ifma"

#ifdef __cplusplus
}
#endif
