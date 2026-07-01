/* SPDX-License-Identifier: Apache-2.0 */
#ifndef VFHE_ARITH_H
#define VFHE_ARITH_H
#include <stdint.h>

typedef struct
{
    uint64_t n;       /* transform length */
    uint64_t modulus; /* prime modulus */
} NTT_proc;

void ntt_forward_32(uint64_t *out, uint64_t *in, NTT_proc proc);

/* Low 64 bits of a*b, implemented in hand-written assembly (see c/src/arch/). */
uint64_t asm_mul64(uint64_t a, uint64_t b);

#endif
