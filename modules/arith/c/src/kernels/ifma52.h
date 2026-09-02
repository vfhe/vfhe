// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
//
// The 52-bit limb radix, and a scalar model of the IFMA multiply-accumulate.
//
// Several implementations here hold big values as base-2^52 limbs because that
// is the radix AVX-512 IFMA multiplies in: VPMADD52LUQ / VPMADD52HUQ read the
// low 52 bits of each 64-bit lane of BOTH operands, multiply, and accumulate
// the low or high 52 bits of the product. `madd52lo` and `madd52hi` below are
// the exact scalar equivalent, truncation included -- a lane holding more than
// 52 bits is silently narrowed on the way in, so a portable path that skipped
// the mask would compute something the tuned path does not.
#ifndef VFHE_IFMA52_H
#define VFHE_IFMA52_H

#include <stdint.h>

#define IFMA52_BITS 52
#define IFMA52_MASK 0x000fffffffffffffULL

static inline uint64_t madd52lo(uint64_t a, uint64_t b, uint64_t c)
{
    unsigned __int128 prod = (unsigned __int128)(b & IFMA52_MASK) * (c & IFMA52_MASK);
    return a + (uint64_t)(prod & IFMA52_MASK);
}

static inline uint64_t madd52hi(uint64_t a, uint64_t b, uint64_t c)
{
    unsigned __int128 prod = (unsigned __int128)(b & IFMA52_MASK) * (c & IFMA52_MASK);
    return a + (uint64_t)(prod >> IFMA52_BITS);
}

#endif // VFHE_IFMA52_H
