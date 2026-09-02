// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
//
// One group of pseudo-Mersenne elements, as a word the kernels compute on.
//
// A `pv_t` holds PMF_VEC_GROUP independent elements' worth of one limb: eight
// lanes of an AVX-512 register on the tuned engine, a single uint64_t on the
// portable one. Every operation below is lane-wise, so a kernel written against
// them says the same thing at either width and both engines run the same
// algorithm -- which is what lets the portable build be a real test of the
// tuned one rather than a separate implementation.
//
// Two rules keep that property:
//
// - No cross-lane operation. Carries travel between limbs of one element, and
//   an element lives in one lane, so nothing here needs lanes to interact.
// - Conditionals produce arithmetic masks -- a word that is all zeros or all
//   ones -- not AVX-512 predicate registers, which have no portable width-1
//   counterpart. `pv_mask_from_flag` and `pv_select` are the whole vocabulary.
#ifndef VFHE_PMF_VEC_OPS_H
#define VFHE_PMF_VEC_OPS_H

#include <arith.h>

#include "kernels/ifma52.h"

#if VFHE_HAVE_AVX512IFMA
#include <immintrin.h>

#define PMF_VEC_GROUP 8
typedef __m512i pv_t;

static inline pv_t pv_zero(void) { return _mm512_setzero_si512(); }
static inline pv_t pv_bcast(uint64_t x) { return _mm512_set1_epi64((long long)x); }
static inline pv_t pv_load(const uint64_t *p) { return _mm512_load_si512((const void *)p); }
static inline void pv_store(uint64_t *p, pv_t v) { _mm512_store_si512((void *)p, v); }
static inline pv_t pv_add(pv_t a, pv_t b) { return _mm512_add_epi64(a, b); }
static inline pv_t pv_sub(pv_t a, pv_t b) { return _mm512_sub_epi64(a, b); }
static inline pv_t pv_and(pv_t a, pv_t b) { return _mm512_and_si512(a, b); }
static inline pv_t pv_or(pv_t a, pv_t b) { return _mm512_or_si512(a, b); }
static inline pv_t pv_xor(pv_t a, pv_t b) { return _mm512_xor_si512(a, b); }
static inline pv_t pv_srl(pv_t a, uint64_t count)
{
    return _mm512_srl_epi64(a, _mm_cvtsi64_si128((long long)count));
}
static inline pv_t pv_madd52lo(pv_t acc, pv_t a, pv_t b)
{
    return _mm512_madd52lo_epu64(acc, a, b);
}
static inline pv_t pv_madd52hi(pv_t acc, pv_t a, pv_t b)
{
    return _mm512_madd52hi_epu64(acc, a, b);
}
static inline int pv_is_zero(pv_t a) { return _mm512_test_epi64_mask(a, a) == 0; }

#else

#define PMF_VEC_GROUP 1
typedef uint64_t pv_t;

static inline pv_t pv_zero(void) { return 0; }
static inline pv_t pv_bcast(uint64_t x) { return x; }
static inline pv_t pv_load(const uint64_t *p) { return *p; }
static inline void pv_store(uint64_t *p, pv_t v) { *p = v; }
static inline pv_t pv_add(pv_t a, pv_t b) { return a + b; }
static inline pv_t pv_sub(pv_t a, pv_t b) { return a - b; }
static inline pv_t pv_and(pv_t a, pv_t b) { return a & b; }
static inline pv_t pv_or(pv_t a, pv_t b) { return a | b; }
static inline pv_t pv_xor(pv_t a, pv_t b) { return a ^ b; }
static inline pv_t pv_srl(pv_t a, uint64_t count) { return a >> count; }
static inline pv_t pv_madd52lo(pv_t acc, pv_t a, pv_t b) { return madd52lo(acc, a, b); }
static inline pv_t pv_madd52hi(pv_t acc, pv_t a, pv_t b) { return madd52hi(acc, a, b); }
static inline int pv_is_zero(pv_t a) { return a == 0; }

#endif

// The 52-bit limb mask, as a word.
static inline pv_t pv_mask52(void) { return pv_bcast(PMF_LIMB_MASK); }

// Turn a per-lane 0/1 flag into a per-lane 0 / all-ones mask.
static inline pv_t pv_mask_from_flag(pv_t flag) { return pv_sub(pv_zero(), flag); }

// `mask` lanes choose `on`, the rest choose `off`.
static inline pv_t pv_select(pv_t mask, pv_t on, pv_t off)
{
    return pv_xor(off, pv_and(pv_xor(off, on), mask));
}

// The borrow out of `a - b - borrow_in`, as a per-lane 0/1 flag, given that both
// limbs are below 2^52: the difference then either fits 52 bits or wraps, and
// bit 63 of the wrapped word is what says which.
static inline pv_t pv_borrow(pv_t difference) { return pv_srl(difference, 63); }

#endif // VFHE_PMF_VEC_OPS_H
