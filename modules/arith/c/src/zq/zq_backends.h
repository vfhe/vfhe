// SPDX-License-Identifier: Apache-2.0
/**
 * @file zq_backends.h
 * @brief Internal registry of element-wise kernel tables (one per backend).
 *
 * Each backend translation unit exports exactly one ::zq_ops table;
 * ::zq_ctx_init picks one based on the build and the prime's bit width.
 * Adding a backend = adding one file + one extern line here.
 */
#ifndef VFHE_ZQ_BACKENDS_H
#define VFHE_ZQ_BACKENDS_H

#include <arith/config.h>
#include <arith/zq.h>

/** Scalar C11 kernels; correct for any q < 2^63. Always available. */
extern const zq_ops zq_ops_portable;

#if VFHE_MP_SIMD
/** AVX-512 kernels for q < 2^32 (mullo products fit 64 bits). */
extern const zq_ops zq_ops_avx512_32;
/** AVX-512 IFMA kernels for q < 2^50 (52-bit multiply-add). */
extern const zq_ops zq_ops_avx512_50;
/** AVX-512 kernels for q < 2^63 (emulated 128-bit products). */
extern const zq_ops zq_ops_avx512_64;
#endif

#endif // VFHE_ZQ_BACKENDS_H
