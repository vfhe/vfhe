// SPDX-License-Identifier: Apache-2.0
/**
 * @file arith/config.h
 * @brief Build-variant selection for the arithmetic engine.
 *
 * The engine ships two implementations of every hot kernel:
 *
 *  - a portable scalar baseline (plain C11, runs everywhere), and
 *  - an AVX-512 IFMA fast path (x86-64 only, opt-in at compile time).
 *
 * ::VFHE_MP_SIMD is the single switch the rest of the engine branches on. It
 * is 1 exactly when the compiler advertises AVX-512 IFMA and no portable
 * override (`PORTABLE_BUILD` / `PORTABLE`) was requested; otherwise 0.
 *
 * The switch is compile-time on purpose: it also selects the digit type of
 * the multi-precision layer (see arith/mp.h), which changes struct layouts.
 * Backend selection *within* a build (32/50/64-bit reduction tiers) is done
 * at runtime, per prime, when a ::zq_ctx is created -- see arith/zq.h.
 */
#ifndef VFHE_ARITH_CONFIG_H
#define VFHE_ARITH_CONFIG_H

#if defined(__AVX512IFMA__) && !defined(PORTABLE_BUILD) && !defined(PORTABLE)
#define VFHE_MP_SIMD 1
#else
#define VFHE_MP_SIMD 0
#endif

#endif // VFHE_ARITH_CONFIG_H
