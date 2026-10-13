// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file x86_64.h
 * @brief Which x86-64 vector extensions the engine being compiled may use.
 *
 * The compiler predefines `__AVX512IFMA__` and friends from the engine's `-m`
 * flags, set in the root meson.build. The portable engine forces every macro
 * here to 0, so its scalar path compiles on any host.
 *
 * Every macro is always defined, to 0 or 1. Test them with `#if`; `#ifdef` is
 * true for all of them. The build passes `-Wundef`, so a misspelt name is a
 * warning.
 *
 * @see x86_64_crypto.h for the instructions the crypto module uses.
 */
#ifndef VFHE_X86_64_H
#define VFHE_X86_64_H

/**
 * @def VFHE_HAVE_X86_64
 * @brief 1 on the x86-64 vector engines; 0 on the portable engine and on
 *        every other architecture. The other macros are gated on it.
 */
#if (defined(__x86_64__) || defined(_M_X64)) && !defined(PORTABLE_BUILD)
#define VFHE_HAVE_X86_64 1
#else
#define VFHE_HAVE_X86_64 0
#endif

/**
 * @def VFHE_HAVE_AVX512F
 * @brief 1 when the engine may use AVX-512 Foundation (`-mavx512f`).
 */
#if VFHE_HAVE_X86_64 && defined(__AVX512F__)
#define VFHE_HAVE_AVX512F 1
#else
#define VFHE_HAVE_AVX512F 0
#endif

/**
 * @def VFHE_HAVE_AVX512IFMA
 * @brief 1 when the engine may use AVX-512 IFMA (`-mavx512ifma`), the 52-bit
 *        fused multiply the multiprecision kernels are built on.
 */
#if VFHE_HAVE_X86_64 && defined(__AVX512IFMA__)
#define VFHE_HAVE_AVX512IFMA 1
#else
#define VFHE_HAVE_AVX512IFMA 0
#endif

#endif // VFHE_X86_64_H
