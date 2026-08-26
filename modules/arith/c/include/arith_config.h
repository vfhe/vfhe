// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#ifndef VFHE_ARITH_CONFIG_H
#define VFHE_ARITH_CONFIG_H

// What the engine being compiled can use, as 0/1 macros for `#if`.
//
// An engine is one whole build of the tree for one ISA level: its flags come
// from tools/_engines.py, and the portable engine additionally defines
// PORTABLE_BUILD. Each macro below is therefore "the compiler offers this ISA
// AND this is not the portable engine" — the portable engine must compile the
// scalar path even on a host whose compiler would accept the intrinsics.
//
// Use `#if VFHE_HAVE_x`, never `#ifdef`: every macro is always defined, so a
// misspelling is a compile error under -Wundef rather than a silently
// disabled kernel.
//
// A translation unit guarded by one of these still needs its ISA header
// (<immintrin.h>) included behind an architecture check, since the header
// itself is x86-only.

#if defined(__AVX512IFMA__) && !defined(PORTABLE_BUILD)
#define VFHE_HAVE_AVX512IFMA 1
#else
#define VFHE_HAVE_AVX512IFMA 0
#endif

#if defined(__AVX512F__) && !defined(PORTABLE_BUILD)
#define VFHE_HAVE_AVX512F 1
#else
#define VFHE_HAVE_AVX512F 0
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(PORTABLE_BUILD)
#define VFHE_HAVE_X86_64 1
#else
#define VFHE_HAVE_X86_64 0
#endif

#if VFHE_HAVE_X86_64 && defined(__AES__)
#define VFHE_HAVE_AESNI 1
#else
#define VFHE_HAVE_AESNI 0
#endif

#if VFHE_HAVE_X86_64 && defined(__RDRND__)
#define VFHE_HAVE_RDRAND 1
#else
#define VFHE_HAVE_RDRAND 0
#endif

#endif // VFHE_ARITH_CONFIG_H
