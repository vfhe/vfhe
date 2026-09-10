// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
// What the engine being compiled may use, as 0/1 macros for `#if`.
//
// An engine is one whole build of the tree for one ISA level: its flags come
// from the engine list in the root meson.build, and the portable engine defines
// PORTABLE_BUILD. Each macro below is therefore "the compiler offers this ISA
// AND this is not the portable engine": the portable engine must compile the
// scalar path even on a host whose compiler would accept the intrinsics.
//
// Every macro is always defined, to 0 or 1. Test them with `#if`, not
// `#ifdef`, which would be true for all of them; the build passes -Wundef so a
// misspelt name is a warning rather than a silent 0.
//
// Which engine this binary is, by name, is vfhe_engine_active() in util.h;
// what the running CPU supports is vfhe_cpu.h. This header is about the build.
#ifndef VFHE_ENGINE_H
#define VFHE_ENGINE_H

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(PORTABLE_BUILD)
#define VFHE_HAVE_X86_64 1
#else
#define VFHE_HAVE_X86_64 0
#endif

#if VFHE_HAVE_X86_64 && defined(__AVX512F__)
#define VFHE_HAVE_AVX512F 1
#else
#define VFHE_HAVE_AVX512F 0
#endif

#if VFHE_HAVE_X86_64 && defined(__AVX512IFMA__)
#define VFHE_HAVE_AVX512IFMA 1
#else
#define VFHE_HAVE_AVX512IFMA 0
#endif

#if VFHE_HAVE_X86_64 && defined(__AES__)
#define VFHE_HAVE_AESNI 1
#else
#define VFHE_HAVE_AESNI 0
#endif

#if VFHE_HAVE_X86_64 && defined(__VAES__)
#define VFHE_HAVE_VAES 1
#else
#define VFHE_HAVE_VAES 0
#endif

#if VFHE_HAVE_X86_64 && defined(__RDRND__)
#define VFHE_HAVE_RDRAND 1
#else
#define VFHE_HAVE_RDRAND 0
#endif

#endif // VFHE_ENGINE_H
