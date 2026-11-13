// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
// The gates the kernels test: the x86 instructions they use, on the same 0/1
// contract as <x86_64.h>, always defined and tested with `#if`. Only the
// kernels read them, which is why they live here.
#ifndef VFHE_CRYPTO_KERNELS_GATES_H
#define VFHE_CRYPTO_KERNELS_GATES_H

#include <x86_64.h>

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

#endif // VFHE_CRYPTO_KERNELS_GATES_H
