// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
// The x86 instructions crypto uses, on the same 0/1 contract as <x86_64.h>: always
// defined, tested with `#if`. Nothing outside crypto reads them, which is why
// they are here and not beside the vector extensions.
#ifndef VFHE_X86_64_CRYPTO_H
#define VFHE_X86_64_CRYPTO_H

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

#endif // VFHE_X86_64_CRYPTO_H
