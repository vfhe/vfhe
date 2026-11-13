// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
#include "../gates.h"

#if !VFHE_HAVE_RDRAND
#include <stdio.h>
#include <stdlib.h>
#include <sys/random.h>
#include <unistd.h>

#include "../seed.h"

// getentropy reads the kernel's pool whole, with no file, no buffering and no
// short read.
void seed_source(uint64_t p[4])
{
    if (getentropy(p, 32) != 0)
    {
        perror("vfhe: getentropy"); /* GCOVR_EXCL_LINE */
        exit(EXIT_FAILURE);         /* GCOVR_EXCL_LINE */
    }
}
#endif // !VFHE_HAVE_RDRAND
