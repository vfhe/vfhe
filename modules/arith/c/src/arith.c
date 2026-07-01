/* SPDX-License-Identifier: Apache-2.0 */
#include "arith.h"

void ntt_forward_32(uint64_t *out, uint64_t *in, NTT_proc proc)
{
    for (uint64_t i = 0; i < proc.n; i++)
    {
        out[i] = in[i] % proc.modulus;
    }
}
