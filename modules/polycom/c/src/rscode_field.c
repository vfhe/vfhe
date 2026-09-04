// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
//
// The Reed-Solomon code over an extension field F_p[x]/(x^d - w), on the
// coefficient planes of a FieldVector. The evaluation points lie in F_p, so the
// transform is F_p-linear and runs once per plane with one NTT_Plan over the
// field's modulus; a plane is exactly the uint64_t array arith's NTT takes.
#include "rscode.h"

#include <string.h>

#include "util.h"

NTT_Plan rs_field_new_plan(uint64_t size, Modulus mod) { return ntt_new_plan(size, mod); }

void rs_field_free_plan(NTT_Plan plan) { ntt_free_plan(plan); }

uint64_t rs_field_plan_root(NTT_Plan plan) { return plan->root_of_unity; }

void rs_field_encode(FieldVector out, const FieldVector in, NTT_Plan plan)
{
    const uint64_t size = out->n;
    const uint64_t degree = in->n;
    uint64_t *scratch = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * size);

    for (uint64_t j = 0; j < in->d; j++)
    {
        memcpy(scratch, in->coeffs[j], sizeof(uint64_t) * degree);
        memset(&scratch[degree], 0, sizeof(uint64_t) * (size - degree));
        ntt_forward(scratch, scratch, plan);
        memcpy(out->coeffs[j], scratch, sizeof(uint64_t) * size);
    }

    free(scratch);
}

int rs_field_decode(FieldVector out, const FieldVector in, NTT_Plan plan)
{
    const uint64_t size = in->n;
    const uint64_t degree = out == NULL ? 0 : out->n;
    uint64_t *scratch = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * size);
    int is_codeword = 1;

    for (uint64_t j = 0; j < in->d && is_codeword; j++)
    {
        memcpy(scratch, in->coeffs[j], sizeof(uint64_t) * size);
        ntt_reverse(scratch, scratch, plan);
        if (out != NULL)
        {
            memcpy(out->coeffs[j], scratch, sizeof(uint64_t) * degree);
        }
        // The degree check: a codeword of this code inverts to a message
        // that was zero-padded above `degree`.
        for (uint64_t k = degree; k < size; k++)
        {
            if (scratch[k] != 0)
            {
                is_codeword = 0;
                break;
            }
        }
    }

    free(scratch);
    return is_codeword;
}
