// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
//
// The Reed-Solomon code over an RNS ring. The transform runs per prime and the
// codeword is gathered coefficient by coefficient, so the whole file is tied to
// the representation; only the signatures are generic, so a field version can
// sit beside it under the same entry points.
#include "rscode.h"
#include <arith_generic.h>

#include <string.h>

#include "util.h"

// -------------------------------------------------------------
// Reed-Solomon (foldable) code over R_q, per RNS prime
// -------------------------------------------------------------
// The transform itself is arith's negacyclic NTT of the codeword length:
// gather one (prime, coefficient slot) column of the message vector into a
// scratch buffer, zero-pad it to the codeword length, transform, scatter the
// result back into the same column of the output vector. See rscode.h for the
// evaluation-point layout the fold depends on.

NTT_Plan *rs_new_plans(RNS_Base base, uint64_t rns_mask, uint64_t size)
{
    const uint64_t rows = (uint64_t)(rns_mask_get_last_active_index(rns_mask) + 1);
    NTT_Plan *plans = (NTT_Plan *)safe_malloc(sizeof(NTT_Plan) * rows);
    for (uint64_t i = 0; i < rows; i++)
    {
        // Borrow the RNS_Base's modulus: one set of Barrett constants per prime,
        // shared by this code's plan at every level.
        plans[i] = (rns_mask & (1ULL << i)) ? ntt_new_plan(size, base->mods[i]) : NULL;
    }
    return plans;
}

void rs_free_plans(NTT_Plan *plans, uint64_t count)
{
    for (uint64_t i = 0; i < count; i++)
    {
        if (plans[i] != NULL)
            ntt_free_plan(plans[i]);
    }
    free(plans);
}

uint64_t rs_plans_root(NTT_Plan *plans, uint64_t index)
{
    return plans[index] == NULL ? 0 : plans[index]->root_of_unity;
}

void rs_encode(ArithElement *out, ArithElement *in, uint64_t size, uint64_t degree, NTT_Plan *plans)
{
    RNS_Base base = arith_rns_polynomial(&in[0])->base;
    const uint64_t N = base->N;
    const uint64_t rns_mask = arith_rns_polynomial(&in[0])->rns_mask;
    uint64_t *codeword = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * size);

    for (uint64_t k = 0; k < size; k++)
    {
        arith_rns_polynomial(&out[k])->rns_mask = rns_mask;
    }

    for (uint64_t i = 0; i < base->l; i++)
    {
        if (!(rns_mask & (1ULL << i)))
            continue;
        for (uint64_t j = 0; j < N; j++)
        {
            memset(&codeword[degree], 0, sizeof(uint64_t) * (size - degree));
            for (uint64_t k = 0; k < degree; k++)
            {
                codeword[k] = arith_rns_polynomial(&in[k])->coeffs[i][j];
            }
            ntt_forward(codeword, codeword, plans[i]);
            for (uint64_t k = 0; k < size; k++)
            {
                arith_rns_polynomial(&out[k])->coeffs[i][j] = codeword[k];
            }
        }
    }

    free(codeword);
}

int rs_decode(ArithElement *out, ArithElement *in, uint64_t size, uint64_t degree, NTT_Plan *plans)
{
    RNS_Base base = arith_rns_polynomial(&in[0])->base;
    const uint64_t N = base->N;
    const uint64_t rns_mask = arith_rns_polynomial(&in[0])->rns_mask;
    uint64_t *codeword = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * size);
    int is_codeword = 1;

    if (out != NULL)
    {
        for (uint64_t k = 0; k < degree; k++)
        {
            arith_rns_polynomial(&out[k])->rns_mask = rns_mask;
        }
    }

    for (uint64_t i = 0; i < base->l && is_codeword; i++)
    {
        if (!(rns_mask & (1ULL << i)))
            continue;
        for (uint64_t j = 0; j < N && is_codeword; j++)
        {
            for (uint64_t k = 0; k < size; k++)
            {
                codeword[k] = arith_rns_polynomial(&in[k])->coeffs[i][j];
            }
            ntt_reverse(codeword, codeword, plans[i]);
            if (out != NULL)
            {
                for (uint64_t k = 0; k < degree; k++)
                {
                    arith_rns_polynomial(&out[k])->coeffs[i][j] = codeword[k];
                }
            }
            // The degree check: a codeword of this code inverts to a message
            // that was zero-padded above `degree`.
            for (uint64_t k = degree; k < size; k++)
            {
                if (codeword[k] != 0)
                {
                    is_codeword = 0;
                    break;
                }
            }
        }
    }

    free(codeword);
    return is_codeword;
}
