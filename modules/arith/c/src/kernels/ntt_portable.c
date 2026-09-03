// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include <arith.h>
#include <inttypes.h>

#include "arith_internal.h"

#if !VFHE_HAVE_AVX512IFMA

// This engine has no vectorized transforms, so it is the shared size-generic
// scalar NTT (ntt_scalar.c) at every length.

void ntt_precompute_fwd(uint64_t n, Modulus mod, uint64_t root_of_unity, uint64_t ***out_ws,
                        uint64_t ***out_w_precon)
{
    ntt_scalar_precompute(n, mod, root_of_unity, out_ws);
    *out_w_precon = NULL; // Not used in portable
}

void ntt_precompute_inv(uint64_t n, Modulus mod, uint64_t inv_root_of_unity, uint64_t ***out_ws,
                        uint64_t ***out_w_precon)
{
    ntt_scalar_precompute(n, mod, inv_root_of_unity, out_ws);
    *out_w_precon = NULL;
}

void ntt_free_precompute(uint64_t **ws, uint64_t **w_precon, uint64_t n)
{
    (void)n;
    ntt_scalar_free_precompute(ws);
    if (w_precon)
        free(w_precon);
}

NTT_Plan ntt_new_plan(uint64_t n, Modulus mod)
{
    const uint64_t q = mod->q;

    // Deterministic search for a primitive 2n-th root of unity: raise successive
    // candidates g = 2, 3, 4, ... to the (q-1)/2n power and keep the first whose
    // n-th power is -1 (i.e. whose order is exactly 2n). At least half of the
    // residues qualify, so this terminates quickly.
    // Without 2n | q - 1 no such root exists and the search below would visit
    // every residue of a 61-bit prime before giving up.
    if ((q - 1) % (2 * n) != 0)
    {
        fprintf(stderr,
                "ntt_new_plan: no primitive %" PRIu64 "-th root of unity modulo %" PRIu64
                " (2n does not divide q - 1)\n",
                2 * n, q);
        return NULL;
    }
    uint64_t root_of_unity = 0;
    for (uint64_t g = 2; g < q; g++)
    {
        uint64_t candidate = power_mod(g, (q - 1) / (2 * n), q);
        if (power_mod(candidate, n, q) == q - 1)
        {
            root_of_unity = candidate;
            break;
        }
    }
    uint64_t inv_root_of_unity = inverse_mod(root_of_unity, q);

    NTT_Plan res = (NTT_Plan)malloc(sizeof(struct _NTT_Plan));
    res->mod = mod; // borrowed; ntt_free_plan leaves it alone
    res->n = n;
    res->root_of_unity = root_of_unity;
    res->inv_root_of_unity = inv_root_of_unity;
    ntt_precompute_fwd(n, mod, root_of_unity, (uint64_t ***)&res->ws_fwd,
                       (uint64_t ***)&res->w_precon_fwd);
    ntt_precompute_inv(n, mod, inv_root_of_unity, (uint64_t ***)&res->ws_inv,
                       (uint64_t ***)&res->w_precon_inv);
    return res;
}

void ntt_forward(uint64_t *out, uint64_t *in, NTT_Plan plan)
{
    if (out != in)
    {
        for (size_t i = 0; i < plan->n; i++)
        {
            out[i] = in[i];
        }
    }
    ntt_CT_NR_gen(out, (uint64_t *)plan->ws_fwd[0], plan);
}

void ntt_reverse(uint64_t *out, uint64_t *in, NTT_Plan plan)
{
    if (out != in)
    {
        for (size_t i = 0; i < plan->n; i++)
        {
            out[i] = in[i];
        }
    }
    ntt_GS_RN_gen(out, (uint64_t *)plan->ws_inv[0], plan);
}

void ntt_free_plan(NTT_Plan plan)
{
    ntt_free_precompute((uint64_t **)plan->ws_fwd, (uint64_t **)plan->w_precon_fwd, plan->n);
    ntt_free_precompute((uint64_t **)plan->ws_inv, (uint64_t **)plan->w_precon_inv, plan->n);
    free(plan);
}

#endif
