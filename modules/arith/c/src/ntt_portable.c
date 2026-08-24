// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include <arith.h>
#include "arith_internal.h"

#if !defined(__AVX512IFMA__) || defined(PORTABLE_BUILD) || defined(PORTABLE)

// This engine has no vectorized transforms, so it is the shared size-generic
// scalar NTT (ntt_scalar.c) at every length.

void ntt_precompute_fwd(uint64_t n, uint64_t q, uint64_t root_of_unity, uint64_t ***out_ws,
                        uint64_t ***out_w_precon)
{
    ntt_scalar_precompute(n, q, root_of_unity, out_ws);
    *out_w_precon = NULL; // Not used in portable
}

void ntt_precompute_inv(uint64_t n, uint64_t q, uint64_t inv_root_of_unity, uint64_t ***out_ws,
                        uint64_t ***out_w_precon)
{
    ntt_scalar_precompute(n, q, inv_root_of_unity, out_ws);
    *out_w_precon = NULL;
}

void ntt_free_precompute(uint64_t **ws, uint64_t **w_precon, uint64_t n)
{
    (void)n;
    ntt_scalar_free_precompute(ws);
    if (w_precon)
        free(w_precon);
}

NTT_proc ntt_new_proc(uint64_t n, uint64_t q)
{
    // Deterministic search for a primitive 2n-th root of unity: raise successive
    // candidates g = 2, 3, 4, ... to the (q-1)/2n power and keep the first whose
    // n-th power is -1 (i.e. whose order is exactly 2n). At least half of the
    // residues qualify, so this terminates quickly, and it always terminates,
    // unlike a self-feeding LCG, which can settle into a cycle of non-primitive
    // candidates and spin forever.
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
    uint64_t k = 64;
    unsigned __int128 m_128 = ((unsigned __int128)1 << k) / q;
    while (m_128 < (1ULL << 63))
    {
        k++;
        m_128 = ((unsigned __int128)1 << k) / q;
    }
    uint64_t m = (uint64_t)m_128;
    NTT_proc res = (NTT_proc)malloc(sizeof(struct _NTT_proc));
    res->n = n;
    res->q = q;
    res->root_of_unity = root_of_unity;
    res->inv_root_of_unity = inv_root_of_unity;
    res->k = k;
    res->m = m;
    res->m52 = (k - 52 >= 64) ? 0 : (m >> (k - 52));
    res->mp_w1 = (uint64_t)(((unsigned __int128)1 << 52) % q);
    res->mp_w2 = (uint64_t)(((unsigned __int128)1 << 104) % q);
    ntt_precompute_fwd(n, q, root_of_unity, (uint64_t ***)&res->ws_fwd,
                       (uint64_t ***)&res->w_precon_fwd);
    ntt_precompute_inv(n, q, inv_root_of_unity, (uint64_t ***)&res->ws_inv,
                       (uint64_t ***)&res->w_precon_inv);
    return res;
}

void ntt_forward(uint64_t *out, uint64_t *in, NTT_proc proc)
{
    if (out != in)
    {
        for (size_t i = 0; i < proc->n; i++)
        {
            out[i] = in[i];
        }
    }
    ntt_CT_NR_gen(out, proc->n, proc->q, (uint64_t *)proc->ws_fwd[0], proc);
}

void ntt_reverse(uint64_t *out, uint64_t *in, NTT_proc proc)
{
    if (out != in)
    {
        for (size_t i = 0; i < proc->n; i++)
        {
            out[i] = in[i];
        }
    }
    ntt_GS_RN_gen(out, proc->n, proc->q, (uint64_t *)proc->ws_inv[0], proc);
}

void ntt_free_proc(NTT_proc proc)
{
    ntt_free_precompute((uint64_t **)proc->ws_fwd, (uint64_t **)proc->w_precon_fwd, proc->n);
    ntt_free_precompute((uint64_t **)proc->ws_inv, (uint64_t **)proc->w_precon_inv, proc->n);
    free(proc);
}

#endif
