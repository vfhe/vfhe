#include <arith.h>

#if !defined(__AVX512IFMA__) || defined(PORTABLE_BUILD) || defined(PORTABLE)

static uint64_t reverse_bits(uint64_t x, int bits)
{
    uint64_t res = 0;
    for (int i = 0; i < bits; i++)
    {
        res = (res << 1) | (x & 1);
        x >>= 1;
    }
    return res;
}

void ntt_precompute_fwd(uint64_t n, uint64_t q, uint64_t root_of_unity, uint64_t ***out_ws,
                        uint64_t ***out_w_precon)
{
    int logn = 0;
    while ((1ULL << logn) < n)
        logn++;

    uint64_t *rou = (uint64_t *)malloc(n * sizeof(uint64_t));
    rou[0] = 1;
    uint64_t idx = 0, prev_idx = 0;
    for (size_t i = 1; i < n; i++)
    {
        idx = reverse_bits(i, logn);
        rou[idx] = (uint64_t)(((unsigned __int128)rou[prev_idx] * root_of_unity) % q);
        prev_idx = idx;
    }

    uint64_t **ws = (uint64_t **)malloc(1 * sizeof(uint64_t *));
    ws[0] = rou;
    *out_ws = ws;
    *out_w_precon = NULL; // Not used in portable
}

void ntt_precompute_inv(uint64_t n, uint64_t q, uint64_t inv_root_of_unity, uint64_t ***out_ws,
                        uint64_t ***out_w_precon)
{
    int logn = 0;
    while ((1ULL << logn) < n)
        logn++;

    uint64_t *rou = (uint64_t *)malloc(n * sizeof(uint64_t));
    rou[0] = 1;
    uint64_t idx = 0, prev_idx = 0;
    for (size_t i = 1; i < n; i++)
    {
        idx = reverse_bits(i, logn);
        rou[idx] = (uint64_t)(((unsigned __int128)rou[prev_idx] * inv_root_of_unity) % q);
        prev_idx = idx;
    }

    uint64_t **ws = (uint64_t **)malloc(1 * sizeof(uint64_t *));
    ws[0] = rou;
    *out_ws = ws;
    *out_w_precon = NULL;
}

void ntt_free_precompute(uint64_t **ws, uint64_t **w_precon, uint64_t n)
{
    (void)n;
    if (ws)
    {
        free(ws[0]);
        free(ws);
    }
    if (w_precon)
        free(w_precon);
}

void ntt_CT_NR_portable(uint64_t *a, uint64_t n, uint64_t q, uint64_t *ws, NTT_proc proc)
{
    size_t t = n;
    for (size_t m = 1; m < n; m <<= 1)
    {
        t >>= 1;
        for (size_t i = 0; i < m; i++)
        {
            size_t j1 = 2 * i * t;
            size_t j2 = j1 + t;
            uint64_t w = ws[m + i];
            for (size_t j = j1; j < j2; j++)
            {
                uint64_t u = a[j];
                uint64_t v = modq((unsigned __int128)a[j + t] * w, proc);
                a[j] = (u + v);
                if (a[j] >= q)
                    a[j] -= q;
                a[j + t] = (u + q - v);
                if (a[j + t] >= q)
                    a[j + t] -= q;
            }
        }
    }
}

void ntt_GS_RN_portable(uint64_t *a, uint64_t n, uint64_t q, uint64_t *ws, NTT_proc proc)
{
    size_t t = 1;
    for (size_t m = n; m > 1; m >>= 1)
    {
        size_t h = m >> 1;
        for (size_t i = 0; i < h; i++)
        {
            size_t j1 = 2 * i * t;
            size_t j2 = j1 + t;
            uint64_t w = ws[h + i];
            for (size_t j = j1; j < j2; j++)
            {
                uint64_t u = a[j];
                uint64_t v = a[j + t];
                a[j] = (u + v);
                if (a[j] >= q)
                    a[j] -= q;
                uint64_t diff = (u + q - v);
                if (diff >= q)
                    diff -= q;
                a[j + t] = modq((unsigned __int128)diff * w, proc);
            }
        }
        t <<= 1;
    }

    // Scale by n^-1
    uint64_t inv_n = inverse_mod(n, q);
    for (size_t i = 0; i < n; i++)
    {
        a[i] = modq((unsigned __int128)a[i] * inv_n, proc);
    }
}

NTT_proc ntt_new_proc(uint64_t n, uint64_t q)
{
    uint64_t root_of_unity = 2;
    while (power_mod(root_of_unity, n, q) != q - 1)
    {
        // Basic pseudo-random fallback since _rdrand64_step is not portable
        root_of_unity = (root_of_unity * 31337 + 12345) % q;
        root_of_unity = power_mod(root_of_unity, (q - 1) / (2 * n), q);
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
    ntt_CT_NR_portable(out, proc->n, proc->q, (uint64_t *)proc->ws_fwd[0], proc);
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
    ntt_GS_RN_portable(out, proc->n, proc->q, (uint64_t *)proc->ws_inv[0], proc);
}

void ntt_free_proc(NTT_proc proc)
{
    ntt_free_precompute((uint64_t **)proc->ws_fwd, (uint64_t **)proc->w_precon_fwd, proc->n);
    ntt_free_precompute((uint64_t **)proc->ws_inv, (uint64_t **)proc->w_precon_inv, proc->n);
    free(proc);
}

#endif
