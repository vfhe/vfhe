// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include <arith.h>
#include <inttypes.h>
#include <util.h> /* safe_malloc */

#include "arith_internal.h"

/* The negacyclic transform on coefficients held as `uint32_t`: 16 to a vector
 * instead of 8.
 *
 * Applies when the prime is narrow (q <= 2^30) and the transform length is at
 * least 32. Values live lazily in [0, 4q) between stages and 4q <= 2^32
 * exactly at that bound, so the lazy range precisely fills a 32-bit lane --
 * the word width costs no modulus range that the radix-2^32 Shoup form did not
 * already require.
 *
 * Against the 8 x u64 IFMA transform it halves the bytes and the number of
 * load/store instructions, and puts the last *four* stages in registers rather
 * than three. It pays in ALU: the high half of a 32x32 product needs two
 * `vpmuludq` -- which read only the even 32-bit half of each 64-bit lane --
 * plus a shift and a blend to rebuild it packed. The trade is a win from
 * n = 256 up and a loss at n = 64, which is why the length floor is a floor
 * and not merely a convenience.
 *
 * A plan over a narrow prime carries these tables *in addition to* its 64-bit
 * ones, because the transform's word width is a property of the buffer and not
 * of the modulus: polycom gathers a codeword into a 64-bit array and
 * transforms that over the same plan.
 */

/* A narrow row whose plan has no 32-bit-word tables -- a transform shorter
 * than NTT_W32_MIN_LEN, or an engine without the kernels -- still has to
 * transform, because storage width is derived from the prime and must not
 * depend on the length or on which engine is loaded. Those rows are widened
 * around the 64-bit transform and narrowed back. Keeping that here rather than
 * at the call site is the same contract every other w32 entry point has: the
 * caller asks for the row's width and the kernel decides what it can do. */
static void ntt_w32_via_wide(uint32_t *out, uint32_t *in, NTT_Plan plan, bool inverse)
{
    const uint64_t n = plan->n;
    uint64_t *wide = (uint64_t *)safe_aligned_malloc(n * sizeof(uint64_t));
    for (uint64_t i = 0; i < n; i++)
        wide[i] = in[i];
    if (inverse)
        ntt_reverse(wide, wide, plan);
    else
        ntt_forward(wide, wide, plan);
    for (uint64_t i = 0; i < n; i++)
    {
        assert(wide[i] <= UINT32_MAX);
        out[i] = (uint32_t)wide[i];
    }
    free(wide);
}

#if VFHE_HAVE_AVX512IFMA

// A 32-coefficient block is two vectors; the tail works inside that pair.
#define W32_BLOCK 32
// Below this there is no block to hold the four tail stages.
#define NTT_W32_MIN_LEN 32
// The tail is four stages, so the last four levels hold per-lane twiddles.
#define W32_TAIL_LEVELS 4

/* The in-register tail's lane permutations. `ia`/`ib` bring the partners of
   one stage into matching lanes of two vectors; `oa`/`ob` put them back. */
typedef struct
{
    __m512i ia, ib, oa, ob;
} W32Perm;

// Written and read as whole zmm registers, so the alignment is load-bearing:
// an aligned 64-byte access to an under-aligned address faults.
static _Alignas(64) W32Perm w32_perm[W32_TAIL_LEVELS];
static pthread_once_t w32_perm_once = PTHREAD_ONCE_INIT;

/* Derived from the definition of the stage rather than written as constants.
   A 32-coefficient block sits in two vectors; at stage distance t, butterfly g
   pairs coefficients lo = (g/t)*2t + (g%t) and lo+t, and we want butterfly g in
   lane g of both halves. Hand-written 32-bit shuffle indices are exactly the
   sort of thing that is quietly wrong at one size, so this asserts the indices
   are an exact cover of the block before using them. */
static void w32_build_perms(void)
{
    for (int s = 0; s < W32_TAIL_LEVELS; s++)
    {
        const int t = 8 >> s; // 8, 4, 2, 1
        uint32_t ia[16], ib[16], oa[16], ob[16];
        int seen[W32_BLOCK] = {0};
        for (int g = 0; g < 16; g++)
        {
            const int lo = (g / t) * 2 * t + (g % t);
            ia[g] = (uint32_t)lo;
            ib[g] = (uint32_t)(lo + t);
            seen[lo]++;
            seen[lo + t]++;
        }
        for (int k = 0; k < W32_BLOCK; k++)
            assert(seen[k] == 1);
        // invert: coefficient ia[g] comes back from lane g of the first result
        // vector, ib[g] from lane g of the second (index 16 + g)
        for (int g = 0; g < 16; g++)
        {
            const uint32_t k1 = ia[g], k2 = ib[g];
            if (k1 < 16)
                oa[k1] = (uint32_t)g;
            else
                ob[k1 - 16] = (uint32_t)g;
            if (k2 < 16)
                oa[k2] = (uint32_t)(16 + g);
            else
                ob[k2 - 16] = (uint32_t)(16 + g);
        }
        w32_perm[s].ia = _mm512_loadu_si512(ia);
        w32_perm[s].ib = _mm512_loadu_si512(ib);
        w32_perm[s].oa = _mm512_loadu_si512(oa);
        w32_perm[s].ob = _mm512_loadu_si512(ob);
    }
}

// floor(w * 2^32 / q), which fits a u32 because w < q.
static uint32_t shoup32(uint32_t w, uint32_t q) { return (uint32_t)(((uint64_t)w << 32) / q); }

// Q = the high 32 bits of W' * Y, for all 16 lanes, packed.
static inline __m512i w32_shoup_q(__m512i WP, __m512i Y)
{
    const __m512i pe = _mm512_mul_epu32(WP, Y); // even 32-bit positions
    const __m512i po = _mm512_mul_epu32(_mm512_srli_epi64(WP, 32), _mm512_srli_epi64(Y, 32)); // odd
    return _mm512_mask_blend_epi32(0xAAAA, _mm512_srli_epi64(pe, 32), po);
}

static inline void w32_fwd_bfly(__m512i *X, __m512i *Y, __m512i W, __m512i WP, __m512i q,
                                __m512i q2)
{
    *X = _mm512_min_epu32(*X, _mm512_sub_epi32(*X, q2));
    const __m512i Q = w32_shoup_q(WP, *Y);
    const __m512i T = _mm512_sub_epi32(_mm512_mullo_epi32(W, *Y), _mm512_mullo_epi32(Q, q));
    *Y = _mm512_add_epi32(*X, _mm512_sub_epi32(q2, T));
    *X = _mm512_add_epi32(*X, T);
}

static inline void w32_inv_bfly(__m512i *X, __m512i *Y, __m512i W, __m512i WP, __m512i q,
                                __m512i q2)
{
    const __m512i Ym2q = _mm512_sub_epi32(*Y, q2);
    const __m512i T = _mm512_sub_epi32(*X, Ym2q);
    __m512i x = _mm512_add_epi32(*X, Ym2q);
    /* X + Y - 2q is negative for half the inputs, and the sign-bit test that
       corrects it is valid only while 2q <= 2^31 -- the q <= 2^30 bound again. */
    x = _mm512_mask_add_epi32(x, _mm512_movepi32_mask(x), x, q2);
    *X = x;
    const __m512i Q = w32_shoup_q(WP, T);
    *Y = _mm512_sub_epi32(_mm512_mullo_epi32(W, T), _mm512_mullo_epi32(Q, q));
}

/* The last GS butterfly of the transform, with 1/n folded into its twiddles:
   X' = (X + Y) * inv_n and Y' = (X - Y + 2q) * (inv_n * w), both by Shoup and
   both finishing in [0, q). Inputs are in [0, 2q), so X + Y and X - Y + 2q are
   under 4q <= 2^32 and stay inside a lane. */
static inline void w32_inv_bfly_final(__m512i *X, __m512i *Y, __m512i inv_n, __m512i inv_n_p,
                                      __m512i inv_n_w, __m512i inv_n_w_p, __m512i q, __m512i q2)
{
    const __m512i T = _mm512_sub_epi32(*X, _mm512_sub_epi32(*Y, q2));
    __m512i S = _mm512_add_epi32(*X, *Y);
    S = _mm512_min_epu32(S, _mm512_sub_epi32(S, q2));
    const __m512i Q1 = w32_shoup_q(inv_n_p, S);
    const __m512i x = _mm512_sub_epi32(_mm512_mullo_epi32(inv_n, S), _mm512_mullo_epi32(Q1, q));
    const __m512i Q2 = w32_shoup_q(inv_n_w_p, T);
    const __m512i y = _mm512_sub_epi32(_mm512_mullo_epi32(inv_n_w, T), _mm512_mullo_epi32(Q2, q));
    *X = _mm512_min_epu32(x, _mm512_sub_epi32(x, q));
    *Y = _mm512_min_epu32(y, _mm512_sub_epi32(y, q));
}

static inline void w32_pair_in(const W32Perm *pm, __m512i A, __m512i B, __m512i *VA, __m512i *VB)
{
    *VA = _mm512_permutex2var_epi32(A, pm->ia, B);
    *VB = _mm512_permutex2var_epi32(A, pm->ib, B);
}

static inline void w32_pair_out(const W32Perm *pm, __m512i VA, __m512i VB, __m512i *A, __m512i *B)
{
    *A = _mm512_permutex2var_epi32(VA, pm->oa, VB);
    *B = _mm512_permutex2var_epi32(VA, pm->ob, VB);
}

/* --- the tables ------------------------------------------------------- */

/* Level l has stage distance t = n / 2^(l+1) elements. The tail levels hold
   one vector of 16 per-lane twiddles per 32-element block; every other level
   holds one scalar per butterfly group, broadcast on load. Lane j of block b
   at distance t belongs to group 16b/t + j/t, which is what makes the tail's
   table a plain gather. `src` is the bit-reversed power table for the forward
   direction and the stage-major flattening for the inverse -- the same two
   orderings the 64-bit side builds. */
static void w32_precompute(uint64_t n, uint32_t q, const uint64_t *src, int inverse, void ***out_ws,
                           void ***out_wp)
{
    int logn = 0;
    while ((1ULL << logn) < n)
        logn++;
    void **ws = (void **)safe_malloc((size_t)logn * sizeof(void *));
    void **wp = (void **)safe_malloc((size_t)logn * sizeof(void *));
    const size_t nv = n / W32_BLOCK;

    for (int l = 0; l < logn; l++)
    {
        const int is_tail = inverse ? (l < W32_TAIL_LEVELS) : (l >= logn - W32_TAIL_LEVELS);
        const size_t t = inverse ? ((size_t)1 << l) : (n >> (l + 1));
        const size_t ngroups = n / (2 * t);
        size_t base = 1;
        if (inverse)
        {
            for (int k = 0; k < l; k++)
                base += n >> (k + 1);
        }
        else
        {
            base = (size_t)1 << l;
        }

        if (is_tail)
        {
            __m512i *tsl = (__m512i *)_mm_malloc(nv * sizeof(__m512i), 64);
            __m512i *tpl = (__m512i *)_mm_malloc(nv * sizeof(__m512i), 64);
            for (size_t b = 0; b < nv; b++)
            {
                uint32_t lw[16], lp[16];
                for (int j = 0; j < 16; j++)
                {
                    const size_t g = (16 * b) / t + (size_t)j / t;
                    const uint32_t w = (uint32_t)src[base + g];
                    lw[j] = w;
                    lp[j] = shoup32(w, q);
                }
                tsl[b] = _mm512_loadu_si512(lw);
                tpl[b] = _mm512_loadu_si512(lp);
            }
            ws[l] = tsl;
            wp[l] = tpl;
        }
        else
        {
            uint32_t *bsl = (uint32_t *)_mm_malloc(ngroups * sizeof(uint32_t), 64);
            uint32_t *bpl = (uint32_t *)_mm_malloc(ngroups * sizeof(uint32_t), 64);
            for (size_t i = 0; i < ngroups; i++)
            {
                const uint32_t w = (uint32_t)src[base + i];
                bsl[i] = w;
                bpl[i] = shoup32(w, q);
            }
            ws[l] = bsl;
            wp[l] = bpl;
        }
    }
    *out_ws = ws;
    *out_wp = wp;
}

static uint64_t w32_reverse_bits(uint64_t x, int bits)
{
    uint64_t res = 0;
    for (int i = 0; i < bits; i++)
    {
        res = (res << 1) | (x & 1);
        x >>= 1;
    }
    return res;
}

static void w32_bitrev_powers(uint64_t *out, Modulus mod, uint64_t root, uint64_t n, int logn)
{
    out[0] = 1;
    uint64_t idx = 0, prev = 0;
    for (uint64_t i = 1; i < n; i++)
    {
        idx = w32_reverse_bits(i, logn);
        out[idx] = mul_modq(out[prev], root, mod);
        prev = idx;
    }
}

bool ntt_w32_applies(uint64_t n, uint64_t q)
{
    return rns_prime_is_narrow(q) && n >= NTT_W32_MIN_LEN;
}

void ntt_w32_precompute(NTT_Plan plan)
{
    const uint64_t n = plan->n;
    const uint32_t q = (uint32_t)plan->mod->q;
    int logn = 0;
    while ((1ULL << logn) < n)
        logn++;

    pthread_once(&w32_perm_once, w32_build_perms);

    uint64_t *rou = (uint64_t *)safe_malloc(n * sizeof(uint64_t));
    uint64_t *tmp = (uint64_t *)safe_malloc(n * sizeof(uint64_t));

    w32_bitrev_powers(rou, plan->mod, plan->root_of_unity, n, logn);
    w32_precompute(n, q, rou, 0, &plan->ws_fwd32, &plan->w_precon_fwd32);

    w32_bitrev_powers(rou, plan->mod, plan->inv_root_of_unity, n, logn);
    tmp[0] = 1;
    {
        size_t idx = 1;
        for (uint64_t m = n >> 1; m > 0; m >>= 1)
            for (uint64_t i = 0; i < m; i++)
                tmp[idx++] = rou[m + i];
    }
    w32_precompute(n, q, tmp, 1, &plan->ws_inv32, &plan->w_precon_inv32);

    /* 1/n and (1/n) * w for the folded final butterfly. The last GS stage is
       the single group at inverse level logn-1, so its twiddle is there
       whether or not the depth-first recursion fired. */
    const uint64_t inv_n = inverse_mod(n, plan->mod->q);
    plan->inv_n32 = (uint32_t)inv_n;
    plan->inv_n32p = shoup32(plan->inv_n32, q);
    const uint32_t topw = ((const uint32_t *)plan->ws_inv32[logn - 1])[0];
    plan->inv_nw32 = (uint32_t)(((uint64_t)plan->inv_n32 * topw) % q);
    plan->inv_nw32p = shoup32(plan->inv_nw32, q);

    free(rou);
    free(tmp);
}

void ntt_w32_free(NTT_Plan plan)
{
    if (plan->ws_fwd32 == NULL)
        return;
    int logn = 0;
    while ((1ULL << logn) < plan->n)
        logn++;
    for (int l = 0; l < logn; l++)
    {
        _mm_free(plan->ws_fwd32[l]);
        _mm_free(plan->w_precon_fwd32[l]);
        _mm_free(plan->ws_inv32[l]);
        _mm_free(plan->w_precon_inv32[l]);
    }
    free(plan->ws_fwd32);
    free(plan->w_precon_fwd32);
    free(plan->ws_inv32);
    free(plan->w_precon_inv32);
}

/* --- forward: CT_NR ---------------------------------------------------- */

static void ntt_w32_ct_nr(__m512i *x, void **ws, void **wp, uint64_t sub_n, size_t level,
                          size_t offset_i, __m512i q_v, __m512i q2)
{
    if (sub_n > NTT_LEAF_ELEMENTS_W32)
    {
        const size_t t = sub_n >> 5; // in vectors of 16
        const __m512i w = _mm512_set1_epi32((int)((const uint32_t *)ws[level])[offset_i]);
        const __m512i wpv = _mm512_set1_epi32((int)((const uint32_t *)wp[level])[offset_i]);
        for (size_t j = 0; j < t; j++)
            w32_fwd_bfly(&x[j], &x[j + t], w, wpv, q_v, q2);
        ntt_w32_ct_nr(x, ws, wp, sub_n / 2, level + 1, 2 * offset_i, q_v, q2);
        ntt_w32_ct_nr(x + t, ws, wp, sub_n / 2, level + 1, 2 * offset_i + 1, q_v, q2);
        return;
    }

    size_t l = level;
    for (; (1ULL << (l - level + 4)) < sub_n; l++)
    {
        const uint32_t *wsv = (const uint32_t *)ws[l];
        const uint32_t *wpv = (const uint32_t *)wp[l];
        const uint64_t t = sub_n >> (l - level + 5);
        const size_t m = 1ULL << (l - level);
        const size_t start_i = offset_i << (l - level);
        for (size_t i = 0; i < m; i++)
        {
            const size_t slice = 2 * i * t;
            const __m512i w = _mm512_set1_epi32((int)wsv[start_i + i]);
            const __m512i wpb = _mm512_set1_epi32((int)wpv[start_i + i]);
            for (size_t j = slice; j < slice + t; j++)
                w32_fwd_bfly(&x[j], &x[j + t], w, wpb, q_v, q2);
        }
    }

    // t = 8, 4, 2, 1, all inside the register pair
    const size_t w_offset = offset_i * (sub_n / W32_BLOCK);
    for (int s = 0; s < W32_TAIL_LEVELS; s++, l++)
    {
        const __m512i *wsv = (const __m512i *)ws[l];
        const __m512i *wpv = (const __m512i *)wp[l];
        const W32Perm *pm = &w32_perm[s];
        for (size_t i = 0; i < sub_n / W32_BLOCK; i++)
        {
            __m512i VA, VB;
            w32_pair_in(pm, x[2 * i], x[2 * i + 1], &VA, &VB);
            w32_fwd_bfly(&VA, &VB, wsv[w_offset + i], wpv[w_offset + i], q_v, q2);
            if (s == W32_TAIL_LEVELS - 1)
            {
                /* Values are lazy in [0, 4q) between stages, and t = 1 is the
                   last stage to touch this block -- the recursion runs its own
                   stage before descending -- so bring them home here rather
                   than in a traversal of the whole array. Lane-wise, so it
                   commutes with the permutation below. */
                VA = _mm512_min_epu32(VA, _mm512_sub_epi32(VA, q2));
                VA = _mm512_min_epu32(VA, _mm512_sub_epi32(VA, q_v));
                VB = _mm512_min_epu32(VB, _mm512_sub_epi32(VB, q2));
                VB = _mm512_min_epu32(VB, _mm512_sub_epi32(VB, q_v));
            }
            w32_pair_out(pm, VA, VB, &x[2 * i], &x[2 * i + 1]);
        }
    }
}

void ntt_forward_w32(uint32_t *out, uint32_t *in, NTT_Plan plan)
{
    if (plan->ws_fwd32 == NULL)
    {
        ntt_w32_via_wide(out, in, plan, false);
        return;
    }
    const uint32_t q = (uint32_t)plan->mod->q;
    const __m512i q_v = _mm512_set1_epi32((int)q);
    const __m512i q2 = _mm512_set1_epi32((int)(2 * q));
    if (out != in)
        memcpy(out, in, plan->n * sizeof(uint32_t));
    /* The transform leaves every element in [0, q): the t = 1 tail stage
       reduces as it finishes with a block, so there is no closing sweep. */
    ntt_w32_ct_nr((__m512i *)out, plan->ws_fwd32, plan->w_precon_fwd32, plan->n, 0, 0, q_v, q2);
}

/* --- inverse: GS_RN ---------------------------------------------------- */

#define W32_FINAL_ARGS                                                                             \
    _mm512_set1_epi32((int)plan->inv_n32), _mm512_set1_epi32((int)plan->inv_n32p),                 \
        _mm512_set1_epi32((int)plan->inv_nw32), _mm512_set1_epi32((int)plan->inv_nw32p), q_v, q2

static void ntt_w32_gs_rn(__m512i *x, NTT_Plan plan, uint64_t sub_n, size_t l_merge,
                          size_t offset_i, int is_top, __m512i q_v, __m512i q2)
{
    void **ws = plan->ws_inv32;
    void **wp = plan->w_precon_inv32;

    if (sub_n > NTT_LEAF_ELEMENTS_W32)
    {
        const size_t t = sub_n >> 5;
        ntt_w32_gs_rn(x, plan, sub_n / 2, l_merge - 1, 2 * offset_i, 0, q_v, q2);
        ntt_w32_gs_rn(x + t, plan, sub_n / 2, l_merge - 1, 2 * offset_i + 1, 0, q_v, q2);
        if (is_top)
        {
            for (size_t j = 0; j < t; j++)
                w32_inv_bfly_final(&x[j], &x[j + t], W32_FINAL_ARGS);
        }
        else
        {
            const __m512i w = _mm512_set1_epi32((int)((const uint32_t *)ws[l_merge])[offset_i]);
            const __m512i wpv = _mm512_set1_epi32((int)((const uint32_t *)wp[l_merge])[offset_i]);
            for (size_t j = 0; j < t; j++)
                w32_inv_bfly(&x[j], &x[j + t], w, wpv, q_v, q2);
        }
        return;
    }

    // the forward tail, mirrored: t = 1, 2, 4, 8 first
    size_t l = 0;
    const size_t w_offset = offset_i * (sub_n / W32_BLOCK);
    for (int s = W32_TAIL_LEVELS - 1; s >= 0; s--, l++)
    {
        const __m512i *wsv = (const __m512i *)ws[l];
        const __m512i *wpv = (const __m512i *)wp[l];
        const W32Perm *pm = &w32_perm[s];
        for (size_t i = 0; i < sub_n / W32_BLOCK; i++)
        {
            __m512i VA, VB;
            w32_pair_in(pm, x[2 * i], x[2 * i + 1], &VA, &VB);
            w32_inv_bfly(&VA, &VB, wsv[w_offset + i], wpv[w_offset + i], q_v, q2);
            w32_pair_out(pm, VA, VB, &x[2 * i], &x[2 * i + 1]);
        }
    }

    for (; (1ULL << (l + is_top)) < sub_n; l++)
    {
        const uint32_t *wsv = (const uint32_t *)ws[l];
        const uint32_t *wpv = (const uint32_t *)wp[l];
        const uint64_t t = 1ULL << (l - W32_TAIL_LEVELS);
        const uint64_t m_sub = sub_n >> (l + 1);
        const size_t start_i = offset_i * m_sub;
        for (size_t i = 0; i < m_sub; i++)
        {
            const size_t slice = 2 * i * t;
            const __m512i w = _mm512_set1_epi32((int)wsv[start_i + i]);
            const __m512i wpb = _mm512_set1_epi32((int)wpv[start_i + i]);
            for (size_t j = slice; j < slice + t; j++)
                w32_inv_bfly(&x[j], &x[j + t], w, wpb, q_v, q2);
        }
    }

    if (is_top)
    {
        const uint64_t t = 1ULL << (l - W32_TAIL_LEVELS);
        for (size_t j = 0; j < t; j++)
            w32_inv_bfly_final(&x[j], &x[j + t], W32_FINAL_ARGS);
    }
}

void ntt_reverse_w32(uint32_t *out, uint32_t *in, NTT_Plan plan)
{
    if (plan->ws_inv32 == NULL)
    {
        ntt_w32_via_wide(out, in, plan, true);
        return;
    }
    const uint32_t q = (uint32_t)plan->mod->q;
    const __m512i q_v = _mm512_set1_epi32((int)q);
    const __m512i q2 = _mm512_set1_epi32((int)(2 * q));
    int logn = 0;
    while ((1ULL << logn) < plan->n)
        logn++;
    if (out != in)
        memcpy(out, in, plan->n * sizeof(uint32_t));
    ntt_w32_gs_rn((__m512i *)out, plan, plan->n, (size_t)logn - 1, 0, 1, q_v, q2);
}

#else // !VFHE_HAVE_AVX512IFMA

// This engine has no vectorized transform of either width, so every narrow row
// goes through the widening path above.
bool ntt_w32_applies(uint64_t n, uint64_t q)
{
    (void)n;
    (void)q;
    return false;
}

void ntt_forward_w32(uint32_t *out, uint32_t *in, NTT_Plan plan)
{
    ntt_w32_via_wide(out, in, plan, false);
}

void ntt_reverse_w32(uint32_t *out, uint32_t *in, NTT_Plan plan)
{
    ntt_w32_via_wide(out, in, plan, true);
}

#endif // VFHE_HAVE_AVX512IFMA
