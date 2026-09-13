// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
//
// The negacyclic transform over F_p[x]/(x^d - w), with the root in the
// extension field rather than in F_p.
//
// What this is for, and why the per-plane transform is not it: running arith's
// F_p transform once per coefficient plane evaluates at points of the prime
// subfield, which is right when the evaluation domain may live there and
// useless when it may not. Here psi is an element of F_(p^d) of order 2n, so
// every evaluation point psi^(2 brv(i) + 1) is an odd power of it and stays
// outside F_p whenever psi does.
//
// Same conventions as `ntt_forward`, so a code built on either sees the same
// layout: Cooley-Tukey, natural in and bit-reversed out, position p holding
// P(psi^(2 brv(p) + 1)), and the +/- pairs therefore adjacent.
//
// **It transforms a batch, and the layout is the reason.** A stage's butterfly
// operates on runs of `t` elements, and `t` halves every stage -- so the last
// stages of a single transform are runs of 1, 2, 4 elements, far too short to
// amortise a kernel call, and they dominate. Laid out with the *block* index
// fastest (element (b, i) at i * blocks + b), a butterfly at any `t` is a
// contiguous run of `t * blocks` elements, so every stage of every block runs
// through one kernel call. `blocks == 1` is the ordinary single transform and
// is exactly the slow case this layout exists to avoid.
#include <arith.h>
#include <stdio.h>
#include "arith_internal.h"
#include "util.h"

// Reverse the low `bits` bits, the permutation Cooley-Tukey leaves in its
// output and therefore the order the twiddle tables are stored in.
static uint64_t field_ntt_brv(uint64_t i, uint64_t bits)
{
    uint64_t out = 0;
    for (uint64_t k = 0; k < bits; k++)
    {
        out = (out << 1) | (i & 1);
        i >>= 1;
    }
    return out;
}

int field_ext_root_of_unity(uint64_t *out, uint64_t n, uint64_t d, uint64_t w, Modulus mod)
{
    if (n == 0 || (n & (n - 1)) != 0)
    {
        fprintf(stderr, "field_ext_root_of_unity: n = %llu is not a power of two\n",
                (unsigned long long)n);
        return 0;
    }

    // The multiplicative group of F_(p^d) is cyclic of order p^d - 1, so a
    // primitive 2n-th root exists exactly when 2n divides it. Raising anything
    // to (p^d - 1) / 2n lands in the 2n-torsion; whether it *generates* it is
    // what the psi^n == -1 check decides, and for 2n a power of two half of all
    // candidates do -- so this terminates quickly and is not a search for a
    // generator, which would need p^d - 1 factored.
    const uint64_t p = mod->q;
    uint64_t order_words[2 * FIELD_MAX_STACK_D];
    uint64_t words = 0;

    // p^d - 1 as little-endian 64-bit words, built by repeated multiplication.
    uint64_t acc[2 * FIELD_MAX_STACK_D] = {0};
    acc[0] = 1;
    words = 1;
    for (uint64_t k = 0; k < d; k++)
    {
        unsigned __int128 carry = 0;
        for (uint64_t i = 0; i < words; i++)
        {
            const unsigned __int128 cur = (unsigned __int128)acc[i] * p + carry;
            acc[i] = (uint64_t)cur;
            carry = cur >> 64;
        }
        while (carry)
        {
            acc[words++] = (uint64_t)carry;
            carry >>= 64;
        }
    }
    // minus one, which cannot borrow past word 0 since p^d is odd
    acc[0] -= 1;

    // divide by 2n, a power of two: one right shift across the words
    const uint64_t shift = 1 + (uint64_t)__builtin_ctzll(n);
    if (shift >= 64 * words)
    {
        fprintf(stderr, "field_ext_root_of_unity: 2 * %llu exceeds the group order\n",
                (unsigned long long)n);
        return 0;
    }
    for (uint64_t i = 0; i < words; i++)
    {
        const uint64_t hi = (i + 1 < words) ? acc[i + 1] : 0;
        order_words[i] = (acc[i] >> shift) | (shift ? (hi << (64 - shift)) : 0);
    }
    while (words > 1 && order_words[words - 1] == 0)
        words--;

    // 2n must divide p^d - 1, which is the same as the shift losing no bits.
    const uint64_t lost = acc[0] & ((1ULL << shift) - 1);
    if (lost != 0)
    {
        fprintf(stderr,
                "field_ext_root_of_unity: 2 * %llu does not divide the order of "
                "F_(p^%llu)*, so no primitive root of that order exists\n",
                (unsigned long long)n, (unsigned long long)d);
        return 0;
    }

    uint64_t candidate[FIELD_MAX_STACK_D], power[FIELD_MAX_STACK_D];
    uint64_t exp_n[2] = {n, 0};
    // A deterministic walk, so a plan is reproducible: the same field and the
    // same n give the same psi on every engine and every run.
    for (uint64_t seed = 2; seed < 4096; seed++)
    {
        for (uint64_t i = 0; i < d; i++)
            candidate[i] = 0;
        // Vary the whole element, not just its constant coefficient: for d > 1
        // a scalar candidate stays in F_p, whose 2n-torsion is smaller.
        candidate[0] = seed % p;
        if (d > 1)
            candidate[1] = (seed / p + 1) % p;

        field_ext_pow(out, candidate, order_words, words, d, w, mod);
        field_ext_pow(power, out, exp_n, 2, d, w, mod);

        // psi^n == -1 is exactly "psi has order 2n" when 2n is a power of two.
        int is_minus_one = (power[0] == p - 1);
        for (uint64_t i = 1; i < d && is_minus_one; i++)
            is_minus_one = (power[i] == 0);
        if (is_minus_one)
            return 1;
    }
    fprintf(stderr, "field_ext_root_of_unity: no primitive 2 * %llu-th root found\n",
            (unsigned long long)n);
    return 0;
}

// --- moving between the two batch layouts ----------------------------------
//
// Both directions are a transpose of the same rows-by-columns picture, which is
// why one routine serves them: to interleave is to transpose a blocks-by-n
// array, and to undo it is to transpose the n-by-blocks one. Reading down a
// column while writing along a row touches a new cache line per element, so it
// goes in tiles.
#define FIELD_NTT_TILE 32

static void field_ntt_transpose(uint64_t *out, const uint64_t *in, uint64_t rows, uint64_t cols)
{
    for (uint64_t i0 = 0; i0 < rows; i0 += FIELD_NTT_TILE)
    {
        const uint64_t imax = i0 + FIELD_NTT_TILE < rows ? i0 + FIELD_NTT_TILE : rows;
        for (uint64_t j0 = 0; j0 < cols; j0 += FIELD_NTT_TILE)
        {
            const uint64_t jmax = j0 + FIELD_NTT_TILE < cols ? j0 + FIELD_NTT_TILE : cols;
            for (uint64_t i = i0; i < imax; i++)
                for (uint64_t j = j0; j < jmax; j++)
                    out[j * rows + i] = in[i * cols + j];
        }
    }
}

void field_ntt_to_interleaved(FieldVector out, const FieldVector in, uint64_t blocks)
{
    const uint64_t n = in->n / blocks;
    for (uint64_t k = 0; k < in->d; k++)
        field_ntt_transpose(out->coeffs[k], in->coeffs[k], blocks, n);
}

void field_ntt_to_blocks(FieldVector out, const FieldVector in, uint64_t blocks)
{
    const uint64_t n = in->n / blocks;
    for (uint64_t k = 0; k < in->d; k++)
        field_ntt_transpose(out->coeffs[k], in->coeffs[k], n, blocks);
}

uint64_t field_ext_root_subfield_degree(uint64_t n, Modulus mod)
{
    // Every primitive 2n-th root of unity generates the same field, and which
    // one is not a choice: the 2n-th roots live in F_(p^k) for the smallest k
    // with p^k = 1 mod 2n, because that is when 2n divides p^k - 1. So the
    // caller cannot pick a root that avoids a subfield the arithmetic puts it
    // in -- it has to pick the prime.
    if (n == 0)
        return 0;
    const unsigned __int128 m = 2 * (unsigned __int128)n;
    const unsigned __int128 base = mod->q % m;
    unsigned __int128 x = base;
    uint64_t k = 1;
    while (x % m != 1 % m)
    {
        x = (x * base) % m;
        if (++k > 64)
            return 0; // p is even or shares a factor with 2n: no such k
    }
    return k;
}

// psi^brv(i) for every i, as d planes of n words -- the order the stages read
// twiddles in, so a stage indexes the table directly.
static uint64_t **field_ntt_table(uint64_t n, uint64_t logn, const uint64_t *root, uint64_t d,
                                  uint64_t w, Modulus mod)
{
    uint64_t **planes = (uint64_t **)malloc(d * sizeof(uint64_t *));
    if (!planes)
        return NULL;
    for (uint64_t k = 0; k < d; k++)
        planes[k] = (uint64_t *)safe_aligned_malloc(n * sizeof(uint64_t));

    uint64_t cur[FIELD_MAX_STACK_D], next[FIELD_MAX_STACK_D];
    for (uint64_t k = 0; k < d; k++)
        cur[k] = 0;
    cur[0] = 1;
    for (uint64_t i = 0; i < n; i++)
    {
        const uint64_t at = field_ntt_brv(i, logn);
        for (uint64_t k = 0; k < d; k++)
            planes[k][at] = cur[k];
        field_ext_mul(next, cur, root, d, w, mod);
        for (uint64_t k = 0; k < d; k++)
            cur[k] = next[k];
    }
    return planes;
}

FieldNTTPlan field_ntt_new_plan(uint64_t n, const uint64_t *root_of_unity, uint64_t d, uint64_t w,
                                Modulus mod)
{
    if (n == 0 || (n & (n - 1)) != 0)
    {
        fprintf(stderr, "field_ntt_new_plan: n = %llu is not a power of two\n",
                (unsigned long long)n);
        return NULL;
    }
    if (d > FIELD_MAX_STACK_D)
    {
        fprintf(stderr, "field_ntt_new_plan: degree %llu is above the supported %d\n",
                (unsigned long long)d, FIELD_MAX_STACK_D);
        return NULL;
    }

    // The root must be primitive of order 2n, which psi^n == -1 decides.
    uint64_t check[FIELD_MAX_STACK_D];
    const uint64_t exp_n[2] = {n, 0};
    field_ext_pow(check, root_of_unity, exp_n, 2, d, w, mod);
    int ok = (check[0] == mod->q - 1);
    for (uint64_t i = 1; i < d && ok; i++)
        ok = (check[i] == 0);
    if (!ok)
    {
        fprintf(stderr,
                "field_ntt_new_plan: the root is not a primitive 2 * %llu-th one "
                "(psi^n != -1)\n",
                (unsigned long long)n);
        return NULL;
    }

    FieldNTTPlan plan = (FieldNTTPlan)malloc(sizeof(*plan));
    if (!plan)
        return NULL;
    plan->n = n;
    plan->logn = (uint64_t)__builtin_ctzll(n);
    plan->d = d;
    plan->w = w;
    plan->mod = mod;

    for (uint64_t k = 0; k < d; k++)
        plan->root[k] = root_of_unity[k];
    if (!field_ext_inv(plan->inv_root, plan->root, d, w, mod))
    {
        free(plan);
        return NULL;
    }
    // n^-1, which the inverse transform applies at the end.
    for (uint64_t k = 0; k < d; k++)
        plan->inv_n[k] = 0;
    plan->inv_n[0] = n % mod->q;
    if (!field_ext_inv(plan->inv_n, plan->inv_n, d, w, mod))
    {
        free(plan);
        return NULL;
    }

    plan->ws_fwd = field_ntt_table(n, plan->logn, plan->root, d, w, mod);
    plan->ws_inv = field_ntt_table(n, plan->logn, plan->inv_root, d, w, mod);
    if (!plan->ws_fwd || !plan->ws_inv)
    {
        field_ntt_free_plan(plan);
        return NULL;
    }
    return plan;
}

void field_ntt_free_plan(FieldNTTPlan plan)
{
    if (!plan)
        return;
    for (uint64_t side = 0; side < 2; side++)
    {
        uint64_t **planes = side ? plan->ws_inv : plan->ws_fwd;
        if (!planes)
            continue;
        for (uint64_t k = 0; k < plan->d; k++)
            free(planes[k]);
        free(planes);
    }
    free(plan);
}

// One twiddle of a table, as the d words the butterflies take.
static void field_ntt_twiddle(uint64_t *out, uint64_t *const *ws, uint64_t index, uint64_t d)
{
    for (uint64_t k = 0; k < d; k++)
        out[k] = ws[k][index];
}

// The butterfly over a run, in whichever of the three forms the run admits.
//
// The run length is what decides, and it is not free to be anything: the tuned
// eltwise kernels load a whole vector at a time from a 64-byte boundary and
// have no tail, so a run that is not a whole number of them can go through
// neither the fused kernel nor the whole-vector one. A batch makes every run
// `t * blocks`, so `blocks` a multiple of the vector width puts every stage on
// the fast path; the short runs below are what a single transform is made of,
// and are the reason the batched layout exists.
static void field_ntt_butterfly(uint64_t *const *lo, uint64_t *const *hi, const uint64_t *root,
                                uint64_t run, FieldNTTPlan plan, int inverse, uint64_t *scratch)
{
    const uint64_t d = plan->d, w = plan->w;
    const Modulus mod = plan->mod;

    if (run % MOD_MIN_VECTOR_LEN != 0)
    {
        // Element at a time. Both the length and the alignment fail here -- the
        // run starts at a multiple of itself -- so there is nothing to salvage
        // by splitting it, and a short run is short precisely when the per-call
        // cost would have dominated anyway.
        uint64_t u[FIELD_MAX_STACK_D], v[FIELD_MAX_STACK_D], t[FIELD_MAX_STACK_D];
        for (uint64_t i = 0; i < run; i++)
        {
            for (uint64_t k = 0; k < d; k++)
            {
                u[k] = lo[k][i];
                v[k] = hi[k][i];
            }
            if (inverse)
            {
                field_ext_sub(t, u, v, d, mod->q);
                field_ext_add(u, u, v, d, mod->q);
                field_ext_mul(v, t, root, d, w, mod);
            }
            else
            {
                field_ext_mul(t, v, root, d, w, mod);
                field_ext_sub(v, u, t, d, mod->q);
                field_ext_add(u, u, t, d, mod->q);
            }
            for (uint64_t k = 0; k < d; k++)
            {
                lo[k][i] = u[k];
                hi[k][i] = v[k];
            }
        }
        return;
    }

    if (field_fused_applies(mod, d, run))
    {
        if (inverse)
            field_fused_gs(lo, hi, root, run, d, w, mod);
        else
            field_fused_ct(lo, hi, root, run, d, w, mod);
        return;
    }

    // The same butterfly through the whole-vector operations: one scratch
    // vector for the product, which is what the fused kernel exists to avoid.
    uint64_t *planes[FIELD_MAX_STACK_D];
    struct _FieldVector lo_v, hi_v, tmp_v;
    for (uint64_t k = 0; k < d; k++)
        planes[k] = scratch + k * run;

    lo_v.coeffs = (uint64_t **)lo;
    hi_v.coeffs = (uint64_t **)hi;
    tmp_v.coeffs = planes;
    struct _FieldVector *views[3] = {&lo_v, &hi_v, &tmp_v};
    for (int v = 0; v < 3; v++)
    {
        views[v]->n = run;
        views[v]->allocated_n = run;
        views[v]->d = d;
        views[v]->w = w;
        views[v]->mod = mod;
    }

    if (inverse)
    {
        field_vec_sub(&tmp_v, &lo_v, &hi_v);
        field_vec_add(&lo_v, &lo_v, &hi_v);
        field_vec_scale(&hi_v, &tmp_v, root);
    }
    else
    {
        field_vec_scale(&tmp_v, &hi_v, root);
        field_vec_sub(&hi_v, &lo_v, &tmp_v);
        field_vec_add(&lo_v, &lo_v, &tmp_v);
    }
}

// Plane pointers offset to element `at` of the batched layout.
static void field_ntt_at(uint64_t **out, uint64_t *const *base, uint64_t at, uint64_t d)
{
    for (uint64_t k = 0; k < d; k++)
        out[k] = base[k] + at;
}

// out[i] = psi^(2 * r(i) + 1); the contract is stated at the declaration.
//
// Taken in the natural order the entries are psi, psi^3, psi^5, ... -- each the
// one before it times psi^2 -- so a single running product yields them all.
// The bit reversal is an involution, so the j-th of those belongs at position
// r(j) and writing it straight there is the whole permutation.
void field_ntt_points(FieldVector out, uint64_t psi, uint64_t bits)
{
    const uint64_t n = 1ULL << bits, d = out->d;
    const Modulus mod = out->mod;
    uint64_t power = modq(psi, mod);
    const uint64_t step = mul_modq(power, power, mod);

    for (uint64_t j = 0; j < d; j++)
        memset(out->coeffs[j], 0, out->allocated_n * sizeof(uint64_t));
    for (uint64_t j = 0; j < n; j++)
    {
        out->coeffs[0][field_ntt_brv(j, bits)] = power;
        power = mul_modq(power, step, mod);
    }
}

uint64_t ntt_plan_root(NTT_Plan plan) { return plan == NULL ? 0 : plan->root_of_unity; }

// --- the transform when psi lies in F_p ------------------------------------
//
// Then it is F_p-linear, and an F_p-linear map of an extension element is the
// same map applied to each coefficient: the whole transform splits into d
// independent F_p transforms, which arith's own kernels already do well. That
// is worth about 3x over the extension butterfly, and the reason is not the
// arithmetic -- it is that each block's transform fits in cache and runs there,
// where the batched form streams the whole batch once per stage.
//
// **This path takes the other layout**: block b is `n` consecutive elements at
// `b * n`, because that is what puts one transform in one cache-resident run.
// The batched layout exists to give the extension butterfly long runs, which is
// a vectorisation problem this path does not have.
static void field_ntt_base(uint64_t *const *planes, uint64_t blocks, uint64_t d, NTT_Plan plan,
                           int inverse)
{
    const uint64_t n = plan->n;
    uint64_t *tmp = (uint64_t *)safe_aligned_malloc(n * sizeof(uint64_t));
    for (uint64_t k = 0; k < d; k++)
        for (uint64_t b = 0; b < blocks; b++)
        {
            uint64_t *block = planes[k] + b * n;
            if (inverse)
                ntt_reverse(tmp, block, plan);
            else
                ntt_forward(tmp, block, plan);
            memcpy(block, tmp, n * sizeof(uint64_t));
        }
    free(tmp);
}

void field_ntt_forward_base(uint64_t *const *planes, uint64_t blocks, uint64_t d, NTT_Plan plan)
{
    field_ntt_base(planes, blocks, d, plan, 0);
}

void field_ntt_inverse_base(uint64_t *const *planes, uint64_t blocks, uint64_t d, NTT_Plan plan)
{
    field_ntt_base(planes, blocks, d, plan, 1);
}

void field_ntt_forward(uint64_t *const *planes, uint64_t blocks, FieldNTTPlan plan)
{
    const uint64_t n = plan->n, d = plan->d;
    uint64_t *lo[FIELD_MAX_STACK_D], *hi[FIELD_MAX_STACK_D], root[FIELD_MAX_STACK_D];
    uint64_t *scratch = (uint64_t *)safe_aligned_malloc(d * (n / 2) * blocks * sizeof(uint64_t));

    uint64_t t = n;
    for (uint64_t m = 1; m < n; m <<= 1)
    {
        t >>= 1;
        for (uint64_t i = 0; i < m; i++)
        {
            field_ntt_twiddle(root, plan->ws_fwd, m + i, d);
            // A group's two halves are `t` consecutive positions, which the
            // layout makes `t * blocks` consecutive elements.
            const uint64_t first = 2 * i * t * blocks, run = t * blocks;
            field_ntt_at(lo, planes, first, d);
            field_ntt_at(hi, planes, first + run, d);
            field_ntt_butterfly(lo, hi, root, run, plan, 0, scratch);
        }
    }
    free(scratch);
}

void field_ntt_inverse(uint64_t *const *planes, uint64_t blocks, FieldNTTPlan plan)
{
    const uint64_t n = plan->n, d = plan->d;
    uint64_t *lo[FIELD_MAX_STACK_D], *hi[FIELD_MAX_STACK_D], root[FIELD_MAX_STACK_D];
    uint64_t *scratch = (uint64_t *)safe_aligned_malloc(d * (n / 2) * blocks * sizeof(uint64_t));

    uint64_t t = 1;
    for (uint64_t m = n; m > 1; m >>= 1)
    {
        const uint64_t h = m >> 1;
        for (uint64_t i = 0; i < h; i++)
        {
            field_ntt_twiddle(root, plan->ws_inv, h + i, d);
            const uint64_t first = 2 * i * t * blocks, run = t * blocks;
            field_ntt_at(lo, planes, first, d);
            field_ntt_at(hi, planes, first + run, d);
            field_ntt_butterfly(lo, hi, root, run, plan, 1, scratch);
        }
        t <<= 1;
    }

    // n^-1 over the whole batch. Same length rule as a butterfly's run, so a
    // total that is not a whole number of vectors goes element at a time.
    const uint64_t total = n * blocks;
    if (total % MOD_MIN_VECTOR_LEN == 0)
    {
        struct _FieldVector all;
        all.coeffs = (uint64_t **)planes;
        all.n = total;
        all.allocated_n = total;
        all.d = d;
        all.w = plan->w;
        all.mod = plan->mod;
        field_vec_scale(&all, &all, plan->inv_n);
    }
    else
    {
        uint64_t e[FIELD_MAX_STACK_D];
        for (uint64_t i = 0; i < total; i++)
        {
            for (uint64_t k = 0; k < d; k++)
                e[k] = planes[k][i];
            field_ext_mul(e, e, plan->inv_n, d, plan->w, plan->mod);
            for (uint64_t k = 0; k < d; k++)
                planes[k][i] = e[k];
        }
    }
    free(scratch);
}
