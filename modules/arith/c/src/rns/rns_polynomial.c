// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
#include "arith.h"
#include "alloc.h"
#include "memory_pool.h"
#include <crypto.h>
#include <blake3.h>

// Row width, the row accessors and the per-row operation macros.
#include "arith_internal.h"
#if VFHE_HAVE_AVX512IFMA
#include <immintrin.h>
#endif

// How the block product writes its destination.
typedef enum
{
    RNS_BLOCK_SET = 0,
    RNS_BLOCK_ADDTO,
    RNS_BLOCK_SUBTO
} RnsBlockAcc;

/* The row bodies, once per storage width. Everything below dispatches on the
   narrow bit once per row and then calls the matching instantiation, so no
   coefficient loop tests a width and no row is converted to reuse a body. */
#define RNS_W uint32_t
#define RNS_SFX _narrow
#define RNS_ADD mod_eltwise_add_w32
#define RNS_SUB mod_eltwise_sub_w32
#define RNS_MUL mod_eltwise_mul_w32
#define RNS_MULADDTO mod_eltwise_mul_addto_w32
#define RNS_NEGATE mod_eltwise_negate_w32
#include "rns_row_ops.inc"

#define RNS_W uint64_t
#define RNS_SFX _wide
#define RNS_ADD mod_eltwise_add
#define RNS_SUB mod_eltwise_sub
#define RNS_MUL mod_eltwise_mul
#define RNS_MULADDTO mod_eltwise_mul_addto
#define RNS_NEGATE mod_eltwise_negate
#include "rns_row_ops.inc"

/* Reducing a row into a buffer of a given width. Only a cross-modulus
   reduction can change width, so these two are the only places in this file
   that do -- everything else keeps both sides at one width. */
static void rns_reduce_row_to32(uint32_t *out, RNS_Polynomial p, size_t i, uint64_t n, Modulus mod)
{
    if (rns_row_is_narrow(p->base, i))
        mod_eltwise_reduce_w32(out, p->rows32[i], n, mod);
    else
        mod_eltwise_reduce_narrow_from_wide(out, p->rows64[i], n, mod);
}

static void rns_reduce_row_to64(uint64_t *out, RNS_Polynomial p, size_t i, uint64_t n, Modulus mod)
{
    if (rns_row_is_narrow(p->base, i))
        mod_eltwise_reduce_wide_from_narrow(out, p->rows32[i], n, mod);
    else
        mod_eltwise_reduce(out, p->rows64[i], n, mod);
}

#define RNS_ROW_SCATTER_SPLIT(out, src, i, n, sd, ps)                                              \
    do                                                                                             \
    {                                                                                              \
        if (rns_row_is_narrow((out)->base, (i)))                                                   \
            rns_row_scatter_split_narrow((out)->rows32[(i)], (src), (n), (sd), (ps));              \
        else                                                                                       \
            rns_row_scatter_split_wide((out)->rows64[(i)], (src), (n), (sd), (ps));                \
    } while (0)

static bool rns_row_zero(RNS_Polynomial p, size_t i, uint64_t n)
{
    return rns_row_is_narrow(p->base, i) ? rns_row_is_zero_narrow(p->rows32[i], n)
                                         : rns_row_is_zero_wide(p->rows64[i], n);
}

// Computes (Z_q[i](Q/q[i]))**-1, for i in [0,l)
void compute_RNS_Qhat_array(uint64_t *out, uint64_t *p, uint64_t l)
{
    for (size_t i = 0; i < l; i++)
    {
        out[i] = 1;
        for (size_t j = 0; j < l; j++)
        {
            if (i != j)
            {
                const uint64_t inv = inverse_mod(p[j], p[i]);
                out[i] = (uint64_t)(((unsigned __int128)out[i] * inv) % p[i]);
            }
        }
    }
}

static Modulus *new_modulus_list(uint64_t *primes, uint64_t l)
{
    Modulus *mods = (Modulus *)safe_malloc(sizeof(Modulus) * l);
    for (size_t i = 0; i < l; i++)
    {
        mods[i] = mod_new(primes[i]);
    }
    return mods;
}

// The plans borrow `mods`, so the caller keeps owning it and must outlive them.
static NTT_Plan *new_ntt_plan_list(Modulus *mods, uint64_t N, uint64_t l)
{
    NTT_Plan *plans = (NTT_Plan *)safe_malloc(sizeof(NTT_Plan) * l);
    for (size_t i = 0; i < l; i++)
    {
        plans[i] = ntt_new_plan(N, mods[i]);
    }
    return plans;
}

/* A narrow copy of prime i's twiddle row, or NULL for a wide prime. The block
   product multiplies a row by these, so they are held at the row's width and
   the product never converts. */
static uint32_t *rns_narrow_twiddles(RNS_Base base, size_t i, uint64_t poly_size)
{
    if (!rns_row_is_narrow(base, i))
        return NULL;
    uint32_t *w32 = (uint32_t *)safe_aligned_malloc(poly_size * sizeof(uint32_t));
    mod_narrow_w32(w32, base->w[i], poly_size);
    return w32;
}

RNS_Base new_rns_base(uint64_t *primes, uint64_t split_degree, uint64_t N, uint64_t l)
{
    const uint64_t poly_size = N / split_degree;
    const uint64_t log_poly_size = (uint64_t)log2(poly_size);
    uint64_t *w_p = (uint64_t *)safe_malloc(2 * poly_size * sizeof(uint64_t));
    RNS_Base base = (RNS_Base)safe_malloc(sizeof(*base));
    base->N = N;
    base->l = l;
    base->narrow_mask = 0;
    // The moduli first: the plans borrow them, so they must outlive the plans.
    base->mods = new_modulus_list(primes, l);
    for (size_t i = 0; i < l; i++)
    {
        if (rns_prime_is_narrow(primes[i]))
            base->narrow_mask |= 1ULL << i;
    }
    base->plans = new_ntt_plan_list(base->mods, poly_size, l);
    base->split_degree = split_degree;
    base->w = (uint64_t **)safe_malloc(sizeof(uint64_t *) * l);
    base->w32 = (uint32_t **)safe_malloc(sizeof(uint32_t *) * l);
    for (size_t i = 0; i < l; i++)
    {
        base->w[i] = (uint64_t *)safe_aligned_malloc(poly_size * sizeof(uint64_t));
        uint64_t w1 = base->plans[i]->root_of_unity;
        w_p[0] = w1;
        for (size_t j = 1; j < 2 * poly_size; j++)
        {
            w_p[j] = mul_modq(w_p[j - 1], w1, base->mods[i]);
        }
        bit_rev(base->w[i], w_p, poly_size, log_poly_size + 1);
        base->w32[i] = rns_narrow_twiddles(base, i, poly_size);
    }
    free(w_p);
    return base;
}

void rns_base_extend_with_primes(RNS_Base base, uint64_t *new_primes, uint64_t count)
{
    if (count == 0)
        return;
    uint64_t new_l = base->l + count;
    const uint64_t poly_size = base->N / base->split_degree;
    const uint64_t log_poly_size = (uint64_t)log2(poly_size);
    uint64_t *w_p = (uint64_t *)safe_malloc(2 * poly_size * sizeof(uint64_t));

    base->mods = (Modulus *)safe_realloc(base->mods, sizeof(Modulus) * new_l);
    base->plans = (NTT_Plan *)safe_realloc(base->plans, sizeof(NTT_Plan) * new_l);
    base->w = (uint64_t **)safe_realloc(base->w, sizeof(uint64_t *) * new_l);
    base->w32 = (uint32_t **)safe_realloc(base->w32, sizeof(uint32_t *) * new_l);

    for (size_t i = base->l; i < new_l; i++)
    {
        uint64_t prime = new_primes[i - base->l];
        if (rns_prime_is_narrow(prime))
            base->narrow_mask |= 1ULL << i;
        base->mods[i] = mod_new(prime);
        base->plans[i] = ntt_new_plan(poly_size, base->mods[i]);
        base->w[i] = (uint64_t *)safe_aligned_malloc(poly_size * sizeof(uint64_t));
        uint64_t w1 = base->plans[i]->root_of_unity;
        w_p[0] = w1;
        for (size_t j = 1; j < 2 * poly_size; j++)
        {
            w_p[j] = mul_modq(w_p[j - 1], w1, base->mods[i]);
        }
        bit_rev(base->w[i], w_p, poly_size, log_poly_size + 1);
        base->w32[i] = rns_narrow_twiddles(base, i, poly_size);
    }

    base->l = new_l;
    free(w_p);
}

void rns_base_free(RNS_Base base)
{
    if (!base)
        return;
    for (size_t i = 0; i < base->l; i++)
    {
        ntt_free_plan(base->plans[i]);
        mod_free(base->mods[i]);
        free(base->w[i]);
        free(base->w32[i]);
    }
    free(base->plans);
    free(base->mods);
    free(base->w);
    free(base->w32);
    free(base);
}

uint64_t **rns_base_get_rou_matrix(RNS_Base base) { return base->w; }

RNS_Polynomial polynomial_new_RNS_polynomial(uint64_t N, uint64_t rns_mask, RNS_Base base)
{
    RNS_Polynomial res;
    res = (RNS_Polynomial)safe_malloc(sizeof(*res));
    /* One row per prime, at that prime's width: exactly one of the two arrays
       holds a buffer for each index, the other a NULL. */
    res->rows64 = (uint64_t **)safe_malloc(sizeof(uint64_t *) * base->l);
    res->rows32 = (uint32_t **)safe_malloc(sizeof(uint32_t *) * base->l);
    for (size_t i = 0; i < base->l; i++)
    {
        if (rns_row_is_narrow(base, i))
        {
            res->rows64[i] = NULL;
            res->rows32[i] = (uint32_t *)mempool_aligned_malloc(sizeof(uint32_t) * base->N);
        }
        else
        {
            res->rows64[i] = (uint64_t *)mempool_aligned_malloc(sizeof(uint64_t) * base->N);
            res->rows32[i] = NULL;
        }
    }
    res->base = base;
    res->rns_mask = rns_mask;
    res->allocated_l = base->l;
    return res;
}

RNS_Polynomial *polynomial_new_RNS_polynomial_array(uint64_t size, uint64_t N, uint64_t rns_mask,
                                                    RNS_Base base)
{
    RNS_Polynomial *res;
    res = (RNS_Polynomial *)safe_malloc(sizeof(RNS_Polynomial) * size);
    for (size_t i = 0; i < size; i++)
    {
        res[i] = polynomial_new_RNS_polynomial(N, rns_mask, base);
    }
    return res;
}

bool polynomial_eq(RNS_Polynomial a, RNS_Polynomial b)
{
    const uint64_t max_l = a->base->l > b->base->l ? a->base->l : b->base->l;
    for (size_t i = 0; i < max_l; i++)
    {
        bool active_a = (i < a->base->l) && (a->rns_mask & (1ULL << i));
        bool active_b = (i < b->base->l) && (b->rns_mask & (1ULL << i));
        if (active_a && active_b)
        {
            if (!RNS_ROW_EQ(a, b, i, a->base->N))
                return false;
        }
        else if (active_a)
        {
            if (!rns_row_zero(a, i, a->base->N))
                return false;
        }
        else if (active_b)
        {
            if (!rns_row_zero(b, i, b->base->N))
                return false;
        }
    }
    return true;
}

void polynomial_copy_RNS_polynomial(RNS_Polynomial out, RNS_Polynomial in)
{
    out->rns_mask = in->rns_mask;
    for (size_t i = 0; i < out->base->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            if (rns_row_is_narrow(out->base, i))
                memcpy(out->rows32[i], in->rows32[i], sizeof(uint32_t) * out->base->N);
            else
                memcpy(out->rows64[i], in->rows64[i], sizeof(uint64_t) * out->base->N);
        }
    }
}

void polynomial_copy_RNSc_polynomial(RNSc_Polynomial out, RNSc_Polynomial in)
{
    polynomial_copy_RNS_polynomial((RNS_Polynomial)out, (RNS_Polynomial)in);
}

void polynomial_RNS_zero(RNS_Polynomial p)
{
    for (size_t i = 0; i < p->base->l; i++)
    {
        if (p->rns_mask & (1ULL << i))
        {
            RNS_ROW_ZERO(p, i, p->base->N);
        }
    }
}

void free_RNS_polynomial(void *p)
{
    RNS_Polynomial pp = (RNS_Polynomial)p;
    // One of the two is NULL for each row, and releasing NULL is a no-op, so
    // this needs no width test and cannot go wrong if the base grew meanwhile.
    for (size_t i = 0; i < pp->allocated_l; i++)
    {
        mempool_free(pp->rows64[i], sizeof(uint64_t) * pp->base->N);
        mempool_free(pp->rows32[i], sizeof(uint32_t) * pp->base->N);
    }
    free(pp->rows64);
    free(pp->rows32);
    free(pp);
}

void free_RNS_polynomial_array(uint64_t size, RNS_Polynomial *p)
{
    for (size_t i = 0; i < size; i++)
    {
        free_RNS_polynomial(p[i]);
    }
    free(p);
}

RNS_Polynomial *polynomial_new_array_of_RNS_polynomials(uint64_t N, uint64_t rns_mask,
                                                        uint64_t size, RNS_Base base)
{
    RNS_Polynomial *res = (RNS_Polynomial *)safe_malloc(sizeof(RNS_Polynomial) * size);
    for (size_t i = 0; i < size; i++)
        res[i] = polynomial_new_RNS_polynomial(N, rns_mask, base);
    return res;
}

void polynomial_to_RNS(RNS_Polynomial out, IntPolynomial in)
{
    const uint64_t poly_size = out->base->N / out->base->split_degree;
    uint64_t *temp = (uint64_t *)safe_aligned_malloc(out->base->N * sizeof(uint64_t));
    for (size_t i = 0; i < out->base->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            Modulus mod = out->base->mods[i];
            mod_eltwise_reduce_signed(temp, (int64_t *)in->coeffs, out->base->N, mod);
            RNS_ROW_SCATTER_SPLIT(out, temp, i, out->base->N, out->base->split_degree, poly_size);
        }
    }
    free(temp);
    polynomial_RNSc_to_RNS(out, (RNSc_Polynomial)out);
}

void int_array_to_RNS(RNS_Polynomial out, uint64_t *in)
{
    int64_t *limbs[1] = {(int64_t *)in};
    int_limbs_to_RNS(out, limbs, 1, 0);
}

void int_limbs_to_RNS(RNS_Polynomial out, int64_t *const *limbs, uint64_t count, uint64_t limb_bits)
{
    const uint64_t N = out->base->N;
    const uint64_t poly_size = N / out->base->split_degree;
    uint64_t *acc = (uint64_t *)safe_aligned_malloc(N * sizeof(uint64_t));
    uint64_t *next = count > 1 ? (uint64_t *)safe_aligned_malloc(N * sizeof(uint64_t)) : NULL;
    for (size_t i = 0; i < out->base->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            Modulus mod = out->base->mods[i];
            // Horner from the top limb: acc = acc 2^limb_bits + limbs[j].
            mod_eltwise_reduce_signed(acc, limbs[count - 1], N, mod);
            const uint64_t radix = count > 1 ? power_mod(2, limb_bits, mod->q) : 0;
            for (uint64_t j = count - 1; j-- > 0;)
            {
                mod_eltwise_reduce_signed(next, limbs[j], N, mod);
                mod_eltwise_fma(next, acc, radix, N, mod);
                uint64_t *t = acc;
                acc = next;
                next = t;
            }
            RNS_ROW_SCATTER_SPLIT(out, acc, i, N, out->base->split_degree, poly_size);
        }
    }
    free(next);
    free(acc);
    polynomial_RNSc_to_RNS(out, (RNSc_Polynomial)out);
}

void array_to_RNS(RNS_Polynomial out, uint64_t **in)
{
    const uint64_t poly_size = out->base->N / out->base->split_degree;
    for (size_t i = 0; i < out->base->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            RNS_ROW_SCATTER_SPLIT(out, in[i], i, out->base->N, out->base->split_degree, poly_size);
        }
    }
    polynomial_RNSc_to_RNS(out, (RNSc_Polynomial)out);
}

void polynomial_gen_random_RNSc_polynomial(RNSc_Polynomial out)
{
    // The sampler writes 64-bit words, so a narrow row is drawn into a wide
    // scratch row and narrowed after.
    uint64_t *scratch = (uint64_t *)safe_aligned_malloc(out->base->N * sizeof(uint64_t));
    for (size_t i = 0; i < out->base->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            const uint64_t p = out->base->mods[i]->q;
            const bool narrow = rns_row_is_narrow(out->base, i);
            uint64_t *dst = narrow ? scratch : out->rows64[i];
            generate_uniform_below(dst, out->base->N, p);
            if (narrow)
                mod_narrow_w32(out->rows32[i], scratch, out->base->N);
        }
    }
    free(scratch);
}

void polynomial_gen_gaussian_RNSc_polynomial(RNSc_Polynomial out, double sigma)
{
    int64_t *noise_arr = (int64_t *)safe_aligned_malloc(out->base->N * sizeof(int64_t));
    for (size_t j = 0; j < out->base->N; j++)
    {
        noise_arr[j] = (int64_t)round(generate_normal_random(sigma));
    }
    for (size_t i = 0; i < out->base->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            Modulus mod = out->base->mods[i];
            RNS_ROW_REDUCE_SIGNED(out, noise_arr, i, out->base->N, mod);
        }
    }
    free(noise_arr);
}

/* The per-prime product. `split_degree == 1` is not a fast path bolted on: it
   is the whole incomplete-NTT product when the ring is not split, and there a
   row is one element-wise multiply at its own width. Above 1 the blocks are
   convolved and the wrap-around terms scaled by the base's twiddle row, which
   is held at both widths for exactly this reason -- so a narrow row runs the
   whole product in 32-bit words and nothing is converted. */
static void rns_product(RNS_Polynomial out, RNS_Polynomial in1, RNS_Polynomial in2, RnsBlockAcc acc)
{
    RNS_Base b = out->base;
    const uint64_t sd = b->split_degree, N = b->N, ps = N / sd;

    if (sd == 1)
    {
        for (size_t i = 0; i < b->l; i++)
        {
            if (!(out->rns_mask & (1ULL << i)))
                continue;
            Modulus mod = b->mods[i];
            if (rns_row_is_narrow(b, i))
            {
                uint32_t *o = out->rows32[i], *x = in1->rows32[i], *y = in2->rows32[i];
                if (acc == RNS_BLOCK_ADDTO)
                    mod_eltwise_mul_addto_w32(o, x, y, N, mod);
                else if (acc == RNS_BLOCK_SUBTO)
                    mod_eltwise_mul_subto_w32(o, x, y, N, mod);
                else
                    mod_eltwise_mul_w32(o, x, y, N, mod);
            }
            else
            {
                uint64_t *o = out->rows64[i], *x = in1->rows64[i], *y = in2->rows64[i];
                if (acc == RNS_BLOCK_ADDTO)
                    mod_eltwise_mul_addto(o, x, y, N, mod);
                else if (acc == RNS_BLOCK_SUBTO)
                    mod_eltwise_mul_subto(o, x, y, N, mod);
                else
                    mod_eltwise_mul(o, x, y, N, mod);
            }
        }
        return;
    }

    uint32_t *t32 = (uint32_t *)safe_aligned_malloc(ps * sizeof(uint32_t));
    uint64_t *t64 = (uint64_t *)safe_aligned_malloc(ps * sizeof(uint64_t));
    for (size_t i = 0; i < b->l; i++)
    {
        if (!(out->rns_mask & (1ULL << i)))
            continue;
        Modulus mod = b->mods[i];
        if (rns_row_is_narrow(b, i))
            rns_row_block_mul_narrow(out->rows32[i], in1->rows32[i], in2->rows32[i], t32, b->w32[i],
                                     sd, ps, mod, acc);
        else
            rns_row_block_mul_wide(out->rows64[i], in1->rows64[i], in2->rows64[i], t64, b->w[i], sd,
                                   ps, mod, acc);
    }
    free(t32);
    free(t64);
}

void polynomial_mul_RNS_polynomial(RNS_Polynomial out, RNS_Polynomial in1, RNS_Polynomial in2)
{
    assert(out != in1);
    assert(out != in2);
    out->rns_mask = in1->rns_mask & in2->rns_mask;
    rns_product(out, in1, in2, RNS_BLOCK_SET);
}

void polynomial_mul_addto_RNS_polynomial(RNS_Polynomial out, RNS_Polynomial in1, RNS_Polynomial in2)
{
    assert(out != in1);
    assert(out != in2);
    out->rns_mask = in1->rns_mask & in2->rns_mask;
    rns_product(out, in1, in2, RNS_BLOCK_ADDTO);
}

void polynomial_mul_subto_RNS_polynomial(RNS_Polynomial out, RNS_Polynomial in1, RNS_Polynomial in2)
{
    assert(out != in1);
    assert(out != in2);
    out->rns_mask = in1->rns_mask & in2->rns_mask;
    rns_product(out, in1, in2, RNS_BLOCK_SUBTO);
}

// The size of one tile of all the b operands of polynomial_matrix_mul_addto_RNS_polynomial,
// small enough to stay in cache while every row of a uses it.
#define MATRIX_TILE_BYTES (1 << 17)

void polynomial_matrix_mul_addto_RNS_polynomial(RNS_Polynomial *out, RNS_Polynomial const *a,
                                                RNS_Polynomial const *b, uint64_t rows,
                                                uint64_t inner, uint64_t cols)
{
    if (rows == 0 || cols == 0 || inner == 0)
        return;
    RNS_Base base = out[0]->base;
    const uint64_t N = base->N, mask = out[0]->rns_mask;
    // Over a ring that does not split fully a product is a convolution of blocks: term by term.
    if (base->split_degree != 1)
    {
        for (uint64_t r = 0; r < rows; r++)
            for (uint64_t c = 0; c < cols; c++)
                for (uint64_t t = 0; t < inner; t++)
                {
                    RNS_Polynomial o = out[r * cols + c];
                    const uint64_t keep = o->rns_mask;
                    polynomial_mul_addto_RNS_polynomial(o, a[r * inner + t], b[c * inner + t]);
                    o->rns_mask = keep;
                }
        return;
    }
    uint64_t tile = MATRIX_TILE_BYTES / (cols * inner * sizeof(uint64_t));
    tile = tile < MOD_MIN_VECTOR_LEN ? MOD_MIN_VECTOR_LEN : tile - tile % MOD_MIN_VECTOR_LEN;
    uint64_t **a_rows = (uint64_t **)safe_malloc(2 * inner * sizeof(uint64_t *));
    uint64_t **b_rows = a_rows + inner;
    for (size_t i = 0; i < base->l; i++)
    {
        if (!(mask & (1ULL << i)))
            continue;
        Modulus mod = base->mods[i];
        if (rns_row_is_narrow(base, i)) // 32-bit rows: term by term
        {
            for (uint64_t r = 0; r < rows; r++)
                for (uint64_t c = 0; c < cols; c++)
                    for (uint64_t t = 0; t < inner; t++)
                        mod_eltwise_mul_addto_w32(out[r * cols + c]->rows32[i],
                                                  a[r * inner + t]->rows32[i],
                                                  b[c * inner + t]->rows32[i], N, mod);
            continue;
        }
        for (uint64_t start = 0; start < N; start += tile)
        {
            const uint64_t len = N - start < tile ? N - start : tile;
            for (uint64_t r = 0; r < rows; r++)
            {
                for (uint64_t t = 0; t < inner; t++)
                    a_rows[t] = a[r * inner + t]->rows64[i] + start;
                for (uint64_t c = 0; c < cols; c++)
                {
                    for (uint64_t t = 0; t < inner; t++)
                        b_rows[t] = b[c * inner + t]->rows64[i] + start;
                    mod_eltwise_inner_product_addto(out[r * cols + c]->rows64[i] + start, a_rows,
                                                    b_rows, inner, len, mod);
                }
            }
        }
    }
    free(a_rows);
}

RNS_Polynomial polynomial_new_RNS_polynomial_rows(uint64_t N, uint64_t rns_mask, RNS_Base base)
{
    RNS_Polynomial res = (RNS_Polynomial)safe_malloc(sizeof(*res));
    res->rows64 = (uint64_t **)safe_malloc(sizeof(uint64_t *) * base->l);
    res->rows32 = (uint32_t **)safe_malloc(sizeof(uint32_t *) * base->l);
    for (size_t i = 0; i < base->l; i++)
    {
        res->rows64[i] = NULL;
        res->rows32[i] = NULL;
        if (!(rns_mask & (1ULL << i)))
            continue;
        if (rns_row_is_narrow(base, i))
            res->rows32[i] = (uint32_t *)mempool_aligned_malloc(sizeof(uint32_t) * base->N);
        else
            res->rows64[i] = (uint64_t *)mempool_aligned_malloc(sizeof(uint64_t) * base->N);
    }
    res->base = base;
    res->rns_mask = rns_mask;
    res->allocated_l = base->l;
    return res;
}

/* out *= in. The element-wise kernels read a lane before writing it, so the
   unsplit case aliases safely; the block product does not, and there the row
   is copied first -- at its own width. */
void polynomial_multo_RNS_polynomial(RNS_Polynomial out, RNS_Polynomial in)
{
    RNS_Base b = out->base;
    out->rns_mask = out->rns_mask & in->rns_mask;
    if (b->split_degree == 1)
    {
        rns_product(out, out, in, RNS_BLOCK_SET);
        return;
    }

    const uint64_t sd = b->split_degree, N = b->N, ps = N / sd;
    uint32_t *t32 = (uint32_t *)safe_aligned_malloc(ps * sizeof(uint32_t));
    uint64_t *t64 = (uint64_t *)safe_aligned_malloc(ps * sizeof(uint64_t));
    uint32_t *prev32 = (uint32_t *)safe_aligned_malloc(N * sizeof(uint32_t));
    uint64_t *prev64 = (uint64_t *)safe_aligned_malloc(N * sizeof(uint64_t));
    for (size_t i = 0; i < b->l; i++)
    {
        if (!(out->rns_mask & (1ULL << i)))
            continue;
        Modulus mod = b->mods[i];
        if (rns_row_is_narrow(b, i))
        {
            memcpy(prev32, out->rows32[i], N * sizeof(uint32_t));
            rns_row_block_mul_narrow(out->rows32[i], in->rows32[i], prev32, t32, b->w32[i], sd, ps,
                                     mod, RNS_BLOCK_SET);
        }
        else
        {
            memcpy(prev64, out->rows64[i], N * sizeof(uint64_t));
            rns_row_block_mul_wide(out->rows64[i], in->rows64[i], prev64, t64, b->w[i], sd, ps, mod,
                                   RNS_BLOCK_SET);
        }
    }
    free(t32);
    free(t64);
    free(prev32);
    free(prev64);
}

void polynomial_sub_RNS_polynomial(RNS_Polynomial out, RNS_Polynomial in1, RNS_Polynomial in2)
{
    out->rns_mask = in1->rns_mask & in2->rns_mask;
    for (size_t i = 0; i < out->base->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            RNS_ROW_BINOP(mod_eltwise_sub, out, in1, in2, i, out->base->N, out->base->mods[i]);
        }
    }
}

void polynomial_sub_RNSc_polynomial(RNSc_Polynomial out, RNSc_Polynomial in1, RNSc_Polynomial in2)
{
    polynomial_sub_RNS_polynomial((RNS_Polynomial)out, (RNS_Polynomial)in1, (RNS_Polynomial)in2);
}

void polynomial_add_RNSc_polynomial(RNSc_Polynomial out, RNSc_Polynomial in1, RNSc_Polynomial in2)
{
    out->rns_mask = in1->rns_mask & in2->rns_mask;
    for (size_t i = 0; i < out->base->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            RNS_ROW_BINOP(mod_eltwise_add, out, in1, in2, i, out->base->N, out->base->mods[i]);
        }
    }
}

void polynomial_add_RNS_polynomial(RNS_Polynomial out, RNS_Polynomial in1, RNS_Polynomial in2)
{
    polynomial_add_RNSc_polynomial((RNSc_Polynomial)out, (RNSc_Polynomial)in1,
                                   (RNSc_Polynomial)in2);
}

void polynomial_RNSc_add_integer(RNSc_Polynomial out, RNSc_Polynomial in1, uint64_t in2)
{
    out->rns_mask = in1->rns_mask;
    for (size_t i = 0; i < out->base->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            if (out != in1)
                RNS_ROW_COPY(out, in1, i, out->base->N);
            const uint64_t q = out->base->mods[i]->q;
            const uint64_t in_mod_q = in2 & (1ULL << 63) ? q - ((-in2) % q) : in2 % q;
            if (rns_row_is_narrow(out->base, i))
                rns_row_add_at0_narrow(out->rows32[i], in_mod_q, q);
            else
                rns_row_add_at0_wide(out->rows64[i], in_mod_q, q);
        }
    }
}

void polynomial_RNS_add_integer(RNS_Polynomial out, RNS_Polynomial in1, uint64_t in2)
{
    out->rns_mask = in1->rns_mask;
    const uint64_t poly_size = out->base->N / out->base->split_degree;
    for (size_t i = 0; i < out->base->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            Modulus mod = out->base->mods[i];
            if (out != in1)
                RNS_ROW_COPY(out, in1, i, out->base->N);
            const uint64_t q = mod->q;
            uint64_t in_mod_q;
            if (in2 & (1ULL << 63))
            {
                in_mod_q = negate_modq(modq(-in2, mod), q);
            }
            else
            {
                in_mod_q = modq(in2, mod);
            }
            RNS_ROW_SCALAROP(mod_eltwise_add_scalar, out, out, in_mod_q, i, poly_size, mod);
        }
    }
}

void polynomial_scale_RNSc_polynomial(RNSc_Polynomial out, RNSc_Polynomial in1, uint64_t scale)
{
    out->rns_mask = in1->rns_mask;
    for (size_t i = 0; i < out->base->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            RNS_ROW_SCALAROP(mod_eltwise_scale, out, in1, scale, i, out->base->N,
                             out->base->mods[i]);
        }
    }
}

void polynomial_scale_addto_RNSc_polynomial(RNSc_Polynomial out, RNSc_Polynomial in1,
                                            uint64_t scale)
{
    out->rns_mask = in1->rns_mask;
    uint64_t mask = out->rns_mask;
    for (size_t i = 0; i < out->base->l; i++)
    {
        if (mask & (1ULL << i))
        {
            RNS_ROW_SCALAROP(mod_eltwise_fma, out, in1, scale, i, out->base->N, out->base->mods[i]);
        }
    }
}

void polynomial_scale_addto_RNS_polynomial(RNS_Polynomial out, RNS_Polynomial in1, uint64_t scale)
{
    polynomial_scale_addto_RNSc_polynomial((RNSc_Polynomial)out, (RNSc_Polynomial)in1, scale);
}

void polynomial_scale_RNS_polynomial(RNS_Polynomial out, RNS_Polynomial in1, uint64_t scale)
{
    polynomial_scale_RNSc_polynomial((RNSc_Polynomial)out, (RNSc_Polynomial)in1, scale);
}

void polynomial_scale_RNS_polynomial_RNS(RNS_Polynomial out, RNS_Polynomial in1, uint64_t *scale)
{
    out->rns_mask = in1->rns_mask;
    for (size_t i = 0; i < out->base->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            RNS_ROW_SCALAROP(mod_eltwise_scale, out, in1, scale[i], i, out->base->N,
                             out->base->mods[i]);
        }
    }
}

void polynomial_RNSc_negate(RNSc_Polynomial out, RNSc_Polynomial in)
{
    out->rns_mask = in->rns_mask;
    for (size_t i = 0; i < out->base->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            RNS_ROW_UNOP(mod_eltwise_negate, out, in, i, out->base->N, out->base->mods[i]);
        }
    }
}

void polynomial_RNS_negate(RNS_Polynomial out, RNS_Polynomial in)
{
    polynomial_RNSc_negate((RNSc_Polynomial)out, (RNSc_Polynomial)in);
}

void polynomial_RNSc_to_RNS(RNS_Polynomial out, RNSc_Polynomial in)
{
    out->rns_mask = in->rns_mask;
    const uint64_t poly_size = out->base->N / out->base->split_degree;
    /* A narrow row transforms in 32-bit words: 16 coefficients per vector and
       half the memory traffic. The plan carries both table sets, so all that is
       asked here is the row's width; a length or an engine without the 32-bit
       kernels is the entry point's business, not this loop's. */
    for (size_t i = 0; i < out->base->l; i++)
    {
        if (!(out->rns_mask & (1ULL << i)))
            continue;
        NTT_Plan plan = out->base->plans[i];
        if (rns_row_is_narrow(out->base, i))
        {
            for (size_t k = 0; k < out->base->split_degree; k++)
            {
                ntt_forward_w32(&out->rows32[i][k * poly_size], &in->rows32[i][k * poly_size],
                                plan);
            }
        }
        else
        {
            for (size_t k = 0; k < out->base->split_degree; k++)
            {
                ntt_forward(&out->rows64[i][k * poly_size], &in->rows64[i][k * poly_size], plan);
            }
        }
    }
}

void polynomial_RNS_to_RNSc(RNSc_Polynomial out, RNS_Polynomial in)
{
    out->rns_mask = in->rns_mask;
    const uint64_t poly_size = out->base->N / out->base->split_degree;
    /* A narrow row transforms in 32-bit words: 16 coefficients per vector and
       half the memory traffic. The plan carries both table sets, so all that is
       asked here is the row's width; a length or an engine without the 32-bit
       kernels is the entry point's business, not this loop's. */
    for (size_t i = 0; i < out->base->l; i++)
    {
        if (!(out->rns_mask & (1ULL << i)))
            continue;
        NTT_Plan plan = out->base->plans[i];
        if (rns_row_is_narrow(out->base, i))
        {
            for (size_t k = 0; k < out->base->split_degree; k++)
            {
                ntt_reverse_w32(&out->rows32[i][k * poly_size], &in->rows32[i][k * poly_size],
                                plan);
            }
        }
        else
        {
            for (size_t k = 0; k < out->base->split_degree; k++)
            {
                ntt_reverse(&out->rows64[i][k * poly_size], &in->rows64[i][k * poly_size], plan);
            }
        }
    }
}

void polynomial_RNSc_add_noise(RNSc_Polynomial out, RNSc_Polynomial in, double sigma)
{
    int64_t *noise_arr = (int64_t *)safe_aligned_malloc(out->base->N * sizeof(int64_t));
    for (size_t j = 0; j < out->base->N; j++)
    {
        noise_arr[j] = (int64_t)round(generate_normal_random(sigma));
    }
    /* The noise is drawn once and reduced per prime, so it lands in a buffer
       at that row's width and the add stays there too. */
    uint64_t *nr64 = (uint64_t *)safe_aligned_malloc(out->base->N * sizeof(uint64_t));
    uint32_t *nr32 = (uint32_t *)safe_aligned_malloc(out->base->N * sizeof(uint32_t));
    out->rns_mask = in->rns_mask;
    for (size_t i = 0; i < out->base->l; i++)
    {
        if (!(out->rns_mask & (1ULL << i)))
            continue;
        Modulus mod = out->base->mods[i];
        if (rns_row_is_narrow(out->base, i))
        {
            mod_eltwise_reduce_signed_w32(nr32, noise_arr, out->base->N, mod);
            mod_eltwise_add_w32(out->rows32[i], in->rows32[i], nr32, out->base->N, mod);
        }
        else
        {
            mod_eltwise_reduce_signed(nr64, noise_arr, out->base->N, mod);
            mod_eltwise_add(out->rows64[i], in->rows64[i], nr64, out->base->N, mod);
        }
    }
    free(nr64);
    free(nr32);
    free(noise_arr);
}

static RNS_BaseConversionParams init_conversion_params(RNS_Base base, uint64_t in_mask,
                                                       uint64_t out_mask, int exact)
{
    RNS_BaseConversionParams params = (RNS_BaseConversionParams)safe_malloc(sizeof(*params));
    params->in_mask = in_mask;
    params->out_mask = out_mask;
    params->exact = exact;
    params->inv_q = NULL;
    params->uM_mod_p = NULL;

    params->D = (uint32_t *)safe_malloc(sizeof(uint32_t) * base->l);
    params->P = (uint32_t *)safe_malloc(sizeof(uint32_t) * base->l);
    params->w = 0;
    params->v = 0;

    for (size_t i = 0; i < base->l; i++)
    {
        if (in_mask & (1ULL << i))
        {
            params->D[params->w++] = (uint32_t)i;
        }
    }

    for (size_t i = 0; i < base->l; i++)
    {
        if ((out_mask & (1ULL << i)) && !(in_mask & (1ULL << i)))
        {
            params->P[params->v++] = (uint32_t)i;
        }
    }

    if (params->v == 0)
    {
        params->Dhat = NULL;
        params->D_mod_p = NULL;
        params->exact = 0;
        return params;
    }

    assert(params->w > 0);

    params->Dhat = (uint64_t *)safe_malloc(sizeof(uint64_t) * params->w);
    for (size_t j = 0; j < params->w; j++)
    {
        uint64_t idx_j = params->D[j];
        Modulus mod_j = base->mods[idx_j];
        uint64_t q_j = mod_j->q;
        uint64_t prod = 1;
        for (size_t k = 0; k < params->w; k++)
        {
            if (k == j)
                continue;
            uint64_t idx_k = params->D[k];
            uint64_t q_k = base->mods[idx_k]->q;
            prod = mul_modq(prod, modq(q_k, mod_j), mod_j);
        }
        params->Dhat[j] = inverse_mod(prod, q_j);
    }

    params->D_mod_p = (uint64_t **)safe_malloc(sizeof(uint64_t *) * params->v);
    for (size_t i = 0; i < params->v; i++)
    {
        params->D_mod_p[i] = (uint64_t *)safe_malloc(sizeof(uint64_t) * params->w);
        uint64_t idx_i = params->P[i];
        Modulus mod_i = base->mods[idx_i];
        for (size_t j = 0; j < params->w; j++)
        {
            uint64_t prod = 1;
            for (size_t k = 0; k < params->w; k++)
            {
                if (k == j)
                    continue;
                uint64_t idx_k = params->D[k];
                uint64_t q_k = base->mods[idx_k]->q;
                prod = mul_modq(prod, modq(q_k, mod_i), mod_i);
            }
            params->D_mod_p[i][j] = prod;
        }
    }

    if (exact)
    {
        /* `polynomial_base_conversion_RNSc` recovers u by summing r_j/q_j
           over the input primes. Storing reciprocals makes that a multiply per
           residue instead of a divide; the extra rounding is far inside the
           margin the caller has to leave anyway. */
        params->inv_q = (double *)safe_malloc(sizeof(double) * params->w);
        for (size_t j = 0; j < params->w; j++)
        {
            params->inv_q[j] = 1.0 / (double)base->mods[params->D[j]]->q;
        }
        /* u stays below w, so the multiples of M that removing it can need
           fit in a table of w+1 entries per output prime. That keeps the
           removal to a lookup and a subtraction per coefficient, with no
           u * (M mod p_i) product to overflow 64 bits. */
        params->uM_mod_p = (uint64_t **)safe_malloc(sizeof(uint64_t *) * params->v);
        for (size_t i = 0; i < params->v; i++)
        {
            Modulus mod_i = base->mods[params->P[i]];
            uint64_t m_mod_p = 1;
            for (size_t j = 0; j < params->w; j++)
            {
                m_mod_p = mul_modq(m_mod_p, modq(base->mods[params->D[j]]->q, mod_i), mod_i);
            }
            params->uM_mod_p[i] = (uint64_t *)safe_malloc(sizeof(uint64_t) * (params->w + 1));
            uint64_t acc = 0;
            for (size_t u = 0; u <= params->w; u++)
            {
                params->uM_mod_p[i][u] = acc;
                acc = add_modq(acc, m_mod_p, mod_i->q);
            }
        }
    }

    return params;
}

RNS_BaseConversionParams init_base_conversion_params(RNS_Base base, uint64_t in_mask,
                                                     uint64_t out_mask)
{
    return init_conversion_params(base, in_mask, out_mask, 0);
}

RNS_BaseConversionParams init_base_conversion_params_exact(RNS_Base base, uint64_t in_mask,
                                                           uint64_t out_mask)
{
    return init_conversion_params(base, in_mask, out_mask, 1);
}

void free_base_conversion_params(RNS_BaseConversionParams params)
{
    if (params == NULL)
        return;
    if (params->D_mod_p != NULL)
    {
        for (size_t i = 0; i < params->v; i++)
        {
            free(params->D_mod_p[i]);
        }
        free(params->D_mod_p);
    }
    if (params->uM_mod_p != NULL)
    {
        for (size_t i = 0; i < params->v; i++)
        {
            free(params->uM_mod_p[i]);
        }
        free(params->uM_mod_p);
    }
    free(params->inv_q);
    if (params->Dhat != NULL)
    {
        free(params->Dhat);
    }
    free(params->D);
    free(params->P);
    free(params);
}

void polynomial_base_conversion_RNSc(RNSc_Polynomial out, RNSc_Polynomial in,
                                     RNS_BaseConversionParams params)
{
    assert(in->base->l == out->base->l);
    assert(in->base->N == out->base->N);

    uint64_t in_mask = in->rns_mask;
    uint64_t out_mask = out->rns_mask;

    // 1. Copy matching active RNS components from in to out (if not in-place)
    for (size_t i = 0; i < out->base->l; i++)
    {
        if ((in_mask & (1ULL << i)) && (out_mask & (1ULL << i)))
        {
            if (out != in)
            {
                RNS_ROW_COPY(out, in, i, out->base->N);
            }
        }
    }

    RNS_BaseConversionParams local_params = params;
    if (local_params == NULL)
    {
        local_params = init_base_conversion_params(in->base, in_mask, out_mask);
    }
    else
    {
        assert(local_params->in_mask == in_mask);
        assert(local_params->out_mask == out_mask);
    }

    uint32_t w = local_params->w;
    uint32_t v = local_params->v;
    uint32_t *D = local_params->D;
    uint32_t *P = local_params->P;
    uint64_t *Dhat = local_params->Dhat;
    uint64_t **D_mod_p = local_params->D_mod_p;

    if (v == 0)
    {
        out->rns_mask = out_mask;
        if (params == NULL)
        {
            free_base_conversion_params(local_params);
        }
        return;
    }

    // 2. Initialize the output target polynomial coefficients to zero
    for (size_t i = 0; i < v; i++)
    {
        uint64_t idx_i = P[i];
        RNS_ROW_ZERO(out, idx_i, out->base->N);
    }

    // 3. Perform Fast Base Extension
    uint64_t *v_tmp = (uint64_t *)safe_aligned_malloc(out->base->N * sizeof(uint64_t));
    uint64_t *v_tmp2 = (uint64_t *)safe_aligned_malloc(out->base->N * sizeof(uint64_t));
    uint32_t *v_tmp_32 = (uint32_t *)safe_aligned_malloc(out->base->N * sizeof(uint32_t));
    uint32_t *v_tmp2_32 = (uint32_t *)safe_aligned_malloc(out->base->N * sizeof(uint32_t));
    const uint64_t N = out->base->N;
    /* The loop below writes x + u*M, not x (see arith.h). Recovering u costs
       no extra pass: the scaled residues r_j it already computes satisfy

           sum_j (r_j / q_j) = u + x/M,

       so the integer part of that sum is u -- provided x/M cannot reach 0 or
       1, which is the caller's guarantee. */
    double *overflow = NULL;
    if (local_params->exact)
    {
        overflow = (double *)safe_aligned_malloc(N * sizeof(double));
        for (uint64_t c = 0; c < N; c++)
            overflow[c] = 0.0;
    }

    for (size_t j = 0; j < w; j++)
    {
        uint64_t idx_j = D[j];
        Modulus mod_j = in->base->mods[idx_j];
        // the scaled residue is still mod p_j, so it keeps that row's width
        const bool j_narrow = rns_row_is_narrow(in->base, idx_j);
        if (j_narrow)
            mod_eltwise_scale_w32(v_tmp_32, in->rows32[idx_j], Dhat[j], N, mod_j);
        else
            mod_eltwise_scale(v_tmp, in->rows64[idx_j], Dhat[j], N, mod_j);

        /* This prime's term r_j/q_j in that sum. */
        if (overflow != NULL)
        {
            if (j_narrow)
                mod_basecvt_accumulate_w32(overflow, v_tmp_32, local_params->inv_q[j], N);
            else
                mod_basecvt_accumulate(overflow, v_tmp, local_params->inv_q[j], N);
        }

        for (size_t i = 0; i < v; i++)
        {
            uint64_t idx_i = P[i];
            Modulus mod_i = out->base->mods[idx_i];
            /* Here the modulus changes, and with it possibly the width: this
               reduction is the only width-crossing step in the conversion. */
            if (rns_row_is_narrow(out->base, idx_i))
            {
                if (j_narrow)
                    mod_eltwise_reduce_w32(v_tmp2_32, v_tmp_32, N, mod_i);
                else
                    mod_eltwise_reduce_narrow_from_wide(v_tmp2_32, v_tmp, N, mod_i);
                mod_eltwise_fma_w32(out->rows32[idx_i], v_tmp2_32, D_mod_p[i][j], N, mod_i);
            }
            else
            {
                if (j_narrow)
                    mod_eltwise_reduce_wide_from_narrow(v_tmp2, v_tmp_32, N, mod_i);
                else
                    mod_eltwise_reduce(v_tmp2, v_tmp, N, mod_i);
                mod_eltwise_fma(out->rows64[idx_i], v_tmp2, D_mod_p[i][j], N, mod_i);
            }
        }
    }

    /* Subtract u*M from every row the conversion wrote, leaving x. */
    if (overflow != NULL)
    {
#ifndef NDEBUG
        for (uint64_t c = 0; c < N; c++)
            assert((size_t)overflow[c] <= w);
#endif
        for (size_t i = 0; i < v; i++)
        {
            const uint64_t idx_i = P[i];
            Modulus mod_i = out->base->mods[idx_i];
            const uint64_t *uM = local_params->uM_mod_p[i];
            if (rns_row_is_narrow(out->base, idx_i))
                mod_basecvt_sub_indexed_w32(out->rows32[idx_i], overflow, uM, w + 1, N, mod_i);
            else
                mod_basecvt_sub_indexed(out->rows64[idx_i], overflow, uM, w + 1, N, mod_i);
        }
        free(overflow);
    }

    free(v_tmp);
    free(v_tmp2);
    free(v_tmp_32);
    free(v_tmp2_32);

    if (params == NULL)
    {
        free_base_conversion_params(local_params);
    }

    out->rns_mask = out_mask;
}

void polynomial_RNSc_mod_reduce_lifted(RNSc_Polynomial out, RNSc_Polynomial in, uint64_t idx)
{
    /* Row `idx` of the input is reduced into every active row of the output,
       and the two rows need not be the same width, so the source is read once
       through a 64-bit view and reduced from there. */
    for (size_t i = 0; i < out->base->l; i++)
    {
        if (!(out->rns_mask & (1ULL << i)))
            continue;
        if (i == idx)
        {
            // the source is already the residue mod p_idx, so reducing mod
            // p_i (== p_idx) is the identity: copy instead of Barrett
            if (out->rows64[i] != in->rows64[idx] || out->rows32[i] != in->rows32[idx])
                RNS_ROW_COPY(out, in, i, out->base->N);
        }
        else
        {
            RNS_ROW_REDUCE(out, in, i, idx, out->base->N, out->base->mods[i]);
        }
    }
}

void polynomial_RNSc_mod_reduce_lifted_centered(RNSc_Polynomial out, RNSc_Polynomial in,
                                                uint64_t idx)
{
    const uint64_t q_in = in->base->mods[idx]->q;
    for (size_t i = 0; i < out->base->l; i++)
    {
        if (!(out->rns_mask & (1ULL << i)))
            continue;
        if (i == idx)
        {
            if (out->rows64[i] != in->rows64[idx] || out->rows32[i] != in->rows32[idx])
                RNS_ROW_COPY(out, in, i, out->base->N);
        }
        else
        {
            RNS_ROW_REDUCE_CENTERED(out, in, i, idx, out->base->N, q_in, out->base->mods[i]);
        }
    }
}

// Row `k` of `in` reduced modulo prime `i` of `base`, read as a centered
// residue if `centered`, into `buf` at row i's width.
static void lift_row(void *buf, RNSc_Polynomial in, size_t k, RNS_Base base, size_t i,
                     bool centered)
{
    const uint64_t N = base->N, q_in = in->base->mods[k]->q;
    const Modulus mod = base->mods[i];
    const bool out_narrow = rns_row_is_narrow(base, i), in_narrow = rns_row_is_narrow(in->base, k);
    if (out_narrow && in_narrow)
    {
        if (centered)
            mod_eltwise_reduce_centered_w32((uint32_t *)buf, in->rows32[k], q_in, N, mod);
        else
            mod_eltwise_reduce_w32((uint32_t *)buf, in->rows32[k], N, mod);
    }
    else if (out_narrow)
    {
        if (centered)
            mod_eltwise_reduce_centered_narrow_from_wide((uint32_t *)buf, in->rows64[k], q_in, N,
                                                         mod);
        else
            mod_eltwise_reduce_narrow_from_wide((uint32_t *)buf, in->rows64[k], N, mod);
    }
    else if (in_narrow)
    {
        if (centered)
            mod_eltwise_reduce_centered_wide_from_narrow((uint64_t *)buf, in->rows32[k], q_in, N,
                                                         mod);
        else
            mod_eltwise_reduce_wide_from_narrow((uint64_t *)buf, in->rows32[k], N, mod);
    }
    else
    {
        if (centered)
            mod_eltwise_reduce_centered((uint64_t *)buf, in->rows64[k], q_in, N, mod);
        else
            mod_eltwise_reduce((uint64_t *)buf, in->rows64[k], N, mod);
    }
}

// One term of a row combination: residue `prime` of `p` times `factor`.
typedef struct
{
    RNSc_Polynomial p;
    size_t prime;
    uint64_t factor;
} RowTerm;

// Row `i` of `out` as the sum of the `m` terms, each lifted to p_i (centered
// if `centered`), by the fused kernel. Returns false, writing nothing, when a
// row involved is narrow or the engine has no body for these moduli.
static bool combine_row(RNSc_Polynomial out, size_t i, const RowTerm *terms, uint64_t m,
                        bool centered)
{
    RNS_Base base = out->base;
    const uint64_t *rows[MOD_COMBINATION_MAX_TERMS];
    uint64_t q_in[MOD_COMBINATION_MAX_TERMS], scale[MOD_COMBINATION_MAX_TERMS];
    if (rns_row_is_narrow(base, i) || m > MOD_COMBINATION_MAX_TERMS)
        return false;
    for (uint64_t t = 0; t < m; t++)
    {
        if (rns_row_is_narrow(terms[t].p->base, terms[t].prime))
            return false;
        rows[t] = terms[t].p->rows64[terms[t].prime];
        q_in[t] = terms[t].p->base->mods[terms[t].prime]->q;
        scale[t] = terms[t].factor;
    }
    if (m == 0)
    {
        RNS_ROW_ZERO(out, i, base->N);
        return true;
    }
    return mod_eltwise_lifted_combination(out->rows64[i], rows, q_in, scale, m, centered, base->N,
                                          base->mods[i]);
}

// combine_row over the residues of `in` in `mask`, row i's factors in
// `factors` (as polynomial_RNSc_lifted_combination takes them).
static bool lifted_combination_row(RNSc_Polynomial out, size_t i, RNSc_Polynomial in, uint64_t mask,
                                   const uint64_t *factors, bool centered)
{
    const uint64_t l = out->base->l;
    RowTerm terms[MOD_COMBINATION_MAX_TERMS];
    uint64_t m = 0;
    for (size_t k = 0; k < l; k++)
    {
        const uint64_t factor = (mask >> k) & 1 ? factors[k * l + i] : 0;
        if (!factor)
            continue;
        if (m == MOD_COMBINATION_MAX_TERMS)
            return false;
        terms[m++] = (RowTerm){in, k, factor};
    }
    return combine_row(out, i, terms, m, centered);
}

void polynomial_RNSc_lifted_combination(RNSc_Polynomial out, RNSc_Polynomial in, uint64_t mask,
                                        const uint64_t *factors, bool centered)
{
    RNS_Base base = out->base;
    const uint64_t N = base->N, l = base->l;
    assert(out != in && (in->rns_mask & mask) == mask);
    void *lifted = NULL;
    for (size_t i = 0; i < l; i++)
    {
        if (!(out->rns_mask & (1ULL << i)))
            continue;
        if (lifted_combination_row(out, i, in, mask, factors, centered))
            continue;
        // Term by term: lift, then multiply-add.
        if (!lifted)
            lifted = mempool_aligned_malloc(N * sizeof(uint64_t));
        const bool narrow = rns_row_is_narrow(base, i);
        RNS_ROW_ZERO(out, i, N);
        for (size_t k = 0; k < l; k++)
        {
            const uint64_t factor = (mask >> k) & 1 ? factors[k * l + i] : 0;
            if (!factor)
                continue;
            // Row i of `in` already holds the residue modulo p_i, centered or not.
            const void *term = narrow ? (const void *)in->rows32[i] : (const void *)in->rows64[i];
            if (k != i)
            {
                lift_row(lifted, in, k, base, i, centered);
                term = lifted;
            }
            if (narrow)
                mod_eltwise_fma_w32(out->rows32[i], (uint32_t *)term, factor, N, base->mods[i]);
            else
                mod_eltwise_fma(out->rows64[i], (uint64_t *)term, factor, N, base->mods[i]);
        }
    }
    if (lifted)
        mempool_free(lifted, N * sizeof(uint64_t));
}

void polynomial_RNSc_mod_reduce(RNSc_Polynomial out, RNSc_Polynomial in)
{
    for (size_t i = 0; i < out->base->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            RNS_ROW_COPY(out, in, i, out->base->N);
        }
    }
}

// Digit `level` of residue `idx` plus `offset` mod its prime, broadcast to
// every active row of `out`.
static void decompose_digit(RNSc_Polynomial out, RNSc_Polynomial in, uint64_t idx,
                            uint64_t log_base, uint64_t level, uint64_t offset)
{
    const uint64_t N = out->base->N;
    uint64_t *tmp = (uint64_t *)mempool_aligned_malloc(N * sizeof(uint64_t));
    const uint64_t mask = (1ULL << log_base) - 1;
    const uint64_t shift = log_base * level;
    const uint64_t q = in->base->mods[idx]->q;
    assert(in->rns_mask & (1ULL << idx));
    assert(offset < q);
    if (rns_row_is_narrow(in->base, (size_t)idx))
    {
        if (offset)
            rns_row_digit_offset_narrow(tmp, in->rows32[idx], shift, mask, offset, q, N);
        else
            rns_row_digit_narrow(tmp, in->rows32[idx], shift, mask, N);
    }
    else
    {
        if (offset)
            rns_row_digit_offset_wide(tmp, in->rows64[idx], shift, mask, offset, q, N);
        else
            rns_row_digit_wide(tmp, in->rows64[idx], shift, mask, N);
    }
    /* The digit is below 2^log_base and so below every prime, which is why one
       array serves every row: no reduction, just a store at the row's width. */
    for (size_t i = 0; i < out->base->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            assert((1ULL << log_base) <= out->base->mods[i]->q);
            if (rns_row_is_narrow(out->base, i))
                mod_narrow_w32(out->rows32[i], tmp, N);
            else
                memcpy(out->rows64[i], tmp, sizeof(uint64_t) * N);
        }
    }
    mempool_free(tmp, N * sizeof(uint64_t));
}

void polynomial_RNSc_decompose_digit(RNSc_Polynomial out, RNSc_Polynomial in, uint64_t idx,
                                     uint64_t log_base, uint64_t level)
{
    decompose_digit(out, in, idx, log_base, level, 0);
}

void polynomial_RNSc_decompose_digit_rounded(RNSc_Polynomial out, RNSc_Polynomial in, uint64_t idx,
                                             uint64_t log_base, uint64_t level, uint64_t dropped)
{
    assert(level >= dropped);
    decompose_digit(out, in, idx, log_base, level, dropped ? 1ULL << (log_base * dropped - 1) : 0);
}

void polynomial_RNSc_decompose_digit_balanced(RNSc_Polynomial out, RNSc_Polynomial in, uint64_t idx,
                                              uint64_t log_base, uint64_t level, uint64_t dropped)
{
    const uint64_t N = out->base->N;
    const uint64_t q = in->base->mods[idx]->q;
    const uint64_t bits = (uint64_t)(64 - __builtin_clzll(q));
    const uint64_t digits = (bits + log_base - 1) / log_base;
    assert(in->rns_mask & (1ULL << idx));
    assert(log_base > 0 && dropped <= level && level < digits);

    // The offset turns the digits below and at `level` into unsigned ones
    // (half a base each), and those under `dropped` into the rounding of what
    // they leave out (half their span). The top digit gets no offset of its
    // own, so it absorbs the carry and the value stays exact. Every offset is
    // below 2^(log_base * (digits - 1)) < q, so nothing overflows.
    const int64_t half = (int64_t)1 << (log_base - 1);
    const uint64_t top = digits - 1;
    int64_t offset = dropped ? (int64_t)1 << (log_base * dropped - 1) : 0;
    for (uint64_t j = dropped; j <= level && j < top; j++)
        offset += half << (log_base * j);
    const int64_t mask = level < top ? (int64_t)((1ULL << log_base) - 1) : -1;

    int64_t *tmp = (int64_t *)mempool_aligned_malloc(N * sizeof(int64_t));
    if (rns_row_is_narrow(in->base, (size_t)idx))
        rns_row_digit_balanced_narrow(tmp, in->rows32[idx], q, offset, log_base * level, mask,
                                      level < top ? half : 0, N);
    else
        rns_row_digit_balanced_wide(tmp, in->rows64[idx], q, offset, log_base * level, mask,
                                    level < top ? half : 0, N);
    // A digit may be negative, so unlike the unsigned one it takes the row's
    // prime added when it is. It is at most 2^(log_base - 1) in absolute
    // value, below every prime that fits an unsigned digit.
    for (size_t i = 0; i < out->base->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            const uint64_t p = out->base->mods[i]->q;
            assert((1ULL << log_base) <= p);
            if (rns_row_is_narrow(out->base, i))
                rns_row_store_small_signed_narrow(out->rows32[i], tmp, p, N);
            else
                rns_row_store_small_signed_wide(out->rows64[i], tmp, p, N);
        }
    }
    mempool_free(tmp, N * sizeof(int64_t));
}

void polynomial_RNSc_decompose_small(RNSc_Polynomial out, RNSc_Polynomial in, uint64_t log_base,
                                     uint64_t level)
{
    int last_active = rns_mask_get_last_active_index(in->rns_mask);
    assert(last_active >= 0);
    polynomial_RNSc_decompose_digit(out, in, (uint64_t)last_active, log_base, level);
}

void polynomial_RNSc_to_multiprecision(RNSc_Polynomial out, RNSc_Polynomial in, uint64_t log_base,
                                       uint64_t level)
{
    polynomial_RNSc_decompose_small(out, in, log_base, level);
}

void polynomial_RNS_hash_update(void *state, RNS_Polynomial p)
{
    for (size_t i = 0; i < p->base->l; i++)
    {
        if (!(p->rns_mask & (1ULL << i)))
            continue;
        // one update per row, over the row as it is stored
        if (rns_row_is_narrow(p->base, i))
            hash_stream_update(state, (const uint8_t *)p->rows32[i], p->base->N * sizeof(uint32_t));
        else
            hash_stream_update(state, (const uint8_t *)p->rows64[i], p->base->N * sizeof(uint64_t));
    }
}

void polynomial_RNS_get_hash(uint64_t *out, RNS_Polynomial p)
{
    // a hash_stream state is a blake3_hasher (hash_stream_size), so it can
    // live on the stack here
    blake3_hasher hasher;
    hash_stream_init(&hasher);
    polynomial_RNS_hash_update(&hasher, p);
    hash_stream_digest(&hasher, (uint8_t *)out);
}

uint64_t *polynomial_RNS_get_hash_p(RNS_Polynomial p)
{
    uint64_t *out = (uint64_t *)safe_malloc(4 * sizeof(uint64_t));
    polynomial_RNS_get_hash(out, p);
    return out;
}

void polynomial_floor_division_RNSc_wo_free(RNSc_Polynomial out, uint64_t divide_mask)
{
    uint64_t mask = divide_mask & out->rns_mask;
    if (mask == 0)
        return;

    const uint64_t N = out->base->N;
    uint64_t *tmp = (uint64_t *)safe_aligned_malloc(N * sizeof(uint64_t));
    /* The dropped row is reduced into each surviving one; that reduce is the
       only step that crosses a modulus, so it is the only one that can change
       width. The subtract and the scale stay at the destination's width. */
    uint32_t *tmp32 = (uint32_t *)safe_aligned_malloc(N * sizeof(uint32_t));

    for (size_t idx = 0; idx < out->base->l; idx++)
    {
        if (mask & (1ULL << idx))
        {
            const uint64_t p = out->base->mods[idx]->q;
            for (size_t i = 0; i < out->base->l; i++)
            {
                if (out->rns_mask & (1ULL << i))
                {
                    if (i == idx)
                        continue;
                    const uint64_t q = out->base->mods[i]->q;
                    const uint64_t inv_p = inverse_mod(p, q);
                    Modulus mod_i = out->base->mods[i];
                    if (rns_row_is_narrow(out->base, i))
                    {
                        rns_reduce_row_to32(tmp32, (RNS_Polynomial)out, idx, N, mod_i);
                        mod_eltwise_sub_w32(out->rows32[i], out->rows32[i], tmp32, N, mod_i);
                        mod_eltwise_scale_w32(out->rows32[i], out->rows32[i], inv_p, N, mod_i);
                    }
                    else
                    {
                        rns_reduce_row_to64(tmp, (RNS_Polynomial)out, idx, N, mod_i);
                        mod_eltwise_sub(out->rows64[i], out->rows64[i], tmp, N, mod_i);
                        mod_eltwise_scale(out->rows64[i], out->rows64[i], inv_p, N, mod_i);
                    }
                }
            }
            RNS_ROW_ZERO(out, idx, N);
            out->rns_mask &= ~(1ULL << idx);
        }
    }
    free(tmp);
    free(tmp32);
}

// Row i of `out` divided by the prime of row idx, rounding: row idx already
// holds x + floor(p/2) mod p, and the two may differ in width, so the reduce
// goes through a buffer of row i's width.
static void round_division_row(RNSc_Polynomial out, size_t idx, size_t i, uint64_t *tmp,
                               uint32_t *tmp32)
{
    const uint64_t N = out->base->N;
    const uint64_t p = out->base->mods[idx]->q, half_p = p / 2;
    const uint64_t q = out->base->mods[i]->q;
    const uint64_t inv_p = inverse_mod(p, q);
    const uint64_t half_p_mod_q = half_p % q;
    Modulus mod_i = out->base->mods[i];
    if (rns_row_is_narrow(out->base, i))
    {
        rns_reduce_row_to32(tmp32, (RNS_Polynomial)out, idx, N, mod_i);
        mod_eltwise_add_scalar_w32(out->rows32[i], out->rows32[i], half_p_mod_q, N, mod_i);
        mod_eltwise_sub_w32(out->rows32[i], out->rows32[i], tmp32, N, mod_i);
        mod_eltwise_scale_w32(out->rows32[i], out->rows32[i], inv_p, N, mod_i);
    }
    else
    {
        rns_reduce_row_to64(tmp, (RNS_Polynomial)out, idx, N, mod_i);
        mod_eltwise_add_scalar(out->rows64[i], out->rows64[i], half_p_mod_q, N, mod_i);
        mod_eltwise_sub(out->rows64[i], out->rows64[i], tmp, N, mod_i);
        mod_eltwise_scale(out->rows64[i], out->rows64[i], inv_p, N, mod_i);
    }
}

// Each prime that leaves divides every row after it in turn, so a row that
// leaves later is first divided by the ones before it.
void polynomial_round_division_RNSc_prepare(RNSc_Polynomial out, uint64_t divide_mask)
{
    const uint64_t mask = divide_mask & out->rns_mask;
    if (mask == 0)
        return;
    const uint64_t N = out->base->N;
    uint64_t *tmp = (uint64_t *)safe_aligned_malloc(N * sizeof(uint64_t));
    uint32_t *tmp32 = (uint32_t *)safe_aligned_malloc(N * sizeof(uint32_t));
    for (size_t idx = 0; idx < out->base->l; idx++)
    {
        if (!(mask & (1ULL << idx)))
            continue;
        const uint64_t half_p = out->base->mods[idx]->q / 2;
        RNS_ROW_SCALAROP(mod_eltwise_add_scalar, out, out, half_p, idx, N, out->base->mods[idx]);
        for (size_t i = idx + 1; i < out->base->l; i++)
        {
            if (mask & (1ULL << i))
                round_division_row(out, idx, i, tmp, tmp32);
        }
    }
    free(tmp);
    free(tmp32);
}

void polynomial_round_division_RNSc_rows(RNSc_Polynomial out, uint64_t divide_mask, uint64_t rows)
{
    const uint64_t mask = divide_mask & out->rns_mask;
    const uint64_t staying = rows & out->rns_mask & ~mask;
    if (mask == 0 || staying == 0)
        return;
    const uint64_t N = out->base->N;
    uint64_t *tmp = (uint64_t *)safe_aligned_malloc(N * sizeof(uint64_t));
    uint32_t *tmp32 = (uint32_t *)safe_aligned_malloc(N * sizeof(uint32_t));
    for (size_t i = 0; i < out->base->l; i++)
    {
        if (!(staying & (1ULL << i)))
            continue;
        for (size_t idx = 0; idx < out->base->l; idx++)
        {
            if (mask & (1ULL << idx))
                round_division_row(out, idx, i, tmp, tmp32);
        }
    }
    free(tmp);
    free(tmp32);
}

void polynomial_round_division_RNSc_finish(RNSc_Polynomial out, uint64_t divide_mask)
{
    const uint64_t mask = divide_mask & out->rns_mask;
    for (size_t idx = 0; idx < out->base->l; idx++)
    {
        if (mask & (1ULL << idx))
            RNS_ROW_ZERO(out, idx, out->base->N);
    }
    out->rns_mask &= ~mask;
}

// The product of the primes in `mask` other than `skip`, modulo `mod`.
static uint64_t product_of_primes_mod(RNS_Base base, uint64_t mask, size_t skip, Modulus mod)
{
    uint64_t f = 1;
    for (size_t m = 0; m < base->l; m++)
        if (m != skip && (mask & (1ULL << m)))
            f = mul_modq(f, modq(base->mods[m]->q, mod), mod);
    return f;
}

void polynomial_mod_down_RNSc_prepare(RNSc_Polynomial out, uint64_t divide_mask)
{
    const uint64_t mask = divide_mask & out->rns_mask;
    for (size_t k = 0; k < out->base->l; k++)
    {
        if (!(mask & (1ULL << k)))
            continue;
        Modulus mod = out->base->mods[k];
        const uint64_t inverse =
            inverse_mod(product_of_primes_mod(out->base, mask, k, mod), mod->q);
        RNS_ROW_SCALAROP(mod_eltwise_scale, out, out, inverse, k, out->base->N, mod);
    }
}

void polynomial_mod_down_RNSc_rows(RNSc_Polynomial out, RNSc_Polynomial in, RNSc_Polynomial addend,
                                   uint64_t divide_mask, uint64_t rows)
{
    RNS_Base base = in->base;
    const uint64_t mask = divide_mask & in->rns_mask, l = base->l, N = base->N;
    const uint64_t staying = rows & in->rns_mask & ~mask;
    if (mask == 0 || staying == 0)
        return;
    // Row i becomes (x_i - [x]_P lifted) P^-1 (+ the addend), and the lift's
    // factor (P/p_k) times P^-1 is p_k^-1: one combination of the row itself,
    // the prepared rows that leave and the addend's row.
    uint64_t factors[l * l];
    for (size_t i = 0; i < l; i++)
    {
        if (!(staying & (1ULL << i)))
            continue;
        Modulus mod = base->mods[i];
        RowTerm terms[MOD_COMBINATION_MAX_TERMS];
        uint64_t m = 0;
        const uint64_t p_inverse = inverse_mod(product_of_primes_mod(base, mask, l, mod), mod->q);
        terms[m++] = (RowTerm){in, i, p_inverse};
        for (size_t k = 0; k < l; k++)
        {
            factors[k * l + i] =
                (mask & (1ULL << k))
                    ? negate_modq(inverse_mod(modq(base->mods[k]->q, mod), mod->q), mod->q)
                    : 0;
            if (factors[k * l + i] && m < MOD_COMBINATION_MAX_TERMS)
                terms[m++] = (RowTerm){in, k, factors[k * l + i]};
        }
        if (addend && m < MOD_COMBINATION_MAX_TERMS)
            terms[m++] = (RowTerm){addend, i, 1};
        if (m == 1 + (uint64_t)__builtin_popcountll(mask) + (addend != NULL) &&
            combine_row(out, i, terms, m, true))
            continue;
        // Term by term, as the combination falls back to.
        RNSc_Polynomial lift = (RNSc_Polynomial)polynomial_new_RNS_polynomial(N, 1ULL << i, base);
        polynomial_RNSc_lifted_combination(lift, in, mask, factors, true);
        if (rns_row_is_narrow(base, i))
        {
            mod_eltwise_scale_w32(out->rows32[i], in->rows32[i], p_inverse, N, mod);
            mod_eltwise_add_w32(out->rows32[i], out->rows32[i], lift->rows32[i], N, mod);
            if (addend)
                mod_eltwise_add_w32(out->rows32[i], out->rows32[i], addend->rows32[i], N, mod);
        }
        else
        {
            mod_eltwise_scale(out->rows64[i], in->rows64[i], p_inverse, N, mod);
            mod_eltwise_add(out->rows64[i], out->rows64[i], lift->rows64[i], N, mod);
            if (addend)
                mod_eltwise_add(out->rows64[i], out->rows64[i], addend->rows64[i], N, mod);
        }
        free_RNS_polynomial((RNS_Polynomial)lift);
    }
}

void polynomial_round_division_RNSc_wo_free(RNSc_Polynomial out, uint64_t divide_mask)
{
    polynomial_round_division_RNSc_prepare(out, divide_mask);
    polynomial_round_division_RNSc_rows(out, divide_mask, out->rns_mask);
    polynomial_round_division_RNSc_finish(out, divide_mask);
}

RNS_Polynomial polynomial_RNS_view(struct _RNS_Polynomial *storage, RNS_Polynomial p, uint64_t mask)
{
    *storage = *p;
    storage->rns_mask = p->rns_mask & mask;
    return storage;
}

void polynomial_floor_division_RNSc(RNSc_Polynomial out)
{
    int last_active = rns_mask_get_last_active_index(out->rns_mask);
    if (last_active >= 0)
    {
        polynomial_floor_division_RNSc_wo_free(out, 1ULL << last_active);
    }
}

void polynomial_round_division_RNSc(RNSc_Polynomial out)
{
    int last_active = rns_mask_get_last_active_index(out->rns_mask);
    if (last_active >= 0)
    {
        polynomial_round_division_RNSc_wo_free(out, 1ULL << last_active);
    }
}

void polynomial_RNSc_permute_rows(RNSc_Polynomial out, RNSc_Polynomial in, uint64_t gen,
                                  uint64_t rows)
{
    assert(out != in);
    const uint64_t N = out->base->N, split_degree = out->base->split_degree;
    int split_degree_log = 0;
    while ((1ULL << split_degree_log) < split_degree)
        split_degree_log++;
    const uint64_t poly_size = out->base->N / out->base->split_degree;
    assert(gen < 2 * N);
    assert(gen > 0);
    assert((rows & ~in->rns_mask) == 0);

    int64_t *temp_signed = (int64_t *)safe_aligned_malloc(N * sizeof(int64_t));
    for (size_t j = 0; j < out->base->l; j++)
    {
        if (!(rows & (1ULL << j)))
            continue;
        Modulus mod = out->base->mods[j];
        if (rns_row_is_narrow(in->base, j))
            rns_row_permute_gather_narrow(temp_signed, in->rows32[j], gen, N, split_degree,
                                          split_degree_log, poly_size);
        else
            rns_row_permute_gather_wide(temp_signed, in->rows64[j], gen, N, split_degree,
                                        split_degree_log, poly_size);
        RNS_ROW_REDUCE_SIGNED(out, temp_signed, j, N, mod);
    }
    free(temp_signed);
}

void polynomial_RNSc_permute(RNSc_Polynomial out, RNSc_Polynomial in, uint64_t gen)
{
    assert(out != in);
    polynomial_RNS_zero((RNS_Polynomial)out);
    out->rns_mask = in->rns_mask;
    polynomial_RNSc_permute_rows(out, in, gen, in->rns_mask);
    for (size_t j = 0; j < out->base->l; j++)
    {
        if (!(out->rns_mask & (1ULL << j)))
            RNS_ROW_ZERO(out, j, out->base->N);
    }
}

// Position p holds P(psi^(2 brv(p) + 1)); P(X) -> P(X^gen) takes position p
// to the position holding that point raised to `gen`.
void polynomial_RNS_automorphism_index(uint32_t *idx, uint64_t N, uint64_t gen)
{
    assert(gen < 2 * N && (gen & 1));
    uint64_t log_n = 0;
    while ((1ULL << log_n) < N)
        log_n++;
    const uint64_t mask_2n = 2 * N - 1;
    for (uint64_t p = 0; p < N; p++)
    {
        const uint64_t point = (2 * arith_bit_reverse(p, log_n) + 1) * gen & mask_2n;
        idx[p] = (uint32_t)arith_bit_reverse(point >> 1, log_n);
    }
}

void polynomial_RNS_permute_rows(RNS_Polynomial out, RNS_Polynomial in, const uint32_t *idx,
                                 uint64_t rows)
{
    assert(out != in);
    assert(out->base == in->base && out->base->split_degree == 1);
    assert((rows & ~in->rns_mask) == 0);
    const uint64_t N = out->base->N;
    for (size_t j = 0; j < out->base->l; j++)
    {
        if (!(rows & (1ULL << j)))
            continue;
        if (rns_row_is_narrow(out->base, j))
            rns_row_gather_narrow(out->rows32[j], in->rows32[j], idx, N);
        else
            rns_row_gather_wide(out->rows64[j], in->rows64[j], idx, N);
    }
}

void polynomial_RNS_permute(RNS_Polynomial out, RNS_Polynomial in, const uint32_t *idx)
{
    out->rns_mask = in->rns_mask;
    polynomial_RNS_permute_rows(out, in, idx, in->rns_mask);
    for (size_t j = 0; j < out->base->l; j++)
    {
        if (!(out->rns_mask & (1ULL << j)))
            RNS_ROW_ZERO(out, j, out->base->N);
    }
}

// a * w mod q for a below 2^64 and w below q < 2^63, with wp = floor(w 2^64 / q).
static inline uint64_t shoup_mul_mod(uint64_t a, uint64_t w, uint64_t wp, uint64_t q)
{
    const uint64_t hi = (uint64_t)(((unsigned __int128)a * wp) >> 64);
    const uint64_t r = a * w - hi * q;
    return r >= q ? r - q : r;
}

static inline uint64_t shoup_constant(uint64_t w, uint64_t q)
{
    return (uint64_t)(((unsigned __int128)w << 64) / q);
}

// Garner: the mixed-radix digits a_i < q_i, x = a_0 + a_1 q_0 + a_2 q_0 q_1 +
// ..., of the value with residues x_i, as
// a_i = (((x_i - a_0) q_0^-1 - a_1) q_1^-1 - ...) mod q_i.
static void rns_mixed_radix_digits(uint64_t *a, const uint64_t *x, const uint64_t *q,
                                   const uint64_t *one_p, const uint64_t *inv,
                                   const uint64_t *inv_p, uint64_t L)
{
    for (uint64_t i = 0; i < L; i++)
    {
        uint64_t t = x[i];
        for (uint64_t j = 0; j < i; j++)
        {
            const uint64_t aj = shoup_mul_mod(a[j], 1, one_p[i], q[i]);
            t = t >= aj ? t - aj : t + q[i] - aj;
            t = shoup_mul_mod(t, inv[j * L + i], inv_p[j * L + i], q[i]);
        }
        a[i] = t;
    }
}

void polynomial_RNSc_to_centered_doubles(double *out, RNSc_Polynomial in, double scale)
{
    const uint64_t N = in->base->N;
    uint64_t L = 0, idx[64];
    for (size_t j = 0; j < in->base->l; j++)
        if (in->rns_mask & (1ULL << j))
            idx[L++] = j;

    // One prime: the residue is the value.
    if (L == 1)
    {
        const uint64_t q = in->base->mods[idx[0]]->q, half = (q - 1) / 2;
        const int narrow = rns_row_is_narrow(in->base, idx[0]);
        const uint32_t *row32 = in->rows32[idx[0]];
        const uint64_t *row64 = in->rows64[idx[0]];
        uint64_t k = 0;
#if VFHE_HAVE_AVX512IFMA
        // AVX-512DQ (on every IFMA CPU) converts 8 int64 to double at a time.
        const __m512i qv = _mm512_set1_epi64((long long)q);
        const __m512i hv = _mm512_set1_epi64((long long)half);
        const __m512d sv = _mm512_set1_pd(scale);
        for (; k + 8 <= N; k += 8)
        {
            __m512i x =
                narrow ? _mm512_cvtepu32_epi64(_mm256_loadu_si256((const __m256i *)(row32 + k)))
                       : _mm512_loadu_si512((const void *)(row64 + k));
            x = _mm512_mask_sub_epi64(x, _mm512_cmpgt_epu64_mask(x, hv), x, qv);
            _mm512_storeu_pd(out + k, _mm512_mul_pd(_mm512_cvtepi64_pd(x), sv));
        }
#endif
        for (; k < N; k++)
        {
            const uint64_t x = narrow ? row32[k] : row64[k];
            out[k] = (double)(x > half ? (int64_t)(x - q) : (int64_t)x) * scale;
        }
        return;
    }

    uint64_t *q = (uint64_t *)safe_malloc(L * sizeof(uint64_t));
    uint64_t *one_p = (uint64_t *)safe_malloc(L * sizeof(uint64_t));
    uint64_t *inv = (uint64_t *)safe_malloc(L * L * sizeof(uint64_t));
    uint64_t *inv_p = (uint64_t *)safe_malloc(L * L * sizeof(uint64_t));
    uint64_t *residues = (uint64_t *)safe_aligned_malloc(L * N * sizeof(uint64_t));
    uint64_t *x = (uint64_t *)safe_malloc(L * sizeof(uint64_t));
    uint64_t *a = (uint64_t *)safe_malloc(L * sizeof(uint64_t));
    uint64_t *half = (uint64_t *)safe_malloc(L * sizeof(uint64_t));
    for (uint64_t i = 0; i < L; i++)
    {
        q[i] = in->base->mods[idx[i]]->q;
        one_p[i] = shoup_constant(1, q[i]);
        for (uint64_t j = 0; j < i; j++)
        {
            inv[j * L + i] = inverse_mod(q[j] % q[i], q[i]);
            inv_p[j * L + i] = shoup_constant(inv[j * L + i], q[i]);
        }
        // Widen each row to 64 bits once, so the loop below has one type.
        if (rns_row_is_narrow(in->base, idx[i]))
            for (uint64_t k = 0; k < N; k++)
                residues[i * N + k] = in->rows32[idx[i]][k];
        else
            memcpy(&residues[i * N], in->rows64[idx[i]], N * sizeof(uint64_t));
    }
    // (Q - 1) / 2 is (q_i - 1) / 2 modulo each q_i; values above it are
    // negative.
    for (uint64_t i = 0; i < L; i++)
        x[i] = (q[i] - 1) / 2;
    rns_mixed_radix_digits(half, x, q, one_p, inv, inv_p, L);

    for (uint64_t k = 0; k < N; k++)
    {
        for (uint64_t i = 0; i < L; i++)
            x[i] = residues[i * N + k];
        rns_mixed_radix_digits(a, x, q, one_p, inv, inv_p, L);
        int negative = 0;
        for (uint64_t i = L; i-- > 0;)
        {
            if (a[i] != half[i])
            {
                negative = a[i] > half[i];
                break;
            }
        }
        // A negative value is -((Q - 1 - x) + 1); Q - 1 has digits q_i - 1, so
        // no borrows. Evaluating non-negative digits from the top keeps the
        // error relative to the value, not to Q.
        double v = 0.0;
        for (uint64_t i = L; i-- > 0;)
            v = v * (double)q[i] + (double)(negative ? q[i] - 1 - a[i] : a[i]);
        out[k] = (negative ? -(v + 1.0) : v) * scale;
    }
    free(q);
    free(one_p);
    free(inv);
    free(inv_p);
    free(residues);
    free(x);
    free(a);
    free(half);
}

void polynomial_int_permute_mod_Q(IntPolynomial out, IntPolynomial in, uint64_t gen)
{
    const uint64_t N = in->N;
    uint64_t idx = 0;
    for (size_t i = 0; i < N; i++)
    {
        out->coeffs[idx] = in->coeffs[i];
        idx = (idx + gen) % N;
    }
}

void polynomial_RNSc_mul_by_xai(RNSc_Polynomial out, RNSc_Polynomial in, uint64_t a)
{
    assert(in != out);
    assert(out->base->split_degree == 1);
    const uint64_t N = out->base->N;
    a &= ((N << 1) - 1);
    if (a == 0)
    {
        polynomial_copy_RNSc_polynomial(out, in);
        return;
    }
    out->rns_mask = in->rns_mask;
    for (size_t j = 0; j < out->base->l; j++)
    {
        if (!(out->rns_mask & (1ULL << j)))
            continue;
        Modulus mod = in->base->mods[j];
        if (rns_row_is_narrow(out->base, j))
            rns_row_mul_by_xai_narrow(out->rows32[j], in->rows32[j], a, N, mod);
        else
            rns_row_mul_by_xai_wide(out->rows64[j], in->rows64[j], a, N, mod);
    }
}

void polynomial_RNSc_mul_by_xai_minus1(RNSc_Polynomial out, RNSc_Polynomial in, uint64_t a)
{
    assert(in != out);
    assert(out->base->split_degree == 1);
    const uint64_t N = out->base->N;
    a &= ((N << 1) - 1);
    out->rns_mask = in->rns_mask;
    if (a == 0)
    {
        // x^0 - 1 == 0
        for (size_t j = 0; j < out->base->l; j++)
        {
            RNS_ROW_ZERO(out, j, N);
        }
        return;
    }
    for (size_t j = 0; j < out->base->l; j++)
    {
        if (!(out->rns_mask & (1ULL << j)))
            continue;
        Modulus mod = in->base->mods[j];
        if (rns_row_is_narrow(out->base, j))
            rns_row_mul_by_xai_minus1_narrow(out->rows32[j], in->rows32[j], a, N, mod);
        else
            rns_row_mul_by_xai_minus1_wide(out->rows64[j], in->rows64[j], a, N, mod);
    }
}

void polynomial_int_decompose_i(IntPolynomial out, IntPolynomial in, uint64_t Bg_bit, uint64_t l,
                                uint64_t q, uint64_t bit_size, uint64_t i)
{
    const uint64_t N = in->N;
    const uint64_t h_mask = (1UL << Bg_bit) - 1;
    const uint64_t h_bit = bit_size - (i + 1) * Bg_bit;
    uint64_t offset = 1ULL << (bit_size - l * Bg_bit - 1);
    for (size_t c = 0; c < N; c++)
    {
        const uint64_t coeff_off = in->coeffs[c] + offset;
        out->coeffs[c] = (coeff_off >> h_bit) & h_mask;
    }
}

IntPolynomial polynomial_new_int_polynomial(uint64_t N)
{
    IntPolynomial res;
    res = (IntPolynomial)safe_malloc(sizeof(*res));
    res->coeffs = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * N);
    res->N = N;
    return res;
}

IntPolynomial *polynomial_new_int_polynomial_array(uint64_t size, uint64_t N)
{
    IntPolynomial *res = (IntPolynomial *)safe_malloc(sizeof(IntPolynomial) * size);
    for (size_t i = 0; i < size; i++)
    {
        res[i] = polynomial_new_int_polynomial(N);
    }
    return res;
}

void free_polynomial(void *p)
{
    free(((IntPolynomial)p)->coeffs);
    free(p);
}

void free_polynomial_array(uint64_t size, IntPolynomial *p)
{
    for (size_t i = 0; i < size; i++)
    {
        free_polynomial(p[i]);
    }
    free(p);
}

void polynomial_RNS_broadcast_slot(RNS_Polynomial out, RNS_Polynomial in, uint64_t slot_idx)
{
    const uint64_t poly_size = out->base->N / out->base->split_degree;
    out->rns_mask = in->rns_mask;
    for (size_t i = 0; i < in->base->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            if (rns_row_is_narrow(out->base, i))
                rns_row_broadcast_slot_narrow(out->rows32[i], in->rows32[i], slot_idx,
                                              out->base->split_degree, poly_size);
            else
                rns_row_broadcast_slot_wide(out->rows64[i], in->rows64[i], slot_idx,
                                            out->base->split_degree, poly_size);
        }
    }
}

void polynomial_RNS_broadcast_RNS_comp(RNS_Polynomial out, RNS_Polynomial in, uint64_t rns_comp)
{
    out->rns_mask = in->rns_mask;
    for (size_t i = 0; i < in->base->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            /* Row `rns_comp` copied verbatim into row i. Both belong to the
               same base, so if their primes differ in width this is a
               cross-modulus move and the values must be reduced -- which is
               what makes it a reduce rather than a copy. */
            RNS_ROW_REDUCE(out, in, i, rns_comp, out->base->N, out->base->mods[i]);
        }
    }
}

void polynomial_RNS_rotate_slot(RNS_Polynomial out, RNS_Polynomial in, uint64_t rot)
{
    assert(out != in);
    const uint64_t poly_size = out->base->N / out->base->split_degree;
    out->rns_mask = in->rns_mask;
    for (size_t i = 0; i < in->base->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            if (rns_row_is_narrow(out->base, i))
                rns_row_rotate_slot_narrow(out->rows32[i], in->rows32[i], rot,
                                           out->base->split_degree, poly_size);
            else
                rns_row_rotate_slot_wide(out->rows64[i], in->rows64[i], rot,
                                         out->base->split_degree, poly_size);
        }
    }
}

void polynomial_RNS_copy_slot(RNS_Polynomial out, uint64_t dst, RNS_Polynomial in, uint64_t src)
{
    const uint64_t poly_size = out->base->N / out->base->split_degree;
    out->rns_mask = in->rns_mask;
    for (size_t i = 0; i < in->base->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            if (rns_row_is_narrow(out->base, i))
                rns_row_copy_slot_narrow(out->rows32[i], in->rows32[i], dst, src,
                                         out->base->split_degree, poly_size);
            else
                rns_row_copy_slot_wide(out->rows64[i], in->rows64[i], dst, src,
                                       out->base->split_degree, poly_size);
        }
    }
}

/* The tables of rns_crt_maps_new, in the notation of
   polynomial_RNS_crt_components (arith.h): p is the prime of base row `row`,
   d the split degree, k = N/d, w_s = base->w[row][s], P_s = (p, X^d - w_s)
   and R_q/P_s = F_p[x]/(x^d - w_s).

   The base stores w_s = psi^(x_s), with psi a primitive 2k-th root of unity
   mod p and x_s = brv(s) + 1, brv the reversal of log2(2k) bits; so
   w_0 = psi, and x_s is odd. The isomorphism R_q/P_s -> R_q/P_0 induced by
   X -> X^(x_s) sends x^j to x^(j x_s) = w_0^floor(j x_s / d) x^(j x_s mod d),
   as x^d = w_0 in R_q/P_0. Its inverse is induced by X -> X^(y_s),
   y_s = x_s^-1 mod 2N, and sends x^j to w_s^floor(j y_s / d) x^(j y_s mod d),
   as x^d = w_s in R_q/P_s.

   Let e_s be x_s if `to_slot_zero` and y_s otherwise. The map then sends the
   coefficient of x^j to the coefficient of x^dest_s(j), dest_s(j) = j e_s mod
   d, multiplied by factor[j * k + s] (w_0^floor(j x_s / d), or
   w_s^floor(j y_s / d)); e_s is odd and d a power of two, so dest_s is a
   permutation of [0, d). slot_dest[j * k + s] is dest_s(j). As dest_s
   depends on s only through e_s mod d, it is constant on runs of consecutive
   slots: slots run_start[r] .. run_start[r + 1] - 1 form run r, and
   run_dest[r * d + j] is dest_s(j) on it. `factor32` is `factor` narrowed
   for a narrow prime, and NULL otherwise. */
struct _RNS_CrtMaps
{
    RNS_Base base;
    uint64_t row;
    bool to_slot_zero;
    uint64_t runs;
    uint64_t *run_start;
    uint64_t *run_dest;
    uint8_t *slot_dest;
    uint64_t *factor;
    uint32_t *factor32;
};

/* a^-1 modulo 2^m, `pow2` = 2^m, for odd a: Newton's iteration doubles the
   number of correct low bits from the 3 that a^-1 = a gives modulo 8. */
static uint64_t inverse_odd_mod_pow2(uint64_t a, uint64_t pow2)
{
    uint64_t inv = a;
    for (int i = 0; i < 5; i++)
        inv *= 2 - a * inv;
    return inv & (pow2 - 1);
}

RNS_CrtMaps rns_crt_maps_new(RNS_Base base, uint64_t row, bool to_slot_zero)
{
    const uint64_t d = base->split_degree, k = base->N / d, two_n = 2 * base->N;
    const Modulus mod = base->mods[row];
    const uint64_t log_2k = (uint64_t)__builtin_ctzll(2 * k);
    assert(d <= 256); // slot_dest holds a coefficient index in a byte

    /* psi^m for m < 2k: every factor is one of these. */
    uint64_t *psi_pow = (uint64_t *)safe_malloc(2 * k * sizeof(uint64_t));
    psi_pow[0] = 1;
    for (uint64_t m = 1; m < 2 * k; m++)
        psi_pow[m] = mul_modq(psi_pow[m - 1], base->w[row][0], mod);

    RNS_CrtMaps maps = (RNS_CrtMaps)safe_malloc(sizeof(*maps));
    maps->base = base;
    maps->row = row;
    maps->to_slot_zero = to_slot_zero;
    maps->factor = (uint64_t *)safe_aligned_malloc(d * k * sizeof(uint64_t));
    maps->run_start = (uint64_t *)safe_malloc((k + 1) * sizeof(uint64_t));
    maps->run_dest = (uint64_t *)safe_malloc(k * d * sizeof(uint64_t));
    maps->slot_dest = (uint8_t *)safe_malloc(d * k);
    maps->runs = 0;
    uint64_t previous = UINT64_MAX;
    for (uint64_t s = 0; s < k; s++)
    {
        const uint64_t x = (int_rev((uint32_t)s) >> (32 - log_2k)) + 1;
        const uint64_t e = to_slot_zero ? x : inverse_odd_mod_pow2(x, two_n);
        if (e % d != previous)
        {
            previous = e % d;
            maps->run_start[maps->runs] = s;
            for (uint64_t j = 0; j < d; j++)
                maps->run_dest[maps->runs * d + j] = j * e % d;
            maps->runs++;
        }
        for (uint64_t j = 0; j < d; j++)
        {
            /* w_0 = psi and w_s = psi^(x_s), so both factors are powers of psi. */
            const uint64_t exponent = to_slot_zero ? j * e / d : (j * e / d) * x;
            maps->factor[j * k + s] = psi_pow[exponent & (2 * k - 1)];
            maps->slot_dest[j * k + s] = (uint8_t)(j * e % d);
        }
    }
    maps->run_start[maps->runs] = k;
    maps->factor32 = NULL;
    if (rns_row_is_narrow(base, row))
    {
        maps->factor32 = (uint32_t *)safe_aligned_malloc(d * k * sizeof(uint32_t));
        mod_narrow_w32(maps->factor32, maps->factor, d * k);
    }
    free(psi_pow);
    return maps;
}

void rns_crt_maps_free(RNS_CrtMaps maps)
{
    free(maps->run_start);
    free(maps->run_dest);
    free(maps->slot_dest);
    free(maps->factor);
    free(maps->factor32);
    free(maps);
}

/* The two kernels below work on tiles of CRT_TILE_ELEMENTS elements by
   CRT_TILE_SLOTS slots, one coefficient plane at a time. In the vector of
   components, one slot of a tile is CRT_TILE_ELEMENTS consecutive words
   (eight cache lines in one page); in a row, one element of a tile is
   CRT_TILE_SLOTS consecutive words, a whole number of lane groups for the
   element-wise product. A tile is 128 KiB at 64-bit words, sized to stay in
   L2; like FIELD_VEC_MUL_TILE, the sizes are properties of the machine's
   caches rather than of the algorithm. */
#define CRT_TILE_ELEMENTS 64
#define CRT_TILE_SLOTS 256

void polynomial_RNS_crt_components(FieldVector out, RNS_Polynomial *in, uint64_t count,
                                   RNS_CrtMaps maps, bool element_major)
{
    assert(maps->to_slot_zero);
    if (count == 0)
        return;
    const RNS_Base base = maps->base;
    const uint64_t row = maps->row;
    const uint64_t d = base->split_degree, k = base->N / d;
    const uint64_t slots = k < CRT_TILE_SLOTS ? k : CRT_TILE_SLOTS;
    const Modulus mod = base->mods[row];
    assert(out->d == d && out->n == count * k);
    assert(out->mod->q == mod->q && out->w == base->w[row][0]);
    for (uint64_t b = 0; b < count; b++)
        assert(in[b]->base == base && ((in[b]->rns_mask >> row) & 1));
    const bool narrow = rns_row_is_narrow(base, row);
    if (element_major)
    {
        /* Entry b * k + s: the components of one element are contiguous, and
           a run of slots sharing a destination is a contiguous run of that
           element's row plane and of the output plane. Each run is one
           product of the row by its factors, written in place; a narrow row
           is multiplied at its width and widened. */
        uint32_t *scratch32 = narrow ? (uint32_t *)safe_aligned_malloc(k * sizeof(uint32_t)) : NULL;
        for (uint64_t b = 0; b < count; b++)
        {
            for (uint64_t j = 0; j < d; j++)
            {
                for (uint64_t r = 0; r < maps->runs; r++)
                {
                    const uint64_t lo = maps->run_start[r], len = maps->run_start[r + 1] - lo;
                    const uint64_t at = j * k + lo;
                    uint64_t *dst = out->coeffs[maps->run_dest[r * d + j]] + b * k + lo;
                    if (narrow)
                    {
                        mod_eltwise_mul_w32(scratch32, rns_row32(in[b], row) + at,
                                            maps->factor32 + at, len, mod);
                        mod_widen_w32(dst, scratch32, len);
                    }
                    else
                        mod_eltwise_mul(dst, rns_row64(in[b], row) + at, maps->factor + at, len,
                                        mod);
                }
            }
        }
        free(scratch32);
        return;
    }
    /* For the current s0 and j, row t of the tile holds the coefficients of
       x^j in slots s0 .. s0 + slots - 1 of element b0 + t, each times its
       factor, at the row's width. */
    void *tile = safe_aligned_malloc(CRT_TILE_ELEMENTS * slots *
                                     (narrow ? sizeof(uint32_t) : sizeof(uint64_t)));
    uint32_t *tile32 = (uint32_t *)tile;
    uint64_t *tile64 = (uint64_t *)tile;
    for (uint64_t b0 = 0; b0 < count; b0 += CRT_TILE_ELEMENTS)
    {
        const uint64_t nb = count - b0 < CRT_TILE_ELEMENTS ? count - b0 : CRT_TILE_ELEMENTS;
        for (uint64_t s0 = 0; s0 < k; s0 += slots)
        {
            for (uint64_t j = 0; j < d; j++)
            {
                const uint64_t at = j * k + s0;
                for (uint64_t t = 0; t < nb; t++)
                {
                    if (narrow)
                        mod_eltwise_mul_w32(tile32 + t * slots, rns_row32(in[b0 + t], row) + at,
                                            maps->factor32 + at, slots, mod);
                    else
                        mod_eltwise_mul(tile64 + t * slots, rns_row64(in[b0 + t], row) + at,
                                        maps->factor + at, slots, mod);
                }
                /* Tile entry (t, s - s0) is the coefficient of x^dest_s(j) in
                   component s of element b0 + t, which is entry
                   s * count + b0 + t of plane dest_s(j): column s - s0 of the
                   tile becomes nb consecutive entries. */
                for (uint64_t i = 0; i < slots; i++)
                {
                    const uint64_t s = s0 + i;
                    uint64_t *dst = out->coeffs[maps->slot_dest[j * k + s]] + s * count + b0;
                    if (narrow)
                        for (uint64_t t = 0; t < nb; t++)
                            dst[t] = tile32[t * slots + i];
                    else
                        for (uint64_t t = 0; t < nb; t++)
                            dst[t] = tile64[t * slots + i];
                }
            }
        }
    }
    free(tile);
}

void polynomial_RNS_from_crt_components(RNS_Polynomial *out, FieldVector in, uint64_t count,
                                        RNS_CrtMaps maps)
{
    assert(!maps->to_slot_zero);
    if (count == 0)
        return;
    const RNS_Base base = maps->base;
    const uint64_t row = maps->row;
    const uint64_t d = base->split_degree, k = base->N / d;
    const uint64_t slots = k < CRT_TILE_SLOTS ? k : CRT_TILE_SLOTS;
    const Modulus mod = base->mods[row];
    assert(in->d == d && in->n == count * k);
    assert(in->mod->q == mod->q && in->w == base->w[row][0]);
    for (uint64_t b = 0; b < count; b++)
        assert(out[b]->base == base);
    const bool narrow = rns_row_is_narrow(base, row);
    void *tile = safe_aligned_malloc(CRT_TILE_ELEMENTS * slots *
                                     (narrow ? sizeof(uint32_t) : sizeof(uint64_t)));
    uint32_t *tile32 = (uint32_t *)tile;
    uint64_t *tile64 = (uint64_t *)tile;
    for (uint64_t b0 = 0; b0 < count; b0 += CRT_TILE_ELEMENTS)
    {
        const uint64_t nb = count - b0 < CRT_TILE_ELEMENTS ? count - b0 : CRT_TILE_ELEMENTS;
        for (uint64_t s0 = 0; s0 < k; s0 += slots)
        {
            for (uint64_t j = 0; j < d; j++)
            {
                /* The coefficients of x^j in components s0 .. s0 + slots - 1
                   of elements b0 .. b0 + nb - 1: entries s * count + b0 ..
                   s * count + b0 + nb - 1 of plane j become column s - s0 of
                   the tile. The values are below p, so a narrow tile holds
                   them. */
                for (uint64_t i = 0; i < slots; i++)
                {
                    const uint64_t *src = in->coeffs[j] + (s0 + i) * count + b0;
                    if (narrow)
                        for (uint64_t t = 0; t < nb; t++)
                            tile32[t * slots + i] = (uint32_t)src[t];
                    else
                        for (uint64_t t = 0; t < nb; t++)
                            tile64[t * slots + i] = src[t];
                }
                /* Times the factors, tile entry (t, s - s0) is the
                   coefficient of x^dest_s(j) in slot s of element b0 + t. */
                for (uint64_t t = 0; t < nb; t++)
                {
                    if (narrow)
                        mod_eltwise_mul_w32(tile32 + t * slots, tile32 + t * slots,
                                            maps->factor32 + j * k + s0, slots, mod);
                    else
                        mod_eltwise_mul(tile64 + t * slots, tile64 + t * slots,
                                        maps->factor + j * k + s0, slots, mod);
                }
                for (uint64_t r = 0; r < maps->runs; r++)
                {
                    const uint64_t lo = maps->run_start[r] > s0 ? maps->run_start[r] : s0;
                    const uint64_t hi =
                        maps->run_start[r + 1] < s0 + slots ? maps->run_start[r + 1] : s0 + slots;
                    if (lo >= hi)
                        continue;
                    const uint64_t at = maps->run_dest[r * d + j] * k + lo;
                    for (uint64_t t = 0; t < nb; t++)
                    {
                        if (narrow)
                            memcpy(rns_row32(out[b0 + t], row) + at, tile32 + t * slots + lo - s0,
                                   (hi - lo) * sizeof(uint32_t));
                        else
                            memcpy(rns_row64(out[b0 + t], row) + at, tile64 + t * slots + lo - s0,
                                   (hi - lo) * sizeof(uint64_t));
                    }
                }
            }
        }
    }
    free(tile);
}

int polynomial_RNS_inverse_generic(RNS_Polynomial out, RNS_Polynomial in)
{
    const uint64_t N = in->base->N;
    const uint64_t d = in->base->split_degree;
    const uint64_t poly_size = N / d;
    out->rns_mask = in->rns_mask;
    assert(out->base->N == N);

    uint64_t *prefix = (uint64_t *)safe_aligned_malloc(poly_size * d * sizeof(uint64_t));
    uint64_t *B = (uint64_t *)safe_aligned_malloc(poly_size * d * sizeof(uint64_t));
    uint64_t *T = (uint64_t *)safe_aligned_malloc(d * sizeof(uint64_t));
    uint64_t *tmp_field = (uint64_t *)safe_aligned_malloc(d * sizeof(uint64_t));

    int status = 0;

    for (size_t i = 0; i < in->base->l; i++)
    {
        if (in->rns_mask & (1ULL << i))
        {
            Modulus mod = in->base->mods[i];
            const uint64_t *w_i = in->base->w[i];
            uint64_t w0 = w_i[0];
            /* One width test for the row; the gathers and scatters below are a
               slot's column across the split blocks, which is one element of
               the degree-d extension. */
            const bool narrow = rns_row_is_narrow(in->base, i);

            for (size_t s = 0; s < poly_size; s++)
            {
                uint64_t A_s[d];
                if (narrow)
                    rns_row_gather_slot_column_narrow(A_s, in->rows32[i], s, d, poly_size);
                else
                    rns_row_gather_slot_column_wide(A_s, in->rows64[i], s, d, poly_size);
                field_base_conversion(&B[s * d], A_s, s, 0, d, poly_size, w_i, mod);
            }

            for (size_t j = 0; j < d; j++)
            {
                prefix[j] = B[j];
            }
            for (size_t s = 1; s < poly_size; s++)
            {
                field_ext_mul(&prefix[s * d], &prefix[(s - 1) * d], &B[s * d], d, w0, mod);
            }

            if (!field_ext_inv(T, &prefix[(poly_size - 1) * d], d, w0, mod))
            {
                status = -2;
                break;
            }

            for (size_t s = poly_size - 1; s > 0; s--)
            {
                uint64_t O_s[d];
                field_ext_mul(O_s, T, &prefix[(s - 1) * d], d, w0, mod);

                uint64_t A_inv_s[d];
                field_base_conversion(A_inv_s, O_s, 0, s, d, poly_size, w_i, mod);
                if (narrow)
                    rns_row_scatter_slot_column_narrow(out->rows32[i], A_inv_s, s, d, poly_size);
                else
                    rns_row_scatter_slot_column_wide(out->rows64[i], A_inv_s, s, d, poly_size);

                field_ext_mul(tmp_field, T, &B[s * d], d, w0, mod);
                memcpy(T, tmp_field, d * sizeof(uint64_t));
            }

            uint64_t A_inv_0[d];
            field_base_conversion(A_inv_0, T, 0, 0, d, poly_size, w_i, mod);
            if (narrow)
                rns_row_scatter_slot_column_narrow(out->rows32[i], A_inv_0, 0, d, poly_size);
            else
                rns_row_scatter_slot_column_wide(out->rows64[i], A_inv_0, 0, d, poly_size);
        }
    }

    free(prefix);
    free(B);
    free(T);
    free(tmp_field);

    return status;
}

int polynomial_RNS_inverse(RNS_Polynomial out, RNS_Polynomial in)
{
    if (in->base->split_degree != 1)
        return polynomial_RNS_inverse_generic(out, in);
    const uint64_t N = in->base->N;
    out->rns_mask = in->rns_mask;
    assert(out->base->N == N);

    uint64_t *prefix = (uint64_t *)safe_aligned_malloc(N * sizeof(uint64_t));

    for (size_t i = 0; i < in->base->l; i++)
    {
        if (in->rns_mask & (1ULL << i))
        {
            Modulus mod = in->base->mods[i];
            int rc;
            if (rns_row_is_narrow(out->base, i))
                rc = rns_row_inverse_narrow(out->rows32[i], in->rows32[i], prefix, N, mod);
            else
                rc = rns_row_inverse_wide(out->rows64[i], in->rows64[i], prefix, N, mod);
            if (rc != 0)
            {
                free(prefix);
                return rc;
            }
        }
    }

    free(prefix);
    return 0;
}

int rns_mask_get_active_index(uint64_t mask, uint64_t i)
{
    uint64_t count = 0;
    for (int idx = 0; idx < 64; idx++)
    {
        if (mask & (1ULL << idx))
        {
            if (count == i)
            {
                return idx;
            }
            count++;
        }
    }
    return -1;
}

int rns_mask_get_last_active_index(uint64_t mask)
{
    for (int idx = 63; idx >= 0; idx--)
    {
        if (mask & (1ULL << idx))
        {
            return idx;
        }
    }
    return -1;
}

void rns_compute_scaling_factors(uint64_t *delta_out, RNS_Base base, uint64_t in_mask,
                                 uint64_t out_mask)
{
    uint64_t diff_mask = out_mask & ~in_mask;
    for (size_t i = 0; i < base->l; i++)
    {
        if (out_mask & (1ULL << i))
        {
            Modulus mod_i = base->mods[i];
            if (in_mask & (1ULL << i))
            {
                uint64_t delta_i = 1;
                for (size_t j = 0; j < base->l; j++)
                {
                    if (diff_mask & (1ULL << j))
                    {
                        uint64_t p_j = base->mods[j]->q;
                        delta_i = mul_modq(delta_i, modq(p_j, mod_i), mod_i);
                    }
                }
                delta_out[i] = delta_i;
            }
            else
            {
                delta_out[i] = 0;
            }
        }
        else
        {
            delta_out[i] = 0;
        }
    }
}

void polynomial_RNSc_scaled_lift(RNSc_Polynomial out, RNSc_Polynomial in, uint64_t *delta)
{
    assert(in->base->N == out->base->N);
    const uint64_t N = out->base->N;

    uint64_t local_delta[64];
    uint64_t *actual_delta = delta;
    if (actual_delta == NULL)
    {
        rns_compute_scaling_factors(local_delta, out->base, in->rns_mask, out->rns_mask);
        actual_delta = local_delta;
    }

    for (size_t i = 0; i < out->base->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            Modulus mod_i = out->base->mods[i];
            if (in->rns_mask & (1ULL << i))
            {
                RNS_ROW_SCALAROP(mod_eltwise_scale, out, in, actual_delta[i], i, N, mod_i);
            }
            else
            {
                RNS_ROW_ZERO(out, i, N);
            }
        }
    }
}
