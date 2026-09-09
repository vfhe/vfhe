// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "arith.h"
#include "util.h"
#include <crypto.h>
#include <blake3.h>

// Row width, the row accessors and the per-row operation macros.
#include "arith_internal.h"

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
            res->rows32[i] = (uint32_t *)safe_aligned_malloc(sizeof(uint32_t) * base->N);
        }
        else
        {
            res->rows64[i] = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * base->N);
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
    // One of the two is NULL for each row, and free(NULL) is a no-op, so this
    // needs no width test and cannot go wrong if the base grew meanwhile.
    for (size_t i = 0; i < pp->allocated_l; i++)
    {
        free(pp->rows64[i]);
        free(pp->rows32[i]);
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
    const uint64_t modMask = out->base->split_degree - 1,
                   poly_size = out->base->N / out->base->split_degree;
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
    const uint64_t modMask = out->base->split_degree - 1,
                   poly_size = out->base->N / out->base->split_degree;
    uint64_t *temp = (uint64_t *)safe_aligned_malloc(out->base->N * sizeof(uint64_t));
    for (size_t i = 0; i < out->base->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            Modulus mod = out->base->mods[i];
            mod_eltwise_reduce_signed(temp, (int64_t *)in, out->base->N, mod);
            RNS_ROW_SCATTER_SPLIT(out, temp, i, out->base->N, out->base->split_degree, poly_size);
        }
    }
    free(temp);
    polynomial_RNSc_to_RNS(out, (RNSc_Polynomial)out);
}

void array_to_RNS(RNS_Polynomial out, uint64_t **in)
{
    const uint64_t modMask = out->base->split_degree - 1,
                   poly_size = out->base->N / out->base->split_degree;
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
    // The sampler and the mod switch both work in 64 bits, so a narrow row is
    // produced in a wide view and narrowed after.
    /* The sampler wants 64 bits of entropy per coefficient and the mod switch
       is defined on that, so a narrow row is produced wide and stored narrow
       in one vectorized pass -- narrowing the sampler instead would change the
       distribution. */
    uint64_t *scratch = (uint64_t *)safe_aligned_malloc(out->base->N * sizeof(uint64_t));
    for (size_t i = 0; i < out->base->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            const uint64_t p = out->base->mods[i]->q;
            const bool narrow = rns_row_is_narrow(out->base, i);
            uint64_t *dst = narrow ? scratch : out->rows64[i];
            generate_random_bytes(sizeof(uint64_t) * out->base->N, (uint8_t *)dst);
            array_mod_switch_from_2k(dst, dst, p, p, out->base->N);
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

RNS_BaseConversionParams init_base_conversion_params(RNS_Base base, uint64_t in_mask,
                                                     uint64_t out_mask)
{
    RNS_BaseConversionParams params = (RNS_BaseConversionParams)safe_malloc(sizeof(*params));
    params->in_mask = in_mask;
    params->out_mask = out_mask;

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

    return params;
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

void polynomial_RNSc_decompose_small(RNSc_Polynomial out, RNSc_Polynomial in, uint64_t log_base,
                                     uint64_t level)
{
    uint64_t *tmp = (uint64_t *)safe_aligned_malloc(out->base->N * sizeof(uint64_t));
    const uint64_t mask = (1ULL << log_base) - 1;
    const uint64_t shift = log_base * level;
    int last_active = rns_mask_get_last_active_index(in->rns_mask);
    assert(last_active >= 0);
    if (rns_row_is_narrow(in->base, (size_t)last_active))
        rns_row_digit_narrow(tmp, in->rows32[last_active], shift, mask, out->base->N);
    else
        rns_row_digit_wide(tmp, in->rows64[last_active], shift, mask, out->base->N);
    /* The digit is below 2^log_base and so below every prime, which is why one
       array serves every row: no reduction, just a store at the row's width. */
    for (size_t i = 0; i < out->base->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            if (rns_row_is_narrow(out->base, i))
                mod_narrow_w32(out->rows32[i], tmp, out->base->N);
            else
                memcpy(out->rows64[i], tmp, sizeof(uint64_t) * out->base->N);
        }
    }
    free(tmp);
}

void polynomial_RNSc_to_multiprecision(RNSc_Polynomial out, RNSc_Polynomial in, uint64_t log_base,
                                       uint64_t level)
{
    polynomial_RNSc_decompose_small(out, in, log_base, level);
}

void polynomial_RNS_get_hash(uint64_t *out, RNS_Polynomial p)
{
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    for (size_t i = 0; i < p->base->l; i++)
    {
        if (p->rns_mask & (1ULL << i))
        {
            /* Eight bytes per coefficient whatever the row's width, so the
               digest is a function of the element and not of how it is
               stored -- the canonical-domain contract. */
            if (rns_row_is_narrow(p->base, i))
            {
                for (size_t j = 0; j < p->base->N; j++)
                {
                    const uint64_t v = p->rows32[i][j];
                    blake3_hasher_update(&hasher, &v, sizeof(uint64_t));
                }
            }
            else
            {
                blake3_hasher_update(&hasher, p->rows64[i], p->base->N * sizeof(uint64_t));
            }
        }
    }
    blake3_hasher_finalize(&hasher, (uint8_t *)out, BLAKE3_OUT_LEN);
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

void polynomial_round_division_RNSc_wo_free(RNSc_Polynomial out, uint64_t divide_mask)
{
    uint64_t mask = divide_mask & out->rns_mask;
    if (mask == 0)
        return;

    const uint64_t N = out->base->N;
    uint64_t *tmp = (uint64_t *)safe_aligned_malloc(N * sizeof(uint64_t));
    // as in the floor case, only the cross-modulus reduce can change width
    uint32_t *tmp32 = (uint32_t *)safe_aligned_malloc(N * sizeof(uint32_t));

    for (size_t idx = 0; idx < out->base->l; idx++)
    {
        if (mask & (1ULL << idx))
        {
            const uint64_t p = out->base->mods[idx]->q, half_p = p / 2;
            RNS_ROW_SCALAROP(mod_eltwise_add_scalar, out, out, half_p, idx, N,
                             out->base->mods[idx]);
            for (size_t i = 0; i < out->base->l; i++)
            {
                if (out->rns_mask & (1ULL << i))
                {
                    if (i == idx)
                        continue;
                    const uint64_t q = out->base->mods[i]->q;
                    const uint64_t inv_p = inverse_mod(p, q);
                    const uint64_t half_p_mod_q = half_p % q;
                    Modulus mod_i = out->base->mods[i];
                    if (rns_row_is_narrow(out->base, i))
                    {
                        rns_reduce_row_to32(tmp32, (RNS_Polynomial)out, idx, N, mod_i);
                        mod_eltwise_add_scalar_w32(out->rows32[i], out->rows32[i], half_p_mod_q, N,
                                                   mod_i);
                        mod_eltwise_sub_w32(out->rows32[i], out->rows32[i], tmp32, N, mod_i);
                        mod_eltwise_scale_w32(out->rows32[i], out->rows32[i], inv_p, N, mod_i);
                    }
                    else
                    {
                        rns_reduce_row_to64(tmp, (RNS_Polynomial)out, idx, N, mod_i);
                        mod_eltwise_add_scalar(out->rows64[i], out->rows64[i], half_p_mod_q, N,
                                               mod_i);
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

void polynomial_RNSc_permute(RNSc_Polynomial out, RNSc_Polynomial in, uint64_t gen)
{
    assert(out != in);
    const uint64_t N = out->base->N, split_degree = out->base->split_degree;
    int split_degree_log = 0;
    while ((1ULL << split_degree_log) < split_degree)
        split_degree_log++;
    const uint64_t poly_size = out->base->N / out->base->split_degree;
    assert(gen < 2 * N);
    assert(gen > 0);
    polynomial_RNS_zero((RNS_Polynomial)out);

    int64_t *temp_signed = (int64_t *)safe_aligned_malloc(N * sizeof(int64_t));
    out->rns_mask = in->rns_mask;
    for (size_t j = 0; j < out->base->l; j++)
    {
        if (out->rns_mask & (1ULL << j))
        {
            Modulus mod = out->base->mods[j];
            if (rns_row_is_narrow(in->base, j))
                rns_row_permute_gather_narrow(temp_signed, in->rows32[j], gen, N, split_degree,
                                              split_degree_log, poly_size);
            else
                rns_row_permute_gather_wide(temp_signed, in->rows64[j], gen, N, split_degree,
                                            split_degree_log, poly_size);
            RNS_ROW_REDUCE_SIGNED(out, temp_signed, j, N, mod);
        }
        else
        {
            RNS_ROW_ZERO(out, j, N);
        }
    }
    free(temp_signed);
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
            const uint64_t q = in->base->mods[i]->q;
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
