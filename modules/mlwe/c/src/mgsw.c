// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
#include <assert.h>
#include <string.h>

#include "gadget_rns.h"
#include "mlwe.h"
#include "rns_rows.h"
#include "alloc.h"

// The external product, left canonical in `out`.
static void external_product_canonical(RNSc_MLWE out, RNS_MLWE *mgsw, RNSc_MLWE in, uint64_t ell,
                                       const GadgetParams *gadget_params)
{
    const uint64_t r = in->r;

    // The products accumulate in the ring the MGSW key lives in, which is
    // wider than `out`'s: `out` is only allocated for its own ring. The
    // rescale afterwards brings the result back, and which primes leave
    // follows from the two rings rather than from counting special ones.
    RNS_MLWE acc = mlwe_alloc_sample(mgsw[0]->ring, out->r);
    mlwe_RNS_trivial_sample_of_zero(acc);

    for (size_t j = 0; j < r; j++)
    {
        gadget_mul_addto_polynomial(acc, &mgsw[j * ell], &in->a[j], gadget_params);
    }
    gadget_mul_addto_polynomial(acc, &mgsw[r * ell], &in->b, gadget_params);

    mlwe_RNS_to_RNSc(acc, acc);
    mlwe_mod_down(out, acc, NULL);
    free_mlwe_RNS_sample(acc);
}

void mgsw_external_product(RNS_MLWE out, RNS_MLWE *mgsw, RNSc_MLWE in, uint64_t ell,
                           uint64_t special_primes, const GadgetParams *gadget_params)
{
    (void)special_primes;
    external_product_canonical(out, mgsw, in, ell, gadget_params);
    mlwe_RNSc_to_RNS(out, out);
}

void mgsw_CMUX(RNS_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2, RNS_MLWE *mgsw, uint64_t ell,
               uint64_t special_primes, const GadgetParams *gadget_params)
{
    const uint64_t r = in1->r;
    ArithRing ring = in1->ring;

    RNSc_MLWE diff = mlwe_alloc_sample(ring, r);
    mlwe_sub_RNSc_sample(diff, in2, in1);

    mgsw_external_product(out, mgsw, diff, ell, special_primes, gadget_params);

    RNS_MLWE in1_NTT = mlwe_alloc_sample(ring, r);
    mlwe_copy_RNS_sample(in1_NTT, in1);
    mlwe_RNSc_to_RNS(in1_NTT, in1_NTT);
    mlwe_add_RNS_sample(out, out, in1_NTT);

    free_mlwe_RNS_sample(diff);
    free_mlwe_RNS_sample(in1_NTT);
}

void mgsw_NCMUX(RNS_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2, RNS_MLWE *mgsw, RNS_MLWE_KS_Key ksk,
                uint64_t ell, uint64_t special_primes, const GadgetParams *gadget_params)
{
    const uint64_t r = in1->r;
    ArithRing ring = in1->ring;
    const uint64_t gen = 2 * ring->N - 1;

    RNSc_MLWE tmp = mlwe_alloc_sample(ring, r);

    mlwe_automorphism_RNSc_GHS(tmp, in2, gen, ksk, ell);

    mgsw_CMUX(out, in1, tmp, mgsw, ell, special_primes, gadget_params);

    free_mlwe_RNS_sample(tmp);
}

// Canonical-output variants of CMUX / NCMUX: out = ExtProduct(in2 - in1) + in1 with every
// operand canonical. The external product's rescale already leaves it canonical, so neither it
// nor `in1` passes through the mul domain.
void mgsw_CMUX_to_coeff(RNS_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2, RNS_MLWE *mgsw, uint64_t ell,
                        uint64_t special_primes, const GadgetParams *gadget_params)
{
    const uint64_t r = in1->r;
    ArithRing ring = in1->ring;

    RNSc_MLWE diff = mlwe_alloc_sample(ring, r);
    mlwe_sub_RNSc_sample(diff, in2, in1);

    (void)special_primes;
    external_product_canonical(out, mgsw, diff, ell, gadget_params);
    mlwe_addto_RNSc_sample(out, in1);

    free_mlwe_RNS_sample(diff);
}

typedef struct
{
    RNSc_MLWE *out, *in1, *in2;
    RNS_MLWE *mgsw;
    uint64_t ell;
    const GadgetParams *gadget_params;
} CMUXJobs;

static void cmux_job(void *ctx, uint64_t i)
{
    const CMUXJobs *jobs = (const CMUXJobs *)ctx;
    mgsw_CMUX_to_coeff(jobs->out[i], jobs->in1[i], jobs->in2[i], jobs->mgsw, jobs->ell, 0,
                       jobs->gadget_params);
}

void mgsw_CMUX_batch(RNSc_MLWE *out, RNSc_MLWE *in1, RNSc_MLWE *in2, uint64_t n, RNS_MLWE *mgsw,
                     uint64_t ell, const GadgetParams *gadget_params, uint64_t n_threads)
{
    CMUXJobs jobs = {out, in1, in2, mgsw, ell, gadget_params};
    vfhe_parallel_for(n, one_thread_on_generic_path(n_threads), cmux_job, &jobs);
}

void mgsw_NCMUX_to_coeff(RNS_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2, RNS_MLWE *mgsw,
                         RNS_MLWE_KS_Key ksk, uint64_t ell, uint64_t special_primes,
                         const GadgetParams *gadget_params)
{
    const uint64_t r = in1->r;
    ArithRing ring = in1->ring;
    const uint64_t gen = 2 * ring->N - 1;

    RNSc_MLWE tmp = mlwe_alloc_sample(ring, r);

    mlwe_automorphism_RNSc_GHS(tmp, in2, gen, ksk, ell);

    mgsw_CMUX_to_coeff(out, in1, tmp, mgsw, ell, special_primes, gadget_params);

    free_mlwe_RNS_sample(tmp);
}

void mgsw_trivial(RNS_MLWE *out, ArithRing ring, uint64_t r, const ArithElement *msg,
                  uint64_t *const *scales, uint64_t width)
{
    for (uint64_t k = 0; k < width; k++)
    {
        ArithScalar scale;
        arith_scalar_new(ring, scales[k], &scale);
        for (uint64_t j = 0; j <= r; j++)
        {
            RNS_MLWE row = mlwe_alloc_sample(ring, r);
            mlwe_RNS_trivial_sample_of_zero(row);
            arith_scale_by(ring, j < r ? &row->a[j] : &row->b, msg, scale);
            out[j * width + k] = row;
        }
        arith_scalar_free(ring, &scale);
    }
}

typedef struct
{
    RNS_MLWE *out;
    RNS_MLWE *a;
    RNS_MLWE *b;
    uint64_t ell;
    const GadgetParams *gadget_params;
} InternalProduct;

static void internal_product_row(void *ctx, uint64_t i)
{
    const InternalProduct *p = (const InternalProduct *)ctx;
    RNS_MLWE row = p->b[i];
    RNSc_MLWE canonical = mlwe_alloc_sample(row->ring, row->r);
    mlwe_RNS_to_RNSc(canonical, row);
    p->out[i] = mlwe_alloc_sample(row->ring, row->r);
    mgsw_external_product(p->out[i], p->a, canonical, p->ell, 0, p->gadget_params);
    free_mlwe_RNS_sample(canonical);
}

void mgsw_internal_product(RNS_MLWE *out, RNS_MLWE *a, uint64_t ell, RNS_MLWE *b, uint64_t rows,
                           const GadgetParams *gadget_params, uint64_t n_threads)
{
    InternalProduct p = {out, a, b, ell, gadget_params};
    vfhe_parallel_for(rows, one_thread_on_generic_path(n_threads), internal_product_row, &p);
}

typedef struct
{
    RNS_MLWE *out;
    RNS_MLWE *in;
    uint64_t width, gen;
    RNS_MLWE_KS_Key aut, rlk;
} Automorphism;

static void automorphism_column(void *ctx, uint64_t k)
{
    const Automorphism *A = (const Automorphism *)ctx;
    const RNS_MLWE source = A->in[0];
    const uint64_t r = source->r, width = A->width;
    ArithRing ring = source->ring;

    // The message row, permuted and switched back to the key.
    RNSc_MLWE canonical = mlwe_alloc_sample(ring, r);
    mlwe_RNS_to_RNSc(canonical, A->in[r * width + k]);
    RNSc_MLWE z = mlwe_alloc_sample(ring, r);
    mlwe_automorphism_RNSc_GHS(z, canonical, A->gen, A->aut, 0);

    // Row j holds -s_j times it: the extended sample with z's mask on the
    // quadratic terms s_i * s_j and its body on the linear term s_j decrypts
    // to sum_i a_i s_i s_j - b s_j, which relinearization brings back to s.
    const uint64_t R = mlwe_extended_rank(r);
    for (uint64_t j = 0; j < r; j++)
    {
        RNSc_MLWE ext = mlwe_alloc_sample(ring, R);
        mlwe_RNS_trivial_sample_of_zero(ext);
        mlwe_RNS_to_RNSc(ext, ext);
        for (uint64_t i = 0; i < r; i++)
        {
            const uint64_t p = i < j ? i : j, q = i < j ? j : i;
            arith_copy(ring, &ext->a[mlwe_quadratic_slot(r, p, q)], &z->a[i]);
        }
        arith_copy(ring, &ext->a[R - r + j], &z->b);
        RNS_MLWE row = mlwe_alloc_sample(ring, r);
        mlwe_RNSc_GHS_hybrid_keyswitch(row, ext, A->rlk, 0);
        mlwe_RNSc_to_RNS(row, row);
        A->out[j * width + k] = row;
        free_mlwe_RNS_sample(ext);
    }
    mlwe_RNSc_to_RNS(z, z);
    A->out[r * width + k] = z;
    free_mlwe_RNS_sample(canonical);
}

void mgsw_automorphism(RNS_MLWE *out, RNS_MLWE *in, uint64_t width, uint64_t gen,
                       RNS_MLWE_KS_Key aut, RNS_MLWE_KS_Key rlk, uint64_t n_threads)
{
    Automorphism A = {out, in, width, gen, aut, rlk};
    vfhe_parallel_for(width, n_threads, automorphism_column, &A);
}

/* ------------------------------------------------------------------------------------------------
 * Shifting sequences by an encrypted amount (mgsw_shift_sequences). With w = width, output i of a
 * sequence x_0, x_1, ... is
 *
 *   y_i = x_(i+w) + sum_(t=1..w) K_t (x) (x_(i+w-t) - x_(i+w)),
 *
 * K_t the MGSW key of [d = t]. A gadget decomposition D is linear up to the choice of digit
 * representatives, which the gadget elements do not see, so with the lag keys L_0 = -(K_1 + ... +
 * K_w) and L_t = K_t,
 *
 *   y_i = x_(i+w) + ModDown(sum_(t=0..w) D(x_(i+w-t)) * L_t).
 *
 * Every input is decomposed once, and its decomposition serves the w + 1 outputs that read it.
 *
 * A sequence is one task. Its inputs are taken in blocks of BLOCK_INPUTS, and each block is worked
 * one row of the key ring at a time: the block's digit rows (each digit lifted to that row and
 * transformed), then all of the block's outputs on that row as one matrix product. The digit
 * rows of the block's last w inputs are kept for the next block, whose first outputs read them.
 * A digit row is therefore used while it is in cache, a key row is read once per block, and no
 * whole decomposition is held.
 * ------------------------------------------------------------------------------------------------
 */

// The inputs of a block: each key row is read once per block, and a block's digit rows are held
// at once.
enum
{
    BLOCK_INPUTS = 8
};

// What every sequence shares.
typedef struct
{
    RNSc_MLWE *out, *in;
    const uint64_t *gens;
    RNS_MLWE_KS_Key aut;
    const uint64_t *input_offset, *output_offset, *length; // per sequence
    RNS_MLWE **lag_keys;                                   // L_0, ..., L_w
    uint64_t width, ell;
    const GadgetParams *gadget_params;
} Shift;

// One sequence being shifted. A digit is numbered d = j * ell + i, digit i of component j, the
// order of an MGSW key's rows. Digit rows are one-row polynomials, recycled through a stack of
// spare ones per key row.
typedef struct
{
    const Shift *shift;
    uint64_t components, digits, N, key_rows;
    RNS_Base base;
    uint64_t row_index[64]; // the position of each key row among the key's rows
    RNS_Polynomial *kept;   // [(row_index * digits + d) * w + a]: the last w inputs, oldest first
    RNS_Polynomial *spare;  // spare_capacity per key row
    uint64_t *n_spare, spare_capacity;
    uint64_t block_start, block_end, n_kept; // the block's inputs, and how many before it are kept
    ElementDecomposition **decompositions;   // [(q - block_start) * components + j]
    RNS_MLWE *acc;                           // [q - block_start], over the key ring
} Sequence;

static RNS_Polynomial take_digit_row(Sequence *S, uint64_t row)
{
    const uint64_t r = S->row_index[row];
    if (S->n_spare[r] > 0)
        return S->spare[r * S->spare_capacity + --S->n_spare[r]];
    return polynomial_new_RNS_polynomial_rows(S->N, 1ULL << row, S->base);
}

static void return_digit_row(Sequence *S, uint64_t row, RNS_Polynomial x)
{
    const uint64_t r = S->row_index[row];
    S->spare[r * S->spare_capacity + S->n_spare[r]++] = x;
}

// One key row of the current block. The window is the kept inputs followed by the block's; the
// row of digit d of window input a is digit_rows[d * window + a]. Output q - w, for an input q of
// the block, reads the inputs q - t, t = 0..w, so its row is
//
//   sum_t sum_d (row of digit d of input q - t) * (row of L_t[d]),
//
// and the block's outputs are one matrix product: outputs x (t, d) by (t, d) x components.
static void block_row(void *ctx, uint64_t row, uint64_t part)
{
    (void)part;
    Sequence *S = (Sequence *)ctx;
    const Shift *shift = S->shift;
    const uint64_t w = shift->width, c = S->components, digits = S->digits;
    const uint64_t new_inputs = S->block_end - S->block_start, window = S->n_kept + new_inputs;
    const uint64_t first_own = S->block_start > w ? S->block_start : w;
    const uint64_t outputs = S->block_end > first_own ? S->block_end - first_own : 0;
    const uint64_t terms = (w + 1) * digits, row_mask = 1ULL << row;

    RNS_Polynomial *digit_rows = (RNS_Polynomial *)safe_malloc(
        (digits * window + (outputs + c) * terms + outputs * c) * sizeof(RNS_Polynomial));
    RNS_Polynomial *lhs = digit_rows + digits * window; // [o * terms + t * digits + d]
    RNS_Polynomial *rhs = lhs + outputs * terms;        // [k * terms + t * digits + d]
    RNS_Polynomial *acc_rows = rhs + c * terms;         // [o * c + k]
    struct _RNS_Polynomial *views = (struct _RNS_Polynomial *)safe_malloc(
        (outputs * c > 0 ? outputs * c : 1) * sizeof(struct _RNS_Polynomial));

    for (uint64_t d = 0; d < digits; d++)
    {
        RNS_Polynomial *kept = &S->kept[(S->row_index[row] * digits + d) * w];
        for (uint64_t a = 0; a < S->n_kept; a++)
            digit_rows[d * window + a] = kept[w - S->n_kept + a];
        for (uint64_t b = 0; b < new_inputs; b++)
        {
            RNS_Polynomial x = take_digit_row(S, row);
            const ElementDecomposition *e = S->decompositions[b * c + d / shift->ell];
            if (d % shift->ell < gadget_decomposition_digits(e))
                gadget_decomposition_digit_row(x, e, d % shift->ell);
            else
                polynomial_RNS_zero(x);
            digit_rows[d * window + S->n_kept + b] = x;
        }
    }

    for (uint64_t o = 0; o < outputs; o++)
    {
        const uint64_t slot = first_own + o - S->block_start; // of its input q in the block
        const uint64_t q = S->n_kept + slot;                  // and in the window
        for (uint64_t t = 0; t <= w; t++)
            for (uint64_t d = 0; d < digits; d++)
                lhs[o * terms + t * digits + d] = digit_rows[d * window + q - t];
        for (uint64_t k = 0; k < c; k++)
            acc_rows[o * c + k] = polynomial_RNS_view(
                &views[o * c + k], mlwe_component_rows(S->acc[slot], k), row_mask);
    }
    for (uint64_t k = 0; k < c; k++)
        for (uint64_t t = 0; t <= w; t++)
            for (uint64_t d = 0; d < digits; d++)
                rhs[k * terms + t * digits + d] = mlwe_component_rows(shift->lag_keys[t][d], k);
    polynomial_matrix_mul_addto_RNS_polynomial(acc_rows, lhs, rhs, outputs, terms, c);

    // Keep the last w inputs' digit rows; recycle the others.
    const uint64_t keep = window < w ? window : w;
    for (uint64_t d = 0; d < digits; d++)
    {
        RNS_Polynomial *kept = &S->kept[(S->row_index[row] * digits + d) * w];
        for (uint64_t a = 0; a < window - keep; a++)
            return_digit_row(S, row, digit_rows[d * window + a]);
        for (uint64_t a = 0; a < keep; a++)
            kept[w - keep + a] = digit_rows[d * window + window - keep + a];
    }
    free(views);
    free(digit_rows);
}

// Sequence s, as one task: the first block is its first w inputs, which own no output; the
// blocks after it have BLOCK_INPUTS inputs and as many outputs.
static void shift_sequence(void *ctx, uint64_t s)
{
    const Shift *shift = (const Shift *)ctx;
    const uint64_t w = shift->width, ell = shift->ell;
    RNSc_MLWE *x = shift->in + shift->input_offset[s], *y = shift->out + shift->output_offset[s];
    const uint64_t *gens = shift->gens != NULL ? shift->gens + shift->input_offset[s] : NULL;
    const uint64_t n_inputs = shift->length[s] + w;
    RNS_MLWE key = shift->lag_keys[1][0]; // any key row: its ring is the key ring
    RNS_Polynomial key_body = arith_rns_polynomial(&key->b);

    Sequence S = {.shift = shift,
                  .components = x[0]->r + 1,
                  .digits = (x[0]->r + 1) * ell,
                  .N = key_body->base->N,
                  .key_rows = key_body->rns_mask,
                  .base = key_body->base};
    const uint64_t rows = (uint64_t)__builtin_popcountll(S.key_rows), c = S.components;
    for (uint64_t row = 0, r = 0; row < 64; row++)
        if (S.key_rows & (1ULL << row))
            S.row_index[row] = r++;
    S.kept = (RNS_Polynomial *)safe_malloc(rows * S.digits * w * sizeof(RNS_Polynomial));
    S.spare_capacity = (w + BLOCK_INPUTS) * S.digits;
    S.spare = (RNS_Polynomial *)safe_malloc(rows * S.spare_capacity * sizeof(RNS_Polynomial));
    S.n_spare = (uint64_t *)safe_malloc(rows * sizeof(uint64_t));
    memset(S.n_spare, 0, rows * sizeof(uint64_t));
    const uint64_t most_inputs = w > BLOCK_INPUTS ? w : BLOCK_INPUTS;
    S.decompositions = (ElementDecomposition **)safe_malloc(most_inputs * c * sizeof(void *));
    S.acc = (RNS_MLWE *)safe_malloc(BLOCK_INPUTS * sizeof(RNS_MLWE));
    for (uint64_t g = 0; g < BLOCK_INPUTS; g++)
        S.acc[g] = mlwe_alloc_sample(key->ring, key->r);
    RNSc_MLWE automorphed[BLOCK_INPUTS > 3 ? BLOCK_INPUTS : 3];
    const ArithElement **own_input = (const ArithElement **)safe_malloc(c * sizeof(ArithElement *));

    for (S.block_start = 0; S.block_start < n_inputs; S.block_start = S.block_end)
    {
        const uint64_t start = S.block_start;
        S.block_end =
            start == 0 ? w : (n_inputs - start < BLOCK_INPUTS ? n_inputs : start + BLOCK_INPUTS);
        S.n_kept = start < w ? start : w;
        const uint64_t first_own = start > w ? start : w; // the first input that owns an output

        // Decompose the block's inputs, each through its automorphism if it has one.
        for (uint64_t q = start; q < S.block_end; q++)
        {
            RNSc_MLWE input = x[q];
            automorphed[q - start] = NULL;
            if (gens != NULL && gens[q] != 1)
            {
                automorphed[q - start] = mlwe_alloc_sample(input->ring, input->r);
                mlwe_automorphism_RNSc_GHS(automorphed[q - start], input, gens[q], shift->aut, 0);
                input = automorphed[q - start];
            }
            for (uint64_t j = 0; j < c; j++)
            {
                ElementDecomposition *e = gadget_decomposition_new(
                    &shift->lag_keys[1][j * ell], mlwe_component(input, j), shift->gadget_params);
                assert(gadget_decomposition_digits(e) <= ell);
                S.decompositions[(q - start) * c + j] = e;
            }
        }
        for (uint64_t q = first_own; q < S.block_end; q++)
            mlwe_RNS_trivial_sample_of_zero(S.acc[q - start]);

        rns_rows_for(S.N, S.key_rows, 1, block_row, &S);

        // y_(q-w) = x_q + ModDown(acc), with x_q as it was decomposed.
        for (uint64_t q = first_own; q < S.block_end; q++)
        {
            RNSc_MLWE input = automorphed[q - start] != NULL ? automorphed[q - start] : x[q];
            for (uint64_t j = 0; j < c; j++)
                own_input[j] = mlwe_component(input, j);
            mlwe_RNS_to_RNSc(S.acc[q - start], S.acc[q - start]);
            mlwe_mod_down(y[q - w], S.acc[q - start], own_input);
        }
        for (uint64_t q = start; q < S.block_end; q++)
        {
            for (uint64_t j = 0; j < c; j++)
                gadget_decomposition_free(S.decompositions[(q - start) * c + j]);
            if (automorphed[q - start] != NULL)
                free_mlwe_RNS_sample(automorphed[q - start]);
        }
    }

    for (uint64_t r = 0; r < rows; r++)
    {
        for (uint64_t a = 0; a < S.n_spare[r]; a++)
            free_RNS_polynomial(S.spare[r * S.spare_capacity + a]);
        for (uint64_t a = 0; a < S.digits * w; a++)
            free_RNS_polynomial(S.kept[r * S.digits * w + a]);
    }
    for (uint64_t g = 0; g < BLOCK_INPUTS; g++)
        free_mlwe_RNS_sample(S.acc[g]);
    free(own_input);
    free(S.acc);
    free(S.decompositions);
    free(S.n_spare);
    free(S.spare);
    free(S.kept);
}

void mgsw_shift_sequences(RNSc_MLWE *out, RNSc_MLWE *in, const uint64_t *gens, RNS_MLWE_KS_Key aut,
                          const uint64_t *lengths, uint64_t n_sequences, RNS_MLWE *const *keys,
                          uint64_t width, uint64_t ell, const GadgetParams *gadget_params,
                          uint64_t n_threads)
{
    if (n_sequences == 0)
        return;
    assert(width >= 1 && width <= 3);
    const uint64_t rows = (in[0]->r + 1) * ell;

    // L_0 = -(K_1 + ... + K_w), row by row; L_t = K_t.
    RNS_MLWE *negated_sum = (RNS_MLWE *)safe_malloc(rows * sizeof(RNS_MLWE));
    for (uint64_t i = 0; i < rows; i++)
    {
        negated_sum[i] = mlwe_alloc_sample(keys[0][i]->ring, keys[0][i]->r);
        mlwe_RNS_trivial_sample_of_zero(negated_sum[i]);
        for (uint64_t t = 0; t < width; t++)
            mlwe_sub_RNSc_sample(negated_sum[i], negated_sum[i], keys[t][i]);
    }
    RNS_MLWE *lag_keys[4] = {negated_sum, NULL, NULL, NULL};
    for (uint64_t t = 1; t <= width; t++)
        lag_keys[t] = keys[t - 1];

    uint64_t *offsets = (uint64_t *)safe_malloc(2 * n_sequences * sizeof(uint64_t));
    for (uint64_t s = 0, a = 0, b = 0; s < n_sequences; s++)
    {
        offsets[s] = a;
        offsets[n_sequences + s] = b;
        a += lengths[s] + width;
        b += lengths[s];
    }
    Shift shift = {.out = out,
                   .in = in,
                   .gens = gens,
                   .aut = aut,
                   .input_offset = offsets,
                   .output_offset = offsets + n_sequences,
                   .length = lengths,
                   .lag_keys = lag_keys,
                   .width = width,
                   .ell = ell,
                   .gadget_params = gadget_params};
    vfhe_parallel_for(n_sequences, n_threads, shift_sequence, &shift);
    free(offsets);
    for (uint64_t i = 0; i < rows; i++)
        free_mlwe_RNS_sample(negated_sum[i]);
    free(negated_sum);
}
