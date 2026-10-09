// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "mlwe.h"
#include "util.h"

// The external product, left canonical in `out`.
static void external_product_canonical(RNSc_MLWE out, RNS_MLWE *mgsw, RNSc_MLWE in, uint64_t ell,
                                       uint64_t log_base, bool balanced)
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
        gadget_mul_addto_polynomial(acc, &mgsw[j * ell], &in->a[j], log_base, balanced);
    }
    gadget_mul_addto_polynomial(acc, &mgsw[r * ell], &in->b, log_base, balanced);

    mlwe_RNS_to_RNSc(acc, acc);
    mlwe_round_division(acc, out->ring);
    mlwe_copy_RNSc_sample(out, acc);
    free_mlwe_RNS_sample(acc);
}

void mgsw_external_product(RNS_MLWE out, RNS_MLWE *mgsw, RNSc_MLWE in, uint64_t ell,
                           uint64_t special_primes, uint64_t log_base, bool balanced)
{
    (void)special_primes;
    external_product_canonical(out, mgsw, in, ell, log_base, balanced);
    mlwe_RNSc_to_RNS(out, out);
}

void mgsw_CMUX(RNS_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2, RNS_MLWE *mgsw, uint64_t ell,
               uint64_t special_primes, uint64_t log_base, bool balanced)
{
    const uint64_t r = in1->r;
    ArithRing ring = in1->ring;

    RNSc_MLWE diff = mlwe_alloc_sample(ring, r);
    mlwe_sub_RNSc_sample(diff, in2, in1);

    mgsw_external_product(out, mgsw, diff, ell, special_primes, log_base, balanced);

    RNS_MLWE in1_NTT = mlwe_alloc_sample(ring, r);
    mlwe_copy_RNS_sample(in1_NTT, in1);
    mlwe_RNSc_to_RNS(in1_NTT, in1_NTT);
    mlwe_add_RNS_sample(out, out, in1_NTT);

    free_mlwe_RNS_sample(diff);
    free_mlwe_RNS_sample(in1_NTT);
}

void mgsw_NCMUX(RNS_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2, RNS_MLWE *mgsw, RNS_MLWE_KS_Key ksk,
                uint64_t ell, uint64_t special_primes, uint64_t log_base, bool balanced)
{
    const uint64_t r = in1->r;
    ArithRing ring = in1->ring;
    const uint64_t gen = 2 * ring->N - 1;

    RNSc_MLWE tmp = mlwe_alloc_sample(ring, r);

    mlwe_automorphism_RNSc_GHS(tmp, in2, gen, ksk, ell);

    mgsw_CMUX(out, in1, tmp, mgsw, ell, special_primes, log_base, balanced);

    free_mlwe_RNS_sample(tmp);
}

// Canonical-output variants of CMUX / NCMUX: out = ExtProduct(in2 - in1) + in1 with every
// operand canonical. The external product's rescale already leaves it canonical, so neither it
// nor `in1` passes through the mul domain.
void mgsw_CMUX_to_coeff(RNS_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2, RNS_MLWE *mgsw, uint64_t ell,
                        uint64_t special_primes, uint64_t log_base, bool balanced)
{
    const uint64_t r = in1->r;
    ArithRing ring = in1->ring;

    RNSc_MLWE diff = mlwe_alloc_sample(ring, r);
    mlwe_sub_RNSc_sample(diff, in2, in1);

    (void)special_primes;
    external_product_canonical(out, mgsw, diff, ell, log_base, balanced);
    mlwe_addto_RNSc_sample(out, in1);

    free_mlwe_RNS_sample(diff);
}

void mgsw_NCMUX_to_coeff(RNS_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2, RNS_MLWE *mgsw,
                         RNS_MLWE_KS_Key ksk, uint64_t ell, uint64_t special_primes,
                         uint64_t log_base, bool balanced)
{
    const uint64_t r = in1->r;
    ArithRing ring = in1->ring;
    const uint64_t gen = 2 * ring->N - 1;

    RNSc_MLWE tmp = mlwe_alloc_sample(ring, r);

    mlwe_automorphism_RNSc_GHS(tmp, in2, gen, ksk, ell);

    mgsw_CMUX_to_coeff(out, in1, tmp, mgsw, ell, special_primes, log_base, balanced);

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
    uint64_t ell, log_base;
    bool balanced;
} InternalProduct;

static void internal_product_row(void *ctx, uint64_t i)
{
    const InternalProduct *p = (const InternalProduct *)ctx;
    RNS_MLWE row = p->b[i];
    RNSc_MLWE canonical = mlwe_alloc_sample(row->ring, row->r);
    mlwe_RNS_to_RNSc(canonical, row);
    p->out[i] = mlwe_alloc_sample(row->ring, row->r);
    mgsw_external_product(p->out[i], p->a, canonical, p->ell, 0, p->log_base, p->balanced);
    free_mlwe_RNS_sample(canonical);
}

void mgsw_internal_product(RNS_MLWE *out, RNS_MLWE *a, uint64_t ell, RNS_MLWE *b, uint64_t rows,
                           uint64_t log_base, bool balanced, uint64_t n_threads)
{
    InternalProduct p = {out, a, b, ell, log_base, balanced};
    vfhe_parallel_for(rows, n_threads, internal_product_row, &p);
}

typedef struct
{
    RNS_MLWE *out;
    RNS_MLWE *in;
    uint64_t width, gen;
    RNS_MLWE_KS_Key aut, rlk;
} Automorphism;

// Slot of the quadratic term s_p * s_q (p <= q) in an extended sample of rank r, as
// mlwe_tensor_product lays them out.
static uint64_t quadratic_slot(uint64_t r, uint64_t p, uint64_t q)
{
    return p * r - p * (p - 1) / 2 + (q - p);
}

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
            arith_copy(ring, &ext->a[quadratic_slot(r, p, q)], &z->a[i]);
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
