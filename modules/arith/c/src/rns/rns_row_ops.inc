// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
//
// Row bodies that are identical apart from the storage width. Included once
// per width from rns_polynomial.c, so there is one source and two
// instantiations rather than a width test inside a coefficient loop -- and so
// a narrow row is never widened to reuse the 64-bit body.
//
// The includer defines:
//   RNS_W          the word type
//   RNS_SFX        the name suffix (_narrow / _wide)
//   RNS_{ADD,SUB,MUL,MULADDTO,NEGATE}  the element-wise kernels at this width
//
// Everything here takes bare row pointers. The caller has already resolved the
// width once for the row, which is the whole point.

#define RNS_CAT_(a, b) a##b
#define RNS_CAT(a, b) RNS_CAT_(a, b)
#define RNS_FN(name) RNS_CAT(name, RNS_SFX)

/* The incomplete-NTT block product: out = in1 * in2 in
   Z_q[X]/(X^N+1) split into `sd` blocks of `poly_size`, the wrap-around terms
   scaled by the per-block twiddle. `acc` says whether the result replaces the
   destination or accumulates into it.

   `tmp` is `poly_size` words at this width, supplied by the caller; nothing
   here converts a width. */
static void RNS_FN(rns_row_block_mul)(RNS_W *out, RNS_W *in1, RNS_W *in2, RNS_W *tmp, RNS_W *tw,
                                      uint64_t sd, uint64_t poly_size, Modulus mod, RnsBlockAcc acc)
{
    for (uint64_t j = 0; j < sd; j++)
    {
        for (uint64_t k = 0; k < sd - j; k++)
        {
            RNS_W *dst = out + (j + k) * poly_size;
            if (acc == RNS_BLOCK_SET && j == 0)
            {
                /* j == 0 reaches every block and its wrap loop below is empty,
                   so this pass initializes the whole row: no zeroing needed. */
                RNS_MUL(dst, in1, in2 + k * poly_size, poly_size, mod);
            }
            else if (acc == RNS_BLOCK_SUBTO)
            {
                RNS_MUL(tmp, in1 + j * poly_size, in2 + k * poly_size, poly_size, mod);
                RNS_SUB(dst, dst, tmp, poly_size, mod);
            }
            else
            {
                RNS_MULADDTO(dst, in1 + j * poly_size, in2 + k * poly_size, poly_size, mod);
            }
        }
        // X^sd is the twiddle, so these terms wrap onto the low blocks
        for (uint64_t k = sd - j; k < sd; k++)
        {
            RNS_W *dst = out + (j + k - sd) * poly_size;
            RNS_MUL(tmp, in1 + j * poly_size, in2 + k * poly_size, poly_size, mod);
            RNS_MUL(tmp, tmp, tw, poly_size, mod);
            if (acc == RNS_BLOCK_SUBTO)
                RNS_SUB(dst, dst, tmp, poly_size, mod);
            else
                RNS_ADD(dst, dst, tmp, poly_size, mod);
        }
    }
}

/* out = in * X^a in Z_q[X]/(X^N+1), so a shift by `a` with the wrapped part
   negated. The `a % 8 == 0` forms let the kernels do the whole run; otherwise
   the shift is unaligned and the moves are scalar. */
static void RNS_FN(rns_row_mul_by_xai)(RNS_W *out, RNS_W *in, uint64_t a, uint64_t N, Modulus mod)
{
    const uint64_t q = mod->q;
    if (a < N)
    {
        if (a % 8 == 0)
        {
            RNS_NEGATE(out, in + N - a, a, mod);
            memcpy(out + a, in, (N - a) * sizeof(RNS_W));
        }
        else
        {
            for (uint64_t i = 0; i < a; i++)
                out[i] = (RNS_W)negate_modq(in[i - a + N], q);
            for (uint64_t i = a; i < N; i++)
                out[i] = in[i - a];
        }
    }
    else
    {
        if (a % 8 == 0)
        {
            memcpy(out, in + 2 * N - a, (a - N) * sizeof(RNS_W));
            RNS_NEGATE(out + a - N, in, 2 * N - a, mod);
        }
        else
        {
            for (uint64_t i = 0; i < a - N; i++)
                out[i] = in[i - a + 2 * N];
            for (uint64_t i = a - N; i < N; i++)
                out[i] = (RNS_W)negate_modq(in[i - a + N], q);
        }
    }
}

// out = in * (X^a - 1), the same shift with the un-shifted row subtracted.
static void RNS_FN(rns_row_mul_by_xai_minus1)(RNS_W *out, RNS_W *in, uint64_t a, uint64_t N,
                                              Modulus mod)
{
    const uint64_t q = mod->q;
    if (a < N)
    {
        if (a % 8 == 0)
        {
            RNS_NEGATE(out, in + N - a, a, mod);
            RNS_SUB(out, out, in, a, mod);
            RNS_SUB(out + a, in, in + a, N - a, mod);
        }
        else
        {
            for (uint64_t i = 0; i < a; i++)
                out[i] = (RNS_W)sub_modq(negate_modq(in[i - a + N], q), in[i], q);
            for (uint64_t i = a; i < N; i++)
                out[i] = (RNS_W)sub_modq(in[i - a], in[i], q);
        }
    }
    else
    {
        if (a % 8 == 0)
        {
            RNS_SUB(out, in + 2 * N - a, in, a - N, mod);
            RNS_NEGATE(out + a - N, in, 2 * N - a, mod);
            RNS_SUB(out + a - N, out + a - N, in + a - N, 2 * N - a, mod);
        }
        else
        {
            for (uint64_t i = 0; i < a - N; i++)
                out[i] = (RNS_W)sub_modq(in[i - a + 2 * N], in[i], q);
            for (uint64_t i = a - N; i < N; i++)
                out[i] = (RNS_W)sub_modq(negate_modq(in[i - a + N], q), in[i], q);
        }
    }
}

/* The Galois automorphism's gather: coefficient (i, i2) of the split layout
   moves to `dst`, negated when the exponent wrapped. Signed because the
   caller reduces the whole array in one pass afterwards. */
static void RNS_FN(rns_row_permute_gather)(int64_t *out, const RNS_W *in, uint64_t gen, uint64_t N,
                                           uint64_t split_degree, uint64_t split_degree_log,
                                           uint64_t poly_size)
{
    const uint64_t split_degree_mod = split_degree - 1, mod_mask = N - 1;
    for (uint64_t i = 0; i < split_degree; i++)
    {
        for (uint64_t i2 = 0; i2 < poly_size; i2++)
        {
            const uint64_t idx = ((i + (i2 << split_degree_log)) * gen);
            const uint64_t dst =
                (idx & split_degree_mod) * poly_size + ((idx & mod_mask) >> split_degree_log);
            const int64_t val = (int64_t)in[i * poly_size + i2];
            out[dst] = (idx & N) ? -val : val;
        }
    }
}

// Slot moves: strided copies within a row, one block at a time.
static void RNS_FN(rns_row_broadcast_slot)(RNS_W *out, const RNS_W *in, uint64_t slot_idx,
                                           uint64_t split_degree, uint64_t poly_size)
{
    for (uint64_t j = 0; j < split_degree; j++)
        for (uint64_t k = 0; k < poly_size; k++)
            out[j * poly_size + k] = in[j * poly_size + slot_idx];
}

static void RNS_FN(rns_row_rotate_slot)(RNS_W *out, const RNS_W *in, uint64_t rot,
                                        uint64_t split_degree, uint64_t poly_size)
{
    for (uint64_t j = 0; j < split_degree; j++)
    {
        for (uint64_t k = 0; k < poly_size - rot; k++)
            out[j * poly_size + k] = in[j * poly_size + k + rot];
        for (uint64_t k = poly_size - rot; k < poly_size; k++)
            out[j * poly_size + k] = in[j * poly_size + k + rot - poly_size];
    }
}

static void RNS_FN(rns_row_copy_slot)(RNS_W *out, const RNS_W *in, uint64_t dst, uint64_t src,
                                      uint64_t split_degree, uint64_t poly_size)
{
    for (uint64_t j = 0; j < split_degree; j++)
        out[j * poly_size + dst] = in[j * poly_size + src];
}

/* A slot's coefficients across the split blocks, gathered out and scattered
   back: that column is one element of the degree-`split_degree` extension, and
   the caller inverts it there. */
static void RNS_FN(rns_row_gather_slot_column)(uint64_t *out, const RNS_W *in, uint64_t s,
                                               uint64_t split_degree, uint64_t poly_size)
{
    for (uint64_t j = 0; j < split_degree; j++)
        out[j] = in[j * poly_size + s];
}

static void RNS_FN(rns_row_scatter_slot_column)(RNS_W *out, const uint64_t *in, uint64_t s,
                                                uint64_t split_degree, uint64_t poly_size)
{
    for (uint64_t j = 0; j < split_degree; j++)
        out[j * poly_size + s] = (RNS_W)in[j];
}

/* Coefficient-wise inverse of a whole row, by the prefix-product trick: one
   pass up accumulating products, one inversion, one pass down. Returns -2 if
   any coefficient is zero, in which case `out` is untouched. */
static int RNS_FN(rns_row_inverse)(RNS_W *out, const RNS_W *in, uint64_t *prefix, uint64_t N,
                                   Modulus mod)
{
    if (in[0] == 0)
        return -2;
    prefix[0] = in[0];
    for (uint64_t k = 1; k < N; k++)
    {
        if (in[k] == 0)
            return -2;
        prefix[k] = mul_modq(prefix[k - 1], in[k], mod);
    }

    uint64_t t = inverse_mod(prefix[N - 1], mod->q);
    for (uint64_t k = N - 1; k > 0; k--)
    {
        out[k] = (RNS_W)mul_modq(t, prefix[k - 1], mod);
        t = mul_modq(t, in[k], mod);
    }
    out[0] = (RNS_W)t;
    return 0;
}

// Digit `level` of every coefficient, in base 2^log_base.
static void RNS_FN(rns_row_digit)(uint64_t *out, const RNS_W *in, uint64_t shift, uint64_t mask,
                                  uint64_t N)
{
    for (uint64_t i = 0; i < N; i++)
        out[i] = (in[i] >> shift) & mask;
}

// Scatter a reduced 64-bit array into the split layout of one row.
static void RNS_FN(rns_row_scatter_split)(RNS_W *out, const uint64_t *in, uint64_t N,
                                          uint64_t split_degree, uint64_t poly_size)
{
    const uint64_t mod_mask = split_degree - 1;
    for (uint64_t j = 0; j < N; j++)
        out[(j & mod_mask) * poly_size + j / split_degree] = (RNS_W)in[j];
}

static bool RNS_FN(rns_row_is_zero)(const RNS_W *row, uint64_t N)
{
    for (uint64_t i = 0; i < N; i++)
        if (row[i])
            return false;
    return true;
}

// One coefficient, for the integer-addend path: only index 0 is ever touched.
static void RNS_FN(rns_row_add_at0)(RNS_W *row, uint64_t addend, uint64_t q)
{
    row[0] = (RNS_W)((row[0] + addend) % q);
}

#undef RNS_CAT_
#undef RNS_CAT
#undef RNS_FN
#undef RNS_W
#undef RNS_SFX
#undef RNS_ADD
#undef RNS_SUB
#undef RNS_MUL
#undef RNS_MULADDTO
#undef RNS_NEGATE
