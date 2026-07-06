/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_poly.c
 * @brief The ring/poly layer: multiplication against a schoolbook oracle,
 *        fused-kernel equivalences, monomial shifts, automorphisms, slot-wise
 *        inversion and slot moves, gadget decomposition, the digest, and the
 *        status-code error paths.
 */
#include <stdint.h>
#include <string.h>

#include "arith.h"
#include "test_arith_support.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* ---------------------------------------------------- mul vs schoolbook ---- */

static void poly_mul_against_oracle(uint64_t split_degree)
{
    const uint64_t N = 16;
    const uint64_t n_ntt = N / split_degree;
    uint64_t primes[2];
    primes[0] = nt_next_ntt_prime(1ULL << 30, n_ntt, false);
    primes[1] = nt_next_ntt_prime(primes[0], n_ntt, false);
    ring_t r = ring_new(primes, split_degree, N, 2);
    TEST_ASSERT_NOT_NULL(r);

    uint64_t a_c[16], b_c[16];
    for (uint64_t i = 0; i < N; i++)
    {
        a_c[i] = i + 1;
        b_c[i] = 3 * i + 2;
    }

    rns_poly_t a = poly_new(r, 0x3);
    rns_poly_t b = poly_new(r, 0x3);
    rns_poly_t c = poly_new(r, 0x3);
    TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_from_int_array(a, a_c));
    TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_from_int_array(b, b_c));

    /* Domain guard: multiplying coefficient-form operands must be rejected. */
    rns_poly_t a_coeff = poly_new(r, 0x3);
    TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_to_coeff(a_coeff, a));
    TEST_ASSERT_EQUAL_INT(VFHE_ERR_DOMAIN, poly_mul(c, a_coeff, b));

    TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_mul(c, a, b));
    TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_to_coeff(c, c));

    /* Undo the split-block layout and compare against the schoolbook oracle. */
    const uint64_t ps = N / split_degree;
    for (int limb = 0; limb < 2; limb++)
    {
        uint64_t expect[16], a_red[16], b_red[16];
        for (uint64_t i = 0; i < N; i++)
        {
            a_red[i] = a_c[i] % primes[limb];
            b_red[i] = b_c[i] % primes[limb];
        }
        negacyclic_schoolbook(expect, a_red, b_red, N, primes[limb]);
        const uint64_t *row = poly_limb_data(c, (uint64_t)limb);
        for (uint64_t j = 0; j < N; j++)
        {
            uint64_t stored = row[(j % split_degree) * ps + j / split_degree];
            TEST_ASSERT_EQUAL_UINT64(expect[j], stored);
        }
    }

    poly_free(a);
    poly_free(b);
    poly_free(c);
    poly_free(a_coeff);
    ring_free(r);
}

void test_poly_mul_split1(void) { poly_mul_against_oracle(1); }
void test_poly_mul_split2(void) { poly_mul_against_oracle(2); }

void test_poly_add_scalar_domains(void)
{
    const uint64_t N = 16;
    uint64_t primes[1] = {nt_next_ntt_prime(1ULL << 30, N, false)};
    ring_t r = ring_new(primes, 1, N, 1);

    uint64_t v[16] = {5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    rns_poly_t p = poly_new(r, 0x1);
    TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_from_int_array(p, v)); /* ends in EVAL */

    /* Adding the constant 7 in EVAL, then converting back, must add 7 to the
     * constant coefficient only. */
    TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_add_scalar(p, p, 7));
    TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_to_coeff(p, p));
    TEST_ASSERT_EQUAL_UINT64(12, poly_limb_data(p, 0)[0]);
    TEST_ASSERT_EQUAL_UINT64(1, poly_limb_data(p, 0)[1]);

    poly_free(p);
    ring_free(r);
}

/* ------------------------------------------------ fused mul equivalences ---- */

static void fused_mul_equivalences(uint64_t split_degree)
{
    enum
    {
        N = 16
    };
    uint64_t primes[2];
    ring_t r = make_ring(N, split_degree, primes);

    uint64_t a_c[N], b_c[N], c_c[N];
    for (uint64_t i = 0; i < N; i++)
    {
        a_c[i] = lcg() % primes[0];
        b_c[i] = lcg() % primes[0];
        c_c[i] = lcg() % primes[0];
    }
    rns_poly_t a = poly_new(r, 0x3), b = poly_new(r, 0x3), c = poly_new(r, 0x3);
    rns_poly_t ab = poly_new(r, 0x3), acc = poly_new(r, 0x3), ref = poly_new(r, 0x3);
    poly_from_int_array(a, a_c);
    poly_from_int_array(b, b_c);
    poly_from_int_array(c, c_c);

    TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_mul(ab, a, b));

    /* mul_addto == c + a*b */
    poly_copy(acc, c);
    TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_mul_addto(acc, a, b));
    TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_add(ref, c, ab));
    TEST_ASSERT_TRUE(poly_eq(acc, ref));

    /* mul_subto == c - a*b */
    poly_copy(acc, c);
    TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_mul_subto(acc, a, b));
    TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_sub(ref, c, ab));
    TEST_ASSERT_TRUE(poly_eq(acc, ref));

    /* mul_into == mul */
    poly_copy(acc, a);
    TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_mul_into(acc, b));
    TEST_ASSERT_TRUE(poly_eq(acc, ab));

    poly_free(a);
    poly_free(b);
    poly_free(c);
    poly_free(ab);
    poly_free(acc);
    poly_free(ref);
    ring_free(r);
}

void test_poly_fused_mul_split1(void) { fused_mul_equivalences(1); }
void test_poly_fused_mul_split2(void) { fused_mul_equivalences(2); }

/* ----------------------------------------------- monomial / automorphism ---- */

void test_poly_mul_xai(void)
{
    enum
    {
        N = 16
    };
    uint64_t primes[2];
    ring_t r = make_ring(N, 1, primes);

    uint64_t v[N];
    for (uint64_t i = 0; i < N; i++)
        v[i] = lcg() % primes[0];
    rns_poly_t p = poly_new(r, 0x3), shifted = poly_new(r, 0x3), diff = poly_new(r, 0x3);
    poly_from_int_array(p, v);
    poly_to_coeff(p, p);

    /* Cover the vectorized (a % 8 == 0) and scalar paths, both halves of the
     * 2N range. */
    const uint64_t shifts[7] = {0, 1, 5, 8, 16, 24, 31};
    for (int t = 0; t < 7; t++)
    {
        const uint64_t a = shifts[t];
        TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_mul_xai(shifted, p, a));
        for (int limb = 0; limb < 2; limb++)
        {
            const uint64_t q = primes[limb];
            const uint64_t *in_row = poly_limb_data(p, (uint64_t)limb);
            const uint64_t *out_row = poly_limb_data(shifted, (uint64_t)limb);
            for (uint64_t i = 0; i < N; i++)
            {
                /* X^i * X^a = (sign) X^((i + a) mod N), sign flips per wrap. */
                uint64_t pos = (i + a) % N;
                int wraps = (int)((i + a) / N) & 1;
                uint64_t expect = wraps ? (q - in_row[i]) % q : in_row[i];
                TEST_ASSERT_EQUAL_UINT64(expect, out_row[pos]);
            }
        }
        /* (X^a - 1) * p == X^a * p - p. */
        TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_mul_xai_minus1(diff, p, a));
        rns_poly_t ref = poly_new(r, 0x3);
        poly_sub(ref, shifted, p);
        TEST_ASSERT_TRUE(poly_eq(diff, ref));
        poly_free(ref);
    }

    poly_free(p);
    poly_free(shifted);
    poly_free(diff);
    ring_free(r);
}

void test_poly_permute_composition(void)
{
    enum
    {
        N = 16
    };
    uint64_t primes[2];
    ring_t r = make_ring(N, 1, primes);

    uint64_t v[N];
    for (uint64_t i = 0; i < N; i++)
        v[i] = lcg() % primes[0];
    rns_poly_t p = poly_new(r, 0x3), s1 = poly_new(r, 0x3), s2 = poly_new(r, 0x3);
    poly_from_int_array(p, v);
    poly_to_coeff(p, p);

    /* gen = 3 and gen^-1 = 11 (3 * 11 = 33 == 1 mod 2N = 32): the
     * automorphisms must compose to the identity. */
    TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_permute(s1, p, 3));
    TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_permute(s2, s1, 11));
    TEST_ASSERT_TRUE(poly_eq(p, s2));

    poly_free(p);
    poly_free(s1);
    poly_free(s2);
    ring_free(r);
}

/* ------------------------------------------------------ slot operations ---- */

void test_poly_inverse_slots(void)
{
    enum
    {
        N = 16
    };
    uint64_t primes[2];
    ring_t r = make_ring(N, 1, primes);

    rns_poly_t p = poly_new(r, 0x3), inv = poly_new(r, 0x3), one = poly_new(r, 0x3);
    /* Nonzero evaluation slots, loaded directly in EVAL form. */
    for (int limb = 0; limb < 2; limb++)
    {
        for (uint64_t i = 0; i < N; i++)
            poly_limb_data(p, (uint64_t)limb)[i] = 1 + (lcg() % (primes[limb] - 1));
    }
    poly_assume_domain(p, VFHE_EVAL);

    TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_inverse(inv, p));
    TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_mul(one, p, inv));
    for (int limb = 0; limb < 2; limb++)
    {
        for (uint64_t i = 0; i < N; i++)
            TEST_ASSERT_EQUAL_UINT64(1, poly_limb_data(one, (uint64_t)limb)[i]);
    }

    poly_free(p);
    poly_free(inv);
    poly_free(one);
    ring_free(r);
}

void test_poly_slot_ops(void)
{
    enum
    {
        N = 16
    };
    uint64_t primes[2];
    ring_t r = make_ring(N, 2, primes); /* split_degree 2: 2 blocks x 8 slots */
    const uint64_t ps = N / 2;

    rns_poly_t p = poly_new(r, 0x3), rot = poly_new(r, 0x3), back = poly_new(r, 0x3);
    for (int limb = 0; limb < 2; limb++)
    {
        for (uint64_t i = 0; i < N; i++)
            poly_limb_data(p, (uint64_t)limb)[i] = lcg() % primes[limb];
    }
    poly_assume_domain(p, VFHE_EVAL);

    /* Rotate r then ps - r is the identity on each block. */
    TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_rotate_slots(rot, p, 3));
    TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_rotate_slots(back, rot, ps - 3));
    TEST_ASSERT_TRUE(poly_eq(p, back));

    /* Rotation moves slot k + 3 into slot k, blockwise. */
    for (uint64_t k = 0; k < ps; k++)
    {
        TEST_ASSERT_EQUAL_UINT64(poly_limb_data(p, 0)[(k + 3) % ps], poly_limb_data(rot, 0)[k]);
        TEST_ASSERT_EQUAL_UINT64(poly_limb_data(p, 0)[ps + (k + 3) % ps],
                                 poly_limb_data(rot, 0)[ps + k]);
    }

    /* Broadcast fills every slot of each block with slot 2's value. */
    rns_poly_t bc = poly_new(r, 0x3);
    TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_broadcast_slot(bc, p, 2));
    for (uint64_t k = 0; k < ps; k++)
    {
        TEST_ASSERT_EQUAL_UINT64(poly_limb_data(p, 0)[2], poly_limb_data(bc, 0)[k]);
        TEST_ASSERT_EQUAL_UINT64(poly_limb_data(p, 0)[ps + 2], poly_limb_data(bc, 0)[ps + k]);
    }

    /* copy_slot writes exactly one slot per block. */
    rns_poly_t cp = poly_new(r, 0x3);
    poly_copy(cp, p);
    TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_copy_slot(cp, 5, p, 1));
    TEST_ASSERT_EQUAL_UINT64(poly_limb_data(p, 0)[1], poly_limb_data(cp, 0)[5]);
    TEST_ASSERT_EQUAL_UINT64(poly_limb_data(p, 0)[0], poly_limb_data(cp, 0)[0]);

    poly_free(p);
    poly_free(rot);
    poly_free(back);
    poly_free(bc);
    poly_free(cp);
    ring_free(r);
}

/* --------------------------------------------------- gadget decomposition ---- */

void test_poly_decompose_digit_reconstruction(void)
{
    enum
    {
        N = 16
    };
    uint64_t primes[2];
    ring_t r = make_ring(N, 1, primes);

    /* Composite values ride on the last active limb (limb 1). */
    rns_poly_t p = poly_new(r, 0x3);
    uint64_t vals[N];
    for (uint64_t i = 0; i < N; i++)
    {
        vals[i] = lcg() & 0xFFFFF; /* 20-bit values, 5 digits of 4 bits */
        poly_limb_data(p, 1)[i] = vals[i];
        poly_limb_data(p, 0)[i] = 0;
    }
    poly_assume_domain(p, VFHE_COEFF);

    const uint64_t log_base = 4, levels = 5;
    rns_poly_t digit = poly_new(r, 0x3);
    uint64_t recon[N];
    memset(recon, 0, sizeof(recon));
    for (uint64_t l = 0; l < levels; l++)
    {
        TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_decompose_digit(digit, p, log_base, l));
        for (uint64_t i = 0; i < N; i++)
        {
            /* Digits replicate into every active limb. */
            TEST_ASSERT_EQUAL_UINT64(poly_limb_data(digit, 0)[i], poly_limb_data(digit, 1)[i]);
            recon[i] |= poly_limb_data(digit, 0)[i] << (log_base * l);
        }
    }
    TEST_ASSERT_EQUAL_UINT64_ARRAY(vals, recon, N);

    poly_free(p);
    poly_free(digit);
    ring_free(r);
}

/* ------------------------------------------------------------- digest ---- */

void test_poly_digest(void)
{
    enum
    {
        N = 16
    };
    uint64_t primes[2];
    ring_t r = make_ring(N, 1, primes);
    uint64_t v[N];
    for (uint64_t i = 0; i < N; i++)
        v[i] = i * 3 + 1;

    rns_poly_t p = poly_new(r, 0x3);
    poly_from_int_array(p, v);

    uint64_t d1[VFHE_POLY_DIGEST_WORDS], d2[VFHE_POLY_DIGEST_WORDS];
    poly_digest(d1, p);
    poly_digest(d2, p);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(d1, d2, VFHE_POLY_DIGEST_WORDS); /* deterministic */

    poly_limb_data(p, 0)[0] ^= 1; /* single-bit sensitivity */
    poly_digest(d2, p);
    TEST_ASSERT_TRUE(memcmp(d1, d2, sizeof(d1)) != 0);

    poly_free(p);
    ring_free(r);
}

/* ----------------------------------------------------------- error paths ---- */

void test_error_paths(void)
{
    enum
    {
        N = 16
    };
    uint64_t primes[2];
    ring_t r = make_ring(N, 2, primes);
    uint64_t v[N];
    for (uint64_t i = 0; i < N; i++)
        v[i] = i;

    rns_poly_t a = poly_new(r, 0x3), b = poly_new(r, 0x3), out = poly_new(r, 0x3);
    poly_from_int_array(a, v);
    poly_from_int_array(b, v);

    /* Aliased multiplication is rejected. */
    TEST_ASSERT_EQUAL_INT(VFHE_ERR_ARG, poly_mul(a, a, b));
    /* Galois permutation needs COEFF operands. */
    TEST_ASSERT_EQUAL_INT(VFHE_ERR_DOMAIN, poly_permute(out, a, 3));
    /* Monomial shift needs a complete NTT (split_degree == 1). */
    poly_to_coeff(a, a);
    TEST_ASSERT_EQUAL_INT(VFHE_ERR_UNSUPPORTED, poly_mul_xai(out, a, 1));
    /* Slot rotation cannot run in place. */
    poly_to_eval(a, a);
    TEST_ASSERT_EQUAL_INT(VFHE_ERR_ARG, poly_rotate_slots(a, a, 1));
    /* Tower division needs COEFF. */
    TEST_ASSERT_EQUAL_INT(VFHE_ERR_DOMAIN, poly_div_round(a, 0x2));
    /* Slot inversion needs a complete NTT on this split-degree-2 ring... */
    TEST_ASSERT_EQUAL_INT(VFHE_ERR_UNSUPPORTED, poly_inverse(out, a));

    poly_free(a);
    poly_free(b);
    poly_free(out);
    ring_free(r);

    /* ...and rejects zero slots on a split-degree-1 ring. */
    ring_t r1 = make_ring(N, 1, primes);
    rns_poly_t z = poly_new(r1, 0x3), zout = poly_new(r1, 0x3);
    poly_zero(z);
    poly_assume_domain(z, VFHE_EVAL);
    TEST_ASSERT_EQUAL_INT(VFHE_ERR_NOT_INVERTIBLE, poly_inverse(zout, z));
    poly_free(z);
    poly_free(zout);
    ring_free(r1);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_poly_mul_split1);
    RUN_TEST(test_poly_mul_split2);
    RUN_TEST(test_poly_add_scalar_domains);
    RUN_TEST(test_poly_fused_mul_split1);
    RUN_TEST(test_poly_fused_mul_split2);
    RUN_TEST(test_poly_mul_xai);
    RUN_TEST(test_poly_permute_composition);
    RUN_TEST(test_poly_inverse_slots);
    RUN_TEST(test_poly_slot_ops);
    RUN_TEST(test_poly_decompose_digit_reconstruction);
    RUN_TEST(test_poly_digest);
    RUN_TEST(test_error_paths);
    return UNITY_END();
}
