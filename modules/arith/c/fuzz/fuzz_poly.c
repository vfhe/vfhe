/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file fuzz_poly.c
 * @brief libFuzzer harness for the arith polynomial load / transform surface.
 *
 * Each input byte string is decoded into a ring shape (N, split_degree) and two
 * polynomials, then driven through the memory-heavy paths -- coefficient load
 * (forward NTT), evaluation-domain multiplication, the NTT roundtrip, the
 * digest, and gadget decomposition. Two properties are asserted so a violation
 * aborts (and the crash is minimized by the fuzzer):
 *
 *   1. forward-then-inverse NTT is the identity;
 *   2. none of the exercised kernels read/write out of bounds (ASan/UBSan).
 *
 * Built by scripts/build_fuzzers.py with -fsanitize=fuzzer,address,undefined
 * and -DVFHE_RNG_TESTING, so any sampling is reproducible from the input.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "arith.h"
#include "rng_internal.h" /* rng_test_set_seed */

/* Little cursor over the fuzzer input; reads zero-padded once exhausted. */
typedef struct
{
    const uint8_t *p;
    size_t n, i;
} cursor;

static uint64_t take_u64(cursor *c)
{
    uint64_t v = 0;
    for (int b = 0; b < 8; b++)
        v = (v << 8) | (uint64_t)(c->i < c->n ? c->p[c->i++] : 0);
    return v;
}

static uint8_t take_u8(cursor *c) { return (uint8_t)(c->i < c->n ? c->p[c->i++] : 0); }

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    cursor c = {data, size, 0};
    rng_test_set_seed(take_u64(&c)); /* deterministic sampling per input */

    static const uint64_t Ns[4] = {8, 16, 32, 64};
    const uint64_t N = Ns[take_u8(&c) & 3];
    uint64_t split = (take_u8(&c) & 1) ? 2 : 1;
    const uint64_t n_ntt = N / split;

    uint64_t primes[2];
    primes[0] = nt_next_ntt_prime((uint64_t)1 << 30, n_ntt, false);
    primes[1] = nt_next_ntt_prime(primes[0], n_ntt, false);
    ring_t r = ring_new(primes, split, N, 2);
    if (!r)
        return 0;

    uint64_t *v = (uint64_t *)malloc(N * sizeof(uint64_t));
    if (!v)
    {
        ring_free(r);
        return 0;
    }

    rns_poly_t a = poly_new(r, 0x3), b = poly_new(r, 0x3);
    for (uint64_t i = 0; i < N; i++)
        v[i] = take_u64(&c);
    poly_from_int_array(a, v); /* forward NTT, ends EVAL */
    for (uint64_t i = 0; i < N; i++)
        v[i] = take_u64(&c);
    poly_from_int_array(b, v);

    /* Evaluation-domain multiplication (split-1 or split-k cross-term kernel). */
    rns_poly_t prod = poly_new(r, 0x3);
    poly_mul(prod, a, b);

    /* NTT roundtrip must be the identity. */
    rns_poly_t chk = poly_new(r, 0x3);
    poly_copy(chk, a);
    poly_to_coeff(chk, chk);
    poly_to_eval(chk, chk);
    if (!poly_eq(chk, a))
        abort();

    /* Digest never faults. */
    uint64_t dig[VFHE_POLY_DIGEST_WORDS];
    poly_digest(dig, prod);

    /* Gadget decomposition (needs a complete NTT, i.e. split_degree == 1). */
    if (split == 1)
    {
        rns_poly_t d = poly_new(r, 0x3);
        poly_to_coeff(a, a);
        poly_decompose_digit(d, a, 8, 0);
        poly_free(d);
    }

    poly_free(a);
    poly_free(b);
    poly_free(prod);
    poly_free(chk);
    free(v);
    ring_free(r);
    return 0;
}
