/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file fuzz_ntt.c
 * @brief libFuzzer harness for the arith NTT surface.
 *
 * Each input is decoded into a ring size N, a special prime, and N coefficients,
 * then driven through the forward and inverse transforms. Two properties are
 * asserted so a violation aborts (and the fuzzer minimizes the crash):
 *
 *   1. forward-then-inverse NTT is the identity;
 *   2. none of the exercised kernels read/write out of bounds (ASan/UBSan).
 *
 * Deterministic per input (no RNG), so a saved crash reproduces exactly. Built
 * by scripts/build_fuzzers.py with -fsanitize=fuzzer,address,undefined.
 */
#include <stdint.h>
#include <stdlib.h>

#include <arith.h>

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

    static const uint64_t Ns[4] = {8, 16, 64, 256};
    const uint64_t n = Ns[take_u8(&c) & 3];
    const uint64_t qbits = 20 + (take_u8(&c) % 43); /* 20..62 */
    const uint64_t q = next_special_prime((uint64_t)1 << qbits, n, true);

    NTT_proc proc = ntt_new_proc(n, q);
    if (!proc)
        return 0;

    uint64_t *in = malloc(n * sizeof(uint64_t));
    uint64_t *fwd = malloc(n * sizeof(uint64_t));
    uint64_t *back = malloc(n * sizeof(uint64_t));
    if (in && fwd && back)
    {
        for (uint64_t i = 0; i < n; i++)
            in[i] = take_u64(&c) % q;
        ntt_forward(fwd, in, proc);
        ntt_reverse(back, fwd, proc);
        for (uint64_t i = 0; i < n; i++)
            if (back[i] != in[i])
                abort(); /* NTT roundtrip must be the identity */
    }

    free(in);
    free(fwd);
    free(back);
    ntt_free_proc(proc);
    return 0;
}
