// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
//
// Sample operations over an RNS ring, run in parallel over its primes. Each
// step these cover treats every prime's row on its own, so one row is one
// task and the result does not depend on how many threads run. The tasks go
// through vfhe_parallel_for, so they stay within the library limit and the
// calling thread's local number of threads.
#ifndef VFHE_MLWE_RNS_ROWS_H
#define VFHE_MLWE_RNS_ROWS_H

#include "mlwe.h"
#include <arith_trace.h>
#include <parallel.h>

#include <stdbool.h>

// Component k of a sample: a[k] for k < r, and the body b for k == r.
static inline ArithElement *mlwe_component(MLWE c, uint64_t k)
{
    return k < c->r ? &c->a[k] : &c->b;
}

// The RNS polynomial of component k of a sample over an RNS ring.
static inline RNS_Polynomial mlwe_component_rows(MLWE c, uint64_t k)
{
    return arith_rns_polynomial(mlwe_component(c, k));
}

// The slot of the quadratic term s_p * s_q, p <= q < r, in a tensor product
// of rank-r samples. mlwe_tensor_product lays the slots out as the quadratic
// pairs in lexicographic order, then the r linear terms, then the constant.
static inline uint64_t mlwe_quadratic_slot(uint64_t r, uint64_t p, uint64_t q)
{
    return p * r - p * (p - 1) / 2 + (q - p);
}

// The factors of slot k <= mlwe_extended_rank(r) of the tensor product of x
// and y, component r standing for the body: x[*i] * y[*j], plus
// x[*j] * y[*i] when it returns true.
static inline bool mlwe_tensor_slot_factors(uint64_t r, uint64_t k, uint64_t *i, uint64_t *j)
{
    const uint64_t quadratic = r * (r + 1) / 2;
    if (k >= quadratic)
    {
        *i = k - quadratic;
        *j = r;
        return *i < r;
    }
    uint64_t p = 0;
    while (k >= mlwe_quadratic_slot(r, p + 1, p + 1))
        p++;
    *i = p;
    *j = p + (k - mlwe_quadratic_slot(r, p, p));
    return *i != *j;
}

// `n_threads` as a batch of samples may use them, or one thread when the
// samples go through the generic interface (ARITH_GENERIC_PATH), so that
// their operations are recorded in the order of the batch.
static inline uint64_t one_thread_on_generic_path(uint64_t n_threads)
{
    return ARITH_GENERIC_PATH() ? 1 : n_threads;
}

// Below this ring dimension a row is too little work to hand to another
// thread: the hand-off costs more than the row.
#define ROWS_MIN_N 4096

// Calls body(ctx, row, part) for every row in `mask` and every part < parts,
// in parallel unless the ring dimension N makes a row too little work.
void rns_rows_for(uint64_t N, uint64_t mask, uint64_t parts,
                  void (*body)(void *ctx, uint64_t row, uint64_t part), void *ctx);
// rns_rows_for over several items at once, each with its own rows and part
// count: calls body(ctx, item, row, part) for every item < n, every row in
// masks[item] and every part < parts[item], all as one parallel loop on up to
// n_threads threads (0: the library limit).
void rns_rows_batch_for(uint64_t n, const uint64_t *masks, const uint64_t *parts,
                        void (*body)(void *ctx, uint64_t item, uint64_t row, uint64_t part),
                        void *ctx, uint64_t n_threads);
// How many parts to cut the work on each row into, for a loop over `rows`
// rows x parts on up to n_threads threads (0: the library limit), when a
// row's work is `items` independent pieces that a part can take a share of.
// 1 -- one task per row -- when the rows alone are at least as many as the
// threads or N is below ROWS_MIN_N; otherwise enough parts for about two
// tasks per thread, and never more than `items`.
uint64_t rns_rows_parts(uint64_t N, uint64_t rows, uint64_t items, uint64_t n_threads);

// mlwe_RNSc_to_RNS / mlwe_RNS_to_RNSc over an RNS ring.
void mlwe_rns_convert(MLWE out, MLWE in, ArithDomain to);
// The automorphism X -> X^gen of every element (arith_permute) over an RNS
// ring; `out` and `in` are distinct.
void mlwe_rns_permute(MLWE out, MLWE in, uint64_t gen);
// mlwe_round_division over an RNS ring: `out` moves to `to`, a quotient ring.
void mlwe_rns_round_division(MLWE out, ArithRing to);
// mlwe_mod_down over an RNS ring, where `in`'s ring has primes `out`'s does
// not.
void mlwe_rns_mod_down(MLWE out, MLWE in, const ArithElement *const *addend);
// mlwe_tensor_product over an RNS ring. Returns -1, doing nothing, unless
// every operand element is in the mul domain over the same primes.
int mlwe_rns_tensor_product(ArithElement *out, MLWE in1, MLWE in2);

// Batches of hybrid key switches, run step by step over the rows of all
// their samples (keyswitch_batch.c). Results are those of the one-at-a-time
// operations, bit for bit, whatever the number of threads.

// Whether a key switch into `out` with `ksk` is a hybrid (GHS) key switch
// over RNS rings, the only kind the batches below take: both rings are RNS
// rings, and the key's ring has every prime of out's ring and more -- the
// special primes, which the ModDown divides by. A BV key switch (the key
// over out's ring itself) or another ring implementation is not one.
bool mlwe_is_rns_hybrid_keyswitch(RNSc_MLWE out, RNS_MLWE_KS_Key ksk);

// One key switch of a batch: `out` gets the key switch of `in` with `ksk`,
// as mlwe_RNSc_GHS_hybrid_keyswitch computes it. Then, unless `rescale_to` is
// NULL, it is round-divided to that ring (as mlwe_round_division, so a
// quotient of out's ring), and with `to_mul_domain` it ends in the mul
// domain rather than the coefficient one. `in` is in the coefficient domain,
// over out's ring, and is not `out`.
typedef struct
{
    RNSc_MLWE out;
    RNSc_MLWE in;
    RNS_MLWE_KS_Key ksk;
    ArithRing rescale_to;
    bool to_mul_domain;
} KeySwitch;

// The n key switches, on up to n_threads threads (0: the library limit).
void mlwe_rns_hybrid_keyswitch_batch(KeySwitch *ks, uint64_t n, uint64_t n_threads);
// out[j] = in1[j] * in2[j] relinearized with `ksk`, as mlwe_multiply, then
// round-divided to `rescale_to` and taken to the mul domain as a KeySwitch
// says. Operands not in the mul domain are converted in place.
void mlwe_rns_multiply_batch(RNSc_MLWE *out, RNS_MLWE *in1, RNS_MLWE *in2, RNS_MLWE_KS_Key ksk,
                             ArithRing rescale_to, bool to_mul_domain, uint64_t n,
                             uint64_t n_threads);
// out[j] = Aut_gens[j](in[j]) key-switched with ksks[j], as
// mlwe_automorphism_RNSc_GHS; in[j] may be in either domain, and is left
// unchanged unless it is out[j].
void mlwe_rns_automorphism_batch(RNSc_MLWE *out, RNSc_MLWE *in, const uint64_t *gens,
                                 RNS_MLWE_KS_Key *ksks, uint64_t n, uint64_t n_threads);

// Multiplies every element of `c` by 2^-1 modulo its ring's modulus, in
// place and in either domain; exact, since every prime is odd.
void mlwe_rns_halve(MLWE c);

#endif // VFHE_MLWE_RNS_ROWS_H
