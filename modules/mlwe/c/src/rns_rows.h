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
#include "util.h"

// Calls body(ctx, row, part) for every row in `mask` and every part < parts,
// in parallel unless the ring dimension N makes a row too little work.
void rns_rows_for(uint64_t N, uint64_t mask, uint64_t parts,
                  void (*body)(void *ctx, uint64_t row, uint64_t part), void *ctx);

// mlwe_RNSc_to_RNS / mlwe_RNS_to_RNSc over an RNS ring.
void mlwe_rns_convert(MLWE out, MLWE in, ArithDomain to);
// The automorphism X -> X^gen of every element (arith_permute) over an RNS
// ring; `out` and `in` are distinct.
void mlwe_rns_permute(MLWE out, MLWE in, uint64_t gen);
// mlwe_round_division over an RNS ring: `out` moves to `to`, a quotient ring.
void mlwe_rns_round_division(MLWE out, ArithRing to);
// mlwe_tensor_product over an RNS ring. Returns -1, doing nothing, unless
// every operand element is in the mul domain over the same primes.
int mlwe_rns_tensor_product(ArithElement *out, MLWE in1, MLWE in2);

// Multiplies every element of `c` by 2^-1 modulo its ring's modulus, in
// place and in either domain; exact, since every prime is odd.
void mlwe_rns_halve(MLWE c);

#endif // VFHE_MLWE_RNS_ROWS_H
