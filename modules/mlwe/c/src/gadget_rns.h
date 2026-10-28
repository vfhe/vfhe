// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
//
// Gadget products as row tasks, for callers that schedule the tasks
// themselves (gadget_mul_*to_polynomial, and the batched key switches of
// keyswitch_batch.c).
#ifndef VFHE_MLWE_GADGET_RNS_H
#define VFHE_MLWE_GADGET_RNS_H

#include "mlwe.h"

typedef struct ElementDecomposition ElementDecomposition;

// A sum of gadget products, out -= sum_k <digits(poly[k]), ksk[k]> (or +=,
// see `subtract`): every poly[k] is decomposed and each digit multiplied by
// its key.
//
// The work is cut into tasks by row of the key ring and by digit group. All
// the digits of all the terms are numbered in key order, and digit d belongs
// to group d mod `groups`. gadget_sum_row does the digits of one group on
// one row, into an accumulator the caller gives it; the caller then adds the
// groups' accumulators. That sum is exact (it is modular addition), so the
// result does not depend on how many groups there were.
typedef struct
{
    uint64_t n_terms;
    ElementDecomposition *terms;
    RNS_MLWE **ksk; // a copy of the key arrays given
    // Per digit, in order: its term and its index among the term's digits.
    uint64_t n_digits;
    uint64_t *term_of, *digit_of;
    // The rows of the key ring, and the number of components of a sample.
    uint64_t key_mask, components;
    int subtract;
} GadgetSum;

// Prepares the sum of the n_terms products poly[k] x ksk[k]: ksk[k] is the
// key array (one key per digit, as gadget_mul_addto_polynomial takes it) for
// poly[k], all of them over one key ring and with `gadget_params`. Only what
// the decomposition needs is computed here; the digits themselves are
// computed row by row in gadget_sum_row. A poly[k] with no rows adds nothing.
void gadget_sum_init(GadgetSum *s, RNS_MLWE *const *ksk, const ArithElement *const *poly,
                     uint64_t n_terms, const GadgetParams *gadget_params, int subtract);
// Computes digits group, group + groups, ... on row `row` of the key ring and
// accumulates their products into that row of `out`, a sample over the key
// ring with as many components as the keys.
void gadget_sum_row(const GadgetSum *s, RNS_MLWE out, uint64_t group, uint64_t groups,
                    uint64_t row);
// The mask component k of an accumulator should be given once all its rows
// are done (a row task works on a view, which leaves the mask alone).
uint64_t gadget_sum_mask(const GadgetSum *s, uint64_t k);
void gadget_sum_free(GadgetSum *s);

// The gadget decomposition of one element, computed one digit and one row at a
// time, for a caller that multiplies each digit row as soon as it has it.
// gadget_decomposition_new prepares it for keys like `ksk` (one key per digit,
// as gadget_mul_addto_polynomial takes them) and reads `poly` until freed;
// gadget_decomposition_digit_row writes digit t < gadget_decomposition_digits
// into `out`, a polynomial over one row of the key ring, lifted to that row
// and in the domain gadget_sum_row multiplies digits in. An element with no
// rows has no digits.
ElementDecomposition *gadget_decomposition_new(RNS_MLWE *ksk, const ArithElement *poly,
                                               const GadgetParams *gadget_params);
uint64_t gadget_decomposition_digits(const ElementDecomposition *e);
// The parameters of digit t, as arith_decompose_digit takes them: the digit
// is of the residue modulo `primes` (one prime, or a group), unsigned or
// `centered`, and with a `log_base` digit `level` of it in that radix, the
// `dropped` lowest digits rounded away.
void gadget_decomposition_parameters(const ElementDecomposition *e, uint64_t t, uint64_t *primes,
                                     uint64_t *log_base, uint64_t *level, uint64_t *dropped,
                                     int *centered);
// Allocates `n` digits over the ring of `key`, in the domain gadget_decompose
// leaves them in, holding no value yet.
void gadget_digits_new(GadgetDigits *out, uint64_t n, RNS_Polynomial key);

// The same products through the generic interface (gadget_generic.c), one
// whole element at a time: what a ring without the row kernels would run,
// and the path taken on ARITH_GENERIC_PATH().
void gadget_mul_accumulate_generic(RNS_MLWE out, RNS_MLWE *ksk, const ArithElement *poly,
                                   int subtract, const GadgetParams *gadget_params);
void gadget_mul_automorphism_generic(RNS_MLWE out, RNS_MLWE *ksk, const GadgetDigits *digits,
                                     uint64_t gen, int subtract);
void gadget_decompose_generic(GadgetDigits *out, RNS_MLWE *ksk, const ArithElement *poly,
                              const GadgetParams *gadget_params);
void gadget_decomposition_digit_row(RNS_Polynomial out, const ElementDecomposition *e, uint64_t t);
void gadget_decomposition_free(ElementDecomposition *e);

#endif // VFHE_MLWE_GADGET_RNS_H
