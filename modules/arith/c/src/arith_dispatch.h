// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#ifndef VFHE_ARITH_DISPATCH_H
#define VFHE_ARITH_DISPATCH_H

#include <arith_generic.h>

// The compiled-in implementations' slot functions, for the dispatcher's
// switch arms. Internal to the arith module: nothing here is API, the
// public boundary is arith_generic.h, and this header lives under c/src on
// purpose.
//
// The contract for adding an implementation: pick a tag in ArithImpl, define
// every function below with your prefix -- a slot you do not implement
// returns ARITH_UNIMPLEMENTED rather than being absent, since a switch arm,
// unlike a method-table entry, cannot be NULL -- declare them here, and add
// your case to ARITH_DISPATCH in arith_generic.c. Names match the method
// table's members exactly.

ArithStatus arith_rns_new_element(ArithRing ring, ArithElement *out);
ArithStatus arith_rns_new_like(ArithRing ring, const ArithElement *model, ArithElement *out);
void arith_rns_free_element(ArithRing ring, ArithElement *element);
ArithStatus arith_rns_copy(ArithRing ring, ArithElement *out, const ArithElement *in);
ArithStatus arith_rns_zero(ArithRing ring, ArithElement *out);
ArithStatus arith_rns_to_mul(ArithRing ring, ArithElement *element);
ArithStatus arith_rns_to_canonical(ArithRing ring, ArithElement *element);
ArithStatus arith_rns_add(ArithRing ring, ArithElement *out, const ArithElement *a,
                          const ArithElement *b);
ArithStatus arith_rns_sub(ArithRing ring, ArithElement *out, const ArithElement *a,
                          const ArithElement *b);
ArithStatus arith_rns_mul(ArithRing ring, ArithElement *out, const ArithElement *a,
                          const ArithElement *b);
ArithStatus arith_rns_mul_addto(ArithRing ring, ArithElement *out, const ArithElement *a,
                                const ArithElement *b);
ArithStatus arith_rns_scale_int(ArithRing ring, ArithElement *out, const ArithElement *a,
                                uint64_t scale);
ArithStatus arith_rns_mul_subto(ArithRing ring, ArithElement *out, const ArithElement *a,
                                const ArithElement *b);
ArithStatus arith_rns_scale_addto(ArithRing ring, ArithElement *out, const ArithElement *a,
                                  uint64_t scale);
ArithStatus arith_rns_scale_by(ArithRing ring, ArithElement *out, const ArithElement *a,
                               ArithScalar scale);
ArithStatus arith_rns_permute(ArithRing ring, ArithElement *out, const ArithElement *a,
                              uint64_t gen);
ArithStatus arith_rns_mul_by_monomial(ArithRing ring, ArithElement *out, const ArithElement *a,
                                      uint64_t power, int minus_one);
ArithStatus arith_rns_sample_uniform(ArithRing ring, ArithElement *out);
ArithStatus arith_rns_sample_gaussian(ArithRing ring, ArithElement *out, double sigma);
ArithStatus arith_rns_from_int_array(ArithRing ring, ArithElement *out, const uint64_t *values,
                                     uint64_t count);
ArithStatus arith_rns_round_division(ArithRing ring, ArithElement *element, ArithRing to);
ArithStatus arith_rns_mod_reduce_lifted(ArithRing ring, ArithElement *out, const ArithElement *a,
                                        ArithRing from);
ArithStatus arith_rns_scalar_new(ArithRing ring, const uint64_t *per_component, ArithScalar *out);
void arith_rns_scalar_free(ArithRing ring, ArithScalar *scalar);

#endif // VFHE_ARITH_DISPATCH_H
