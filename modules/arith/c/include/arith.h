// SPDX-License-Identifier: Apache-2.0
/**
 * @file arith.h
 * @brief Umbrella header for the vfhe arithmetic engine.
 *
 * The engine is layered; each layer has its own header and depends only on the
 * layers below it:
 *
 *   arith/nt.h     number theory on plain integers (primality, roots of unity)
 *   arith/zq.h     Z_q scalar/array arithmetic for one prime (backend vtable)
 *   arith/ntt.h    negacyclic NTT plan for one prime (twiddles + transform)
 *   arith/ring.h   ring context: RNS prime pool x ring layout x mul strategy
 *   arith/poly.h   RNS polynomials (domain-tagged) and their operations
 *   arith/tower.h  RNS tower ops: base conversion, rescaling, lifting
 *   arith/mp.h     multi-precision (base-2^52) polynomials + CRT bridge
 *   arith/cfft.h   complex FFT for CKKS-style encoding
 *
 * Include this header for the whole C API, or the individual headers for a
 * single layer. The Python-facing ABI is the strict subset declared in
 * python/cdef/arith.cdef (opaque handles + accessor functions only).
 */
#ifndef VFHE_ARITH_H
#define VFHE_ARITH_H

#include "arith/config.h"
#include "arith/error.h"
#include "arith/nt.h"
#include "arith/zq.h"
#include "arith/ntt.h"
#include "arith/ring.h"
#include "arith/poly.h"
#include "arith/tower.h"
#include "arith/mp.h"
#include "arith/cfft.h"
#include "arith/zqvec.h"

#endif // VFHE_ARITH_H
