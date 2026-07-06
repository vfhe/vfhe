// SPDX-License-Identifier: Apache-2.0
/**
 * @file ntt_backends.h
 * @brief Internal NTT backend interface + shared twiddle-schedule helpers.
 *
 * A backend owns two things for its tier: the packed memory layout of the
 * twiddle tables and the butterfly kernels that consume them. Its single
 * entry point fills the function pointers and table handles of an ::ntt_plan
 * whose n / zq / root fields are already set. ::ntt_plan_init selects the
 * backend that matches the zq context's kernel tier.
 */
#ifndef VFHE_NTT_BACKENDS_H
#define VFHE_NTT_BACKENDS_H

#include <arith/config.h>
#include <arith/ntt.h>

// --- Shared raw twiddle schedules (ntt_tables.c) -----------------------------

/**
 * Bit-reversed power table: out[bitrev(i)] = root^i mod q, for i in [0, n).
 * This is the schedule the forward (CT, natural->bitrev) pass consumes.
 * Caller frees.
 */
uint64_t *ntt_tables_bitrev_powers(uint64_t n, uint64_t q, uint64_t root);

/**
 * Gentleman-Sande schedule: flatten the bit-reversed inverse-root powers in
 * per-stage order (stage sizes n/2, n/4, ..., 1), as consumed by the SIMD
 * inverse kernels. Caller frees.
 */
uint64_t *ntt_tables_gs_schedule(uint64_t n, const uint64_t *bitrev_rou);

// --- Backends ----------------------------------------------------------------

/** Portable scalar backend (any q). */
void ntt_backend_portable_init(ntt_plan *plan);

#if VFHE_MP_SIMD
/** AVX-512 backend, q < 2^32 tier. */
void ntt_backend_avx512_32_init(ntt_plan *plan);
/** AVX-512 IFMA backend, q < 2^50 tier. */
void ntt_backend_avx512_50_init(ntt_plan *plan);
/** AVX-512 backend, q < 2^63 tier. */
void ntt_backend_avx512_64_init(ntt_plan *plan);

/**
 * Shared AVX-512 twiddle packing (ntt_avx512_tables.c): replicate/interleave
 * the raw schedules into per-stage __m512i arrays together with their
 * Barrett preconditioners for the given @p shift (32, 52, or 64).
 */
void ntt_avx512_pack_fwd(uint64_t n, uint64_t q, uint64_t shift, const uint64_t *bitrev_rou,
                         void ***out_ws, void ***out_ws_pre);
void ntt_avx512_pack_inv(uint64_t n, uint64_t q, uint64_t shift, const uint64_t *gs_schedule,
                         void ***out_ws, void ***out_ws_pre);
/** Free tables produced by the packers. */
void ntt_avx512_pack_free(void **ws, void **ws_pre, uint64_t n);
#endif

#endif // VFHE_NTT_BACKENDS_H
