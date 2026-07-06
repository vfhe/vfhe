// SPDX-License-Identifier: Apache-2.0
/**
 * @file ntt_avx512_driver.inc
 * @brief Shared AVX-512 NTT driver, included by each tier backend.
 *
 * The three tiers (32/50/64-bit) run the exact same cache-blocked recursion
 * and lane interleaving; only the butterfly arithmetic and the Barrett shift
 * differ. Each backend translation unit defines, *before* including this
 * file:
 *
 *   static inline void bfly_fwd(__m512i *X, __m512i *Y, __m512i W,
 *                               __m512i W_pre, __m512i minus_q, __m512i q2);
 *   static inline void bfly_inv(__m512i *X, __m512i *Y, __m512i W,
 *                               __m512i W_pre, __m512i minus_q, __m512i q2);
 *   static inline void bfly_inv_final(__m512i *X, __m512i *Y, __m512i inv_n,
 *                               __m512i inv_n_pre, __m512i inv_n_w,
 *                               __m512i inv_n_w_pre, __m512i minus_q,
 *                               __m512i q2, __m512i q_vec);
 *   #define NTT_TIER_SHIFT   <32 | 52 | 64>   // Barrett preconditioner shift
 *   #define NTT_BACKEND_INIT <init symbol declared in ntt_backends.h>
 *
 * and then includes this file. Everything here is `static`, so each backend
 * gets its own fully inlined copy (this is the C analogue of a template
 * instantiation).
 *
 * Algorithm notes:
 *  - Forward: Cooley-Tukey, natural -> bit-reversed, values kept lazily in
 *    [0, 4q); a final pass normalizes to [0, q).
 *  - Inverse: Gentleman-Sande, bit-reversed -> natural; the last stage fuses
 *    the 1/n (and 1/n * w) scaling into the butterfly (bfly_inv_final).
 *  - Blocks larger than 1024 lanes are split recursively so working sets
 *    stay in L1/L2; sub-16 tails use the T4/T2/T1 in-register interleaves.
 */

// --- Lane interleaves for the last three stages -------------------------------

static inline void fwd_interleave_t4(__m512i *A, __m512i *B) {
  const __m512i vperm2_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
  __m512i perm_hi = _mm512_permutexvar_epi64(vperm2_idx, *B);
  *B = _mm512_mask_blend_epi64(0xf0, perm_hi, *A);
  *A = _mm512_mask_blend_epi64(0x0f, perm_hi, *A);
  *B = _mm512_permutexvar_epi64(vperm2_idx, *B);
}

static inline void load_inv_interleaved_t2(__m512i *out1, __m512i *out2, __m512i *in) {
  __m512i v1 = _mm512_loadu_si512(in++);
  __m512i v2 = _mm512_loadu_si512(in);
  const __m512i v1_perm_idx = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
  __m512i v1_perm = _mm512_permutexvar_epi64(v1_perm_idx, v1);
  __m512i v2_perm = _mm512_permutexvar_epi64(v1_perm_idx, v2);
  *out1 = _mm512_mask_blend_epi64(0xaa, v1, v2_perm);
  *out2 = _mm512_mask_blend_epi64(0xaa, v1_perm, v2);
}

static inline void fwd_interleave_t2(__m512i *A, __m512i *B) {
  const __m512i v1_perm_idx = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
  __m512i v1_perm = _mm512_permutexvar_epi64(v1_perm_idx, *A);
  __m512i v2_perm = _mm512_permutexvar_epi64(v1_perm_idx, *B);
  *A = _mm512_mask_blend_epi64(0xcc, *A, v2_perm);
  *B = _mm512_mask_blend_epi64(0xcc, v1_perm, *B);
}

static inline void fwd_interleave_t1(__m512i *A, __m512i *B) {
  const __m512i perm_idx = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
  __m512i v1_perm = _mm512_permutexvar_epi64(perm_idx, *A);
  __m512i v2_perm = _mm512_permutexvar_epi64(perm_idx, *B);
  *A = _mm512_mask_blend_epi64(0xaa, *A, v2_perm);
  *B = _mm512_mask_blend_epi64(0xaa, v1_perm, *B);
}

static inline void fwd_reinterleave_t1(__m512i *A, __m512i *B) {
  const __m512i vperm2_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
  const __m512i v_X_out_idx = _mm512_set_epi64(7, 3, 6, 2, 5, 1, 4, 0);
  const __m512i v_Y_out_idx = _mm512_set_epi64(3, 7, 2, 6, 1, 5, 0, 4);
  *B = _mm512_permutexvar_epi64(vperm2_idx, *B);
  __m512i perm_lo = _mm512_mask_blend_epi64(0x0f, *A, *B);
  __m512i perm_hi = _mm512_mask_blend_epi64(0xf0, *A, *B);
  *A = _mm512_permutexvar_epi64(v_X_out_idx, perm_hi);
  *B = _mm512_permutexvar_epi64(v_Y_out_idx, perm_lo);
}

static inline void inv_interleave_t1(__m512i *A, __m512i *B) {
  const __m512i vperm_hi_idx = _mm512_set_epi64(6, 4, 2, 0, 7, 5, 3, 1);
  const __m512i vperm_lo_idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  const __m512i vperm2_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
  __m512i perm_lo = _mm512_permutexvar_epi64(vperm_lo_idx, *A);
  __m512i perm_hi = _mm512_permutexvar_epi64(vperm_hi_idx, *B);
  *A = _mm512_mask_blend_epi64(0x0f, perm_hi, perm_lo);
  *B = _mm512_mask_blend_epi64(0xf0, perm_hi, perm_lo);
  *B = _mm512_permutexvar_epi64(vperm2_idx, *B);
}

static inline void inv_interleave_t4(__m512i *A, __m512i *B) {
  const __m512i perm_idx = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
  __m512i v1_perm = _mm512_permutexvar_epi64(perm_idx, *A);
  __m512i v2_perm = _mm512_permutexvar_epi64(perm_idx, *B);
  *A = _mm512_mask_blend_epi64(0xcc, *A, v2_perm);
  *B = _mm512_mask_blend_epi64(0xcc, v1_perm, *B);
}

static inline void inv_reinterleave_t4(__m512i *A, __m512i *B) {
  __m256i x0 = _mm512_extracti64x4_epi64(*A, 0);
  __m256i x1 = _mm512_extracti64x4_epi64(*A, 1);
  __m256i y0 = _mm512_extracti64x4_epi64(*B, 0);
  __m256i y1 = _mm512_extracti64x4_epi64(*B, 1);
  *A = _mm512_inserti64x4(_mm512_castsi256_si512(x0), y0, 1);
  *B = _mm512_inserti64x4(_mm512_castsi256_si512(x1), y1, 1);
}

// --- Barrett preconditioner for the fused final stage --------------------------

static uint64_t tier_barrett_factor(uint64_t val, uint64_t q) {
  unsigned __int128 num = (unsigned __int128)val << NTT_TIER_SHIFT;
  return (uint64_t)(num / q);
}

// --- Forward recursion ----------------------------------------------------------

static void ct_nr_internal(__m512i *x, __m512i **ws, __m512i **w_precon, uint64_t sub_n,
                           uint64_t q, size_t level, size_t offset_i) {
  __m512i minus_q = _mm512_set1_epi64(-(long long)q);
  __m512i q2 = _mm512_set1_epi64((long long)(2 * q));

  if (sub_n <= 1024) {
    size_t l = level;
    for (; (1ULL << (l - level + 3)) < sub_n; l++) {
      const __m512i *wsv = ws[l];
      const __m512i *ws_precon_v = w_precon[l];
      const uint64_t t = sub_n >> (l - level + 4);
      size_t m = 1ULL << (l - level);
      size_t start_i = offset_i << (l - level);
      for (size_t i = 0; i < m; i++) {
        const uint64_t slice = 2 * i * t;
        const __m512i w_i = wsv[start_i + i];
        const __m512i wp_i = ws_precon_v[start_i + i];
        for (size_t j = slice; j < slice + t; j++) {
          bfly_fwd(&x[j], &x[j + t], w_i, wp_i, minus_q, q2);
        }
      }
    }

    if (sub_n >= 16) {
      size_t w_offset = offset_i * (sub_n / 16);

      {  // t = 4
        const __m512i *wsv = ws[l];
        const __m512i *ws_precon_v = w_precon[l];
        for (size_t i = 0; i < sub_n / 16; i++) {
          const uint64_t slice = 2 * i;
          const __m512i w_i = wsv[w_offset + i];
          const __m512i wp_i = ws_precon_v[w_offset + i];
          fwd_interleave_t4(&x[slice], &x[slice + 1]);
          bfly_fwd(&x[slice], &x[slice + 1], w_i, wp_i, minus_q, q2);
        }
      }
      l++;

      {  // t = 2
        const __m512i *wsv = ws[l];
        const __m512i *ws_precon_v = w_precon[l];
        for (size_t i = 0; i < sub_n / 16; i++) {
          const uint64_t slice = 2 * i;
          const __m512i w_i = wsv[w_offset + i];
          const __m512i wp_i = ws_precon_v[w_offset + i];
          fwd_interleave_t2(&x[slice], &x[slice + 1]);
          bfly_fwd(&x[slice], &x[slice + 1], w_i, wp_i, minus_q, q2);
        }
      }
      l++;

      {  // t = 1
        const __m512i *wsv = ws[l];
        const __m512i *ws_precon_v = w_precon[l];
        for (size_t i = 0; i < sub_n / 16; i++) {
          const uint64_t slice = 2 * i;
          const __m512i w_i = wsv[w_offset + i];
          const __m512i wp_i = ws_precon_v[w_offset + i];
          fwd_interleave_t1(&x[slice], &x[slice + 1]);
          bfly_fwd(&x[slice], &x[slice + 1], w_i, wp_i, minus_q, q2);
          fwd_reinterleave_t1(&x[slice], &x[slice + 1]);
        }
      }
    }
  } else {
    size_t t = sub_n >> 4;
    const __m512i w_i = ws[level][offset_i];
    const __m512i wp_i = w_precon[level][offset_i];
    for (size_t j = 0; j < t; j++) {
      bfly_fwd(&x[j], &x[j + t], w_i, wp_i, minus_q, q2);
    }
    ct_nr_internal(x, ws, w_precon, sub_n / 2, q, level + 1, 2 * offset_i);
    ct_nr_internal(x + t, ws, w_precon, sub_n / 2, q, level + 1, 2 * offset_i + 1);
  }
}

// --- Inverse recursion ------------------------------------------------------------

static void gs_rn_internal(__m512i *x, __m512i **ws, __m512i **w_precon, uint64_t sub_n,
                           uint64_t q, size_t l_merge, size_t offset_i, int is_top_level,
                           uint64_t inv_n) {
  __m512i minus_q = _mm512_set1_epi64(-(long long)q);
  __m512i q2 = _mm512_set1_epi64((long long)(2 * q));

  if (sub_n <= 1024) {
    size_t l = 0;
    if (sub_n >= 16) {
      size_t w_offset = offset_i * (sub_n / 16);

      {  // t = 1
        const __m512i *wsv = ws[l];
        const __m512i *ws_precon_v = w_precon[l];
        for (size_t i = 0; i < sub_n / 16; i++) {
          const uint64_t slice = 2 * i;
          const __m512i w_i = wsv[w_offset + i];
          const __m512i wp_i = ws_precon_v[w_offset + i];
          inv_interleave_t1(&x[slice], &x[slice + 1]);
          bfly_inv(&x[slice], &x[slice + 1], w_i, wp_i, minus_q, q2);
        }
      }
      l++;

      {  // t = 2
        const __m512i *wsv = ws[l];
        const __m512i *ws_precon_v = w_precon[l];
        for (size_t i = 0; i < sub_n / 16; i++) {
          const uint64_t slice = 2 * i;
          const __m512i w_i = wsv[w_offset + i];
          const __m512i wp_i = ws_precon_v[w_offset + i];
          __m512i v_X, v_Y;
          load_inv_interleaved_t2(&v_X, &v_Y, &x[slice]);
          bfly_inv(&v_X, &v_Y, w_i, wp_i, minus_q, q2);
          x[slice] = v_X;
          x[slice + 1] = v_Y;
        }
      }
      l++;

      {  // t = 4
        const __m512i *wsv = ws[l];
        const __m512i *ws_precon_v = w_precon[l];
        for (size_t i = 0; i < sub_n / 16; i++) {
          const uint64_t slice = 2 * i;
          const __m512i w_i = wsv[w_offset + i];
          const __m512i wp_i = ws_precon_v[w_offset + i];
          inv_interleave_t4(&x[slice], &x[slice + 1]);
          bfly_inv(&x[slice], &x[slice + 1], w_i, wp_i, minus_q, q2);
          inv_reinterleave_t4(&x[slice], &x[slice + 1]);
        }
      }
      l++;
    }

    for (; (1ULL << (l + is_top_level)) < sub_n; l++) {
      const __m512i *wsv = ws[l];
      const __m512i *ws_precon_v = w_precon[l];
      const uint64_t t = 1ULL << (l - 3);
      const uint64_t m_sub = sub_n >> (l + 1);
      size_t start_i = offset_i * m_sub;
      for (size_t i = 0; i < m_sub; i++) {
        const uint64_t slice = 2 * i * t;
        const __m512i w_i = wsv[start_i + i];
        const __m512i wp_i = ws_precon_v[start_i + i];
        for (size_t j = slice; j < slice + t; j++) {
          bfly_inv(&x[j], &x[j + t], w_i, wp_i, minus_q, q2);
        }
      }
    }

    if (is_top_level) {
      // Fused last stage: butterfly + multiplication by 1/n (and w/n).
      size_t l_top = l;
      const __m512i *wsv = ws[l_top];
      const uint64_t t = 1ULL << (l_top - 3);
      uint64_t inv_n_prime = tier_barrett_factor(inv_n, q);
      __m512i v_inv_n = _mm512_set1_epi64((long long)inv_n);
      __m512i v_inv_n_prime = _mm512_set1_epi64((long long)inv_n_prime);
      __m512i q_vec = _mm512_set1_epi64((long long)q);
      uint64_t w = (uint64_t)_mm_cvtsi128_si64(_mm512_castsi512_si128(wsv[0]));
      uint64_t inv_n_w = (uint64_t)(((unsigned __int128)inv_n * w) % q);
      uint64_t inv_n_w_prime = tier_barrett_factor(inv_n_w, q);
      __m512i v_inv_n_w = _mm512_set1_epi64((long long)inv_n_w);
      __m512i v_inv_n_w_prime = _mm512_set1_epi64((long long)inv_n_w_prime);
      for (size_t j = 0; j < t; j++) {
        bfly_inv_final(&x[j], &x[j + t], v_inv_n, v_inv_n_prime, v_inv_n_w, v_inv_n_w_prime,
                       minus_q, q2, q_vec);
      }
    }
  } else {
    size_t t = sub_n >> 4;
    gs_rn_internal(x, ws, w_precon, sub_n / 2, q, l_merge - 1, 2 * offset_i, 0, inv_n);
    gs_rn_internal(x + t, ws, w_precon, sub_n / 2, q, l_merge - 1, 2 * offset_i + 1, 0, inv_n);
    if (is_top_level) {
      uint64_t inv_n_prime = tier_barrett_factor(inv_n, q);
      __m512i v_inv_n = _mm512_set1_epi64((long long)inv_n);
      __m512i v_inv_n_prime = _mm512_set1_epi64((long long)inv_n_prime);
      __m512i q_vec = _mm512_set1_epi64((long long)q);
      const __m512i w_i = ws[l_merge][offset_i];
      uint64_t w = (uint64_t)_mm_cvtsi128_si64(_mm512_castsi512_si128(w_i));
      uint64_t inv_n_w = (uint64_t)(((unsigned __int128)inv_n * w) % q);
      uint64_t inv_n_w_prime = tier_barrett_factor(inv_n_w, q);
      __m512i v_inv_n_w = _mm512_set1_epi64((long long)inv_n_w);
      __m512i v_inv_n_w_prime = _mm512_set1_epi64((long long)inv_n_w_prime);
      for (size_t j = 0; j < t; j++) {
        bfly_inv_final(&x[j], &x[j + t], v_inv_n, v_inv_n_prime, v_inv_n_w, v_inv_n_w_prime,
                       minus_q, q2, q_vec);
      }
    } else {
      const __m512i w_i = ws[l_merge][offset_i];
      const __m512i wp_i = w_precon[l_merge][offset_i];
      for (size_t j = 0; j < t; j++) {
        bfly_inv(&x[j], &x[j + t], w_i, wp_i, minus_q, q2);
      }
    }
  }
}

// --- Plan entry points ---------------------------------------------------------------

static void tier_forward(uint64_t *out, const uint64_t *in, const ntt_plan *plan) {
  if (out != in) memcpy(out, in, plan->n * sizeof(uint64_t));
  ct_nr_internal((__m512i *)out, (__m512i **)plan->tw_fwd, (__m512i **)plan->tw_fwd_pre, plan->n,
                 plan->zq->q, 0, 0);

  // Normalize from the lazy [0, 4q) range to [0, q).
  const __m512i q_vec = _mm512_set1_epi64((long long)plan->zq->q);
  const __m512i q2_vec = _mm512_set1_epi64((long long)(2 * plan->zq->q));
  for (size_t i = 0; i < plan->n / 8; i++) {
    __m512i v = ((__m512i *)out)[i];
    v = _mm512_min_epu64(v, _mm512_sub_epi64(v, q2_vec));
    v = _mm512_min_epu64(v, _mm512_sub_epi64(v, q_vec));
    ((__m512i *)out)[i] = v;
  }
}

static void tier_inverse(uint64_t *out, const uint64_t *in, const ntt_plan *plan) {
  if (out != in) memcpy(out, in, plan->n * sizeof(uint64_t));
  uint64_t inv_n = nt_inverse_mod(plan->n, plan->zq->q);
  gs_rn_internal((__m512i *)out, (__m512i **)plan->tw_inv, (__m512i **)plan->tw_inv_pre, plan->n,
                 plan->zq->q, plan->log_n - 1, 0, 1, inv_n);
}

static void tier_free_tables(ntt_plan *plan) {
  ntt_avx512_pack_free(plan->tw_fwd, plan->tw_fwd_pre, plan->n);
  ntt_avx512_pack_free(plan->tw_inv, plan->tw_inv_pre, plan->n);
}

void NTT_BACKEND_INIT(ntt_plan *plan) {
  const uint64_t q = plan->zq->q;

  uint64_t *fwd_raw = ntt_tables_bitrev_powers(plan->n, q, plan->root);
  ntt_avx512_pack_fwd(plan->n, q, NTT_TIER_SHIFT, fwd_raw, &plan->tw_fwd, &plan->tw_fwd_pre);
  free(fwd_raw);

  uint64_t *inv_raw = ntt_tables_bitrev_powers(plan->n, q, plan->inv_root);
  uint64_t *inv_sched = ntt_tables_gs_schedule(plan->n, inv_raw);
  ntt_avx512_pack_inv(plan->n, q, NTT_TIER_SHIFT, inv_sched, &plan->tw_inv, &plan->tw_inv_pre);
  free(inv_raw);
  free(inv_sched);

  plan->forward = tier_forward;
  plan->inverse = tier_inverse;
  plan->free_tables = tier_free_tables;
}
