#ifndef __ARITH_INTERNAL_H__
#define __ARITH_INTERNAL_H__

#include <arith.h>

// 32-bit declarations
void ntt_forward_32(uint64_t *out, uint64_t *in, NTT_proc proc);
void ntt_reverse_32(uint64_t *out, uint64_t *in, NTT_proc proc);
void mod_eltwise_mul_32(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc);
void mod_eltwise_mul_addto_32(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n,
                              NTT_proc proc);
void mod_eltwise_mul_subto_32(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n,
                              NTT_proc proc);
void mod_eltwise_scale_32(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, NTT_proc proc);
void mod_eltwise_fma_32(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, NTT_proc proc);
void mod_eltwise_add_scalar_32(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n,
                               NTT_proc proc);
void mod_eltwise_sub_scalar_32(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n,
                               NTT_proc proc);
void mod_eltwise_negate_32(uint64_t *out, uint64_t *in, uint64_t n, NTT_proc proc);
void mod_eltwise_add_32(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc);
void mod_eltwise_sub_32(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc);
void mod_eltwise_reduce_32(uint64_t *out, uint64_t *in, uint64_t n, NTT_proc proc);
void mod_eltwise_reduce_signed_32(uint64_t *out, int64_t *in, uint64_t n, NTT_proc proc);
void mod_reduce_array_mp_32(uint64_t *out, uint64_t *in_high, uint64_t *in_low, uint64_t n,
                            NTT_proc proc);

// 50-bit declarations
void ntt_forward_50(uint64_t *out, uint64_t *in, NTT_proc proc);
void ntt_reverse_50(uint64_t *out, uint64_t *in, NTT_proc proc);
void mod_eltwise_mul_50(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc);
void mod_eltwise_mul_addto_50(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n,
                              NTT_proc proc);
void mod_eltwise_mul_subto_50(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n,
                              NTT_proc proc);
void mod_eltwise_scale_50(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, NTT_proc proc);
void mod_eltwise_fma_50(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, NTT_proc proc);
void mod_eltwise_add_scalar_50(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n,
                               NTT_proc proc);
void mod_eltwise_sub_scalar_50(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n,
                               NTT_proc proc);
void mod_eltwise_negate_50(uint64_t *out, uint64_t *in, uint64_t n, NTT_proc proc);
void mod_eltwise_add_50(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc);
void mod_eltwise_sub_50(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc);
void mod_eltwise_reduce_50(uint64_t *out, uint64_t *in, uint64_t n, NTT_proc proc);
void mod_eltwise_reduce_signed_50(uint64_t *out, int64_t *in, uint64_t n, NTT_proc proc);
void mod_reduce_array_mp_50(uint64_t *out, uint64_t *in_high, uint64_t *in_low, uint64_t n,
                            NTT_proc proc);

// 64-bit declarations
void ntt_forward_64(uint64_t *out, uint64_t *in, NTT_proc proc);
void ntt_reverse_64(uint64_t *out, uint64_t *in, NTT_proc proc);
void mod_eltwise_mul_64(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc);
void mod_eltwise_mul_addto_64(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n,
                              NTT_proc proc);
void mod_eltwise_mul_subto_64(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n,
                              NTT_proc proc);
void mod_eltwise_scale_64(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, NTT_proc proc);
void mod_eltwise_fma_64(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, NTT_proc proc);
void mod_eltwise_add_scalar_64(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n,
                               NTT_proc proc);
void mod_eltwise_sub_scalar_64(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n,
                               NTT_proc proc);
void mod_eltwise_negate_64(uint64_t *out, uint64_t *in, uint64_t n, NTT_proc proc);
void mod_eltwise_add_64(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc);
void mod_eltwise_sub_64(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc);
void mod_eltwise_reduce_64(uint64_t *out, uint64_t *in, uint64_t n, NTT_proc proc);
void mod_eltwise_reduce_signed_64(uint64_t *out, int64_t *in, uint64_t n, NTT_proc proc);
void mod_reduce_array_mp_64(uint64_t *out, uint64_t *in_high, uint64_t *in_low, uint64_t n,
                            NTT_proc proc);

#endif // __ARITH_INTERNAL_H__
