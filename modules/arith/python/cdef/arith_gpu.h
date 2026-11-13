// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
// Python-facing ABI of arith's CUDA kernels (arith/c/src/cuda/arith_gpu.cu,
// prototyped in arith/c/include/arith_gpu.h). An element is passed as `void *`.

int arith_gpu_rns_new(void **out, uint64_t rows, uint64_t N, const uint64_t *primes);
void arith_gpu_rns_free(void *p);
int arith_gpu_rns_upload(void *p, const uint64_t *words);
int arith_gpu_rns_download(uint64_t *words, const void *p);
int arith_gpu_rns_copy(void *out, const void *in);
int arith_gpu_rns_add(void *out, const void *a, const void *b);
int arith_gpu_rns_sub(void *out, const void *a, const void *b);
int arith_gpu_rns_negate(void *out, const void *in);
