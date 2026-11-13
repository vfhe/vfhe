// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
#include <stdlib.h>

#include "arith_gpu.h"

struct rows
{
    uint64_t *coeffs; // rows * N residues, row-major, on the device
    uint64_t *primes; // one per row, on the device
    uint64_t rows, N;
};

enum op
{
    ADD,
    SUB,
    NEG
};

// One thread per residue; the row's prime comes from its index.
static __global__ void elementwise(enum op op, uint64_t *out, const uint64_t *a, const uint64_t *b,
                                   const uint64_t *primes, uint64_t N, uint64_t n)
{
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const uint64_t p = primes[i / N];
    uint64_t r;
    switch (op)
    {
    case ADD:
        r = a[i] + b[i];
        break;
    case SUB:
        r = a[i] + p - b[i];
        break;
    default:
        r = a[i] ? p - a[i] : 0;
        return (void)(out[i] = r);
    }
    out[i] = r >= p ? r - p : r;
}

static int run(enum op op, struct rows *out, const struct rows *a, const struct rows *b)
{
    const uint64_t n = out->rows * out->N;
    if (n == 0)
        return cudaSuccess;
    const unsigned threads = 256;
    const unsigned blocks = (unsigned)((n + threads - 1) / threads);
    elementwise<<<blocks, threads>>>(op, out->coeffs, a->coeffs, b ? b->coeffs : NULL, out->primes,
                                     out->N, n);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess)
        err = cudaDeviceSynchronize();
    return (int)err;
}

int arith_gpu_rns_new(void **out, uint64_t rows, uint64_t N, const uint64_t *primes)
{
    struct rows *p = (struct rows *)calloc(1, sizeof *p);
    if (p == NULL)
        return (int)cudaErrorMemoryAllocation;
    p->rows = rows;
    p->N = N;
    cudaError_t err = cudaMalloc(&p->coeffs, rows * N * sizeof(uint64_t));
    if (err == cudaSuccess)
        err = cudaMalloc(&p->primes, rows * sizeof(uint64_t));
    if (err == cudaSuccess)
        err = cudaMemcpy(p->primes, primes, rows * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        arith_gpu_rns_free(p);
        return (int)err;
    }
    *out = p;
    return cudaSuccess;
}

void arith_gpu_rns_free(void *p)
{
    struct rows *r = (struct rows *)p;
    if (r == NULL)
        return;
    cudaFree(r->coeffs);
    cudaFree(r->primes);
    free(r);
}

int arith_gpu_rns_upload(void *p, const uint64_t *words)
{
    struct rows *r = (struct rows *)p;
    return (int)cudaMemcpy(r->coeffs, words, r->rows * r->N * sizeof(uint64_t),
                           cudaMemcpyHostToDevice);
}

int arith_gpu_rns_download(uint64_t *words, const void *p)
{
    const struct rows *r = (const struct rows *)p;
    return (int)cudaMemcpy(words, r->coeffs, r->rows * r->N * sizeof(uint64_t),
                           cudaMemcpyDeviceToHost);
}

int arith_gpu_rns_copy(void *out, const void *in)
{
    struct rows *o = (struct rows *)out;
    const struct rows *i = (const struct rows *)in;
    return (int)cudaMemcpy(o->coeffs, i->coeffs, o->rows * o->N * sizeof(uint64_t),
                           cudaMemcpyDeviceToDevice);
}

int arith_gpu_rns_add(void *out, const void *a, const void *b)
{
    return run(ADD, (struct rows *)out, (const struct rows *)a, (const struct rows *)b);
}

int arith_gpu_rns_sub(void *out, const void *a, const void *b)
{
    return run(SUB, (struct rows *)out, (const struct rows *)a, (const struct rows *)b);
}

int arith_gpu_rns_negate(void *out, const void *in)
{
    return run(NEG, (struct rows *)out, (const struct rows *)in, NULL);
}
