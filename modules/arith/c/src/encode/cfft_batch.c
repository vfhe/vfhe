// SPDX-License-Identifier: Apache-2.0
/**
 * @file cfft_batch.c
 * @brief Threaded batch encode pipeline (bit-reverse -> IFFT -> scale -> round).
 *
 * Pure orchestration over the cfft.c kernels and the poly loader; contains
 * no arithmetic of its own.
 */
#include <pthread.h>

#include <arith/cfft.h>

#define CFFT_BATCH_DEFAULT_THREADS 8
#define CFFT_BATCH_MAX_THREADS 64

typedef struct
{
    void **rows_in;
    void **outs;
    uint64_t start, end;
    uint64_t n_complex;
    uint32_t log_prec;
    double *const *ws_inv;
    double delta;
} cfft_batch_arg;

static void *cfft_batch_thread(void *arg)
{
    cfft_batch_arg *a = (cfft_batch_arg *)arg;
    const uint64_t n_complex = a->n_complex;
    // Fold the FFT's 1/n normalization into the CKKS delta scaling.
    const double scale = a->delta / (double)n_complex;
    for (uint64_t i = a->start; i < a->end; i++)
    {
        double *row = (double *)a->rows_in[i];
        cfft_bit_reverse(row, n_complex, a->log_prec);
        cfft_inverse(row, a->ws_inv, n_complex);
        cfft_scale(row, scale, n_complex);
        cfft_round_to_poly((rns_poly_t)a->outs[i], row);
    }
    return NULL;
}

void cfft_ifft_scale_round_batch(void **rows_in, void **outs, uint64_t count, uint64_t n_complex,
                                 uint32_t log_prec, double *const *ws_inv, double delta,
                                 uint64_t n_threads)
{
    if (n_threads == 0)
        n_threads = CFFT_BATCH_DEFAULT_THREADS;
    if (n_threads > CFFT_BATCH_MAX_THREADS)
        n_threads = CFFT_BATCH_MAX_THREADS;

    pthread_t threads[CFFT_BATCH_MAX_THREADS];
    cfft_batch_arg args[CFFT_BATCH_MAX_THREADS];
    const uint64_t batch_size = (count + n_threads - 1) / n_threads;
    for (uint64_t i = 0; i < n_threads; i++)
    {
        args[i].rows_in = rows_in;
        args[i].outs = outs;
        args[i].start = i * batch_size;
        args[i].end = (i + 1) * batch_size < count ? (i + 1) * batch_size : count;
        args[i].n_complex = n_complex;
        args[i].log_prec = log_prec;
        args[i].ws_inv = ws_inv;
        args[i].delta = delta;
        if (args[i].start < count)
            pthread_create(&threads[i], NULL, cfft_batch_thread, &args[i]);
    }
    for (uint64_t i = 0; i < n_threads; i++)
    {
        if (args[i].start < count)
            pthread_join(threads[i], NULL);
    }
}
