#include <stdlib.h>
#include <pthread.h>

#ifdef __APPLE__
// Custom implementation of pthread_barrier for macOS
typedef struct
{
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    unsigned int count;
    unsigned int limit;
    unsigned int trip_count;
} pthread_barrier_t;

#ifndef PTHREAD_BARRIER_SERIAL_THREAD
#define PTHREAD_BARRIER_SERIAL_THREAD 1
#endif

static inline int pthread_barrier_init(pthread_barrier_t *barrier, const void *attr,
                                       unsigned int count)
{
    if (count == 0)
        return -1;
    barrier->count = 0;
    barrier->limit = count;
    barrier->trip_count = 0;
    pthread_mutex_init(&barrier->mutex, NULL);
    pthread_cond_init(&barrier->cond, NULL);
    return 0;
}

static inline int pthread_barrier_destroy(pthread_barrier_t *barrier)
{
    pthread_mutex_destroy(&barrier->mutex);
    pthread_cond_destroy(&barrier->cond);
    return 0;
}

static inline int pthread_barrier_wait(pthread_barrier_t *barrier)
{
    pthread_mutex_lock(&barrier->mutex);
    barrier->count++;
    if (barrier->count >= barrier->limit)
    {
        barrier->trip_count++;
        barrier->count = 0;
        pthread_cond_broadcast(&barrier->cond);
        pthread_mutex_unlock(&barrier->mutex);
        return PTHREAD_BARRIER_SERIAL_THREAD;
    }
    else
    {
        unsigned int current_trip = barrier->trip_count;
        while (current_trip == barrier->trip_count)
        {
            pthread_cond_wait(&barrier->cond, &barrier->mutex);
        }
        pthread_mutex_unlock(&barrier->mutex);
        return 0;
    }
}
#endif

#include "mlwe.h"

void gp25_RGSW_monomial_mul(RNS_MLWE *p0, uint64_t in_N, RNS_MLWE **e, uint64_t r_prec,
                            RNS_MLWE **ksk, uint64_t ell, uint64_t special_primes)
{
    uint64_t N = p0[0]->b->ntt->N;
    uint64_t r = p0[0]->r;
    uint64_t mask = p0[0]->b->rns_mask;
    incNTT ntt = p0[0]->b->ntt;

    // Allocate p1 array of size in_N
    RNS_MLWE *p1 = (RNS_MLWE *)malloc(in_N * sizeof(RNS_MLWE));
    for (size_t i = 0; i < in_N; i++)
    {
        p1[i] = mlwe_alloc_RNS_sample(N, r, mask, ntt);
    }

    RNS_MLWE *p[2] = {p0, p1};

    for (size_t i = 0; i < r_prec; i++)
    {
        uint64_t power = 1ULL << i;
        uint64_t out_idx = (i + 1) & 1;
        uint64_t in_idx = out_idx ^ 1;

        for (size_t j = 0; j < power; j++)
        {
            mgsw_NCMUX_to_coeff(p[out_idx][j], (RNSc_MLWE)p[in_idx][j],
                                (RNSc_MLWE)p[in_idx][in_N - power + j], e[i], ksk, ell,
                                special_primes);
        }

        for (size_t j = 0; j < in_N - power; j++)
        {
            mgsw_CMUX_to_coeff(p[out_idx][j + power], (RNSc_MLWE)p[in_idx][j + power],
                               (RNSc_MLWE)p[in_idx][j], e[i], ell, special_primes);
        }
        // _to_coeff variants already leave each output in coefficient form.
    }

    if ((r_prec & 1) == 1)
    {
        for (size_t i = 0; i < in_N; i++)
        {
            mlwe_copy_RNS_sample(p0[i], p1[i]);
        }
    }

    // Free p1 array
    for (size_t i = 0; i < in_N; i++)
    {
        free_mlwe_RNS_sample(p1[i]);
    }
    free(p1);
}

typedef struct
{
    RNS_MLWE *p0;
    RNS_MLWE *p1;
    uint64_t in_N;
    RNS_MLWE **e;
    uint64_t r_prec;
    RNS_MLWE **ksk;
    uint64_t ell;
    uint64_t special_primes;
    uint64_t start_k;
    uint64_t stride;
    pthread_barrier_t *barrier;
} worker_args_t;

void *monomial_mul_worker(void *arg)
{
    worker_args_t *args = (worker_args_t *)arg;
    RNS_MLWE *p[2] = {args->p0, args->p1};

    for (size_t i = 0; i < args->r_prec; i++)
    {
        uint64_t power = 1ULL << i;
        uint64_t out_idx = (i + 1) & 1;
        uint64_t in_idx = out_idx ^ 1;

        // Strided (round-robin) index assignment: NCMUX ops (k < power, each ~2x a CMUX
        // due to the extra automorphism) cluster in the low indices, so contiguous chunks
        // would dump them all on the first few threads and stall the rest at the barrier.
        // Striding interleaves NCMUX/CMUX evenly across threads.
        // The _to_coeff variants leave the output in coefficient form directly,
        // folding in the per-index inverse NTT and avoiding a forward NTT of in1.
        for (size_t k = args->start_k; k < args->in_N; k += args->stride)
        {
            if (k < power)
            {
                mgsw_NCMUX_to_coeff(p[out_idx][k], (RNSc_MLWE)p[in_idx][k],
                                    (RNSc_MLWE)p[in_idx][args->in_N - power + k], args->e[i],
                                    args->ksk, args->ell, args->special_primes);
            }
            else
            {
                mgsw_CMUX_to_coeff(p[out_idx][k], (RNSc_MLWE)p[in_idx][k],
                                   (RNSc_MLWE)p[in_idx][k - power], args->e[i], args->ell,
                                   args->special_primes);
            }
        }

        // Synchronize before moving to the next stage
        pthread_barrier_wait(args->barrier);
    }

    return NULL;
}

void gp25_RGSW_monomial_mul_mt(RNS_MLWE *p0, uint64_t in_N, RNS_MLWE **e, uint64_t r_prec,
                               RNS_MLWE **ksk, uint64_t ell, uint64_t special_primes,
                               uint64_t num_threads)
{
    if (num_threads <= 1)
    {
        gp25_RGSW_monomial_mul(p0, in_N, e, r_prec, ksk, ell, special_primes);
        return;
    }

    if (num_threads > in_N)
    {
        num_threads = in_N;
    }

    uint64_t N = p0[0]->b->ntt->N;
    uint64_t r = p0[0]->r;
    uint64_t mask = p0[0]->b->rns_mask;
    incNTT ntt = p0[0]->b->ntt;

    // Allocate p1 array of size in_N
    RNS_MLWE *p1 = (RNS_MLWE *)malloc(in_N * sizeof(RNS_MLWE));
    for (size_t i = 0; i < in_N; i++)
    {
        p1[i] = mlwe_alloc_RNS_sample(N, r, mask, ntt);
    }

    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, num_threads);

    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    worker_args_t *args = (worker_args_t *)malloc(num_threads * sizeof(worker_args_t));

    for (size_t t = 0; t < num_threads; t++)
    {
        args[t].p0 = p0;
        args[t].p1 = p1;
        args[t].in_N = in_N;
        args[t].e = e;
        args[t].r_prec = r_prec;
        args[t].ksk = ksk;
        args[t].ell = ell;
        args[t].special_primes = special_primes;
        args[t].barrier = &barrier;
        // strided assignment: thread t handles indices t, t+num_threads, t+2*num_threads, ...
        args[t].start_k = t;
        args[t].stride = num_threads;

        pthread_create(&threads[t], NULL, monomial_mul_worker, &args[t]);
    }

    for (size_t t = 0; t < num_threads; t++)
    {
        pthread_join(threads[t], NULL);
    }

    if ((r_prec & 1) == 1)
    {
        for (size_t i = 0; i < in_N; i++)
        {
            mlwe_copy_RNS_sample(p0[i], p1[i]);
        }
    }

    // Free p1 array
    for (size_t i = 0; i < in_N; i++)
    {
        free_mlwe_RNS_sample(p1[i]);
    }
    free(p1);

    pthread_barrier_destroy(&barrier);
    free(threads);
    free(args);
}

/* ------------------------------------------------------------------------------------------------
 * sub_a, multithreaded. Per accumulator coefficient k (all INDEPENDENT, so no barrier):
 *     p[k] <- p[k] * X^a[k]
 *     tmp  <- p[k] * (X^{(-2 a[k]) mod 2N} - 1)
 *     p[k] <- p[k] + s_sign (X) tmp        (one fixed MGSW `s_sign`, external product)
 * This replaces the per-element Python loop (which serialized 2048 external products through
 * ctypes); the work is identical, just done in one parallel C kernel.
 * ------------------------------------------------------------------------------------------------
 */
typedef struct
{
    RNS_MLWE *p0;
    uint64_t *a;
    RNS_MLWE *s_sign; /* the MGSW: (r+1)*ell rows, in NTT form */
    uint64_t in_N, ell, special_primes, N, start_k, end_k;
} suba_args_t;

static void *suba_worker(void *arg)
{
    suba_args_t *A = (suba_args_t *)arg;
    const uint64_t N = A->N, twoN = 2 * N;
    const uint64_t r = A->p0[0]->r;
    const uint64_t mask = A->p0[0]->b->rns_mask;
    incNTT ntt = A->p0[0]->b->ntt;

    RNSc_MLWE pax = mlwe_alloc_RNSc_sample(N, r, mask, ntt); /* p[k]*X^a (coeff)   */
    RNSc_MLWE tmp = mlwe_alloc_RNSc_sample(N, r, mask, ntt); /* pax*(X^m2a - 1)    */
    RNS_MLWE ext = mlwe_alloc_RNS_sample(N, r, mask, ntt);   /* s_sign (X) tmp (NTT) */
    RNS_MLWE pax_ntt = mlwe_alloc_RNS_sample(N, r, mask, ntt);

    for (size_t k = A->start_k; k < A->end_k; k++)
    {
        RNSc_MLWE pk = (RNSc_MLWE)A->p0[k];
        const uint64_t ai = A->a[k];
        mlwe_RNSc_mul_by_xai(pax, pk, ai);                    /* pax = p[k] * X^a       */
        const uint64_t m2a = (twoN - (2 * ai) % twoN) % twoN; /* (-2a) mod 2N           */
        mlwe_RNSc_mul_by_xai_minus1(tmp, pax, m2a);           /* tmp = pax * (X^m2a - 1) */
        mgsw_external_product(ext, A->s_sign, tmp, A->ell,
                              A->special_primes); /* ext = s_sign (X) tmp */
        mlwe_copy_RNS_sample(pax_ntt, (RNS_MLWE)pax);
        mlwe_RNSc_to_RNS(pax_ntt, (RNSc_MLWE)pax_ntt); /* pax -> NTT for the add  */
        for (size_t i = 0; i < r; i++)
            polynomial_add_RNS_polynomial(ext->a[i], ext->a[i], pax_ntt->a[i]);
        polynomial_add_RNS_polynomial(ext->b, ext->b, pax_ntt->b);
        mlwe_RNS_to_RNSc((RNSc_MLWE)A->p0[k], ext); /* result (coeff) into p[k] */
    }

    free_mlwe_RNS_sample((RNS_MLWE)pax);
    free_mlwe_RNS_sample((RNS_MLWE)tmp);
    free_mlwe_RNS_sample(ext);
    free_mlwe_RNS_sample(pax_ntt);
    return NULL;
}

void gp25_sub_a_mt(RNS_MLWE *p0, uint64_t in_N, uint64_t *a, RNS_MLWE *s_sign, uint64_t ell,
                   uint64_t special_primes, uint64_t N, uint64_t num_threads)
{
    if (num_threads > in_N)
        num_threads = in_N;
    if (num_threads < 1)
        num_threads = 1;

    if (num_threads == 1)
    {
        suba_args_t args = {p0, a, s_sign, in_N, ell, special_primes, N, 0, in_N};
        suba_worker(&args);
        return;
    }

    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    suba_args_t *args = (suba_args_t *)malloc(num_threads * sizeof(suba_args_t));
    uint64_t chunk = in_N / num_threads, rem = in_N % num_threads, cur = 0;
    for (size_t t = 0; t < num_threads; t++)
    {
        uint64_t end = cur + chunk + (t < rem ? 1 : 0);
        args[t] = (suba_args_t){p0, a, s_sign, in_N, ell, special_primes, N, cur, end};
        cur = end;
        pthread_create(&threads[t], NULL, suba_worker, &args[t]);
    }
    for (size_t t = 0; t < num_threads; t++)
        pthread_join(threads[t], NULL);
    free(threads);
    free(args);
}
