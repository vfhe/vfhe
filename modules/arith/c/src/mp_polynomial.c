#include <arith.h>
#include <misc.h>

#if !defined(__AVX512IFMA__) || defined(PORTABLE_BUILD) || defined(PORTABLE)
static inline uint64_t madd52lo(uint64_t a, uint64_t b, uint64_t c)
{
    unsigned __int128 prod =
        (unsigned __int128)(b & 0x000fffffffffffffULL) * (c & 0x000fffffffffffffULL);
    return a + (uint64_t)(prod & 0x000fffffffffffffULL);
}

static inline uint64_t madd52hi(uint64_t a, uint64_t b, uint64_t c)
{
    unsigned __int128 prod =
        (unsigned __int128)(b & 0x000fffffffffffffULL) * (c & 0x000fffffffffffffULL);
    return a + (uint64_t)(prod >> 52);
}
#endif

int get_mp_vector_size(void)
{
#if defined(__AVX512IFMA__) && !defined(PORTABLE_BUILD) && !defined(PORTABLE)
    return 8;
#else
    return 1;
#endif
}

MPPolynomial new_mp_polynomial(uint64_t N, uint64_t d)
{
    MPPolynomial res = (MPPolynomial)safe_malloc(sizeof(*res));
    res->coeffs = (uint64_t **)safe_malloc(sizeof(*res->coeffs) * d);
    res->d = d;
    res->N = N;
    for (size_t i = 0; i < d; i++)
    {
        res->coeffs[i] = (uint64_t *)safe_aligned_malloc(sizeof(**(res->coeffs)) * N);
        memset(res->coeffs[i], 0, sizeof(*(res->coeffs)) * N);
    }
    return res;
}

void free_mp_polynomial(MPPolynomial p)
{
    for (size_t i = 0; i < p->d; i++)
    {
        free(p->coeffs[i]);
    }
    free(p->coeffs);
    free(p);
}

// out = in*X^a mod (X^N + 1)
void mp_polynomial_mul_by_xai(MPPolynomial out, MPPolynomial in, uint64_t a)
{
    const int N = out->N;
    assert(out->d == in->d);
    assert(out->N == in->N);
    a &= ((N << 1) - 1); // a % 2N
    if (!a)
        return;
    for (size_t j = 0; j < in->d; j++)
    {
        if (a < N)
        {
            for (int i = 0; i < a; i++)
                out->coeffs[j][i] = -in->coeffs[j][i - a + N];
            for (int i = a; i < N; i++)
                out->coeffs[j][i] = in->coeffs[j][i - a];
        }
        else
        {
            for (int i = 0; i < a - N; i++)
                out->coeffs[j][i] = in->coeffs[j][i - a + 2 * N];
            for (int i = a - N; i < N; i++)
                out->coeffs[j][i] = -in->coeffs[j][i - a + N];
        }
    }
}

void mp_polynomial_negate(MPPolynomial out, MPPolynomial in)
{
#if defined(__AVX512IFMA__) && !defined(PORTABLE_BUILD) && !defined(PORTABLE)
    __m512i neg_mask = _mm512_set1_epi64(0x000FFFFFFFFFFFFFULL);
    __m512i neg_mask2 = _mm512_set1_epi64(1);
    for (size_t j = 0; j < in->d; j++)
    {
        __m512i *inv = (__m512i *)in->coeffs[j];
        __m512i *outv = (__m512i *)out->coeffs[j];
        for (size_t i = 0; i < in->N / 8; i++)
        {
            outv[i] = _mm512_xor_epi64(inv[i], neg_mask);
            if (j == 0)
                outv[i] = _mm512_add_epi64(outv[i], neg_mask2);
        }
    }
#else
    for (size_t j = 0; j < in->d; j++)
    {
        for (size_t i = 0; i < in->N; i++)
        {
            out->coeffs[j][i] = in->coeffs[j][i] ^ 0x000FFFFFFFFFFFFFULL;
            if (j == 0)
                out->coeffs[j][i] += 1;
        }
    }
#endif
}

void mp_polynomial_add(MPPolynomial out, MPPolynomial a, MPPolynomial b)
{
#if defined(__AVX512IFMA__) && !defined(PORTABLE_BUILD) && !defined(PORTABLE)
    for (size_t j = 0; j < a->d; j++)
    {
        __m512i *av = (__m512i *)a->coeffs[j];
        __m512i *bv = (__m512i *)b->coeffs[j];
        __m512i *outv = (__m512i *)out->coeffs[j];
        for (size_t i = 0; i < a->N / 8; i++)
        {
            outv[i] = _mm512_add_epi64(av[i], bv[i]);
        }
    }
#else
    for (size_t j = 0; j < a->d; j++)
    {
        for (size_t i = 0; i < a->N; i++)
        {
            out->coeffs[j][i] = a->coeffs[j][i] + b->coeffs[j][i];
        }
    }
#endif
}

void mp_polynomial_drop_digits(MPPolynomial p, uint64_t num_digits)
{
    for (size_t i = 1; i <= num_digits; i++)
    {
        free(p->coeffs[p->d - i]);
    }
    p->d -= num_digits;
}

void mp_polynomial_rnd(MPPolynomial poly)
{
#if defined(__AVX512IFMA__) && !defined(PORTABLE_BUILD) && !defined(PORTABLE)
    __m512i and_mask = _mm512_set1_epi64(0x000FFFFFFFFFFFFFULL);
    for (size_t j = 0; j < poly->d; j++)
    {
        generate_random_bytes(sizeof(uint64_t) * poly->N, (uint8_t *)poly->coeffs[j]);
        __m512i *polyv = (__m512i *)poly->coeffs[j];
        for (size_t i = 0; i < poly->N / 8; i++)
        {
            polyv[i] = _mm512_and_epi64(polyv[i], and_mask);
        }
    }
#else
    for (size_t j = 0; j < poly->d; j++)
    {
        generate_random_bytes(sizeof(uint64_t) * poly->N, (uint8_t *)poly->coeffs[j]);
        for (size_t i = 0; i < poly->N; i++)
        {
            poly->coeffs[j][i] &= 0x000FFFFFFFFFFFFFULL;
        }
    }
#endif
}

// MPPolynomials should not have carry bits
void mp_polynomial_scale(MPPolynomial out, MPPolynomial in, uint64_t scale)
{
#if defined(__AVX512IFMA__) && !defined(PORTABLE_BUILD) && !defined(PORTABLE)
    __m512i zero = _mm512_setzero_si512();
    __m512i scalev = _mm512_set1_epi64(scale);
    for (size_t j = 0; j < in->d; j++)
    {
        __m512i *inv = (__m512i *)in->coeffs[j];
        __m512i *outv_prev = (__m512i *)out->coeffs[j - 1];
        __m512i *outv = (__m512i *)out->coeffs[j];
        for (size_t i = 0; i < in->N / 8; i++)
        {
            if (j > 0)
                outv_prev[i] = _mm512_madd52hi_epu64(outv_prev[i], inv[i], scalev);
            outv[i] = _mm512_madd52lo_epu64(zero, inv[i], scalev);
        }
    }
#else
    for (size_t j = 0; j < in->d; j++)
    {
        for (size_t i = 0; i < in->N; i++)
        {
            if (j > 0)
                out->coeffs[j - 1][i] = madd52hi(out->coeffs[j - 1][i], in->coeffs[j][i], scale);
            out->coeffs[j][i] = madd52lo(0, in->coeffs[j][i], scale);
        }
    }
#endif
}

void mp_polynomial_sp_scale_mp(MPPolynomial out, MPPolynomial in, mp_vector_t *scale)
{
#if defined(__AVX512IFMA__) && !defined(PORTABLE_BUILD) && !defined(PORTABLE)
    __m512i zero = _mm512_setzero_si512();
    for (size_t j = 0; j < in->d; j++)
    {
        __m512i *inv_sp = (__m512i *)in->coeffs[0];
        __m512i *outv_prev = (__m512i *)out->coeffs[j - 1];
        __m512i *outv = (__m512i *)out->coeffs[j];
        for (size_t i = 0; i < in->N / 8; i++)
        {
            if (j > 0)
                outv_prev[i] = _mm512_madd52hi_epu64(outv_prev[i], scale[j], inv_sp[i]);
            outv[i] = _mm512_madd52lo_epu64(zero, scale[j], inv_sp[i]);
        }
    }
#else
    for (size_t j = 0; j < in->d; j++)
    {
        for (size_t i = 0; i < in->N; i++)
        {
            if (j > 0)
                out->coeffs[j - 1][i] = madd52hi(out->coeffs[j - 1][i], scale[j], in->coeffs[0][i]);
            out->coeffs[j][i] = madd52lo(0, scale[j], in->coeffs[0][i]);
        }
    }
#endif
}

void mp_polynomial_int_sp_scale_mp(MPPolynomial out, uint64_t *in, MPScalar scale)
{
#if defined(__AVX512IFMA__) && !defined(PORTABLE_BUILD) && !defined(PORTABLE)
    __m512i zero = _mm512_setzero_si512();
    for (size_t j = 0; j < out->d - 1; j++)
    {
        __m512i *inv_sp = (__m512i *)in;
        __m512i *outv_next = (__m512i *)out->coeffs[j + 1];
        __m512i *outv = (__m512i *)out->coeffs[j];
        for (size_t i = 0; i < out->N / 8; i++)
        {
            outv_next[i] = _mm512_madd52hi_epu64(zero, scale->digits[j], inv_sp[i]);
            if (j == 0)
            {
                outv[i] = _mm512_madd52lo_epu64(zero, scale->digits[j], inv_sp[i]);
            }
            else
            {
                outv[i] = _mm512_madd52lo_epu64(outv[i], scale->digits[j], inv_sp[i]);
            }
        }
    }
#else
    for (size_t j = 0; j < out->d - 1; j++)
    {
        for (size_t i = 0; i < out->N; i++)
        {
            out->coeffs[j + 1][i] = madd52hi(0, scale->digits[j], in[i]);
            if (j == 0)
            {
                out->coeffs[j][i] = madd52lo(0, scale->digits[j], in[i]);
            }
            else
            {
                out->coeffs[j][i] = madd52lo(out->coeffs[j][i], scale->digits[j], in[i]);
            }
        }
    }
#endif
}

void mp_polynomial_int_sp_scale_addto_mp(MPPolynomial out, uint64_t *in, MPScalar scale)
{
#if defined(__AVX512IFMA__) && !defined(PORTABLE_BUILD) && !defined(PORTABLE)
    for (size_t j = 0; j < out->d - 1; j++)
    {
        __m512i *inv_sp = (__m512i *)in;
        __m512i *outv_next = (__m512i *)out->coeffs[j + 1];
        __m512i *outv = (__m512i *)out->coeffs[j];
        for (size_t i = 0; i < out->N / 8; i++)
        {
            outv_next[i] = _mm512_madd52hi_epu64(outv_next[i], scale->digits[j], inv_sp[i]);
            outv[i] = _mm512_madd52lo_epu64(outv[i], scale->digits[j], inv_sp[i]);
        }
    }
#else
    for (size_t j = 0; j < out->d - 1; j++)
    {
        for (size_t i = 0; i < out->N; i++)
        {
            out->coeffs[j + 1][i] = madd52hi(out->coeffs[j + 1][i], scale->digits[j], in[i]);
            out->coeffs[j][i] = madd52lo(out->coeffs[j][i], scale->digits[j], in[i]);
        }
    }
#endif
}

void mp_polynomial_scale_addto(MPPolynomial out, MPPolynomial in, uint64_t scale)
{
#if defined(__AVX512IFMA__) && !defined(PORTABLE_BUILD) && !defined(PORTABLE)
    __m512i scalev = _mm512_set1_epi64(scale);
    for (size_t j = 0; j < in->d; j++)
    {
        __m512i *inv = (__m512i *)in->coeffs[j];
        __m512i *outv_prev = (__m512i *)out->coeffs[j - 1];
        __m512i *outv = (__m512i *)out->coeffs[j];
        for (size_t i = 0; i < in->N / 8; i++)
        {
            if (j > 0)
                outv_prev[i] = _mm512_madd52hi_epu64(outv_prev[i], inv[i], scalev);
            outv[i] = _mm512_madd52lo_epu64(outv[i], inv[i], scalev);
        }
    }
#else
    for (size_t j = 0; j < in->d; j++)
    {
        for (size_t i = 0; i < in->N; i++)
        {
            if (j > 0)
                out->coeffs[j - 1][i] = madd52hi(out->coeffs[j - 1][i], in->coeffs[j][i], scale);
            out->coeffs[j][i] = madd52lo(out->coeffs[j][i], in->coeffs[j][i], scale);
        }
    }
#endif
}

void mp_polynomial_zero(MPPolynomial poly)
{
    for (size_t j = 0; j < poly->d; j++)
    {
        memset(poly->coeffs[j], 0, sizeof(uint64_t) * poly->N);
    }
}

void mp_polynomial_propagate_carry(MPPolynomial p)
{
#if defined(__AVX512IFMA__) && !defined(PORTABLE_BUILD) && !defined(PORTABLE)
    __m512i mod_mask = _mm512_set1_epi64(0x000FFFFFFFFFFFFFULL);
    for (uint64_t j = 0; j < p->d - 1; j++)
    {
        __m512i *pv = (__m512i *)p->coeffs[j];
        __m512i *pv_next = (__m512i *)p->coeffs[j + 1];
        for (size_t i = 0; i < p->N / 8; i++)
        {
            // propagate carry
            // this can be cheaper in enchange for some bits of precision using
            // _mm512_madd52hi_epu64
            pv_next[i] = _mm512_add_epi64(pv_next[i], _mm512_srli_epi64(pv[i], 52));
            pv[i] = _mm512_and_epi64(pv[i], mod_mask); // remove MSB
        }
    }
    __m512i *pv = (__m512i *)p->coeffs[p->d - 1];
    for (size_t i = 0; i < p->N / 8; i++)
    {
        pv[i] = _mm512_and_epi64(pv[i], mod_mask); // remove MSB
    }
#else
    for (uint64_t j = 0; j < p->d - 1; j++)
    {
        for (size_t i = 0; i < p->N; i++)
        {
            p->coeffs[j + 1][i] += p->coeffs[j][i] >> 52;
            p->coeffs[j][i] &= 0x000FFFFFFFFFFFFFULL;
        }
    }
    for (size_t i = 0; i < p->N; i++)
    {
        p->coeffs[p->d - 1][i] &= 0x000FFFFFFFFFFFFFULL;
    }
#endif
}

// out = a*b mod (X^N + 1)
void mp_polynomial_mul_addto_sparse_MPPolynomial(MPPolynomial out, MPPolynomial a, uint64_t *b,
                                                 uint64_t size)
{
    const uint64_t N = a->N, N_mask = N - 1;
    MPPolynomial a_neg = new_mp_polynomial(N, a->d);
    mp_polynomial_negate(a_neg, a);
    mp_polynomial_propagate_carry(a_neg);
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            const uint64_t pos = (b[i] + j) & N_mask,
                           sign = (((b[i] & N_mask) + j) >= N) ^ (b[i] >> 63);
            for (size_t k = 0; k < a->d; k++)
            {
                if (sign)
                    out->coeffs[k][j] += a_neg->coeffs[k][pos];
                else
                    out->coeffs[k][j] += a->coeffs[k][pos];
            }
        }
    }
    mp_polynomial_propagate_carry(out);
    free_mp_polynomial(a_neg);
}

void print_MPScalar(MPScalar x)
{
#if defined(__AVX512IFMA__) && !defined(PORTABLE_BUILD) && !defined(PORTABLE)
    uint64_t *v_int = (uint64_t *)x->digits;
    printf("0x");
    for (int64_t i = x->d - 1; i >= 0; i--)
    {
        printf("%013lx", v_int[i * 8]);
    }
    printf("\n");
#else
    printf("0x");
    for (int64_t i = x->d - 1; i >= 0; i--)
    {
        printf("%013lx", x->digits[i]);
    }
    printf("\n");
#endif
}

uint64_t array32_bit_slice52(uint64_t *array, uint64_t start)
{
    uint64_t res = 0;
    const uint64_t word_idx = start / 32, word_shift = 52 - 32 + (start % 32),
                   mod_mask = (1ULL << 52) - 1;
    res |= array[word_idx] << word_shift;
    if (word_shift < 32)
    {
        res |= array[word_idx + 1] >> (32 - word_shift);
    }
    else
    {
        res |= array[word_idx + 1] << (word_shift - 32);
        res |= array[word_idx + 2] >> (64 - word_shift);
    }
    res &= mod_mask;
    return res;
}

mp_vector_t *load_m512(uint64_t in)
{
#if defined(__AVX512IFMA__) && !defined(PORTABLE_BUILD) && !defined(PORTABLE)
    __m512i *res = (__m512i *)safe_aligned_malloc(sizeof(__m512i));
    *res = _mm512_set1_epi64(in);
    return res;
#else
    uint64_t *res = (uint64_t *)safe_malloc(sizeof(uint64_t));
    *res = in;
    return res;
#endif
}

MPScalar mp_load(uint64_t *in, uint64_t d)
{
    MPScalar out;
    out = (MPScalar)safe_malloc(sizeof(*out));
    out->d = d;
#if defined(__AVX512IFMA__) && !defined(PORTABLE_BUILD) && !defined(PORTABLE)
    out->digits = (__m512i *)safe_aligned_malloc(d * sizeof(__m512i));
    for (size_t i = 0; i < d; i++)
        out->digits[i] = _mm512_set1_epi64(in[i]);
#else
    out->digits = (uint64_t *)safe_malloc(d * sizeof(uint64_t));
    for (size_t i = 0; i < d; i++)
        out->digits[i] = in[i];
#endif
    return out;
}

void mp_scale(MPScalar out, MPScalar in, mp_vector_t *m)
{
    assert(in != out);
    assert(out->d >= in->d + 1);
#if defined(__AVX512IFMA__) && !defined(PORTABLE_BUILD) && !defined(PORTABLE)
    __m512i mod_mask = _mm512_set1_epi64(0x000FFFFFFFFFFFFFULL);
    __m512i zero = _mm512_setzero_si512();
    out->digits[0] = zero;
    for (size_t j = 0; j < in->d; j++)
    {
        out->digits[j + 1] = _mm512_madd52hi_epu64(zero, in->digits[j], *m);
        out->digits[j] = _mm512_madd52lo_epu64(out->digits[j], in->digits[j], *m);
        out->digits[j + 1] =
            _mm512_add_epi64(out->digits[j + 1], _mm512_srli_epi64(out->digits[j], 52));
        out->digits[j] = _mm512_and_epi64(out->digits[j], mod_mask); // remove MSB
    }
#else
    out->digits[0] = 0;
    for (size_t j = 0; j < in->d; j++)
    {
        out->digits[j + 1] = madd52hi(0, in->digits[j], *m);
        out->digits[j] = madd52lo(out->digits[j], in->digits[j], *m);
        out->digits[j + 1] += out->digits[j] >> 52;
        out->digits[j] &= 0x000FFFFFFFFFFFFFULL;
    }
#endif
}

void mp_sub(MPScalar out, MPScalar a, MPScalar b)
{
    assert(b != out);
#if defined(__AVX512IFMA__) && !defined(PORTABLE_BUILD) && !defined(PORTABLE)
    __m512i mod_mask = _mm512_set1_epi64(0x000FFFFFFFFFFFFFULL);
    __m512i neg_mask = _mm512_set1_epi64(0x000FFFFFFFFFFFFFULL);
    __m512i one = _mm512_set1_epi64(1), tmp;
    for (size_t j = 0; j < out->d; j++)
    {
        // out = - b
        if (j < b->d)
        {
            tmp = _mm512_xor_epi64(b->digits[j], neg_mask);
        }
        else
        {
            tmp = neg_mask;
        }
        if (j == 0)
            tmp = _mm512_add_epi64(tmp, one);
        // out = -b + a
        out->digits[j] = _mm512_add_epi64(tmp, a->digits[j]);
    }
    for (size_t j = 0; j < out->d - 1; j++)
    {
        out->digits[j + 1] =
            _mm512_add_epi64(out->digits[j + 1], _mm512_srli_epi64(out->digits[j], 52));
        out->digits[j] = _mm512_and_epi64(out->digits[j], mod_mask);
    }
    out->digits[out->d - 1] = _mm512_and_epi64(out->digits[out->d - 1], mod_mask);
#else
    for (size_t j = 0; j < out->d; j++)
    {
        uint64_t tmp;
        if (j < b->d)
        {
            tmp = b->digits[j] ^ 0x000FFFFFFFFFFFFFULL;
        }
        else
        {
            tmp = 0x000FFFFFFFFFFFFFULL;
        }
        if (j == 0)
            tmp += 1;
        out->digits[j] = tmp + a->digits[j];
    }
    for (size_t j = 0; j < out->d - 1; j++)
    {
        out->digits[j + 1] += out->digits[j] >> 52;
        out->digits[j] &= 0x000FFFFFFFFFFFFFULL;
    }
    out->digits[out->d - 1] &= 0x000FFFFFFFFFFFFFULL;
#endif
}

#if defined(__AVX512IFMA__) && !defined(PORTABLE_BUILD) && !defined(PORTABLE)
#define MPScalar_stack_alloc_(var, size, suffix)                                                   \
    struct _MPScalar __p##suffix;                                                                  \
    __p##suffix.d = (size);                                                                        \
    __m512i __tmp##suffix[(size)];                                                                 \
    __p##suffix.digits = __tmp##suffix;                                                            \
    var = &__p##suffix;

#define MPScalar_stack_alloc__(var, size, suffix) MPScalar_stack_alloc_(var, size, suffix)
#define MPScalar_stack_alloc(var, size) MPScalar_stack_alloc__(var, size, __LINE__)
#endif

void mp_polynomial_mod_reduce(MPPolynomial out, MPScalar q, mp_vector_t *m, uint64_t k)
{
#if defined(__AVX512IFMA__) && !defined(PORTABLE_BUILD) && !defined(PORTABLE)
    __m512i zero = _mm512_setzero_si512();
    MPScalar tmp1, tmp2;
    MPScalar_stack_alloc(tmp1, out->d + 1);
    MPScalar_stack_alloc(tmp2, out->d);

    for (size_t i = 0; i < out->N / 8; i++)
    {
        // tmp2 = x
        for (size_t j = 0; j < out->d; j++)
            tmp2->digits[j] = ((__m512i *)(out->coeffs[j]))[i];
        // tmp1 = x*m
        tmp1->d = out->d + 1;
        mp_scale(tmp1, tmp2, m);
        // quo = tmp/2**k = x*m / 2**k
        const uint64_t bit_length = 52 * out->d - k;
        __m512i quo = tmp1->digits[out->d] << bit_length;
        quo |= tmp1->digits[out->d - 1] >> (52 - bit_length);
        // tmp1 = quo*q
        tmp1->d = out->d;
        mp_scale(tmp1, q, &quo);
        // x = x - quo*q
        mp_sub(tmp2, tmp2, tmp1);
        mp_sub(tmp1, tmp2, q);
        __mmask8 select = _mm512_cmpeq_epi64_mask(tmp1->digits[out->d - 1], zero);
        // x = tmp2 or tmp 1
        for (size_t j = 0; j < out->d; j++)
        {
            ((__m512i *)(out->coeffs[j]))[i] =
                _mm512_mask_blend_epi64(select, tmp2->digits[j], tmp1->digits[j]);
        }
    }
#else
    MPScalar tmp1, tmp2;
    uint64_t tmp1_digits[out->d + 1];
    uint64_t tmp2_digits[out->d];
    struct _MPScalar s_tmp1 = {tmp1_digits, out->d + 1};
    struct _MPScalar s_tmp2 = {tmp2_digits, out->d};
    tmp1 = &s_tmp1;
    tmp2 = &s_tmp2;

    for (size_t i = 0; i < out->N; i++)
    {
        for (size_t j = 0; j < out->d; j++)
            tmp2->digits[j] = out->coeffs[j][i];
        tmp1->d = out->d + 1;
        mp_scale(tmp1, tmp2, m);
        const uint64_t bit_length = 52 * out->d - k;
        uint64_t quo = tmp1->digits[out->d] << bit_length;
        quo |= tmp1->digits[out->d - 1] >> (52 - bit_length);
        tmp1->d = out->d;
        mp_scale(tmp1, q, &quo);
        mp_sub(tmp2, tmp2, tmp1);
        mp_sub(tmp1, tmp2, q);
        int select = (tmp1->digits[out->d - 1] == 0);
        for (size_t j = 0; j < out->d; j++)
        {
            out->coeffs[j][i] = select ? tmp1->digits[j] : tmp2->digits[j];
        }
    }
#endif
}

void mp_polynomial_to_RNSc(RNSc_Polynomial out, MPPolynomial in)
{
    const uint64_t N = in->N, D = in->d;
    for (size_t i = 0; i < out->ntt->l; i++)
    {
        if (!(out->rns_mask & (1ULL << i)))
            continue;
        NTT_proc proc = out->ntt->ntt[i];
        const uint64_t w1 = proc->mp_w1, q = proc->q;
        for (uint64_t c = 0; c < N; c++)
        {
            uint64_t res = 0;
            for (int64_t j = (int64_t)D - 1; j >= 0; j--)
            {
                res = mul_modq(res, w1, proc);
                res = add_modq(res, modq((unsigned __int128)in->coeffs[j][c], proc), q);
            }
            out->coeffs[i][c] = res;
        }
    }
}

void mp_polynomial_from_RNS(MPPolynomial out, RNS_Polynomial in, MPScalar *PW, MPScalar q,
                            mp_vector_t *m, uint64_t k)
{
    mp_polynomial_int_sp_scale_mp(out, in->coeffs[0], PW[0]);
    for (size_t i = 1; i < in->ntt->l; i++)
    {
        mp_polynomial_int_sp_scale_addto_mp(out, in->coeffs[i], PW[i]);
        if ((i & 0xFF) == 0)
        {
            mp_polynomial_propagate_carry(out);
        }
    }
    mp_polynomial_propagate_carry(out);
    mp_polynomial_mod_reduce(out, q, m, k);
}

mp_vector_t *_delta = NULL;
uint64_t _p = 0, _d = 0;
void setup_mod_switch_delta(uint64_t d, uint64_t p)
{
    if (p == _p && _d >= d)
    {
        return;
    }
    else
    {
        if (_delta)
            free(_delta);
#if defined(__AVX512IFMA__) && !defined(PORTABLE_BUILD) && !defined(PORTABLE)
        _delta = (__m512i *)safe_aligned_malloc(sizeof(__m512i) * d);
#else
        _delta = (uint64_t *)safe_malloc(sizeof(uint64_t) * d);
#endif
    }
    assert(p < (1ULL << 32));
    uint64_t delta32[2 * d], rem = 1;
    memset(delta32, 0, sizeof(uint64_t) * 2 * d);
    for (size_t i = 0; i < 2 * d; i++)
    {
        delta32[i] = (rem << 32) / p;
        rem = (rem << 32) % p;
    }
    for (size_t i = 0; i < d; i++)
    {
#if defined(__AVX512IFMA__) && !defined(PORTABLE_BUILD) && !defined(PORTABLE)
        _delta[i] = _mm512_set1_epi64(array32_bit_slice52(delta32, 52 * i));
#else
        _delta[i] = array32_bit_slice52(delta32, 52 * i);
#endif
    }
    _p = p;
    _d = d;
}
