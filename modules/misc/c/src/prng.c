#include "misc.h"
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <blake3.h>

#ifdef USE_SHAKE
#include "../third-party/sha3/fips202.c"
#endif

// Forward declaration of aes_prng (only used if AES-NI is available and not in portable build)
#if !defined(PORTABLE_BUILD) && !defined(PORTABLE) && (defined(__x86_64__) || defined(_M_X64)) &&  \
    defined(__AES__)
void aes_prng(uint8_t *output, uint64_t outlen, const uint8_t *input, uint64_t inlen);
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(PORTABLE) && !defined(PORTABLE_BUILD)
// Intel hardware RDRAND seed generator
void generate_rnd_seed(uint64_t *p)
{
    if (0 == _rdrand64_step((unsigned long long *)p) ||
        0 == _rdrand64_step((unsigned long long *)&(p[1])) ||
        0 == _rdrand64_step((unsigned long long *)&(p[2])) ||
        0 == _rdrand64_step((unsigned long long *)&(p[3])))
    {
        printf("Random Generation Failed\n");
        return;
    }
}
#else
// Fallback urandom seed generator
void generate_rnd_seed(uint64_t *p)
{
    FILE *fp;
    fp = fopen("/dev/urandom", "r");
    if (fp)
    {
        size_t unused = fread(p, 1, 32, fp);
        (void)unused;
        fclose(fp);
    }
}
#endif

void get_rnd_from_hash(uint64_t amount, uint8_t *pointer)
{
    uint64_t rnd[4];
    generate_rnd_seed(rnd);

#if !defined(PORTABLE_BUILD) && !defined(PORTABLE) && (defined(__x86_64__) || defined(_M_X64)) &&  \
    defined(__AES__)
    aes_prng(pointer, amount, (uint8_t *)rnd, 32);
#elif defined(USE_SHAKE)
    shake256(pointer, amount, (uint8_t *)rnd, 32);
#else
    // Default fallback is Blake3
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, (uint8_t *)rnd, 32);
    blake3_hasher_finalize(&hasher, pointer, amount);
#endif
}

void get_rnd_from_buffer(uint64_t amount, uint8_t *pointer)
{
    static uint8_t buffer[1024] __attribute__((aligned(64)));
    static uint64_t idx = 1024;
    if (amount > (1024 - idx))
    {
        idx = 0;
        get_rnd_from_hash(1024, buffer);
    }
    memcpy(pointer, buffer + idx, (size_t)amount);
    idx += amount;
}

void generate_random_bytes(uint64_t amount, uint8_t *pointer)
{
    if (amount < 512)
        get_rnd_from_buffer(amount, pointer);
    else
        get_rnd_from_hash(amount, pointer);
}
