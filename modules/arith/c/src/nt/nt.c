// SPDX-License-Identifier: Apache-2.0
/**
 * @file nt.c
 * @brief Number theory on plain 64-bit integers (see arith/nt.h).
 */
#include <arith/nt.h>

uint64_t nt_power_mod(uint64_t base, uint64_t exp, uint64_t mod)
{
    uint64_t res = 1;
    base %= mod;
    while (exp > 0)
    {
        if (exp % 2 == 1)
            res = (uint64_t)(((unsigned __int128)res * base) % mod);
        base = (uint64_t)(((unsigned __int128)base * base) % mod);
        exp /= 2;
    }
    return res;
}

uint64_t nt_inverse_mod(uint64_t a, uint64_t m)
{
    // m prime => a^(m-2) is the inverse (Fermat).
    return nt_power_mod(a, m - 2, m);
}

bool nt_is_prime(uint64_t n)
{
    if (n < 2)
        return false;
    if (n == 2 || n == 3)
        return true;
    if (n % 2 == 0 || n % 3 == 0)
        return false;

    // Miller-Rabin with a base set that is deterministic for all 64-bit inputs.
    uint64_t d = n - 1;
    int s = 0;
    while (d % 2 == 0)
    {
        d /= 2;
        s++;
    }

    static const uint64_t bases[] = {2, 3, 5, 7, 11, 13, 17, 19, 23};
    for (int i = 0; i < 9; i++)
    {
        uint64_t a = bases[i];
        if (n <= a)
            break;
        uint64_t x = nt_power_mod(a, d, n);
        if (x == 1 || x == n - 1)
            continue;
        bool composite = true;
        for (int r = 1; r < s; r++)
        {
            x = (uint64_t)(((unsigned __int128)x * x) % n);
            if (x == n - 1)
            {
                composite = false;
                break;
            }
        }
        if (composite)
            return false;
    }
    return true;
}

uint64_t nt_gen_root_of_unity(uint64_t q, uint64_t n)
{
    // n is a power of two, so w = g^((q-1)/n) is a primitive n-th root exactly
    // when w^(n/2) == -1; scan generators g until that holds.
    const uint64_t phi = q - 1;
    uint64_t g = 2;
    while (true)
    {
        uint64_t root = nt_power_mod(g, phi / n, q);
        if (nt_power_mod(root, n / 2, q) == q - 1)
        {
            return root;
        }
        g++;
    }
}

uint64_t nt_next_ntt_prime(uint64_t x, uint64_t n, bool primitive)
{
    const uint64_t rou_order = 2 * n;
    uint64_t a = x / rou_order;

    // Candidates are a * 2n + 1; ensure the first one is > x.
    if (a * rou_order + 1 <= x)
        a++;

    if (primitive)
    {
        // (a*2n + 1) % 4n == 1 iff a is even, so odd a gives p % 4n != 1.
        if (a % 2 == 0)
            a++;
    }

    while (true)
    {
        uint64_t candidate = a * rou_order + 1;
        if (nt_is_prime(candidate))
        {
            if (!primitive || candidate % (4 * n) != 1)
                return candidate;
        }
        a += (primitive ? 2 : 1);
    }
}
