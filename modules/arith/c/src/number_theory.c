#include <arith.h>
#include <stdbool.h>

uint64_t power_mod(uint64_t base, uint64_t exp, uint64_t mod)
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

uint64_t inverse_mod(uint64_t a, uint64_t m)
{
    // Assuming m is prime (Fermat's Little Theorem)
    return power_mod(a, m - 2, m);
}

bool is_prime(uint64_t n)
{
    if (n < 2)
        return false;
    if (n == 2 || n == 3)
        return true;
    if (n % 2 == 0 || n % 3 == 0)
        return false;

    // Miller-Rabin primality test
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
        uint64_t x = power_mod(a, d, n);
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

uint64_t generate_Nth_root_of_unity(uint64_t q, uint64_t n)
{
    // Finds a primitive n-th root of unity modulo q
    // Assumes q is prime and n divides q-1
    uint64_t phi = q - 1;
    uint64_t g = 2;

    // Simplification: since we usually use n = 2^k, we can just find
    // a primitive root and raise it to (q-1)/n.
    // Or we can just test random elements.

    // For general n, we'd need factors of n to check primitivity.
    // However, if n is a power of 2, a root w is primitive n-th if w^(n/2) == -1 mod q.

    while (true)
    {
        uint64_t root = power_mod(g, phi / n, q);
        if (power_mod(root, n / 2, q) == q - 1)
        {
            return root;
        }
        g++;
    }
}

uint64_t next_special_prime(uint64_t x, uint64_t n, bool primitive)
{
    uint64_t rou_order = 2 * n;
    uint64_t a = x / rou_order;

    // We want candidate = a * rou_order + 1 > x
    // If x % rou_order == 0, then a*rou_order + 1 might be < x or > x.
    // Let's ensure candidate > x.
    if (a * rou_order + 1 <= x)
        a++;

    if (primitive)
    {
        // candidate % 4n != 1
        // candidate = a * 2n + 1
        // (a * 2n + 1) % 4n = 1 if a is even, 2n + 1 if a is odd.
        // So a must be odd.
        if (a % 2 == 0)
            a++;
    }

    while (true)
    {
        uint64_t candidate = a * rou_order + 1;
        if (is_prime(candidate))
        {
            if (primitive)
            {
                if (candidate % (4 * n) != 1)
                    return candidate;
            }
            else
            {
                return candidate;
            }
        }
        a += (primitive ? 2 : 1);
    }
}
