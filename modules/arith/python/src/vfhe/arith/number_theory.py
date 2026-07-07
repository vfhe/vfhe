import random


def is_prime(n):
    if n < 2:
        return False
    # Check small primes
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if n == p:
            return True
        if n % p == 0:
            return False

    # Deterministic bases for Miller-Rabin up to 2^64
    if n < 1373653:
        bases = [2, 3]
    elif n < 9080191:
        bases = [31, 73]
    elif n < 4759123141:
        bases = [2, 7, 61]
    elif n < 1122004669633:
        bases = [2, 13, 23, 1662803]
    elif n < 2152302898747:
        bases = [2, 3, 5, 7, 11]
    elif n < 3474749660383:
        bases = [2, 3, 5, 7, 11, 13]
    elif n < 341550071728321:
        bases = [2, 3, 5, 7, 11, 13, 17]
    else:
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    d = n - 1
    r = 0
    while d % 2 == 0:
        r += 1
        d //= 2

    for a in bases:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def crt(values, moduli):
    if len(moduli) == 1:
        return values[0] % moduli[0]
    N = 1
    for m in moduli:
        N *= m
    result = 0
    for val, m in zip(values, moduli):
        n = N // m
        inv = pow(n, -1, m)
        result = (result + val * n * inv) % N
    return result
