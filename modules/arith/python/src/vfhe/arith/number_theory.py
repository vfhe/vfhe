# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-FileCopyrightText: 2026 Daniele Cozzo <daniele.cozzo@imdea.org>
# SPDX-License-Identifier: Apache-2.0
def is_prime(n: int) -> bool:
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


def _two_adicity(n: int) -> int:
    return (n & -n).bit_length() - 1


def gen_pseudo_mersenne_prime(
    bits: int,
    two_adicity: int = 0,
    max_two_adicity: int | None = None,
    max_c_bits: int = 52,
) -> int:
    """
    Generate the pseudo-Mersenne prime p = 2^bits - c with the smallest c > 0
    such that 2^two_adicity divides p - 1 (i.e. p admits at least
    2^two_adicity-th roots of unity).

    p - 1 = 2^bits - (c + 1), so for bits > two_adicity the 2-adicity of p - 1
    is governed entirely by c + 1: restricting the search to the residue class
    c = 2^v - 1 (mod 2^v) makes 2^v divide c + 1 and hence p - 1. A smaller c is
    strictly better (it is the reduction constant), so the search returns the
    first hit in that class.

    two_adicity is a lower bound, and the prime found may well exceed it -- for
    free, and typically with a smaller c. max_two_adicity caps it instead, so
    max_two_adicity=two_adicity pins the 2-adicity exactly, and a wider pair
    (e.g. 6, 7) searches a band of acceptable root-of-unity counts. c is capped
    at max_c_bits bits, defaulting to the 52-bit IFMA limb width so that the
    reduction constant stays a single-limb value.
    """
    if two_adicity < 0:
        raise ValueError("two_adicity must be non-negative")
    if bits <= two_adicity + 1:
        raise ValueError("bits must exceed two_adicity + 1")
    if max_two_adicity is not None:
        if max_two_adicity < two_adicity:
            raise ValueError("max_two_adicity must be >= two_adicity")
        # p is odd, so p - 1 always has 2-adicity >= 1: a cap of 0 is unmeetable.
        if max_two_adicity == 0:
            raise ValueError("max_two_adicity 0 is unsatisfiable for an odd prime")

    # v = 0 has no residue-class constraint, but c must stay odd to keep p odd.
    step = 1 << two_adicity if two_adicity else 2
    c = step - 1 if two_adicity else 1
    # Keep c single-limb and small enough for p to really have `bits` bits.
    limit = min(1 << max_c_bits, 1 << (bits - 1))

    while c < limit:
        p = (1 << bits) - c
        capped = max_two_adicity is None or _two_adicity(p - 1) <= max_two_adicity
        if capped and is_prime(p):
            return p
        c += step

    raise ValueError(
        f"no pseudo-Mersenne prime 2^{bits} - c with 2-adicity in "
        f"[{two_adicity}, {'inf' if max_two_adicity is None else max_two_adicity}] "
        f"and c < 2^{max_c_bits}"
    )


def crt(values: list[int], moduli: list[int]) -> int:
    if len(moduli) == 1:
        return values[0] % moduli[0]
    N = 1
    for m in moduli:
        N *= m
    result = 0
    for val, m in zip(values, moduli, strict=True):
        n = N // m
        inv = pow(n, -1, m)
        result = (result + val * n * inv) % N
    return result
