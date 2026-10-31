# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Basefold parameters for a target soundness.

`basefold_parameters` chooses the general foldable code (`k0`, `c`, `d`) and
the number of queries `rep` of a basefold opening so that the soundness error
`BasefoldEval.soundness_error` reports is at most `2^-security_bits`, and
among the choices that reach it, the one with the smallest proof.

The bound is [ZCF24]'s soundness theorem as [CCCFGS26] states it (its
appendix on BaseFold): for `gamma, delta` in `(0, 1)` with

    delta < J_gamma(J_gamma(Delta))  and  3 delta - d gamma < Delta,
    J_gamma(x) = 1 - sqrt(1 - x (1 - gamma)),

where `Delta` is the relative minimum distance of the depth-`d` code, an
evaluation proof is accepted with probability at most
`2d / (gamma^3 |F|) + (1 - delta + gamma d)^rep` when the claim is false
(plus the batching terms of a bundle of claims). `delta` is a proximity
parameter derived from the distance, not the distance itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vfhe.arith import ExtensionField, Field, RNSRing
from vfhe.crypto import DIGEST_LEN

from .basefold import basefold_soundness_error
from .code import foldable_relative_distance
from .field_code import TWIST_FIELDS

# The offsets of gamma above its least admissible value that the search tries:
# a half-decade grid, 10^-2 to 10^-7.5.
_GAPS = tuple(10 ** (-i / 2) for i in range(4, 16))


@dataclass(frozen=True)
class BasefoldParameters:
    """What `basefold_parameters` chose.

    `FieldFoldableRS(field, k0, c, d)` or `FoldableRS(ring, k0, c, d)` is the
    code, `Basefold(code, schedule)` the scheme and `BasefoldEval(scheme,
    rep)` the evaluation protocol. `distance` is the code's
    `relative_distance`, `gamma` and `delta` the arguments of
    `BasefoldEval.soundness_error` that give `soundness_error`, and
    `proof_bits` the size estimate the choice minimized.
    """

    k0: int
    c: int
    d: int
    rep: int
    schedule: tuple[int, ...]
    distance: float
    gamma: float
    delta: float
    soundness_error: float
    proof_bits: float

    @property
    def n0(self) -> int:
        """The length of the base codeword."""
        return self.c * self.k0

    @property
    def num_vars(self) -> int:
        """The variables of the committed polynomials."""
        return self.d + self.k0.bit_length() - 1


def _johnson(gamma: float, x: float) -> float:
    """`J_gamma(x) = 1 - sqrt(1 - x (1 - gamma))`."""
    return 1 - math.sqrt(1 - x * (1 - gamma))


def _twist_bits(domain: Field | RNSRing, twist_field: str | None) -> float:
    """`log2` of the size of the field the general code's twists are drawn
    from, the field size of its distance bound: what `relative_distance`
    uses on a `FieldFoldableRS` or `FoldableRS` built over `domain` with
    `twist_field` (the class's default when None)."""
    if twist_field is not None and twist_field not in TWIST_FIELDS:
        raise ValueError(
            f"twist_field must be one of {TWIST_FIELDS}, got {twist_field!r}"
        )
    if isinstance(domain, Field):
        if (
            twist_field == "prime"
            and isinstance(domain, ExtensionField)
            and domain.d > 1
        ):
            return math.log2(domain.prime)
        return math.log2(domain.order)
    bits = math.log2(min(domain.primes))
    if twist_field is None:
        twist_field = "field" if domain.split_degree > 1 else "prime"
    return bits * domain.split_degree if twist_field == "field" else bits


def _element_bits(domain: Field | RNSRing) -> float:
    """The bits of one element of `domain`: `log2 |F|` for a field,
    `N log2 q` for `R_q`."""
    if isinstance(domain, Field):
        return math.log2(domain.order)
    return domain.N * sum(math.log2(p) for p in domain.primes)


def _proof_bits(
    element_bits: float,
    digest_bits: int,
    k0: int,
    n0: int,
    d: int,
    rep: int,
    num_vars: int,
    claims: int,
    polynomials: int,
) -> float:
    """The size of an opening's messages under the default schedule: `d`
    round messages of three elements, `d - 1` roots, the `k0`-entry base
    table, and for each query a pair and its path at every level `l` from
    `d` down to 1 (`log2 n_l - 1` digests), the top level's once per
    polynomial. A bundle of claims adds the common-point sumcheck, counted
    as if the points were distinct: `num_vars` rounds of three elements and
    one value per polynomial."""
    log_n0 = n0.bit_length() - 1
    elements = 3 * d + k0 + 2 * rep * (d - 1 + polynomials)
    digests = (d - 1) + rep * (
        polynomials * (log_n0 + d - 1) + (d - 1) * log_n0 + (d - 1) * (d - 2) // 2
    )
    if claims > 1:
        elements += 3 * num_vars + polynomials
    return elements * element_bits + digests * digest_bits


def basefold_parameters(
    domain: Field | RNSRing,
    num_vars: int,
    security_bits: int = 128,
    *,
    claims: int = 1,
    polynomials: int = 1,
    twist_field: str | None = None,
    bound: str = "cccfgs26",
    max_k0: int = 256,
    inverse_rates: tuple[int, ...] = (2, 4),
    digest_bits: int = 8 * DIGEST_LEN,
) -> BasefoldParameters:
    """The basefold parameters for opening `num_vars`-variate multilinear
    polynomials over `domain` -- a field, or `R_q` as an `RNSRing` -- with
    soundness error at most `2^-security_bits`, by the general foldable code
    [ZCF24] and the default schedule, one fold per committed codeword, which
    is what the soundness theorem covers.

    The search is the one of the parameter estimator of [CCCFGS26], run on
    the library's bounds. For every base dimension
    `k0 = 2^i <= max_k0` that leaves `d = num_vars - i >= 1` folds, and
    every inverse rate `c` in `inverse_rates`:

    - `Delta` is `foldable_relative_distance(k0, c, d, ...)` with `bound`,
      the value `relative_distance` reports on the code these parameters
      build; its field is the one the twists are drawn from, chosen by
      `twist_field` as the code classes do (their own default when None:
      the whole field for `FieldFoldableRS`, and for `FoldableRS` the
      residue fields `F_(p^d)` on a split ring and `F_p` otherwise);
    - `gamma = gamma_0 + g` for every `g` in a half-decade grid from
      `10^-2` to `10^-7.5`, where `gamma_0` makes `2d / (gamma^3 |A|)` the
      whole budget left by the batching terms;
    - `delta = min((Delta + d gamma) / 3, J_gamma(J_gamma(Delta)))`, the
      largest proximity parameter the theorem admits;
    - `rep` is the fewest queries for which `basefold_soundness_error`, the
      expression `BasefoldEval.soundness_error` reports, is at most
      `2^-security_bits`, and it must be below `n0 = c k0`, as `BasefoldEval`
      requires.

    The candidate with the smallest `proof_bits` is returned: the round
    messages, the roots, the base table and the queries' pairs and paths
    of one opening, with elements of `log2 |F|` bits over a field and
    `N log2 q` over `R_q`, and digests of `digest_bits`. `|A|` is the
    domain's `exceptional_set_size`, the set the challenges come from.

    `claims` and `polynomials` describe a bundle opened together, as
    `soundness_error` takes them. `max_k0` bounds the base table the
    verifier re-encodes in full; with `inverse_rates`, its default is the
    estimator's search space.

    Raises `ValueError` when no candidate reaches the target, which is
    always the case when `|A|` is below about `2^security_bits`: the first
    term of the bound then exceeds the target for every `gamma < 1`.
    """
    if num_vars < 1:
        raise ValueError(f"num_vars must be at least 1, got {num_vars}")
    if claims < 1 or polynomials < 1 or polynomials > claims:
        raise ValueError(
            "a bundle has at least one claim per polynomial and one polynomial, "
            f"got claims = {claims}, polynomials = {polynomials}"
        )
    for name, value in (("max_k0", max_k0), *(("c", c) for c in inverse_rates)):
        if value < 1 or value & (value - 1):
            raise ValueError(f"{name} must be a power of two, got {value}")
    field_bits = _twist_bits(domain, twist_field)
    element_bits = _element_bits(domain)
    size = domain.exceptional_set_size
    target = 2.0**-security_bits
    batching = ((2 * num_vars + claims - 1) if claims > 1 else 0) + polynomials - 1
    budget = target - batching / size
    if budget <= 0:
        raise ValueError(
            f"the batching terms alone exceed 2^-{security_bits} over a "
            f"challenge set of 2^{math.log2(size):.1f} elements"
        )
    best: BasefoldParameters | None = None
    for log_k0 in range(min(max_k0.bit_length() - 1, num_vars - 1) + 1):
        k0 = 1 << log_k0
        d = num_vars - log_k0
        for c in inverse_rates:
            n0 = c * k0
            distance = foldable_relative_distance(
                k0, c, d, field_bits, security_bits, bound
            )
            if distance <= 0:
                continue
            gamma_0 = (2 * d / (size * budget)) ** (1 / 3)
            for gap in _GAPS:
                gamma = gamma_0 + gap
                if gamma >= 1:
                    continue
                delta = min(
                    (distance + d * gamma) / 3,
                    _johnson(gamma, _johnson(gamma, distance)),
                )
                base = 1 - delta + gamma * d
                rest = budget - 2 * d / (gamma**3 * size)
                if not (0 < base < 1 and rest > 0):
                    continue

                def error(rep: int, d=d, gamma=gamma, delta=delta) -> float:
                    return basefold_soundness_error(
                        size, d, num_vars, rep, gamma, delta, claims, polynomials
                    )

                rep = max(1, math.ceil(math.log(rest) / math.log(base)))
                while rep < n0 and error(rep) > target:
                    rep += 1
                while rep > 1 and error(rep - 1) <= target:
                    rep -= 1
                if rep >= n0:
                    continue
                bits = _proof_bits(
                    element_bits,
                    digest_bits,
                    k0,
                    n0,
                    d,
                    rep,
                    num_vars,
                    claims,
                    polynomials,
                )
                if best is None or bits < best.proof_bits:
                    best = BasefoldParameters(
                        k0=k0,
                        c=c,
                        d=d,
                        rep=rep,
                        schedule=(1,) * d,
                        distance=distance,
                        gamma=gamma,
                        delta=delta,
                        soundness_error=error(rep),
                        proof_bits=bits,
                    )
    if best is None:
        raise ValueError(
            f"no basefold parameters reach 2^-{security_bits} for {num_vars} "
            f"variables with twists over 2^{field_bits:.1f} elements and "
            f"challenges from 2^{math.log2(size):.1f}"
        )
    return best
