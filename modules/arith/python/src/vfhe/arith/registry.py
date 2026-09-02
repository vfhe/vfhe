# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The registry of arithmetic implementations, backends, and conversions.

Two tables. The first maps ``(implementation, backend)`` to the `Spec`
describing it, and `resolve` picks an entry from a caller's parameters. The
second holds conversions between specs: absent one, mixed-spec arithmetic is
an error the caller resolves explicitly; present and marked implicit, a
binary operation may apply it silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from .spec import Capability, Spec

_SPECS: dict[tuple[str, str], Spec] = {}
_CONVERSIONS: dict[tuple[tuple[str, str], tuple[str, str]], Conversion] = {}


@dataclass(frozen=True)
class Conversion:
    """How to move a value from one spec to another.

    ``implicit`` conversions may be applied by a binary operation without the
    caller asking; every other conversion must be requested explicitly, so
    that an expensive move (a device copy, a change of representation) is
    visible at the call site rather than hidden in an operator.

    ``cost`` orders candidates when more than one route exists; it is a
    relative number with no unit.
    """

    source: tuple[str, str]
    target: tuple[str, str]
    convert: Callable
    implicit: bool = False
    cost: float = 1.0


def register(spec: Spec) -> Spec:
    """Add `spec` to the registry, replacing any entry with the same key."""
    _SPECS[spec.key] = spec
    return spec


def registered() -> dict[tuple[str, str], Spec]:
    """Every registered spec, keyed by ``(implementation, backend)``."""
    return dict(_SPECS)


def implementations() -> list[str]:
    """Names of every registered implementation, sorted."""
    return sorted({key[0] for key in _SPECS})


def backends(implementation: str) -> list[str]:
    """Backend names registered for `implementation`, best-ranked first."""
    matches = [
        spec for spec in _SPECS.values() if spec.implementation == implementation
    ]
    return [spec.backend for spec in sorted(matches, key=lambda s: (s.rank, s.backend))]


def get(implementation: str, backend: str) -> Spec:
    """The spec for one pair, or KeyError naming what is registered."""
    try:
        return _SPECS[(implementation, backend)]
    except KeyError:
        raise KeyError(
            f"no backend '{backend}' for implementation '{implementation}'; "
            f"registered for it: {backends(implementation) or 'none'}"
        ) from None


def resolve(
    implementation: str | None = None,
    backend: str | None = None,
    prime_bits: int | None = None,
    requires: Capability | None = None,
) -> Spec:
    """The spec matching a caller's request, choosing what was left open.

    `implementation` defaults to the only registered one when there is no
    ambiguity, so a caller that does not care never has to name it.
    `backend` defaults to the best-ranked backend of that implementation
    whose constraints `prime_bits` satisfies. `requires` filters to specs
    carrying every named capability.

    Raises LookupError naming the available choices when nothing matches.
    """
    if implementation is None:
        names = implementations()
        if len(names) != 1:
            raise LookupError(
                f"implementation is ambiguous; pass one of {names or 'none registered'}"
            )
        implementation = names[0]

    if implementation not in implementations():
        raise LookupError(
            f"unknown implementation '{implementation}'; "
            f"registered: {implementations() or 'none'}"
        )

    if backend is not None:
        spec = get(implementation, backend)
        _check(spec, prime_bits, requires)
        return spec

    candidates = sorted(
        (s for s in _SPECS.values() if s.implementation == implementation),
        key=lambda s: (s.rank, s.backend),
    )
    for spec in candidates:
        if prime_bits is not None and not spec.constraints.accepts_prime_bits(
            prime_bits
        ):
            continue
        if requires is not None and not spec.has(requires):
            continue
        return spec

    raise LookupError(
        f"no backend of '{implementation}' satisfies "
        f"prime_bits={prime_bits}, requires={requires}; "
        f"tried {[str(s) for s in candidates]}"
    )


def _check(spec: Spec, prime_bits: int | None, requires: Capability | None) -> None:
    """Raise if an explicitly named spec cannot serve the request."""
    if prime_bits is not None and not spec.constraints.accepts_prime_bits(prime_bits):
        raise LookupError(
            f"{spec} accepts primes up to {spec.constraints.max_prime_bits} bits, "
            f"but {prime_bits} were requested"
        )
    if requires is not None and not spec.has(requires):
        missing = requires & ~spec.capabilities
        raise LookupError(f"{spec} lacks {missing!r}")


def register_conversion(
    source: tuple[str, str],
    target: tuple[str, str],
    convert: Callable,
    implicit: bool = False,
    cost: float = 1.0,
) -> Conversion:
    """Declare how to move a value from `source` to `target`."""
    conversion = Conversion(source, target, convert, implicit, cost)
    _CONVERSIONS[(source, target)] = conversion
    return conversion


def find_conversion(
    source: tuple[str, str], target: tuple[str, str]
) -> Conversion | None:
    """The registered conversion between two specs, or None."""
    return _CONVERSIONS.get((source, target))


def common_spec(left: Spec, right: Spec) -> Spec:
    """The spec a binary operation on `left` and `right` should compute in.

    Returns the shared spec when both sides already agree. Otherwise an
    implicit conversion must be registered in one direction, and the target
    of that conversion is the answer; a conversion registered both ways
    resolves toward the cheaper direction. Raises TypeError naming the
    explicit call to make when no implicit conversion exists, so that a
    silent and possibly expensive move never happens behind an operator.
    """
    if left.key == right.key:
        return left

    forward = find_conversion(left.key, right.key)
    backward = find_conversion(right.key, left.key)
    options = [
        (conversion, target)
        for conversion, target in ((forward, right), (backward, left))
        if conversion is not None and conversion.implicit
    ]
    if not options:
        raise TypeError(
            f"cannot combine {left} with {right}: no implicit conversion is "
            f"registered. Convert explicitly first."
        )
    return min(options, key=lambda option: option[0].cost)[1]
