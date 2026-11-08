# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The arithmetization of [CCCFGS26, Sec. 5 and App. A.5]: a proof-friendly
CKKS multiplication as an arithmetic circuit over R_q plus range checks,
with the paper's names.

For a multiplication (PreMult, Eq. 18; KeySwitch, Eq. 19; Rescale, Eq. 20):

- instance: ``ct0`` and ``ct1``, the operands as ``[mask, body]``, and
  ``evk``, the key's components, one entry per ciphertext component ``i``
  holding ``evk[i][t]`` for each digit ``t``;
- intermediate: ``d0``, ``d1``, ``d2``, ``ct'`` (the key switch's output)
  and ``ct''`` (the rescaled output);
- witness: ``w_ks``, the digits, each within the bound of its
  decomposition (``p_t/2`` for vfhe's centered residues, which is within
  the paper's ``< p_t``; ``2^(w-1)`` for balanced radix digits), and
  ``w_quot`` and ``w_rmd`` of each rescale, ``w_rmd`` within ``P/2`` for
  the product ``P`` of the primes dropped and ``w_quot`` within ``q_l/P``,
  the modulus of its ring, which an element of that ring meets by
  construction and which the range check imposes on the integers the
  proof works with.

A tensor product no key switch completes is PreMult alone; a key switch
that completes no tensor product (the relinearization of a sum of extended
products) has the same names as a multiplication's, ``d2`` the value it
decomposes and ``d1``, ``d0`` what it adds to the mask and to the body
(when it adds anything). A rescale between levels that are not nested
divides ``P_l * ct``, the component lifted to the union of the two levels
and scaled by the product ``P_l`` of the primes the lift adds.

Two conventions of vfhe differ from the paper and are stated here rather
than hidden: the products with the key are subtracted,
``c'_i = d_i - sum_t w_ks[t] * evk[i][t]``; and a hybrid key has special
primes, whose division is approximate: its quotient and remainder are
``w_quot_P`` and ``w_rmd_P``, with ``w_rmd_P`` within ``(k+1)P/2`` for
``k`` special primes.

The relations: ``PreMult`` (Eq. 18); at each prime, the digits taken there
recompose to ``d2``: the digit itself for a residue, the sum of the radix
digits times their powers of two for a radix decomposition (up to the
rounding of the digits it leaves out); ``KeySwitch`` (Eq. 19, with the two
conventions); for each division, ``x * z_l = (w_quot * P + w_rmd) * z_l``:
in the quotient's ring the identity holds by construction, so what is
checked is ``x`` and ``w_rmd`` agreeing modulo the primes divided by; and
for a lifted rescale, that the value divided is the lift of its input.
"""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING, Any

from vfhe.arith import Polynomial
from vfhe.vfhe.computation import Division, KeySwitch, Multiplication, Tensor
from vfhe.vfhe.witness import Entry, Relation

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["entries", "relations"]


def _over(value: Any, primes: Iterable[int]) -> Any:
    """``value`` reduced into the ring of ``primes``, a quotient of its own."""
    ring = value.ring
    mask = 0
    for p in primes:
        mask |= 1 << ring.prime_indices[ring.primes.index(p)]
    return value.mod_reduce(ring.quotient_ring(mask=mask))


def _division_entries(
    division: Division, step: Any, index: int | None, suffix: str = ""
) -> list[Entry]:
    k = 0 if division.exact else len(division.primes)
    return [
        Entry(
            "w_quot" + suffix,
            "witness",
            division.quotient,
            step,
            index,
            bound=prod(division.quotient.ring.primes),
        ),
        Entry(
            "w_rmd" + suffix,
            "witness",
            division.remainder,
            step,
            index,
            bound=(k + 1) * division.modulus / 2,
        ),
    ]


def _tensor_entries(step: Multiplication | Tensor) -> list[Entry]:
    return [
        Entry("ct0", "instance", step.operands[0], step),
        Entry("ct1", "instance", step.operands[1], step),
        Entry("d0", "intermediate", step.constant, step),
        Entry("d1", "intermediate", step.linear, step),
        Entry("d2", "intermediate", step.quadratic, step),
    ]


def _switch_entries(switch: KeySwitch, step: Any) -> list[Entry]:
    found = [
        Entry("w_ks", "witness", digit, step, t, bound=switch.bound(t))
        for t, digit in enumerate(switch.digits)
    ]
    found.extend(
        Entry("evk", "instance", [k[i] for k in switch.key], step, i) for i in range(2)
    )
    for i, division in enumerate(switch.mod_down):
        found += _division_entries(division, step, i, "_P")
    found.append(Entry("ct'", "intermediate", switch.output, step))
    return found


def entries(step: Multiplication | Tensor | KeySwitch | Division) -> list[Entry]:
    if isinstance(step, Division):
        return [
            *_division_entries(step, step, None),
            Entry("ct''", "intermediate", step.quotient, step),
        ]
    if isinstance(step, Tensor):
        return _tensor_entries(step)
    if isinstance(step, KeySwitch):
        found = [Entry("d2", "intermediate", step.decomposed, step)]
        for name, slot in zip(("d1", "d0"), step.slots, strict=True):
            if slot is not None:
                found.append(Entry(name, "intermediate", slot, step))
        return found + _switch_entries(step, step)
    found = _tensor_entries(step) + _switch_entries(step.key_switch, step)
    for i, rescale in enumerate(step.rescales):
        found += _division_entries(rescale, step, i)
    if step.rescales:
        found.append(
            Entry("ct''", "intermediate", [r.quotient for r in step.rescales], step)
        )
    return found


def _divides(division: Division) -> bool:
    centered = division.remainder.get_polynomial(signed=True)
    reduced = _over(division.x, division.primes)
    return reduced == Polynomial(reduced.ring).from_bigint_array(centered)


def _lifts(division: Division) -> bool:
    return division.source.scaled_lift(division.x.ring) == division.x


def _premultiplies(step: Multiplication | Tensor) -> bool:
    (a0, b0), (a1, b1) = step.operands
    return (
        step.quadratic == a0 * a1
        and step.constant == b0 * b1
        and step.linear == a0 * b1 + b0 * a1
    )


def _recomposes(switch: KeySwitch, prime: int) -> bool:
    """Whether the digits taken at ``prime``, each times its power of two,
    sum to ``d2`` there: exactly, or within the rounding of the radix
    digits the decomposition leaves out."""
    taken = [t for t, p in enumerate(switch.primes) if p == prime]
    total = None
    for t in taken:
        term = _over(switch.digits[t], [prime]) * switch.power(t)
        total = term if total is None else total + term
    target = _over(switch.decomposed, [prime])
    digit = switch.digit_events[taken[0]]
    if not digit.dropped:
        return total == target
    rounding = 1 << (digit.log_base * digit.dropped - 1)
    difference = (target - total).get_polynomial(signed=True)
    return max(abs(c) for c in difference) <= rounding


def _switches(switch: KeySwitch) -> bool:
    for i in range(2):
        products = None
        for digit, key in zip(switch.digits, switch.key, strict=True):
            term = digit * key[i]
            products = term if products is None else products + term
        if switch.sums[i] + products != 0:
            return False
        switched = switch.mod_down[i].quotient if switch.mod_down else switch.sums[i]
        if switch.slots[i] is not None:
            switched = switched + switch.slots[i]
        if switched != switch.output[i]:
            return False
    return True


def _rescale_relations(division: Division) -> list[Relation]:
    found = [Relation("Rescale", lambda: _divides(division))]
    if division.lift != 1:
        found.append(Relation("the lift by P_l", lambda: _lifts(division)))
    return found


def _switch_relations(switch: KeySwitch) -> list[Relation]:
    found = []
    for prime in dict.fromkeys(switch.primes):
        if switch.digit_events[switch.primes.index(prime)].log_base:
            name = f"the digits at {prime} recompose to d2"
        else:
            name = f"w_ks[{switch.primes.index(prime)}] recomposes to d2"
        found.append(Relation(name, lambda p=prime: _recomposes(switch, p)))
    found.append(Relation("KeySwitch", lambda: _switches(switch)))
    found += [
        Relation("the division by the special primes", lambda d=d: _divides(d))
        for d in switch.mod_down
    ]
    return found


def relations(step: Multiplication | Tensor | KeySwitch | Division) -> list[Relation]:
    if isinstance(step, Division):
        return _rescale_relations(step)
    if isinstance(step, KeySwitch):
        return _switch_relations(step)
    found = [Relation("PreMult", lambda: _premultiplies(step))]
    if isinstance(step, Tensor):
        return found
    found += _switch_relations(step.key_switch)
    for rescale in step.rescales:
        found += _rescale_relations(rescale)
    return found
