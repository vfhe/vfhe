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
- witness: ``w_ks``, the digits, within ``p_t/2`` (vfhe's digits are
  centered, which is within the paper's ``< p_t``), and ``w_quot`` and
  ``w_rmd`` of each rescale, ``w_rmd`` within ``p_l/2``; ``w_quot`` is an
  element of the ring without ``p_l`` by construction, so no bound is
  attached to it.

Two conventions of vfhe differ from the paper and are stated here rather
than hidden: the products with the key are subtracted,
``c'_i = d_i - sum_t w_ks[t] * evk[i][t]``; and a hybrid key has special
primes, whose division is approximate: its quotient and remainder are
``w_quot_P`` and ``w_rmd_P``, with ``w_rmd_P`` within ``(k+1)P/2`` for
``k`` special primes.

The relations: ``PreMult`` (Eq. 18); each digit recomposes to ``d2``
through the CRT map, which is the digit agreeing with ``d2`` at its prime;
``KeySwitch`` (Eq. 19, with the two conventions); and for each division,
``x * z_l = (w_quot * p_l + w_rmd) * z_l``: in the quotient's ring the
identity holds by construction, so what is checked is ``x`` and ``w_rmd``
agreeing modulo the primes divided by.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vfhe.arith import Polynomial
from vfhe.vfhe.computation import Division, KeySwitch, Multiplication
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
        Entry("w_quot" + suffix, "witness", division.quotient, step, index),
        Entry(
            "w_rmd" + suffix,
            "witness",
            division.remainder,
            step,
            index,
            bound=(k + 1) * division.modulus / 2,
        ),
    ]


def entries(step: Multiplication | Division) -> list[Entry]:
    if isinstance(step, Division):
        return [
            *_division_entries(step, step, None),
            Entry("ct''", "intermediate", step.quotient, step),
        ]
    switch = step.key_switch
    found = [
        Entry("ct0", "instance", step.operands[0], step),
        Entry("ct1", "instance", step.operands[1], step),
        Entry("d0", "intermediate", step.constant, step),
        Entry("d1", "intermediate", step.linear, step),
        Entry("d2", "intermediate", step.quadratic, step),
    ]
    for t, (digit, p) in enumerate(zip(switch.digits, switch.primes, strict=True)):
        found.append(Entry("w_ks", "witness", digit, step, t, bound=p / 2))
    found.extend(
        Entry("evk", "instance", [k[i] for k in switch.key], step, i) for i in range(2)
    )
    for i, division in enumerate(switch.mod_down):
        found += _division_entries(division, step, i, "_P")
    found.append(Entry("ct'", "intermediate", switch.output, step))
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


def _premultiplies(step: Multiplication) -> bool:
    (a0, b0), (a1, b1) = step.operands
    return (
        step.quadratic == a0 * a1
        and step.constant == b0 * b1
        and step.linear == a0 * b1 + b0 * a1
    )


def _recomposes(switch: KeySwitch, t: int) -> bool:
    p = switch.primes[t]
    return _over(switch.digits[t], [p]) == _over(switch.quadratic, [p])


def _switches(switch: KeySwitch) -> bool:
    for i in range(2):
        products = None
        for digit, key in zip(switch.digits, switch.key, strict=True):
            term = digit * key[i]
            products = term if products is None else products + term
        if switch.sums[i] + products != 0:
            return False
        switched = switch.mod_down[i].quotient if switch.mod_down else switch.sums[i]
        if switched + switch.slots[i] != switch.output[i]:
            return False
    return True


def relations(step: Multiplication | Division) -> list[Relation]:
    if isinstance(step, Division):
        return [Relation("Rescale", lambda: _divides(step))]
    switch = step.key_switch
    found = [Relation("PreMult", lambda: _premultiplies(step))]
    found.extend(
        Relation(f"w_ks[{t}] recomposes to d2", lambda t=t: _recomposes(switch, t))
        for t in range(len(switch.digits))
    )
    found.append(Relation("KeySwitch", lambda: _switches(switch)))
    found += [
        Relation("the division by the special primes", lambda d=d: _divides(d))
        for d in switch.mod_down
    ]
    found += [Relation("Rescale", lambda r=r: _divides(r)) for r in step.rescales]
    return found
