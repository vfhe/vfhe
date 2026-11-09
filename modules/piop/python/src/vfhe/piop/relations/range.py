# SPDX-FileCopyrightText: 2026 Daniele Cozzo <daniele.cozzo@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The range relations: every value of an oracle lies in `[0, B)`.

`Relation_Range` is the claim on an `R_q` oracle, whose values are
polynomials with `N` coefficients in `Z_q`, every coefficient below `B`;
`Relation_ResidueRange` the same claim on an integer table held in residue
number system form [Gar59], one oracle per prime of `q`. The index is the
bound alone: the relations say what is claimed, not how it is proven.
A decomposition protocol discharges them by digit decomposition and
lookups; the conventions it relies on -- LSB-first table order, which variable block is which -- are set
out in `modules/piop/range.md`.
"""

from __future__ import annotations

from typing import cast

from vfhe.arith import crt
from vfhe.arith.mle import native_table, vector_table
from vfhe.piop.oracle import TableOracle
from vfhe.piop.relation import Relation, Statement, hypercube


class _Range(Relation):
    """What the range relations share: the index `(B,)`, the bound, a
    positive integer -- public, reusable, preprocessable data, which is what
    an index is for. How a protocol proves membership in `[0, B)` (a digit
    decomposition into lookups, an inner-product argument, a comparison) is
    the protocol's business, with its own parameters."""

    fields = ("oracles",)
    index_fields = ("bound",)

    def __init__(self, bound: int) -> None:
        if not isinstance(bound, int) or bound < 1:
            raise ValueError(f"bound B must be a positive integer, got {bound}")
        super().__init__(index={"bound": bound})

    @property
    def bound(self) -> int:
        """The bound `B`: the values lie in `[0, B)`."""
        return cast("int", self.index["bound"])


class Relation_Range(_Range):
    """Every `Z_q` coefficient of every hypercube evaluation of the `R_q`
    oracle lies in `[0, B)`, the coefficients read as their representatives
    in `[0, q)`: the range relation on packed polynomials of [CCCFGS25;
    CCCFGS26]. The index is the bound `B`."""

    name = "range"

    def check(self, statement: Statement) -> bool:
        (v,) = statement.oracles
        if isinstance(v, TableOracle) and native_table(v.polynomial):
            entries = v.polynomial.table
        else:
            entries = [v.query(b) for b in hypercube(v.variables)]
        # get_polynomial() is the unsigned representative in [0, q), so a
        # negative coefficient shows up here as a large one and is rejected.
        return all(max(e.get_polynomial()) < self.bound for e in entries)


def residue_primes(oracles) -> list[int]:
    """The primes of the residue oracles of a `Relation_ResidueRange`
    statement: the characteristic of each oracle's domain (`Oracle.domain`,
    a field with a `prime`), in the statement's order. Every oracle names
    its domain, the primes are distinct and the oracles have one number of
    variables."""
    primes = []
    for oracle in oracles:
        domain = oracle.domain
        prime = getattr(domain, "prime", None)
        if prime is None:
            raise TypeError(
                f"{oracle!r} does not name a field of known characteristic; a "
                "residue oracle is over a field, which its domain says"
            )
        primes.append(int(prime))
    if len(set(primes)) != len(primes):
        raise ValueError("the residue oracles are over fields of distinct primes")
    if len({o.num_vars for o in oracles}) != 1:
        raise ValueError("the residue oracles have one number of variables")
    return primes


class Relation_ResidueRange(_Range):
    """Every entry of an integer table lies in `[0, B)`, the table given by
    its residues modulo the primes `p_1, ..., p_U` of `q`: one oracle per
    prime, over a field of characteristic `p_i` (`F_(p_i)` or an extension
    of it), whose entries are the residues as elements of the prime
    subfield. The index is the bound `B`.

    This is `Relation_Range` on the *unpacked* table `u = unpack(v)` of
    [CCCFGS26] -- the multilinear extension of the coefficients of `v`'s
    values, one entry per `Z_q` coefficient -- written in residue number
    system form [Gar59], the CRT components of `Z_q`: the relation of a
    range proof whose coefficient oracles are committed per prime instead
    of as one `R_q` oracle. An integer below `q` is determined by its
    residues (the Chinese remainder theorem), so the claim is that the
    residue tuple at every hypercube point is that of an integer below `B`;
    an entry outside the prime subfield (`F_(p^d)`, `d > 1`) is the residue
    of no integer.
    """

    name = "residue_range"

    def check(self, statement: Statement) -> bool:
        oracles = statement.oracles
        primes = residue_primes(oracles)
        columns = []
        for oracle in oracles:
            if isinstance(oracle, TableOracle) and vector_table(oracle.polynomial):
                words = oracle.polynomial.table.to_uint64()
                if words is None:  # an entry outside the prime subfield
                    return False
                columns.append(list(words))
            else:
                try:
                    columns.append(
                        [int(oracle.query(b)) for b in hypercube(oracle.variables)]
                    )
                except ValueError:  # an element with no integer value
                    return False
        return all(
            crt(list(residues), primes) < self.bound
            for residues in zip(*columns, strict=True)
        )
