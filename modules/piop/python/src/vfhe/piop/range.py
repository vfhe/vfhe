# SPDX-FileCopyrightText: 2026 Daniele Cozzo <daniele.cozzo@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Range checks over packed polynomials: Figure 13 of [CCCFGS26].

`Relation_Range` is the claim that every `Z_q` coefficient of every hypercube
evaluation of an `R_q` oracle lies in `[0, B)` -- membership in the table
`T_B` of [CCCFGS26] §4. `RangeDecomposition` discharges it by the optimized
protocol of §I.5: the prover sends the base-`beta` digits of those
coefficients as an oracle over a large prime field `F_p`, and what remains is

- that the digits *recompose* to the coefficients (checked here, `kappa` times
  at small integer points, over `Z` and then modulo `q`), and
- that every digit is a member of `t_beta` (emitted as a `Relation_Lookup`).

The trick that makes it cheap is that the digits are indexed by the
*unpacked* polynomial `u = unpack(v)` -- one variable block for the `R_q`
coefficient index -- while `u` itself is never committed: by Lemma 3.1 an
evaluation of `unpack(v)` at integer coordinates is a local `O(N)`
computation from a single evaluation of `v`, which is what `uEval` below
does. So the verifier's one query to the large oracle is a query to `v`.

The conventions this file depends on -- LSB-first table order, which variable
block is which, and the exact form of the recomposition check -- are set out
in `modules/piop/range.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from vfhe.arith import MLE, MLE_Variable
from vfhe.arith.mle import native_table

from .lookup import Relation_Lookup
from .piop import (
    Protocol,
    Prover,
    Rejection,
    Relation,
    Relation_Eval,
    Statement,
    Verifier,
    _constant,
    _hypercube,
)


def eq_int(bits: list[int], point: list[int]) -> int:
    """`eq~(bits, point)` over the integers: `prod_i (b_i z_i + (1-b_i)(1-z_i))`.

    Exact, and generally *not* a small number: the coordinates of `point` are
    the small integer challenges of `Sigma`, so `1 - z_i` is negative and the
    product is bounded only by `sigma^len(point)`. Staying in `Z` is the
    point -- the recomposition check is an identity over the integers that is
    only reduced at the very end.
    """
    total = 1
    for bit, z in zip(bits, point, strict=True):
        total *= z if bit else 1 - z
    return total


def base_digits(value: int, width: int, count: int) -> list[int]:
    """The `count` base-`2^width` digits of `value`, least significant first.

    A `value` that does not fit in `count * width` bits is silently truncated.
    That is deliberate: it is exactly what a prover holding an out-of-range
    coefficient would have to submit, and the recomposition check of
    `RangeDecomposition.verify` is what catches it.
    """
    mask = (1 << width) - 1
    return [(value >> (width * j)) & mask for j in range(count)]


def uEval(element: Any, point: list[int]) -> int:
    """Lemma 3.1's `uEval(a, z) = sum_k a[k] * eq~(k, z)`, over the integers.

    `element` is a ring element -- the answer to one query to `v` -- and
    `point` the `nu` integer coordinates of the `Z` block. The result is the
    evaluation of `unpack(v)` at that point, which is how the verifier queries
    the unpacked oracle without it ever being committed. Kept exact so the
    caller can compare it with the integer the field side produces through
    `balanced`.
    """
    coefficients = element.get_polynomial()
    nu = len(point)
    if len(coefficients) != 1 << nu:
        raise ValueError(
            f"a point of {nu} coordinates needs {1 << nu} coefficients, "
            f"got {len(coefficients)}"
        )
    return sum(
        coefficient * eq_int([(k >> t) & 1 for t in range(nu)], point)
        for k, coefficient in enumerate(coefficients)
    )


def balanced(value: Any, prime: int) -> int:
    """`iota`: the representative of `value` in `I_p = {-|p/2|, ..., |p/2|}`.

    `int()` on a field element is its canonical representative in `[0, p)`;
    this is the inverse of the restriction of `pi_p` to `I_p`, which is what
    lets `Z_p` emulate signed integer arithmetic as long as nothing overflows
    (Proposition I.1).
    """
    residue = int(value)
    return residue - prime if residue > prime // 2 else residue


@dataclass(frozen=True)
class _Shape:
    """The dimensions of one range claim, derived identically by both parties."""

    ell: int  # variables of v
    nu: int  # log2 N, the R_q coefficient block
    gamma: int  # log2 c, the digit block
    beta: int  # the digit alphabet, B ** (1/c)
    chunks: int  # c, the number of digits per coefficient
    ring: Any
    variables: list  # the ell_star variables of mu', in X | Z | Z' order

    @property
    def ell_star(self) -> int:
        return self.ell + self.nu + self.gamma


class Relation_Range(Relation):
    """Every `Z_q` coefficient of every hypercube evaluation of the oracle
    lies in `[0, B)`: the table `T_B` of [CCCFGS26] §4.

    The index is `(B, c)`: the bound, and the number of base-`beta` digits the
    protocol splits each coefficient into (`beta = B ** (1/c)`). Both are
    public, reusable, preprocessable data, which is what an index is for.
    `c` is a pure efficiency knob -- it trades the digit alphabet against the
    size of the digit oracle -- and does not change the language.
    """

    name = "range"
    fields = ("oracles",)

    def __init__(self, bound: int, chunks: int) -> None:
        if bound <= 1 or bound & (bound - 1):
            raise ValueError(f"bound B must be a power of two > 1, got {bound}")
        if chunks <= 0 or chunks & (chunks - 1):
            raise ValueError(f"chunks c must be a power of two, got {chunks}")
        if (bound.bit_length() - 1) % chunks:
            raise ValueError(
                f"c must divide log2(B): log2({bound}) = {bound.bit_length() - 1} "
                f"is not a multiple of {chunks}"
            )
        super().__init__(index=(bound, chunks))

    # `Relation.index` is deliberately untyped in the base class -- it holds
    # whatever a relation needs -- so a subclass narrows it where it reads it,
    # the way `Relation_Circuit.circuit` does (`circuit.py`). The constructor
    # above is what makes these casts honest.

    @property
    def bound(self) -> int:
        """The coefficient bound `B`."""
        return cast("tuple[int, int]", self.index)[0]

    @property
    def chunks(self) -> int:
        """The number of base-`beta` digits per coefficient, `c`."""
        return cast("tuple[int, int]", self.index)[1]

    @property
    def beta(self) -> int:
        """The digit alphabet `B ** (1/c)`."""
        return 1 << ((self.bound.bit_length() - 1) // self.chunks)

    def check(self, statement: Statement) -> bool:
        (v,) = statement.oracles
        entries = (
            v.table
            if native_table(v)
            else [
                _constant(v.evaluate(b, in_place=False))
                for b in _hypercube(v.variables)
            ]
        )
        # get_polynomial() is the unsigned representative in [0, q), so a
        # negative coefficient shows up here as a large one and is rejected.
        return all(max(e.get_polynomial()) < self.bound for e in entries)


class RangeDecomposition(Protocol):
    """`Relation_Range -> Relation_Lookup x Relation_Eval^(2 kappa)`
    (Figure 13 of [CCCFGS26]).

    Round 1: the prover sends `mu'`, the base-`beta` digits of the unpacked
    coefficients, reduced into `F_p`. The verifier replies with `kappa` points
    `r_eta` drawn from the small integer alphabet `Sigma = {0, ..., sigma-1}`
    -- raw coins (`challenge_bits`), not domain elements, because the same
    integers have to be read both in `Z_q` (to query `v`) and in `F_p` (to
    query `mu'`).

    Round 2: the prover sends the partial evaluations
    `psi_eta = mu'(r'_eta, .)` -- binding the first `m` variables, an
    efficiency knob -- and the `kappa` ring values `v(x_eta)`. The verifier
    replies with one spot-check point `s'`.

    It then checks, per repetition, that the digits recompose:

        sum_{delta, j} iota(psi_eta[delta, j]) * eq~(delta; r_eta^{>m})
                                              * beta^{to-int(j)}
          ==  uEval(v(x_eta), z_eta)   (mod q)

    with both sides computed as exact integers first. `iota` is sound there
    because the left side is bounded by `2^m beta sigma^m < p/2`
    (equation 38), so nothing overflows `F_p`.

    What is left over becomes statements: that each `psi_eta` really is a
    partial evaluation of `mu'` (one `Relation_Eval` on `mu'` per repetition,
    at `(r'_eta, s')`), that each `v(x_eta)` really is an evaluation of `v`
    (one `Relation_Eval` on `v` per repetition), and that every digit is in
    `t_beta` (one `Relation_Lookup`).
    """

    name = "range"
    reduces_from: type[Relation] = Relation_Range
    reduces_to: tuple[type[Relation], ...] = (Relation_Lookup, Relation_Eval)

    def __init__(self, field: Any, m: int, alphabet: int, kappa: int) -> None:
        if alphabet <= 1 or alphabet & (alphabet - 1):
            raise ValueError(
                f"the alphabet size sigma must be a power of two > 1, got {alphabet}"
            )
        if kappa < 1:
            raise ValueError(f"kappa must be at least 1, got {kappa}")
        if m < 0:
            raise ValueError(f"m must be non-negative, got {m}")
        self.field = field
        self.m = m
        self.alphabet = alphabet
        self.kappa = kappa

    # --- shape and parameter conditions -----------------------------------

    def _shape(self, statement: Statement) -> _Shape:
        """The claim's dimensions, plus the variables of `mu'`.

        Called once per party per run: the `MLE_Variable`s are compared by
        identity, so each party threads its own through everything it builds.
        """
        (v,) = statement.oracles
        relation = statement.relation
        if not isinstance(relation, Relation_Range):
            raise TypeError("RangeDecomposition discharges Relation_Range")
        ring = v.ring
        if ring is None:
            raise TypeError("a range claim is about an oracle over a Ring")
        ell = v.num_vars
        nu = ring.N.bit_length() - 1
        gamma = relation.chunks.bit_length() - 1
        shape = _Shape(
            ell=ell,
            nu=nu,
            gamma=gamma,
            beta=relation.beta,
            chunks=relation.chunks,
            ring=ring,
            variables=[MLE_Variable(f"mu{i}") for i in range(ell + nu + gamma)],
        )
        self._validate(shape)
        return shape

    def _validate(self, shape: _Shape) -> None:
        """The side conditions of §I.5, checked before anything is proven.

        They mix constructor parameters with claim dimensions, so this is the
        earliest place they can be tested -- and testing them is not optional:
        a prime too small for equation (38) does not fail visibly, it silently
        wraps `iota` and makes the recomposition check meaningless.
        """
        prime = self.field.prime
        if self.m > shape.ell:
            raise ValueError(
                f"m must not exceed the number of variables of v: "
                f"m = {self.m}, ell = {shape.ell}"
            )
        bound = 2 * shape.beta * (2 * self.alphabet) ** self.m
        if prime <= bound:
            raise ValueError(
                f"p must exceed 2*beta*(2*sigma)^m = {bound} (equation 38); "
                f"p has {prime.bit_length()} bits"
            )
        if prime <= 1 << shape.ell_star:
            raise ValueError(
                f"p must exceed 2^(ell+nu+gamma) = {1 << shape.ell_star}; "
                f"p has {prime.bit_length()} bits"
            )
        if self.alphabet > min(shape.ring.primes):
            raise ValueError(
                f"sigma = {self.alphabet} must not exceed the smallest RNS prime "
                f"{min(shape.ring.primes)}, or the challenges leave the "
                "exceptional set of Z_q"
            )

    def soundness_error(self, statement: Statement, domain: Any = None) -> float:
        """`max{ ((ell+nu)/sigma)^kappa, (ell*-m)/p }` (Theorem I.6).

        The theorem's third term is the error of the `Relation_Lookup` this
        reduces to, which that relation's own protocol accounts for.
        """
        shape = self._shape(statement)
        prime = self.field.prime
        return max(
            ((shape.ell + shape.nu) / self.alphabet) ** self.kappa,
            (shape.ell_star - self.m) / prime,
        )

    # --- the two halves ----------------------------------------------------

    async def prove(
        self, prover: Prover, statements: list[Statement]
    ) -> list[Statement]:
        (statement,) = statements
        iop = prover.iop
        if iop is None:
            raise RuntimeError("this party is not bound to an IOP")
        (v,) = statement.oracles
        shape = self._shape(statement)
        label = f"{self.name}{statement.path}"

        mu = MLE(
            field=self.field,
            variables=shape.variables,
            evaluations=[self.field(d) for d in self._digit_table(v, shape)],
        )
        iop.transcript.write(f"{label}/mu", tuple(mu.table))

        points = [self._draw_point(iop, label, shape, eta) for eta in range(self.kappa)]

        psis, values = [], []
        for eta, r in enumerate(points):
            psi = mu.evaluate(self._bind_prefix(shape, r), in_place=False)
            iop.transcript.write(f"{label}/psi{eta}", tuple(psi.table))
            psis.append(psi)
            w = _constant(
                v.evaluate(
                    {var: r[i] for i, var in enumerate(v.variables)}, in_place=False
                )
            )
            iop.transcript.write(f"{label}/w{eta}", w)
            values.append(w)

        spot = self._draw_spot(iop, label, shape)
        return self._outputs(statement, shape, mu, points, psis, values, spot)

    async def verify(
        self, verifier: Verifier, statements: list[Statement]
    ) -> list[Statement]:
        (statement,) = statements
        iop = verifier.iop
        if iop is None:
            raise RuntimeError("this party is not bound to an IOP")
        shape = self._shape(statement)
        label = f"{self.name}{statement.path}"

        table = await iop.transcript.read(f"{label}/mu")
        self._expect(len(table), 1 << shape.ell_star, f"{label}/mu")
        mu = MLE(field=self.field, variables=shape.variables, evaluations=list(table))

        points = [self._draw_point(iop, label, shape, eta) for eta in range(self.kappa)]

        free = shape.variables[self.m :]
        psis, values = [], []
        for eta in range(self.kappa):
            entries = await iop.transcript.read(f"{label}/psi{eta}")
            self._expect(len(entries), 1 << len(free), f"{label}/psi{eta}")
            psis.append(
                MLE(field=self.field, variables=free, evaluations=list(entries))
            )
            values.append(await iop.transcript.read(f"{label}/w{eta}"))

        spot = self._draw_spot(iop, label, shape)

        for eta, (r, psi, w) in enumerate(zip(points, psis, values, strict=True)):
            self._check_recomposition(shape, r, psi, w, label, eta)

        return self._outputs(statement, shape, mu, points, psis, values, spot)

    # --- the pieces both halves share --------------------------------------

    @staticmethod
    def _expect(got: int, want: int, label: str) -> None:
        if got != want:
            raise Rejection(f"{label}: expected {want} entries, got {got}")

    def _digit_table(self, v: MLE, shape: _Shape) -> list[int]:
        """The integer digit table of `mu`, LSB-first over `X | Z | Z'`:
        `index = i + 2^ell * k + 2^(ell+nu) * j` holds the `j`-th base-`beta`
        digit of the `k`-th coefficient of `v(i)` (Lemma 4.1, proof in §C.1).

        Laying `Z'` out as the *slowest* block is what makes the digit oracle
        line up index-for-index with the grand-product leaves the lookup
        argument builds from it.
        """
        if not native_table(v):
            raise TypeError(
                "the prover needs v as a dense ring table in the evaluation basis"
            )
        width = shape.beta.bit_length() - 1
        table = [0] * (1 << shape.ell_star)
        stride_k = 1 << shape.ell
        stride_j = 1 << (shape.ell + shape.nu)
        for i, element in enumerate(v.table):
            for k, coefficient in enumerate(element.get_polynomial()):
                digits = base_digits(coefficient, width, shape.chunks)
                for j, digit in enumerate(digits):
                    table[i + stride_k * k + stride_j * j] = digit
        return table

    def _bind_prefix(self, shape: _Shape, r: list[int]) -> dict:
        """`{X_1: r_1, ..., X_m: r_m}` as field elements, in variable order.

        In order because `MLE.evaluate` binds the entries of the dict one at a
        time: taking them front to back keeps every bind at position 0, which
        is the layout a field-backed table folds fastest in.
        """
        return {shape.variables[i]: self.field(r[i]) for i in range(self.m)}

    def _draw_point(self, iop: Any, label: str, shape: _Shape, eta: int) -> list[int]:
        """One `r_eta` in `Sigma^(ell+nu)`, from raw coins.

        Not `challenge()`: these are integers, read modulo `q` to query `v`
        and modulo `p` to query `mu'`, so they belong to neither domain's
        exceptional set on its own. `sigma` is a power of two, so slicing the
        coins is unbiased.
        """
        width = self.alphabet.bit_length() - 1
        count = shape.ell + shape.nu
        raw = iop.verifier.challenge_bits(f"{label}/r{eta}", width * count)
        coins = int.from_bytes(raw, "little")
        mask = self.alphabet - 1
        return [(coins >> (width * i)) & mask for i in range(count)]

    def _draw_spot(self, iop: Any, label: str, shape: _Shape) -> list:
        """`s'`, one exceptional-set element of `F_p` per free variable."""
        return [
            iop.verifier.challenge(f"{label}/s{i}")
            for i in range(shape.ell_star - self.m)
        ]

    def _check_recomposition(
        self,
        shape: _Shape,
        r: list[int],
        psi: MLE,
        w: Any,
        label: str,
        eta: int,
    ) -> None:
        """The heart of Figure 13: that the digits rebuild the coefficients.

        Both sides are accumulated as exact Python integers -- the left one
        through `iota`, which is what lets a claim about `F_p` values say
        something about `Z` -- and only then reduced mod `q`. Reducing mod `q`
        is reducing mod every `p_l`, which is the form the paper states.
        """
        prime = self.field.prime
        width = shape.ell + shape.nu - self.m
        weights = r[self.m :]
        total = 0
        for index, value in enumerate(psi.table):
            delta = [(index >> t) & 1 for t in range(width)]
            digit = index >> width  # to-int of the Z' block, LSB-first
            total += balanced(value, prime) * eq_int(delta, weights) * shape.beta**digit
        if (total - uEval(w, r[shape.ell :])) % shape.ring.q_l:
            raise Rejection(
                f"{label}: repetition {eta} does not recompose -- the digits of "
                "mu' do not rebuild the coefficients of v at this point"
            )

    def _outputs(
        self,
        statement: Statement,
        shape: _Shape,
        mu: MLE,
        points: list[list[int]],
        psis: list[MLE],
        values: list,
        spot: list,
    ) -> list[Statement]:
        """The bundle this step reduces to, in an order both halves agree on.

        Order matters beyond tidiness: child paths come from a counter on the
        parent, and those paths namespace every transcript label downstream,
        so the two parties must emit the same statements in the same sequence.
        """
        (v,) = statement.oracles
        free = shape.variables[self.m :]
        out = []
        for r, psi in zip(points, psis, strict=True):
            point = self._bind_prefix(shape, r)
            point.update(zip(free, spot, strict=True))
            value = _constant(
                psi.evaluate(
                    dict(zip(psi.variables, spot, strict=True)), in_place=False
                )
            )
            out.append(
                self.reduce(
                    [statement], Relation_Eval, oracles=[mu], point=point, value=value
                )
            )
        for r, w in zip(points, values, strict=True):
            point = {var: r[i] for i, var in enumerate(v.variables)}
            out.append(
                self.reduce(
                    [statement], Relation_Eval, oracles=[v], point=point, value=w
                )
            )
        out.append(self.reduce([statement], Relation_Lookup(shape.beta), oracles=[mu]))
        return out
