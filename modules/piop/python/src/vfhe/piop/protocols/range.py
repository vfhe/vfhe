# SPDX-FileCopyrightText: 2026 Daniele Cozzo <daniele.cozzo@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Range checks over packed polynomials, by digit decomposition and lookups.

A range check -- every value of an oracle lies in `[0, B)` -- is proven by
splitting each value into `c` base-`beta` digits and looking each digit up
in the small table `t_beta = {0, ..., beta - 1}`: bit decomposition in
Bulletproofs [BBBPWM18], digit tables in plookup [GW20], Lasso's
decomposable tables with `c` chunks [STW24]. [CCCFGS25] carried it to
`R_q`, whose values are polynomials with `N` coefficients, and [CCCFGS26]
made it cheap with *packing*: the digits are indexed by the *unpacked*
polynomial `u = unpack(v)`, the multilinear extension of the coefficients
of `v`'s values, with `log2 N` more variables than `v`, so the digit oracle
is a table of `2^(ell + nu + gamma)` field entries and no oracle larger
than `v` is sent.

`RangeDecomposition` discharges `Relation_Range` by [CCCFGS26]'s optimized
range PIOP `Pi*_range`, the variant of its appendix that takes the
auxiliary ring to be a large prime field `F_p`. It reads the relation as a
lookup into `T_B`, the table of all values with coefficients below `B`
[CCCFGS25; CCCFGS26], which the decomposition into `c` digits turns into
lookups in `t_beta`; `c` is the protocol's parameter (`chunks`), as in
Lasso, and `B` must be a power of two. The prover sends the digits as an
oracle over `F_p`, and what remains is

- that the digits *recompose* to the coefficients: an identity between the
  unpacked polynomial and the digit polynomial over the integers, tested
  `kappa` times at random points of a small set of integers `Sigma` that is
  an exceptional set of `Z_q` and of `F_p` at once (the Schwartz-Zippel
  lemma over rings [BCS21; CCCFGS25]); the field side is read back into
  `Z` through the centered representative, faithful while nothing
  overflows, and the two sides are compared modulo `q`;
- that every digit is a member of `t_beta` (emitted as a `Relation_Lookup`),
  and, when the most significant digit is the shorter one, that every such
  digit is a member of `t_top_beta` too (a second `Relation_Lookup`).

`u` itself is never committed: its value at integer coordinates is a local
`O(N)` computation from the one value `v(x)` ([CCCFGS26]'s
unpacked-evaluation lemma; `unpacked_evaluation` below), so the verifier's
one query to the large oracle is a query to `v`.

`ResidueRangeDecomposition` discharges `Relation_ResidueRange` by the same
protocol with `u` given as an oracle in its own right, in residue number
system form [Gar59], one field oracle per prime of `q`: the verifier queries
it directly and checks the recomposition modulo each prime, as an equality
in that oracle's field; the queries become evaluation claims on the residue
oracles.

The conventions this file depends on -- LSB-first table order, which variable
block is which, and the exact form of the recomposition check -- are set out
in `modules/piop/range.md`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vfhe.arith import (
    MLE,
    ExtensionField,
    FieldVector,
    MLE_Variable,
    crt,
    crt_lift_mod_2_128,
)
from vfhe.arith.mle import native_table, vector_table
from vfhe.piop.evaluation import Relation_Eval
from vfhe.piop.messages import OracleMessages
from vfhe.piop.protocol import Protocol, Rejection
from vfhe.piop.relations import Relation_Lookup, Relation_Range, Relation_ResidueRange
from vfhe.piop.relations.range import residue_primes
from vfhe.piop.virtual import VirtualOracle, resolve_table

if TYPE_CHECKING:
    from vfhe.piop.oracle import Oracle
    from vfhe.piop.party import Prover, Verifier
    from vfhe.piop.relation import Relation, Statement


def eq_int(bits: list[int], point: list[int]) -> int:
    """`eq~(bits, point)` over the integers: `prod_i (b_i z_i + (1-b_i)(1-z_i))`,
    the multilinear Lagrange basis polynomial of `bits` [Tha22, Lemma 3.6].

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


def unpacked_evaluation(element: Any, point: list[int]) -> int:
    """`sum_k a[k] * eq~(k, z)` over the integers, for `a = element`: the
    value `u(x, z)` of the unpacked polynomial `u = unpack(v)` from the one
    value `a = v(x)` ([CCCFGS26]'s unpacked-evaluation lemma: `u(x, .)` is
    the multilinear extension of the coefficients of `v(x)`; its `N` basis
    values come from the `O(N)` algorithm of [VSBW13]).

    `element` is a ring element -- the answer to one query to `v` -- and
    `point` the `nu` integer coordinates of the `Z` block. This is how the
    verifier queries the unpacked oracle without it ever being committed.
    This is the exact value; the verifier's check only needs it modulo `q`
    and computes that natively (`RangeDecomposition._unpacked_evaluation_mod_q`).
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


def centered(value: Any, prime: int) -> int:
    """The centered representative of `value`: its representative in
    `I_p = {-|p/2|, ..., |p/2|}` (the least absolute residue of number
    theory; the `iota` of [CCCFGS26]).

    `int()` on a field element is its canonical representative in `[0, p)`;
    this is the inverse of reduction modulo `p` restricted to `I_p`, which
    is what lets `Z_p` emulate signed integer arithmetic as long as no value
    reaches `p/2` in absolute value (the lift proposition of [CCCFGS26]'s
    appendix: the usual no-overflow argument).
    """
    residue = int(value)
    return residue - prime if residue > prime // 2 else residue


def digit_alphabets(bound: int, chunks: int) -> tuple[int, int]:
    """The digit alphabets of the decomposition of values below `bound` into
    `chunks` base-`beta` digits: `(beta, top_beta)` with
    `beta = 2^ceil(log2(B) / c)` -- `B^(1/c)` when `c` divides `log2(B)` --
    and `top_beta = B / beta^(c-1)`, the alphabet of the most significant
    digit: `beta` when `c` divides `log2(B)`, smaller otherwise.

    `c` is an efficiency parameter, the chunk count of Lasso's decomposable
    tables [STW24]: it trades the digit alphabet against the size of the
    digit oracle. `B` and `c` must be powers of two, `B > 1`, and the most
    significant digit must keep at least one bit.
    """
    if bound <= 1 or bound & (bound - 1):
        raise ValueError(f"the decomposition needs a bound B = 2^b > 1, got {bound}")
    if chunks <= 0 or chunks & (chunks - 1):
        raise ValueError(f"chunks c must be a power of two, got {chunks}")
    bits = bound.bit_length() - 1
    width = -(-bits // chunks)
    if bits - (chunks - 1) * width < 1:
        raise ValueError(
            f"c = {chunks} digits of {width} bits leave no bit for the most "
            f"significant digit of log2(B) = {bits} bits"
        )
    return 1 << width, 1 << (bits - (chunks - 1) * width)


@dataclass(frozen=True)
class _Shape:
    """The dimensions of one range claim, derived identically by both parties."""

    ell: int  # log2 of the table length: the variables of v or of the residue oracles
    nu: int  # log2 N, the coefficient block of the unpacked table; 0 for residue oracles
    gamma: int  # log2 c, the digit block
    beta: int  # the digit alphabet, 2^ceil(log2(B) / c)
    top_beta: int  # the most significant digit's alphabet, B / beta^(c-1)
    chunks: int  # c, the number of digits per coefficient
    ring: Any  # R_q; None for residue oracles
    primes: tuple[int, ...]  # the primes of q
    variables: list  # the ell_star variables of mu', in X | Z | Z' order

    @property
    def block(self) -> int:
        """The variables of the unpacked table, `ell + nu`: the `X | Z` block
        of `mu'`, and what a point of `Sigma` has coordinates for."""
        return self.ell + self.nu

    @property
    def ell_star(self) -> int:
        return self.block + self.gamma


class RangeDecomposition(Protocol):
    """`Relation_Range -> Relation_Lookup x Relation_Eval^(2 kappa)`
    ([CCCFGS26]'s optimized range PIOP `Pi*_range`).

    Round 1: the prover sends `mu'`, the base-`beta` digits of the unpacked
    coefficients, reduced into `F_p`. The verifier replies with `kappa` points
    `r_eta` drawn from the challenge set `Sigma = {0, ..., sigma-1}`, a set of
    integers exceptional in `Z_q` and in `F_p` at once -- raw coins
    (`challenge_bits`), not domain elements, because the same integers have
    to be read both in `Z_q` (to query `v`) and in `F_p` (to query `mu'`).

    Round 2: the prover sends the partial evaluations
    `psi_eta = mu'(r'_eta, .)` -- binding the first `m` variables, an
    efficiency parameter -- and the `kappa` ring values `v(x_eta)`. The verifier
    replies with one spot-check point `s'`.

    It then checks, per repetition, that the digits recompose:

        sum_{delta, j} iota(psi_eta[delta, j]) * eq~(delta; r_eta^{>m})
                                              * beta^{to-int(j)}
          ==  u(x_eta, z_eta)   (mod q)

    with `u(x_eta, z_eta)` computed from `v(x_eta)` by `unpacked_evaluation`
    and both sides computed as exact integers first. `iota` is sound there
    because every entry of `psi_eta`, as an integer, is bounded by
    `2^m beta sigma^m < p/2` (the parameter condition `p > 2 beta (2 sigma)^m`),
    so nothing overflows `F_p`.

    What is left over becomes statements: that each `psi_eta` really is a
    partial evaluation of `mu'` (one `Relation_Eval` on `mu'` per repetition,
    at `(r'_eta, s')`), that each `v(x_eta)` really is an evaluation of `v`
    (one `Relation_Eval` on `v` per repetition), and that every digit is in
    `t_beta` (one `Relation_Lookup`).

    When `c` does not divide `log2(B)`, the digits are `c - 1` of
    `log2(beta)` bits and a most significant one of fewer
    (`digit_alphabets`), and a
    `Relation_Lookup(top_beta)` on the restriction of `mu'` to the last
    digit (`Z'` all ones) follows the first: with it, the integer the
    digits rebuild is below `beta^(c-1) * top_beta = B`, not only below
    `beta^c`.

    The steps that read the coefficient oracle -- the digits, the queries
    at the integer points, the check against them and the claims they
    leave -- are methods a subclass replaces for another form of that
    oracle (`ResidueRangeDecomposition`); the rounds, the messages of `mu'`
    and `psi`, the lookups and the soundness bound are shared.

    Parameters: `field`, the prime field `F_p` of the digits; `m`, the number
    of variables of `mu'` bound in each `psi_eta`, an efficiency parameter
    (`0 <= m <= ell + nu`; a larger `m` means a smaller `psi` and a larger
    `p`); `alphabet`, the size `sigma` of the challenge set `Sigma`, a power
    of two at most the smallest prime of `q`; `kappa`, the number of
    repetitions (soundness `((ell + nu) / sigma)^kappa`); `ring`, the `R_q`
    of the claims; `chunks`, the number `c` of digits per coefficient
    (`digit_alphabets`); `messages`, how `mu'` travels (`OracleMessages`).
    """

    name = "range"
    reduces_from: type[Relation] = Relation_Range
    reduces_to: tuple[type[Relation], ...] = (Relation_Lookup, Relation_Eval)

    def __init__(
        self,
        field: Any,
        m: int,
        alphabet: int,
        kappa: int,
        ring: Any,
        chunks: int,
        messages: OracleMessages | None = None,
    ) -> None:
        if chunks <= 0 or chunks & (chunks - 1):
            raise ValueError(f"chunks c must be a power of two, got {chunks}")
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
        # c, the number of base-beta digits each coefficient splits into.
        self.chunks = chunks
        # The R_q the claims are about: N, the primes and q are read from it.
        self.ring = ring
        # How mu' is sent: its entries (the ideal model) by default.
        self.messages = OracleMessages() if messages is None else messages
        # F_p for each RNS prime p the verifier has evaluated at.
        self._prime_fields: dict[int, ExtensionField] = {}

    def parameters(self) -> tuple:
        return (
            self.field,
            self.m,
            self.alphabet,
            self.kappa,
            self.ring,
            self.chunks,
            self.messages,
        )

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
        ring = self.ring
        ell = v.num_vars
        nu = ring.N.bit_length() - 1
        gamma = self.chunks.bit_length() - 1
        beta, top_beta = digit_alphabets(relation.bound, self.chunks)
        shape = _Shape(
            ell=ell,
            nu=nu,
            gamma=gamma,
            beta=beta,
            top_beta=top_beta,
            chunks=self.chunks,
            ring=ring,
            primes=tuple(ring.primes),
            variables=[MLE_Variable(f"mu{i}") for i in range(ell + nu + gamma)],
        )
        self._validate(shape)
        return shape

    def _validate(self, shape: _Shape) -> None:
        """The side conditions of the protocol, checked before anything is
        proven.

        They mix constructor parameters with claim dimensions, so this is the
        earliest place they can be tested -- and testing them is not optional:
        a prime too small for `p > 2 beta (2 sigma)^m` does not fail visibly,
        it silently wraps `iota` and makes the recomposition check meaningless.
        """
        prime = self.field.prime
        if self.m > shape.block:
            raise ValueError(
                f"m must not exceed the variables of the X|Z block: "
                f"m = {self.m}, ell + nu = {shape.block}"
            )
        bound = 2 * shape.beta * (2 * self.alphabet) ** self.m
        if prime <= bound:
            raise ValueError(
                f"p must exceed 2*beta*(2*sigma)^m = {bound}; "
                f"p has {prime.bit_length()} bits"
            )
        if prime <= 1 << shape.ell_star:
            raise ValueError(
                f"p must exceed 2^(ell+nu+gamma) = {1 << shape.ell_star}; "
                f"p has {prime.bit_length()} bits"
            )
        if self.alphabet > min(shape.primes):
            raise ValueError(
                f"sigma = {self.alphabet} must not exceed the smallest RNS prime "
                f"{min(shape.primes)}, or the challenges leave the "
                "exceptional set of Z_q"
            )
        bound = shape.beta ** (shape.chunks - 1) * shape.top_beta
        if bound > math.prod(shape.primes):
            raise ValueError(
                f"the bound B = 2^{bound.bit_length() - 1} must not exceed the "
                f"modulus q, of {math.prod(shape.primes).bit_length()} bits"
            )

    def soundness_error(self, statement: Statement, domain: Any = None) -> float:
        """`max{ ((ell+nu)/sigma)^kappa, (ell*-m)/p }` ([CCCFGS26]'s theorem on
        `Pi*_range`).

        The theorem's third term is the error of the `Relation_Lookup` this
        reduces to, which that relation's own protocol accounts for.
        """
        shape = self._shape(statement)
        prime = self.field.prime
        return max(
            (shape.block / self.alphabet) ** self.kappa,
            (shape.ell_star - self.m) / prime,
        )

    # --- the two halves ----------------------------------------------------

    async def prove(
        self, prover: Prover, statements: list[Statement]
    ) -> list[Statement]:
        (statement,) = statements
        iop = prover.iop
        shape = self._shape(statement)
        label = f"{self.name}{statement.path}"

        mu_table = self._digit_mle(statement, prover, shape)
        mu = self.messages.send(prover, f"{label}/mu", mu_table)

        points = [
            await self._draw_point(iop, label, shape, eta) for eta in range(self.kappa)
        ]

        psis, values = [], []
        for eta, r in enumerate(points):
            psi = mu_table.evaluate(self._bind_prefix(shape, r), in_place=False)
            iop.transcript.write(f"{label}/psi{eta}", tuple(psi.table))
            psis.append(psi)
            w = self._query(statement, prover, shape, r)
            iop.transcript.write(f"{label}/w{eta}", w)
            values.append(w)

        spot = await self._draw_spot(iop, label, shape)
        return self._outputs(statement, shape, mu, points, psis, values, spot)

    async def verify(
        self, verifier: Verifier, statements: list[Statement]
    ) -> list[Statement]:
        (statement,) = statements
        iop = verifier.iop
        shape = self._shape(statement)
        label = f"{self.name}{statement.path}"

        mu = await self.messages.receive(
            verifier, f"{label}/mu", shape.variables, self.field
        )

        points = [
            await self._draw_point(iop, label, shape, eta) for eta in range(self.kappa)
        ]

        free = shape.variables[self.m :]
        psis, values = [], []
        for eta in range(self.kappa):
            entries = await iop.transcript.read(f"{label}/psi{eta}")
            self._expect(len(entries), 1 << len(free), f"{label}/psi{eta}")
            psis.append(
                MLE(field=self.field, variables=free, evaluations=list(entries))
            )
            values.append(await iop.transcript.read(f"{label}/w{eta}"))

        spot = await self._draw_spot(iop, label, shape)

        for eta, (r, psi, w) in enumerate(zip(points, psis, values, strict=True)):
            self._check_recomposition(statement, shape, r, psi, w, label, eta)

        return self._outputs(statement, shape, mu, points, psis, values, spot)

    # --- the coefficient oracle: what a subclass replaces ------------------

    def _digit_mle(self, statement: Statement, prover: Prover, shape: _Shape) -> MLE:
        """The prover's `mu'`: the digit table of the coefficients of `v`,
        over `F_p`."""
        (v,) = statement.oracles
        v_table = resolve_table(v, prover.witnesses)
        digits = self._digit_vector(v_table, shape)
        if digits is None:
            digits = [self.field(d) for d in self._digit_table(v_table, shape)]
        return MLE(field=self.field, variables=shape.variables, evaluations=digits)

    def _query(
        self, statement: Statement, prover: Prover, shape: _Shape, r: list[int]
    ) -> Any:
        """The prover's answer to the query at `r`: `v(x_eta)`, the ring
        element at the `X` block of `r`, read as integers."""
        (v,) = statement.oracles
        v_table = resolve_table(v, prover.witnesses)
        return v_table.evaluate(
            {var: r[i] for i, var in enumerate(v.variables)}, in_place=False
        ).constant()

    def _check_value(
        self,
        statement: Statement,
        shape: _Shape,
        total: int,
        w: Any,
        r: list[int],
        label: str,
        eta: int,
    ) -> None:
        """The recomposed integer `total` against the answer `w`:
        `total == unpacked_evaluation(w, z_eta) (mod q)`, `q` the product of
        the primes."""
        if (
            total - self._unpacked_evaluation_mod_q(w, r[shape.ell :])
        ) % shape.ring.q_l:
            raise Rejection(
                f"{label}: repetition {eta} does not recompose -- the digits of "
                "mu' do not rebuild the coefficients of v at this point"
            )

    def _value_claims(
        self, statement: Statement, shape: _Shape, points: list[list[int]], values
    ) -> list[Statement]:
        """The claims the answers leave: `v(x_eta) = w_eta`, one per
        repetition, at the integer point."""
        (v,) = statement.oracles
        out = []
        for r, w in zip(points, values, strict=True):
            point = {var: r[i] for i, var in enumerate(v.variables)}
            out.append(
                self.reduce(
                    [statement], Relation_Eval, oracles=[v], point=point, value=w
                )
            )
        return out

    # --- the pieces both halves share --------------------------------------

    @staticmethod
    def _expect(got: int, want: int, label: str) -> None:
        if got != want:
            raise Rejection(f"{label}: expected {want} entries, got {got}")

    def _digit_table(self, v: MLE, shape: _Shape) -> list[int]:
        """The integer digit table of `mu` from the prover's table `v`,
        LSB-first over `X | Z | Z'`:
        `index = i + 2^ell * k + 2^(ell+nu) * j` holds the `j`-th base-`beta`
        digit of the `k`-th coefficient of `v(i)` ([CCCFGS26]'s
        digit-decomposition lemma).

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
        stride_j = 1 << shape.block
        for i, element in enumerate(v.table):
            for k, coefficient in enumerate(element.get_polynomial()):
                digits = base_digits(coefficient, width, shape.chunks)
                for j, digit in enumerate(digits):
                    table[i + stride_k * k + stride_j * j] = digit
        return table

    def _digit_vector(self, v: MLE, shape: _Shape) -> FieldVector | None:
        """`_digit_table`, lifted into `F_p`, built with whole-array steps
        from the coefficients of `v` as integers modulo ``2^128``
        (`lift_mod_2_128`): `beta` is a power of two, so the digits are bit
        fields, written into their blocks by one `set_radix_decomposition` call.

        None when a digit is wider than 64 bits, or when the digits reach past
        bit 128 while `q` exceeds ``2^128`` (a coefficient's bits there are
        then unknown); `_digit_table` is the way then.
        """
        if not native_table(v):
            raise TypeError(
                "the prover needs v as a dense ring table in the evaluation basis"
            )
        if not self._digits_fit(shape):
            return None
        # Entry i * N + k is coefficient k of v(i); every block holds them i
        # fastest.
        low, high = shape.ring.lift_mod_2_128(v.table)
        return self._radix_vector(low, high, shape, columns=shape.ring.N)

    def _digits_fit(self, shape: _Shape) -> bool:
        """Whether the digits are bit fields of the integers modulo
        ``2^128``: a digit of at most 64 bits, and no digit past bit 128
        unless `q` is below ``2^128`` (the bits there are then zero)."""
        width = shape.beta.bit_length() - 1
        q_bits = sum(p.bit_length() for p in shape.primes)
        return width <= 64 and (width * shape.chunks <= 128 or q_bits <= 128)

    def _radix_vector(self, low, high, shape: _Shape, columns: int = 1) -> FieldVector:
        """The digit table over `F_p` from the integers ``low + 2^64 high``,
        one block of digits after another (`set_radix_decomposition`, the
        values read as rows of `columns` entries)."""
        width = shape.beta.bit_length() - 1
        table = FieldVector(self.field, len(low) * shape.chunks)
        table.set_radix_decomposition(
            low, width, shape.chunks, high=high, columns=columns
        )
        return table

    def _bind_prefix(self, shape: _Shape, r: list[int]) -> dict:
        """The first `m` variables of the `X|Z` block bound to `r_1, ..., r_m`
        as field elements, in variable order.

        In order because `MLE.evaluate` binds the entries of the dict one at a
        time: taking them front to back keeps every bind at position 0, which
        is the layout a field-backed table folds fastest in.
        """
        return {shape.variables[i]: self.field(r[i]) for i in range(self.m)}

    async def _draw_point(
        self, iop: Any, label: str, shape: _Shape, eta: int
    ) -> list[int]:
        """One `r_eta` in `Sigma^(ell+nu)`, from raw coins.

        Not `challenge()`: these are integers, read modulo the modulus of
        whichever oracle they query -- `q` for `v`, each prime for a residue
        oracle, `p` for `mu'` -- so they belong to no one domain's exceptional
        set; `Sigma` is exceptional in all of them. `sigma` is a power of two,
        so slicing the coins is unbiased.
        """
        width = self.alphabet.bit_length() - 1
        count = shape.block
        raw = await iop.verifier.challenge_bits(f"{label}/r{eta}", width * count)
        coins = int.from_bytes(raw, "little")
        mask = self.alphabet - 1
        return [(coins >> (width * i)) & mask for i in range(count)]

    async def _draw_spot(self, iop: Any, label: str, shape: _Shape) -> list:
        """`s'`, one exceptional-set element of `F_p` per free variable."""
        return [
            await iop.verifier.challenge(f"{label}/s{i}", self.field)
            for i in range(shape.ell_star - self.m)
        ]

    def _check_recomposition(
        self,
        statement: Statement,
        shape: _Shape,
        r: list[int],
        psi: MLE,
        w: Any,
        label: str,
        eta: int,
    ) -> None:
        """The heart of the protocol: that the digits rebuild the coefficients.

        The digits' side is accumulated as an exact Python integer through
        `iota`, which is what lets a claim about `F_p` values say something
        about `Z`, and only then compared with the answer `w` modulo the
        primes (`_check_value`). Reducing mod `q` is reducing mod every
        prime of `q`.
        """
        prime = self.field.prime
        width = shape.block - self.m
        weights = r[self.m :]
        total = 0
        for index, value in enumerate(psi.table):
            delta = [(index >> t) & 1 for t in range(width)]
            digit = index >> width  # to-int of the Z' block, LSB-first
            total += centered(value, prime) * eq_int(delta, weights) * shape.beta**digit
        self._check_value(statement, shape, total, w, r, label, eta)

    def _unpacked_evaluation_mod_q(self, element: Any, point: list[int]) -> int:
        """`unpacked_evaluation(element, point)` modulo `q`, the product of
        the element's ring primes, in `[0, q)`.

        For each prime `p`, the coefficients modulo `p` are a table over `F_p`
        indexed LSB-first like `eq_int`'s bits, so folding it once per
        coordinate of `point`, lowest first, leaves the unpacked evaluation
        modulo `p`. The residues are joined by one CRT.
        """
        ring = element.ring
        if 1 << len(point) != ring.N:
            raise ValueError(
                f"a point of {len(point)} coordinates needs {1 << len(point)} "
                f"coefficients, got {ring.N}"
            )
        words = memoryview(element.to_uint64())
        residues = []
        for i, prime in enumerate(ring.primes):
            field = self._prime_fields.get(prime)
            if field is None:
                field = self._prime_fields[prime] = ExtensionField(prime, 1)
            table = FieldVector(field, ring.N)
            table.set_uint64(words[i * ring.N : (i + 1) * ring.N])
            for z in point:
                table = table.fold(field(z % prime))
            residues.append(int(table[0]))
        return crt(residues, list(ring.primes))

    def _outputs(
        self,
        statement: Statement,
        shape: _Shape,
        mu: Oracle,
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
        free = shape.variables[self.m :]
        out = []
        for r, psi in zip(points, psis, strict=True):
            point = self._bind_prefix(shape, r)
            point.update(zip(free, spot, strict=True))
            value = psi.evaluate(
                dict(zip(psi.variables, spot, strict=True)), in_place=False
            ).constant()
            out.append(
                self.reduce(
                    [statement], Relation_Eval, oracles=[mu], point=point, value=value
                )
            )
        out += self._value_claims(statement, shape, points, values)
        out.append(self.reduce([statement], Relation_Lookup(shape.beta), oracles=[mu]))
        if shape.top_beta != shape.beta:
            block = shape.block
            top = VirtualOracle.restriction(
                mu,
                shape.variables[:block],
                dict.fromkeys(shape.variables[block:], 1),
            )
            out.append(
                self.reduce([statement], Relation_Lookup(shape.top_beta), oracles=[top])
            )
        return out


class ResidueRangeDecomposition(RangeDecomposition):
    """`Relation_ResidueRange -> Relation_Lookup x Relation_Eval^(kappa (U+1))`:
    [CCCFGS26]'s optimized range PIOP `Pi*_range` with the unpacked
    polynomial given as an oracle of its own, in residue number system form
    [Gar59]: `U` residue oracles `u_1, ..., u_U`, one per prime of `q`, each
    over a field of that characteristic (`F_(p_i)` or an extension of it)
    with entries in the prime subfield. An oracle over the auxiliary ring
    `H_q = prod_i F_(p_i^d)` of [CCCFGS26] is such a tuple, which is why
    extension fields are allowed.

    The rounds are `RangeDecomposition`'s: `mu'`, the digits of the integers
    the residues define, then `kappa` points `r_eta` of `Sigma^n`, `n` the
    oracles' variables, then the partial evaluations `psi_eta` and the spot
    point. The query at `r_eta` is answered on every residue oracle: the
    prover sends `(u_1(r_eta), ..., u_U(r_eta))`, each coordinate of
    `r_eta` read in the oracle's field, and the verifier checks, for every
    prime,

        sum_{delta, j} iota(psi_eta[delta, j]) * eq~(delta; r_eta^{>m})
                                              * beta^{to-int(j)}
          ==  u_i(r_eta)   in F_(p_i^d),

    the left side the integer of `RangeDecomposition`'s check reduced
    modulo `p_i`. Holding at every prime, the two sides agree modulo `q`,
    which is `RangeDecomposition`'s check with the unpacked evaluation
    replaced by the oracle's own value; an answer outside the prime subfield
    fails the equality. The claims left are `kappa` on `mu'`, as before, and
    `kappa U` evaluation claims on the residue oracles at the integer
    points, in place of the `kappa` claims on the ring oracle.

    The constructor takes `(field, m, alphabet, kappa, chunks)`: the primes
    come from the oracles' domains (`Oracle.domain`).
    """

    name = "residue_range"
    reduces_from: type[Relation] = Relation_ResidueRange

    def __init__(
        self,
        field: Any,
        m: int,
        alphabet: int,
        kappa: int,
        chunks: int,
        messages: OracleMessages | None = None,
    ) -> None:
        super().__init__(field, m, alphabet, kappa, None, chunks, messages)

    def _shape(self, statement: Statement) -> _Shape:
        oracles = statement.oracles
        relation = statement.relation
        if not isinstance(relation, Relation_ResidueRange):
            raise TypeError(
                "ResidueRangeDecomposition discharges Relation_ResidueRange"
            )
        primes = residue_primes(oracles)
        ell = oracles[0].num_vars
        gamma = self.chunks.bit_length() - 1
        beta, top_beta = digit_alphabets(relation.bound, self.chunks)
        shape = _Shape(
            ell=ell,
            nu=0,
            gamma=gamma,
            beta=beta,
            top_beta=top_beta,
            chunks=self.chunks,
            ring=None,
            primes=tuple(primes),
            variables=[MLE_Variable(f"mu{i}") for i in range(ell + gamma)],
        )
        self._validate(shape)
        return shape

    def _digit_mle(self, statement: Statement, prover: Prover, shape: _Shape) -> MLE:
        """The prover's `mu'`: the digits of the integers the residue tables
        define (`crt_lift_mod_2_128`, then one `set_radix_decomposition`),
        or, when the digits are not bit fields of the integers modulo
        ``2^128``, of the exact integers one entry at a time."""
        residues = []
        for oracle in statement.oracles:
            table = resolve_table(oracle, prover.witnesses)
            words = table.table.to_uint64() if vector_table(table) else None
            if words is None:
                raise TypeError(
                    "the prover needs every residue oracle as a field table in the "
                    "evaluation basis with entries in the prime subfield"
                )
            residues.append(words)
        primes = list(shape.primes)
        if self._digits_fit(shape):
            low, high = crt_lift_mod_2_128(residues, primes)
            digits: Any = self._radix_vector(low, high, shape)
        else:
            width = shape.beta.bit_length() - 1
            table = [0] * (1 << shape.ell_star)
            stride_j = 1 << shape.block
            for i, column in enumerate(zip(*residues, strict=True)):
                value = crt(list(column), primes)
                for j, digit in enumerate(base_digits(value, width, shape.chunks)):
                    table[i + stride_j * j] = digit
            digits = [self.field(d) for d in table]
        return MLE(field=self.field, variables=shape.variables, evaluations=digits)

    def _query(
        self, statement: Statement, prover: Prover, shape: _Shape, r: list[int]
    ) -> tuple:
        """The answers at `r`: every residue oracle evaluated at the point,
        its coordinates read in the oracle's field."""
        answers = []
        for oracle in statement.oracles:
            table = resolve_table(oracle, prover.witnesses)
            field = table.field
            point = {var: field(r[i]) for i, var in enumerate(oracle.variables)}
            answers.append(table.evaluate(point, in_place=False).constant())
        return tuple(answers)

    def _check_value(
        self,
        statement: Statement,
        shape: _Shape,
        total: int,
        w: Any,
        r: list[int],
        label: str,
        eta: int,
    ) -> None:
        """`total == w_i` in the field of residue oracle `i`, for every `i`:
        the recomposition modulo each prime, which also pins each answer to
        the prime subfield."""
        oracles = statement.oracles
        if not isinstance(w, tuple) or len(w) != len(oracles):
            raise Rejection(f"{label}: repetition {eta} needs {len(oracles)} answers")
        for oracle, prime, answer in zip(oracles, shape.primes, w, strict=True):
            field = oracle.domain
            if getattr(answer, "field", None) != field:
                raise Rejection(
                    f"{label}: repetition {eta} has an answer outside its "
                    "oracle's field"
                )
            if field(total % prime) != answer:
                raise Rejection(
                    f"{label}: repetition {eta} does not recompose modulo {prime} -- "
                    "the digits of mu' do not rebuild the residues at this point"
                )

    def _value_claims(
        self, statement: Statement, shape: _Shape, points: list[list[int]], values
    ) -> list[Statement]:
        """`u_i(r_eta) = w_(eta, i)` for every repetition and every residue
        oracle, at the point read in the oracle's field."""
        out = []
        for r, answers in zip(points, values, strict=True):
            for oracle, answer in zip(statement.oracles, answers, strict=True):
                field = oracle.domain
                point = {var: field(r[i]) for i, var in enumerate(oracle.variables)}
                out.append(
                    self.reduce(
                        [statement],
                        Relation_Eval,
                        oracles=[oracle],
                        point=point,
                        value=answer,
                    )
                )
        return out
