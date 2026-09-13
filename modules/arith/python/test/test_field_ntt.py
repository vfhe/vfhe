# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The negacyclic transform over an `ExtensionField`, in both its domains.

The reference is Horner evaluation at ``psi ** (2 * brv(j) + 1)``, worked out
one element at a time: the definition the transform is an algorithm for, and
code the transform shares nothing with. It is quadratic, so the lengths here
are small -- what the transform has to get right is the index arithmetic, the
twiddle order and the batch layout, and those are wrong or right at n = 8 just
as at n = 2048.

**Three implementations meet here**, and the constants below are what select
them. With the root in F_p the transform splits across the coefficient planes
onto arith's own kernels (``domain='base'``); outside F_p the butterflies are
extension multiplications, taking the fused kernel below 2^50 and the generic
whole-vector path above it. Both primes have ``v2(p - 1) == 4``, which puts the
base domain at ``n <= 8`` and the extension domain at ``n`` in 16, 32 -- so one
prime exercises both, and the pairs in `CASES` say which is which.

Both primes are 1 mod 4 with a verified non-residue `w`, so ``x**4 - w`` is
irreducible and these are fields rather than quotient rings with zero divisors
-- which the transform needs and plain ring arithmetic does not.
"""

from __future__ import annotations

import pytest
from vfhe.arith import ExtensionField, ExtensionFieldElement, FieldVector

#: 49 bits: the extension path takes the fused butterfly here.
FUSED_PRIME, FUSED_W = 562949953421201, 3
#: 51 bits: above the fused kernel's family, so the generic path runs.
GENERIC_PRIME, GENERIC_W = 2251799813684753, 3

FIELDS = [
    pytest.param(FUSED_PRIME, FUSED_W, id="fused"),
    pytest.param(GENERIC_PRIME, GENERIC_W, id="generic"),
]
#: (domain, n): v2(p - 1) = 4 for both primes, so 2n divides p - 1 up to n = 8
#: and the extension domain is what is left at 16 and 32.
CASES = [("base", 2), ("base", 8), ("extension", 16), ("extension", 32)]


def brv(i: int, bits: int) -> int:
    return int(format(i, f"0{bits}b")[::-1], 2) if bits else 0


def horner(coefficients, x, field):
    """P(x) for P given by `coefficients`, low degree first."""
    acc = field.zero
    for c in reversed(coefficients):
        acc = acc * x + c
    return acc


def polynomials(field, n, blocks, seed):
    """`blocks` polynomials of `n` coefficients, deterministic in `seed`."""
    out, state = [], seed
    for _ in range(blocks):
        block = []
        for _ in range(n):
            coefficients = []
            for _ in range(field.d):
                state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
                coefficients.append(state % field.prime)
            block.append(ExtensionFieldElement(field, coefficients))
        out.append(block)
    return out


def values(field, n, seed):
    """`n` elements as a flat list, deterministic in `seed`."""
    return polynomials(field, n, 1, seed)[0]


def lay_out(polys, layout):
    """The flat vector a plan of this layout takes.

    The two are not interchangeable and the plan says which it wants:
    ``'blocks'`` puts each transform in one contiguous run, ``'interleaved'``
    puts position ``i`` of every block together.
    """
    blocks, n = len(polys), len(polys[0])
    if layout == "blocks":
        return [polys[b][i] for b in range(blocks) for i in range(n)]
    return [polys[b][i] for i in range(n) for b in range(blocks)]


def read(flat, layout, n, blocks, b, j):
    """Position `j` of block `b` out of a flat vector in `layout`."""
    return flat[b * n + j] if layout == "blocks" else flat[j * blocks + b]


class TestDomainSelection:
    @pytest.mark.parametrize(("prime", "w"), FIELDS)
    def test_auto_takes_the_base_domain_where_it_exists(self, prime, w):
        """It is about 3x the faster, so it is the default -- but only where
        the prime supplies it."""
        field = ExtensionField(prime, 4, w)
        assert field.ntt_plan(8).domain == "base"
        assert field.ntt_plan(16).domain == "extension"

    @pytest.mark.parametrize(("prime", "w"), FIELDS)
    def test_a_domain_the_prime_cannot_supply_is_refused(self, prime, w):
        """Both directions, and the extension one is the one that matters: a
        caller asking for a domain outside F_p must not be handed the subfield
        quietly, because nothing downstream would notice."""
        field = ExtensionField(prime, 4, w)
        with pytest.raises(ValueError, match=r"lies in F_p"):
            field.ntt_plan(8, domain="extension")
        with pytest.raises(ValueError, match="does not divide p - 1"):
            field.ntt_plan(16, domain="base")
        with pytest.raises(ValueError, match="domain must be"):
            field.ntt_plan(8, domain="elsewhere")

    @pytest.mark.parametrize(("prime", "w"), FIELDS)
    def test_the_layout_follows_the_domain(self, prime, w):
        field = ExtensionField(prime, 4, w)
        assert field.ntt_plan(8, domain="base").batch_layout == "blocks"
        assert field.ntt_plan(16, domain="extension").batch_layout == "interleaved"

    @pytest.mark.parametrize(("prime", "w"), FIELDS)
    def test_the_base_root_is_the_one_arith_chose(self, prime, w):
        """`ntt_new_plan` picks its own root and takes none, so a caller's
        twists only match the transform if it reads that one back."""
        field = ExtensionField(prime, 4, w)
        plan = field.ntt_plan(8, domain="base")
        assert field.root_of_unity(8, domain="base") == plan.root_of_unity
        assert plan.root_of_unity**8 == -field.one
        # It is a scalar: that is what "the root lies in F_p" means.
        assert all(plan.root_of_unity.value[j] == 0 for j in range(1, field.d))

    def test_plans_are_memoized_per_length_and_domain(self):
        field = ExtensionField(FUSED_PRIME, 4, FUSED_W)
        assert field.ntt_plan(8) is field.ntt_plan(8)
        assert field.ntt_plan(8) is field.ntt_plan(8, domain="base")
        assert field.ntt_plan(8) is not field.ntt_plan(16)


class TestRootOfUnity:
    @pytest.mark.parametrize(("prime", "w"), FIELDS)
    @pytest.mark.parametrize("n", [1, 2, 8, 16, 32])
    def test_is_primitive_of_order_2n(self, prime, w, n):
        field = ExtensionField(prime, 4, w)
        psi = field.root_of_unity(n)
        assert psi**n == -field.one
        assert psi ** (2 * n) == field.one

    def test_is_deterministic(self):
        field = ExtensionField(FUSED_PRIME, 4, FUSED_W)
        again = ExtensionField(FUSED_PRIME, 4, FUSED_W)
        assert field.root_of_unity(32) == again.root_of_unity(32)

    @pytest.mark.parametrize(("prime", "w"), FIELDS)
    def test_the_subfield_degree_is_forced_by_the_prime(self, prime, w):
        """Which subfield the root lands in is arithmetic, not a choice: the
        least k with p^k = 1 mod 2n. The root must actually lie in F_p exactly
        when that k is 1, which is the thing a caller needs to be able to
        predict before building a domain out of it."""
        field = ExtensionField(prime, 4, w)
        for n in (2, 8, 16, 32):
            order, k, x = 2 * n, 1, field.prime % (2 * n)
            while x != 1:
                x = x * (field.prime % order) % order
                k += 1
            assert field.root_subfield_degree(n) == k

            psi = field.root_of_unity(n)
            in_prime_field = all(psi.value[j] == 0 for j in range(1, field.d))
            assert in_prime_field == (k == 1)

    def test_climbs_through_the_subfields(self):
        """With v2(p - 1) = 4 the degree steps 1, 2, 4 as 2n passes each of
        p - 1, p^2 - 1 and p^4 - 1 -- the whole reason a caller has to ask."""
        field = ExtensionField(FUSED_PRIME, 4, FUSED_W)
        assert [field.root_subfield_degree(n) for n in (8, 16, 32)] == [1, 2, 4]

    @pytest.mark.parametrize(("prime", "w"), FIELDS)
    def test_refuses_a_length_the_field_cannot_serve(self, prime, w):
        field = ExtensionField(prime, 4, w)
        # v2(p^4 - 1) = v2(p - 1) + 2 = 6, so 2n = 2^7 does not divide it.
        with pytest.raises(ValueError, match="no primitive"):
            field.root_of_unity(64)
        for bad in (0, 3, -4):
            with pytest.raises(ValueError, match="power of two"):
                field.root_of_unity(bad)


class TestTransform:
    @pytest.mark.parametrize(("prime", "w"), FIELDS)
    @pytest.mark.parametrize(("domain", "n"), CASES)
    @pytest.mark.parametrize("blocks", [1, 3, 8])
    def test_matches_evaluation_at_the_domain(self, prime, w, domain, n, blocks):
        """Position j of a block must hold P(psi^(2 brv(j) + 1)) for that
        block's polynomial -- the definition, by Horner."""
        field = ExtensionField(prime, 4, w)
        plan = field.ntt_plan(n, domain=domain)
        psi = plan.root_of_unity
        logn = n.bit_length() - 1

        polys = polynomials(field, n, blocks, seed=7)
        vector = FieldVector(field, lay_out(polys, plan.batch_layout))
        plan.forward(vector, blocks)
        got = vector.to_list()

        for b in range(blocks):
            for j in range(n):
                point = psi ** (2 * brv(j, logn) + 1)
                assert read(got, plan.batch_layout, n, blocks, b, j) == horner(
                    polys[b], point, field
                )

    @pytest.mark.parametrize(("prime", "w"), FIELDS)
    @pytest.mark.parametrize(("domain", "n"), CASES)
    @pytest.mark.parametrize("blocks", [1, 5])
    def test_round_trips(self, prime, w, domain, n, blocks):
        field = ExtensionField(prime, 4, w)
        plan = field.ntt_plan(n, domain=domain)
        flat = lay_out(polynomials(field, n, blocks, seed=11), plan.batch_layout)
        vector = FieldVector(field, flat)

        plan.forward(vector, blocks)
        assert vector.to_list() != flat  # it did something
        plan.inverse(vector, blocks)
        assert vector.to_list() == flat

    @pytest.mark.parametrize(("prime", "w"), FIELDS)
    @pytest.mark.parametrize(("domain", "n"), CASES)
    def test_a_batch_is_its_blocks(self, prime, w, domain, n):
        """The batch exists for speed and must change nothing: block b of a
        batch has to equal that block transformed on its own."""
        field = ExtensionField(prime, 4, w)
        blocks = 4
        plan = field.ntt_plan(n, domain=domain)
        polys = polynomials(field, n, blocks, seed=13)

        batched = FieldVector(field, lay_out(polys, plan.batch_layout))
        plan.forward(batched, blocks)
        got = batched.to_list()

        for b in range(blocks):
            alone = FieldVector(field, lay_out([polys[b]], plan.batch_layout))
            plan.forward(alone, 1)
            assert [
                read(got, plan.batch_layout, n, blocks, b, j) for j in range(n)
            ] == alone.to_list()

    def test_the_two_domains_are_two_transforms_of_the_same_shape(self):
        """Not the same values -- the roots differ -- but the same structure:
        each is the evaluation of its own domain, and both round-trip. This is
        the check that the base path is a transform and not a copy."""
        field = ExtensionField(FUSED_PRIME, 4, FUSED_W)
        base = field.ntt_plan(8, domain="base")
        ext = field.ntt_plan(16, domain="extension")
        assert base.root_of_unity != ext.root_of_unity
        assert base.batch_layout != ext.batch_layout
        assert base.subfield_degree == 1
        assert ext.subfield_degree == 2

    @pytest.mark.parametrize("domain", ["base", "extension"])
    def test_rejects_a_batch_that_is_not_the_plan_s_shape(self, domain):
        field = ExtensionField(FUSED_PRIME, 4, FUSED_W)
        n = 8 if domain == "base" else 16
        plan = field.ntt_plan(n, domain=domain)
        with pytest.raises(ValueError, match="elements"):
            plan.forward(FieldVector(field, n + 1), 1)
        with pytest.raises(ValueError, match="elements"):
            plan.forward(FieldVector(field, 2 * n), 1)
        with pytest.raises(ValueError, match="blocks"):
            plan.forward(FieldVector(field, n), 0)
        other = ExtensionField(GENERIC_PRIME, 4, GENERIC_W)
        with pytest.raises(ValueError, match="different field"):
            plan.forward(FieldVector(other, n), 1)


class TestPoints:
    """`ntt_points`: the transform's evaluation domain, as a table.

    Checked against the definition written out -- `root ** (2*brv(i) + 1)` per
    index -- which is the n modular exponentiations the running product
    replaces, and the only independent way to say what the table should hold.
    """

    @pytest.mark.parametrize(("prime", "w"), FIELDS)
    @pytest.mark.parametrize("n", [1, 2, 8, 32])
    def test_matches_the_definition(self, prime, w, n):
        from vfhe.polycom import bit_reverse

        field = ExtensionField(prime, 4, w)
        root = int(field.ntt_plan(8, domain="base").root_of_unity)
        bits = n.bit_length() - 1
        got = [int(value) for value in field.ntt_points(root, n)]
        assert got == [pow(root, 2 * bit_reverse(i, bits) + 1, prime) for i in range(n)]

    def test_the_points_are_the_transform_s_own(self):
        """Position j of a transform evaluates at point j: the table and the
        transform have to be reading the same domain."""
        field = ExtensionField(FUSED_PRIME, 4, FUSED_W)
        n = 8
        plan = field.ntt_plan(n, domain="base")
        points = field.ntt_points(int(plan.root_of_unity), n)

        coefficients = values(field, n, seed=41)
        vector = FieldVector(field, coefficients)
        plan.forward(vector, 1)
        got = vector.to_list()
        for j in range(n):
            assert got[j] == horner(coefficients, points[j], field)

    def test_rejects_a_length_that_is_not_a_power_of_two(self):
        field = ExtensionField(FUSED_PRIME, 4, FUSED_W)
        for bad in (0, 3, -8):
            with pytest.raises(ValueError, match="power of two"):
                field.ntt_points(5, bad)


class TestBatchLayout:
    """`pack` / `unpack`: the conversion the extension domain's layout forces.

    The transform takes a batch interleaved and a caller holding each block's
    elements together has them contiguous, so both directions are needed.
    These are that, and for the base domain -- whose layout is already the
    contiguous one -- they are nothing.
    """

    @pytest.mark.parametrize(("prime", "w"), FIELDS)
    @pytest.mark.parametrize("blocks", [1, 3, 8])
    def test_pack_is_the_index_arithmetic_and_unpack_inverts_it(self, prime, w, blocks):
        field = ExtensionField(prime, 4, w)
        n = 16  # the extension domain for these primes
        plan = field.ntt_plan(n, domain="extension")
        source = values(field, n * blocks, seed=43)
        vector = FieldVector(field, source)

        packed = plan.pack(vector, blocks)
        assert packed.to_list() == [
            source[b * n + i] for i in range(n) for b in range(blocks)
        ]
        assert plan.unpack(packed, blocks).to_list() == source

    @pytest.mark.parametrize(("prime", "w"), FIELDS)
    def test_pack_takes_a_destination(self, prime, w):
        field = ExtensionField(prime, 4, w)
        n, blocks = 16, 4
        plan = field.ntt_plan(n, domain="extension")
        vector = FieldVector(field, values(field, n * blocks, seed=44))
        dest = FieldVector(field, n * blocks)
        assert plan.pack(vector, blocks, out=dest) is dest
        assert dest.to_list() == plan.pack(vector, blocks).to_list()
        with pytest.raises(ValueError, match="into its input"):
            plan.pack(vector, blocks, out=vector)

    def test_the_base_domain_needs_no_conversion(self):
        field = ExtensionField(FUSED_PRIME, 4, FUSED_W)
        plan = field.ntt_plan(8, domain="base")
        vector = FieldVector(field, values(field, 8 * 3, seed=45))
        assert plan.pack(vector, 3) is vector
        assert plan.unpack(vector, 3) is vector

    def test_a_packed_batch_transforms_to_the_same_thing(self):
        """The round trip that makes the pair worth having: hold the batch by
        blocks, pack, transform, unpack, and each block is transformed."""
        field = ExtensionField(FUSED_PRIME, 4, FUSED_W)
        n, blocks = 16, 4
        plan = field.ntt_plan(n, domain="extension")
        polys = polynomials(field, n, blocks, seed=46)
        by_blocks = FieldVector(field, [e for block in polys for e in block])

        work = plan.pack(by_blocks, blocks)
        plan.forward(work, blocks)
        result = plan.unpack(work, blocks).to_list()

        for b in range(blocks):
            alone = FieldVector(field, polys[b])
            plan.forward(alone, 1)
            assert result[b * n : (b + 1) * n] == alone.to_list()
