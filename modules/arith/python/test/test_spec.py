# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The implementation/backend registry and the parent contract."""

from __future__ import annotations

import pytest
from vfhe.arith import (
    ArithParent,
    Capability,
    ComplexRing,
    Constraints,
    Domain,
    ExtensionField,
    Field,
    Multiprecision,
    Polynomial,
    PseudoMersenneField,
    Ring,
    RNSPolynomial,
    RNSRing,
    Spec,
    backends,
    implementations,
    registered,
    resolve,
)
from vfhe.arith.registry import common_spec, register_conversion

PRIME = 0xFFFFFFFF00000001


def test_every_implementation_is_registered():
    assert set(implementations()) == {"complex", "field", "mp", "pmf", "rns"}


@pytest.mark.parametrize(
    ("cls", "key"),
    [
        (RNSRing, ("rns", "ntt")),
        (ExtensionField, ("field", "scalar")),
        (PseudoMersenneField, ("pmf", "limb52")),
        (ComplexRing, ("complex", "fft")),
        (Multiprecision, ("mp", "limb52")),
    ],
)
def test_class_carries_its_spec(cls, key):
    assert cls.spec.key == key
    assert registered()[key].parent_cls is cls


def test_resolve_picks_the_only_backend():
    assert resolve("rns").key == ("rns", "ntt")
    assert backends("rns") == ["ntt"]


def test_resolve_rejects_an_unknown_implementation():
    with pytest.raises(LookupError, match="unknown implementation"):
        resolve("nosuch")


def test_resolve_honours_a_prime_width_constraint():
    # rns/ntt accepts 64-bit primes, so a 200-bit request has no backend.
    resolve("rns", prime_bits=49)
    with pytest.raises(LookupError, match="prime_bits=200"):
        resolve("rns", prime_bits=200)


def test_resolve_honours_a_capability_requirement():
    resolve("rns", requires=Capability.TOWER)
    with pytest.raises(LookupError):
        resolve("field", requires=Capability.TOWER)


def test_named_backend_is_checked_not_just_looked_up():
    with pytest.raises(LookupError, match="accepts primes up to"):
        resolve("rns", backend="ntt", prime_bits=200)


def test_unknown_backend_names_what_exists():
    with pytest.raises(KeyError, match="registered for it"):
        resolve("rns", backend="nosuch")


class TestCapabilities:
    """A ring is a quotient polynomial ring with a tower; a field is neither."""

    def test_ring(self):
        ring = Ring(256, 300, split_degree=1)
        assert ring.supports(Capability.QUOTIENT_POLY_RING)
        assert ring.supports(Capability.TOWER)
        assert ring.supports(Capability.EXACT)
        assert not ring.domains_coincide
        assert ring.mul_domain() is Domain.MUL

    def test_field(self):
        field = Field(PRIME, 4, 7)
        assert not field.supports(Capability.QUOTIENT_POLY_RING)
        assert not field.supports(Capability.TOWER)
        assert field.domains_coincide
        assert field.mul_domain() is Domain.CANONICAL

    def test_require_names_the_missing_flag(self):
        field = Field(PRIME, 4, 7)
        with pytest.raises(TypeError, match="QUOTIENT_POLY_RING"):
            field.require(Capability.QUOTIENT_POLY_RING)


class TestExceptionalSetSize:
    """|A| comes from the domain, not from a caller inspecting its internals."""

    def test_ring_is_min_prime_to_the_split_degree(self):
        ring = Ring(256, 300, split_degree=2)
        assert ring.exceptional_set_size == min(ring.primes) ** ring.split_degree

    def test_field_is_the_whole_field(self):
        field = Field(PRIME, 4, 7)
        assert field.exceptional_set_size == PRIME**4

    def test_pseudo_mersenne_is_the_whole_field(self):
        pmf = PseudoMersenneField.generate(260)
        assert pmf.exceptional_set_size == pmf.prime

    def test_domains_are_arith_parents(self):
        assert isinstance(Ring(256, 300, split_degree=1), ArithParent)
        assert isinstance(Field(PRIME, 4, 7), ArithParent)


class TestConversions:
    """Mixed specs need an explicitly declared route."""

    def test_same_spec_needs_no_conversion(self):
        assert common_spec(RNSRing.spec, RNSRing.spec) is RNSRing.spec

    def test_unrelated_specs_refuse_to_combine(self):
        with pytest.raises(TypeError, match="no implicit conversion"):
            common_spec(RNSRing.spec, ExtensionField.spec)

    def test_an_implicit_conversion_picks_the_target(self):
        narrow = Spec(
            implementation="rns",
            backend="_test_narrow",
            parent_cls=RNSRing,
            capabilities=Capability.CORE,
            constraints=Constraints(max_prime_bits=32),
            rank=99,
        )
        register_conversion(
            narrow.key, RNSRing.spec.key, lambda value: value, implicit=True
        )
        assert common_spec(narrow, RNSRing.spec) is RNSRing.spec
        # and the reverse direction, which is not registered, still resolves
        # to the same target rather than silently narrowing
        assert common_spec(RNSRing.spec, narrow) is RNSRing.spec


def test_mlwe_refuses_a_domain_without_the_ring_capabilities():
    """The fail-early guard: FHE is instantiated over a ring, never a field."""
    from vfhe.mlwe import MLWE_Scheme

    with pytest.raises(TypeError, match="QUOTIENT_POLY_RING"):
        MLWE_Scheme(Field(PRIME, 4, 7))


class TestHierarchy:
    """The generic front classes build their resolved implementation."""

    def test_ring_builds_the_default_implementation(self):
        ring = Ring(256, 300, split_degree=1)
        assert type(ring) is RNSRing
        assert isinstance(ring, Ring)

    def test_polynomial_builds_the_ring_element_type(self):
        ring = Ring(256, 300, split_degree=1)
        poly = Polynomial(ring)
        assert type(poly) is RNSPolynomial
        assert isinstance(poly, Polynomial)

    def test_field_builds_the_default_implementation(self):
        field = Field(PRIME, 4, 7)
        assert type(field) is ExtensionField
        assert isinstance(field, Field)

    def test_pseudo_mersenne_is_a_field(self):
        pmf = PseudoMersenneField.generate(260)
        assert isinstance(pmf, Field)
        # the generic |A| answer comes from the shared base
        assert pmf.exceptional_set_size == pmf.order == pmf.prime

    def test_field_defaults_to_degree_one(self):
        field = Field(PRIME)
        assert type(field) is ExtensionField
        assert field.order == PRIME

    def test_field_routes_a_large_pseudo_mersenne_to_pmf(self):
        prime = PseudoMersenneField.generate(260, two_adicity=8).prime
        field = Field(prime)
        assert type(field) is PseudoMersenneField
        assert field.order == prime

    def test_field_rejects_a_large_modulus_no_backend_covers(self):
        # >64 bits but not pseudo-Mersenne: pmf does not apply, field cannot.
        with pytest.raises(LookupError, match="prime_bits=201"):
            Field(2**200 + 5)

    def test_pmf_rejects_an_extension_degree(self):
        prime = PseudoMersenneField.generate(260, two_adicity=8).prime
        with pytest.raises(ValueError, match="degree=2"):
            Field(prime, 2, implementation="pmf")

    def test_implementation_keyword_names_the_subclass(self):
        assert type(Ring(256, 300, split_degree=1, implementation="rns")) is RNSRing
        with pytest.raises(LookupError, match="unknown implementation"):
            Ring(256, 300, implementation="nosuch")

    def test_concrete_classes_construct_as_themselves(self):
        assert type(RNSRing(256, 300, split_degree=1)) is RNSRing
