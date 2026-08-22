# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Fiat-Shamir layer: the transcript's chained prefix hash and
its caches, the element-digest walker, deterministic challenge derivation
(ring and seeded-stub domains), root-statement binding, and end-to-end FS
sumcheck / sumcheckprod runs — two FS runs of the same statement must
produce byte-identical transcripts.
"""

import pytest
from vfhe.arith import Polynomial, Ring
from vfhe.piop import (
    IOP,
    MLE,
    FS_Verifier,
    MLE_Variable,
    Proof,
    Relation_Sum,
    Relation_SumProd,
    Statement,
    Sumcheck,
    SumcheckProd,
    element_digest,
)
from vfhe.piop.fs import expand_bytes, ring_exceptional_from_seed


def _ring() -> Ring:
    return Ring(1024, prime_size=[49], split_degree=4)


def _dense_sum_statement(ring: Ring, evaluations: list) -> Statement:
    v = [MLE_Variable("x0"), MLE_Variable("x1")]
    f = MLE(ring=ring, variables=v, evaluations=evaluations)
    total = sum(evaluations)
    return Statement(Relation_Sum(), oracles=[f], value=total)


def _fs_sum_iop(ring: Ring) -> IOP:
    iop = IOP(domain=ring, fiat_shamir=True)
    iop.register(Relation_Sum, Sumcheck())
    return iop


def _transcript_digests(iop: IOP) -> list:
    t = iop.transcript
    return [
        (label, element_digest(t.entries[label].result())) for label in t.order
    ]


def test_transcript_state_chaining_and_caches():
    iop = IOP()
    t = iop.transcript
    t.bind(b"seed")
    with pytest.raises(IndexError):
        t.state(1)  # nothing written yet
    empty = t.state()
    t.write("a", 5)
    s1 = t.state()
    assert t.state(1) == s1 and t.state(0) == empty  # prefixes are lookups
    t.write("b", b"msg")
    s2 = t.state()
    assert s2 != s1 and t.state(1) == s1  # chain extends, cache stable
    assert len(t.digests) == 2  # per-entry digest vector
    with pytest.raises(RuntimeError):
        t.bind(b"late")  # the chain has started; the seed is fixed


def test_transcript_seed_changes_states():
    def state_after(seed: bytes) -> bytes:
        iop = IOP()
        iop.transcript.bind(seed)
        iop.transcript.write("a", 1)
        return iop.transcript.state()

    assert state_after(b"one") != state_after(b"two")


def test_element_digest_walker():
    # Primitives are domain-separated and deterministic.
    assert element_digest(5) == element_digest(5)
    assert element_digest(5) != element_digest(b"\x05")
    assert element_digest("x") != element_digest(b"x")
    assert element_digest(None) != element_digest(0)
    assert element_digest((1, 2)) != element_digest((2, 1))
    # Statements digest over relation + canonical fields.
    v = [MLE_Variable("x0"), MLE_Variable("x1")]
    f = MLE(variables=v, evaluations=[1, 2, 3, 4])
    s1 = Statement(Relation_Sum(), oracles=[f], value=10)
    s2 = Statement(Relation_Sum(), oracles=[f], value=10)
    s3 = Statement(Relation_Sum(), oracles=[f], value=11)
    assert element_digest(s1) == element_digest(s2)
    assert element_digest(s1) != element_digest(s3)
    # Ring-backed tables digest through Polynomial.get_hash.
    ring = _ring()
    g = MLE(ring=ring, variables=v, evaluations=[1, 2, 3, 4])
    h = MLE(ring=ring, variables=v, evaluations=[1, 2, 3, 5])
    assert element_digest(g) == element_digest(g)
    assert element_digest(g) != element_digest(h)
    with pytest.raises(TypeError):
        element_digest(object())


def test_ring_exceptional_from_seed_deterministic():
    ring = _ring()
    a = ring_exceptional_from_seed(ring, b"seed")
    b = ring_exceptional_from_seed(ring, b"seed")
    c = ring_exceptional_from_seed(ring, b"other")
    assert a == b
    assert a != c


def test_expand_bytes():
    assert expand_bytes(b"s", 100) == expand_bytes(b"s", 100)
    assert len(expand_bytes(b"s", 100)) == 100
    assert expand_bytes(b"s", 32) == expand_bytes(b"s", 100)[:32]


def test_fs_sumcheck_deterministic_transcripts():
    ring = _ring()
    runs = []
    for _ in range(2):
        iop = _fs_sum_iop(ring)
        assert isinstance(iop.verifier, FS_Verifier)
        assert iop.run(_dense_sum_statement(ring, [1, 2, 3, 4]))
        runs.append(_transcript_digests(iop))
    assert runs[0] == runs[1]  # fully deterministic, byte-identical


def test_fs_sumcheck_rejects_false_claim():
    ring = _ring()
    iop = _fs_sum_iop(ring)
    v = [MLE_Variable("x0"), MLE_Variable("x1")]
    f = MLE(ring=ring, variables=v, evaluations=[1, 2, 3, 4])
    assert not iop.run(Statement(Relation_Sum(), oracles=[f], value=11))


def test_fs_challenges_bind_the_statement():
    # Two true statements with the SAME claimed value but different oracles:
    # the chain seed differs, so the first challenge already differs.
    ring = _ring()
    r0 = []
    for evaluations in ([1, 2, 3, 4], [2, 2, 3, 3]):  # both sum to 10
        iop = _fs_sum_iop(ring)
        assert iop.run(_dense_sum_statement(ring, evaluations))
        r0.append(element_digest(iop.transcript.entries["sumcheck/r0"].result()))
    assert r0[0] != r0[1]


def test_fs_sumcheckprod():
    ring = _ring()
    v = [MLE_Variable("x0"), MLE_Variable("x1")]
    runs = []
    for _ in range(2):
        f = MLE(ring=ring, variables=v, evaluations=[1, 2, 3, 4])
        g = MLE(ring=ring, variables=v, evaluations=[2, 2, 3, 3])
        iop = IOP(domain=ring, fiat_shamir=True)
        iop.register(Relation_SumProd, SumcheckProd())
        assert iop.run(Statement(Relation_SumProd(), oracles=[f, g], value=27))
        runs.append(_transcript_digests(iop))
    assert runs[0] == runs[1]


class _SeededDomain:
    """Interactive-and-FS-capable stub: integers derived from seeds."""

    def random_exceptional(self) -> int:
        raise AssertionError("FS runs must not sample")

    def exceptional_from_seed(self, seed: bytes) -> int:
        return int.from_bytes(seed[:8], "little")


def test_fs_domain_hook_and_plain_tables():
    # A domain-provided exceptional_from_seed takes precedence, and the FS
    # path works over plain (ring-less) tables too.
    v = [MLE_Variable("x0"), MLE_Variable("x1")]
    f = MLE(variables=v, evaluations=[1, 2, 3, 4])
    iop = IOP(domain=_SeededDomain(), fiat_shamir=True)
    iop.register(Relation_Sum, Sumcheck())
    assert iop.run(Statement(Relation_Sum(), oracles=[f], value=10))


def test_proof_holds_only_prover_messages():
    ring = _ring()
    iop = _fs_sum_iop(ring)
    proof = iop.prove(_dense_sum_statement(ring, [1, 2, 3, 4]))
    # The challenges are recomputable, so they are not carried.
    assert proof.labels == ("sumcheck/g0", "sumcheck/g1")
    assert set(iop.transcript.derived) == {"sumcheck/r0", "sumcheck/r1"}
    assert len(proof) == 2


def test_prove_then_verify_accepts():
    ring = _ring()
    statement = _dense_sum_statement(ring, [1, 2, 3, 4])
    proof = _fs_sum_iop(ring).prove(statement)
    # A separate IOP, no prover, no witnesses: only the statement + proof.
    assert _fs_sum_iop(ring).verify(statement, proof)


def test_verify_rejects_tampered_message():
    ring = _ring()
    statement = _dense_sum_statement(ring, [1, 2, 3, 4])
    proof = _fs_sum_iop(ring).prove(statement)
    (label, (g0, g1)), rest = proof.messages[0], proof.messages[1:]
    # Round 0 still sums correctly (g0 + g1 unchanged) but the split moves,
    # so every later challenge changes and the run no longer closes.
    one = Polynomial(ring).from_array([1])
    tampered = Proof([(label, (g0 + one, g1 - one)), *rest])
    assert not _fs_sum_iop(ring).verify(statement, tampered)


def test_verify_rejects_wrong_statement():
    ring = _ring()
    proof = _fs_sum_iop(ring).prove(_dense_sum_statement(ring, [1, 2, 3, 4]))
    other = _dense_sum_statement(ring, [2, 2, 3, 3])  # also sums to 10
    assert not _fs_sum_iop(ring).verify(other, proof)


def test_verify_rejects_malformed_proofs():
    ring = _ring()
    statement = _dense_sum_statement(ring, [1, 2, 3, 4])
    proof = _fs_sum_iop(ring).prove(statement)
    truncated = Proof(proof.messages[:1])
    assert not _fs_sum_iop(ring).verify(statement, truncated)  # runs out
    reordered = Proof(proof.messages[::-1])
    assert not _fs_sum_iop(ring).verify(statement, reordered)  # out of turn
    padded = Proof([*proof.messages, ("sumcheck/g2", (0, 0))])
    assert not _fs_sum_iop(ring).verify(statement, padded)  # trailing message


def test_prove_verify_require_fiat_shamir():
    ring = _ring()
    statement = _dense_sum_statement(ring, [1, 2, 3, 4])
    interactive = IOP(domain=ring)
    interactive.register(Relation_Sum, Sumcheck())
    with pytest.raises(RuntimeError):
        interactive.prove(statement)
    with pytest.raises(RuntimeError):
        interactive.verify(statement, Proof(()))


def test_proof_matches_the_interactive_transcript():
    # prove() and run() are the same execution; the proof is just the part
    # of that transcript the verifier cannot recompute.
    ring = _ring()
    statement = _dense_sum_statement(ring, [1, 2, 3, 4])
    proof = _fs_sum_iop(ring).prove(statement)
    iop = _fs_sum_iop(ring)
    assert iop.run(statement)
    assert Proof.of(iop.transcript).labels == proof.labels
    assert Proof.of(iop.transcript).digest() == proof.digest()


def test_prove_then_verify_sumcheckprod():
    ring = _ring()
    v = [MLE_Variable("x0"), MLE_Variable("x1")]

    def statement() -> Statement:
        f = MLE(ring=ring, variables=v, evaluations=[1, 2, 3, 4])
        g = MLE(ring=ring, variables=v, evaluations=[2, 2, 3, 3])
        return Statement(Relation_SumProd(), oracles=[f, g], value=27)

    def iop() -> IOP:
        one = IOP(domain=ring, fiat_shamir=True)
        one.register(Relation_SumProd, SumcheckProd())
        return one

    proof = iop().prove(statement())
    assert iop().verify(statement(), proof)


def test_solo_prover_cannot_read():
    # A prove half that tried to receive a message would hang forever with
    # no counterparty; the transcript refuses instead.
    ring = _ring()

    class _ReadingSumcheck(Sumcheck):
        async def prove(self, prover, statements):
            await prover.iop.transcript.read("nobody/will/write/this")
            return await super().prove(prover, statements)

    iop = IOP(domain=ring, fiat_shamir=True)
    iop.register(Relation_Sum, _ReadingSumcheck())
    with pytest.raises(RuntimeError, match="no counterparty"):
        iop.prove(_dense_sum_statement(ring, [1, 2, 3, 4]))


def test_fs_challenge_bits_deterministic():
    def draw() -> bytes:
        iop = IOP(domain=_SeededDomain(), fiat_shamir=True)
        iop.transcript.bind(b"root")
        return iop.verifier.challenge_bits("queries", 128)

    one, two = draw(), draw()
    assert one == two and len(one) == 16
