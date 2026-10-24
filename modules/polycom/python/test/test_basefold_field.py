# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the basefold PCS over a field: the same scheme and Eval protocol
as over R_q, run on FieldVector-backed tables and the field RS code. Honest
runs accept (interactive, Fiat-Shamir, and prove/verify on an argument
string), false claims and tampered messages are rejected, the compiled
sumcheck -> basefold chain works, and every whole-codeword step stays a
vector. Layer schedules -- several folds per committed codeword -- accept,
leave the default unchanged, and reject the same tampering. Each case runs over both field implementations that have a transform
-- an extension field and a pseudo-Mersenne one -- since they differ in the
element and vector types the protocol carries as well as in the code."""

import pytest
from vfhe.arith import MLE, Field, FieldVector, MLE_Variable, PseudoMersenneField
from vfhe.crypto import DIGEST_LEN
from vfhe.piop import (
    IOP,
    Proof,
    Protocol,
    Relation,
    Relation_Eval,
    Statement,
    element_digest,
)
from vfhe.piop.protocols import Sumcheck
from vfhe.piop.relations import Relation_Sum
from vfhe.polycom import Basefold, BasefoldEval, FieldFoldableRS
from vfhe.polycom.basefold import Word  # noqa: TC002

_PRIME, _W = 562949948178433, 5  # 50 bits, 2-adicity 20; x^2 - 5 irreducible


@pytest.fixture(params=["extension", "pseudo-mersenne"])
def field(request):
    if request.param == "extension":
        return Field(_PRIME, 2, _W)
    return PseudoMersenneField.generate(260, two_adicity=10)


def _setup(field, num_vars: int, k0: int, c: int, d: int, schedule=None):
    scheme = Basefold(FieldFoldableRS(field, k0=k0, c=c, d=d), schedule)
    variables = [MLE_Variable(f"x{i}") for i in range(num_vars)]
    table = FieldVector(field, 1 << num_vars)
    table.sample_random(b"basefold-field-table")
    f = MLE(field=field, variables=variables, evaluations=table)
    return scheme, f


def _claim(f: MLE, field):
    """A random evaluation point and f's true value there."""
    point = {
        var: field.random_element(f"point-{i}".encode())
        for i, var in enumerate(f.variables)
    }
    value = f.evaluate(point, in_place=False).constant()
    return point, value


def _iop(field, scheme, oracle, opening, rep: int = 4, fiat_shamir=False) -> IOP:
    iop = IOP(domain=field, fiat_shamir=fiat_shamir)
    iop.register(Relation_Eval, BasefoldEval(scheme, rep=rep))
    if opening is not None:
        iop.prover.witnesses[oracle.commitment] = opening
    return iop


def test_commit_then_eval_accepts(field):
    scheme, f = _setup(field, num_vars=4, k0=4, c=4, d=2)
    oracle, opening = scheme.commit(f)
    assert isinstance(opening.word, FieldVector) and len(opening.word) == 64
    assert len(oracle.commitment.root) == DIGEST_LEN and len(opening.tree) == 32
    point, value = _claim(f, field)
    iop = _iop(field, scheme, oracle, opening)
    assert iop.run(
        Statement(Relation_Eval(), oracles=(oracle,), point=point, value=value)
    )
    assert iop.transcript.order == [
        "basefold/g0",
        "basefold/r0",
        "basefold/pi1",
        "basefold/g1",
        "basefold/r1",
        "basefold/h0",
        "basefold/queries",
        "basefold/answers",
    ]
    assert f.num_vars == 4  # the shared opening is folded out of place
    # Round messages are field elements, the base table a field-backed MLE.
    element_cls = type(field.one)
    g0 = iop.transcript.entries["basefold/g0"].result()
    assert all(isinstance(e, element_cls) for e in g0) and len(g0) == 3
    assert isinstance(iop.transcript.entries["basefold/h0"].result().table, FieldVector)


def test_rejects_false_claim_and_wrong_point(field):
    scheme, f = _setup(field, num_vars=4, k0=4, c=4, d=2)
    oracle, opening = scheme.commit(f)
    point, value = _claim(f, field)
    iop = _iop(field, scheme, oracle, opening)
    assert not iop.run(
        Statement(
            Relation_Eval(), oracles=(oracle,), point=point, value=value + field.one
        )
    )
    other_point, _ = _claim(f, field)
    other_point[f.variables[0]] = field.random_element(b"elsewhere")
    iop = _iop(field, scheme, oracle, opening)
    assert not iop.run(
        Statement(Relation_Eval(), oracles=(oracle,), point=other_point, value=value)
    )


def test_fiat_shamir_deterministic_and_prove_verify(field):
    from vfhe.piop import element_digest

    scheme, f = _setup(field, num_vars=4, k0=4, c=4, d=2)
    oracle, opening = scheme.commit(f)
    point, value = _claim(f, field)
    stmt_fields = {"oracles": (oracle,), "point": point, "value": value}
    runs = []
    for _ in range(2):
        iop = _iop(field, scheme, oracle, opening, fiat_shamir=True)
        assert iop.run(Statement(Relation_Eval(), **stmt_fields))
        t = iop.transcript
        runs.append([(lbl, element_digest(t.entries[lbl].result())) for lbl in t.order])
    assert runs[0] == runs[1]

    prover_iop = _iop(field, scheme, oracle, opening, fiat_shamir=True)
    proof = prover_iop.prove(Statement(Relation_Eval(), **stmt_fields))
    checker = _iop(field, scheme, oracle, None, fiat_shamir=True)
    assert not checker.prover.witnesses
    assert checker.verify(Statement(Relation_Eval(), **stmt_fields), proof)
    wrong = Statement(
        Relation_Eval(), oracles=(oracle,), point=point, value=value + field.one
    )
    assert not _iop(field, scheme, oracle, None, fiat_shamir=True).verify(wrong, proof)
    assert not _iop(field, scheme, oracle, None, fiat_shamir=True).verify(
        Statement(Relation_Eval(), **stmt_fields), Proof(proof.messages[:-1])
    )


def test_commit_once_evaluate_many_and_open(field):
    scheme, f = _setup(field, num_vars=4, k0=4, c=4, d=2)
    oracle, opening = scheme.commit(f)
    for i in range(3):
        point = {
            var: field.random_element(bytes([i, j]))
            for j, var in enumerate(f.variables)
        }
        value = f.evaluate(point, in_place=False).constant()
        iop = _iop(field, scheme, oracle, opening)
        assert iop.run(
            Statement(Relation_Eval(), oracles=(oracle,), point=point, value=value)
        )
        assert scheme.openings[oracle.commitment] is opening
    assert scheme.open(oracle.commitment, f)
    other = MLE(field=field, variables=f.variables, evaluations=f.table + field.one)
    assert not scheme.open(oracle.commitment, other)


def test_commit_requires_a_field_or_ring_table(field):
    scheme, f = _setup(field, num_vars=4, k0=4, c=4, d=2)
    plain = MLE(variables=f.variables, evaluations=f.table.to_list())
    with pytest.raises(TypeError, match="ring- or field-backed"):
        scheme.commit(plain)
    with pytest.raises(TypeError, match="ring- or field-backed"):
        scheme.commit(f.to_coefficients())


def test_all_variables_folded_and_deeper_code(field):
    # kappa = 0: the base table is a single element; and d = 3 folds.
    scheme, f = _setup(field, num_vars=3, k0=1, c=16, d=3)
    oracle, opening = scheme.commit(f)
    point, value = _claim(f, field)
    assert _iop(field, scheme, oracle, opening).run(
        Statement(Relation_Eval(), oracles=(oracle,), point=point, value=value)
    )
    scheme, f = _setup(field, num_vars=5, k0=4, c=4, d=3)
    oracle, opening = scheme.commit(f)
    point, value = _claim(f, field)
    assert _iop(field, scheme, oracle, opening).run(
        Statement(Relation_Eval(), oracles=(oracle,), point=point, value=value)
    )


def test_sumcheck_then_basefold_pipeline(field):
    scheme, f = _setup(field, num_vars=3, k0=2, c=8, d=2)
    oracle, _ = scheme.commit(f)
    total = f.table.sum()

    def pipeline_iop() -> IOP:
        iop = IOP(domain=field)
        iop.prover.witnesses[oracle] = f
        iop.register(Relation_Sum, Sumcheck())
        iop.register(Relation_Eval, BasefoldEval(scheme, rep=4))
        return iop

    iop = pipeline_iop()
    assert iop.run(Statement(Relation_Sum(), oracles=[oracle], value=total))
    assert "basefold/0/g0" in iop.transcript.order
    assert not pipeline_iop().run(
        Statement(Relation_Sum(), oracles=[oracle], value=total + field.one)
    )


def _replay_with_tamper(honest: IOP, field, scheme, stmt_fields, tamper):
    """A verifier-only run over the honest transcript with one label's value
    replaced (see test_basefold.py for the rationale)."""
    iop = _iop(field, scheme, None, None)
    for label in honest.transcript.order:
        value = honest.transcript.entries[label].result()
        iop.transcript.write(label, tamper(label, value))
    stmt = Statement(Relation_Eval(), **stmt_fields)
    return iop.loop.run_until_complete(iop.verifier.verify(stmt))


def _honest_run(field):
    scheme, f = _setup(field, num_vars=4, k0=4, c=4, d=2)
    oracle, opening = scheme.commit(f)
    point, value = _claim(f, field)
    honest = _iop(field, scheme, oracle, opening)
    fields = {"oracles": (oracle,), "point": point, "value": value}
    assert honest.run(Statement(Relation_Eval(), **fields))
    return scheme, honest, fields


def test_replay_harness_accepts_an_untampered_transcript(field):
    scheme, honest, fields = _honest_run(field)
    assert _replay_with_tamper(honest, field, scheme, fields, lambda _l, v: v)


def test_rejects_tampered_round_root_and_base_table(field):
    scheme, honest, fields = _honest_run(field)

    def bad_root(label, val):
        return bytes(DIGEST_LEN) if label == "basefold/pi1" else val

    def bad_table(label, val):
        if label == "basefold/h0":
            return MLE(
                field=field, variables=val.variables, evaluations=val.table + field.one
            )
        return val

    assert not _replay_with_tamper(honest, field, scheme, fields, bad_root)
    assert not _replay_with_tamper(honest, field, scheme, fields, bad_table)


def test_rejects_tampered_answer(field):
    scheme, honest, fields = _honest_run(field)

    def tamper(label, val):
        if label == "basefold/answers":
            (pair, path), *rest = val[0]
            bad = ((pair[0] + field.one, pair[1]), path)
            return ((bad, *rest), *val[1:])
        return val

    assert not _replay_with_tamper(honest, field, scheme, fields, tamper)


def test_rejects_a_consistently_committed_wrong_fold(field):
    # Every path verifies (the liar really built a tree over its codeword),
    # so only the fold-consistency check catches the wrong challenge.
    scheme, honest, fields = _honest_run(field)
    opening = scheme.openings[fields["oracles"][0].commitment]
    d = scheme.code.d
    wrong_r = honest.transcript.entries["basefold/r0"].result() + field.one
    # A codeword in whichever form this code produces (see basefold.Word).
    bad_word: Word = scheme.code.fold(opening.word, wrong_r, level=d)
    bad_tree = scheme.merkle_commit(bad_word)
    queries = BasefoldEval(scheme, rep=4).query_positions(
        honest.transcript.entries["basefold/queries"].result()
    )

    def tamper(label, val):
        if label == "basefold/pi1":
            return bad_tree.root
        if label == "basefold/answers":
            return tuple(
                (top, (scheme.code.pair_at(bad_word, j), bad_tree.open(j)))
                for (top, _), j in zip(val, (q // 2 for q in queries), strict=True)
            )
        return val

    assert not _replay_with_tamper(honest, field, scheme, fields, tamper)


# --- layer schedules ---


def _layered_run(field, schedule, rep: int = 4):
    """An honest Fiat-Shamir proof of a depth-4 code under `schedule`."""
    scheme, f = _setup(field, num_vars=6, k0=4, c=4, d=4, schedule=schedule)
    oracle, opening = scheme.commit(f)
    point, value = _claim(f, field)
    fields = {"oracles": (oracle,), "point": point, "value": value}
    iop = _iop(field, scheme, oracle, opening, rep=rep, fiat_shamir=True)
    proof = iop.prove(Statement(Relation_Eval(), **fields))
    return scheme, iop, proof, fields


@pytest.mark.parametrize("schedule", [[2, 2], [4], [1, 3], [3, 1], [2, 1, 1]], ids=str)
def test_layer_schedules_accept_with_one_root_per_layer(field, schedule):
    scheme, iop, proof, fields = _layered_run(field, schedule)
    checker = _iop(field, scheme, None, None, fiat_shamir=True)
    assert checker.verify(Statement(Relation_Eval(), **fields), proof)
    wrong = dict(fields, value=fields["value"] + field.one)
    assert not _iop(field, scheme, None, None, fiat_shamir=True).verify(
        Statement(Relation_Eval(), **wrong), proof
    )
    # A root after the fold that ends each layer but the last.
    ends = [sum(schedule[: i + 1]) for i in range(len(schedule) - 1)]
    roots = [label for label in iop.transcript.order if "/pi" in label]
    assert roots == [f"basefold/pi{k}" for k in ends]
    # Each answer opens one leaf per layer, of 2^s symbols.
    for answer in iop.transcript.entries["basefold/answers"].result():
        assert [len(leaf) for leaf, _ in answer] == [1 << s for s in schedule]


def test_default_schedule_is_one_fold_per_layer(field):
    """No schedule and ``[1] * d`` are the same scheme: same commitment, same
    transcript. A wider first layer changes the commitment, which still
    opens."""
    _, default_iop, _, fields = _layered_run(field, None)
    _, explicit_iop, _, explicit_fields = _layered_run(field, [1, 1, 1, 1])
    root = fields["oracles"][0].commitment.root
    assert explicit_fields["oracles"][0].commitment.root == root
    assert default_iop.transcript.order == explicit_iop.transcript.order
    for label in default_iop.transcript.order:
        assert element_digest(
            default_iop.transcript.entries[label].result()
        ) == element_digest(explicit_iop.transcript.entries[label].result())

    scheme, f = _setup(field, num_vars=6, k0=4, c=4, d=4, schedule=[2, 2])
    oracle, _ = scheme.commit(f)
    assert oracle.commitment.root != Basefold(scheme.code).commit(f)[0].commitment.root
    assert scheme.open(oracle.commitment, f)


def test_schedule_must_be_positive_and_cover_the_depth(field):
    code = FieldFoldableRS(field, k0=4, c=4, d=4)
    for schedule in ([2, 1], [], [0, 4], [2, 3]):
        with pytest.raises(ValueError, match="schedule"):
            Basefold(code, schedule)


def _layered_honest(field, schedule):
    """An honest interactive run under `schedule`, for `_replay_with_tamper`."""
    scheme, f = _setup(field, num_vars=6, k0=4, c=4, d=4, schedule=schedule)
    oracle, opening = scheme.commit(f)
    point, value = _claim(f, field)
    honest = _iop(field, scheme, oracle, opening)
    fields = {"oracles": (oracle,), "point": point, "value": value}
    assert honest.run(Statement(Relation_Eval(), **fields))
    return scheme, honest, fields


def test_layer_schedule_rejects_tampering(field):
    scheme, honest, fields = _layered_honest(field, [2, 2])
    assert _replay_with_tamper(honest, field, scheme, fields, lambda _l, v: v)

    def first_leaf(change):
        """Replace the first query's top-layer leaf by `change(leaf)`."""

        def tamper(label, val):
            if label != "basefold/answers":
                return val
            (top, rest), *others = val
            leaf, path = top
            return (((change(leaf), path), rest), *others)

        return tamper

    def bump(leaf):
        changed = FieldVector.concat([leaf])
        changed[1] = changed[1] + field.one
        return changed

    def bad_root(label, val):
        return bytes(DIGEST_LEN) if label == "basefold/pi2" else val

    for tamper in (
        first_leaf(bump),
        first_leaf(lambda leaf: leaf.query(range(2))),
        first_leaf(lambda leaf: tuple(leaf.to_list())),
        bad_root,
    ):
        assert not _replay_with_tamper(honest, field, scheme, fields, tamper)


def test_layer_schedule_rejects_a_consistently_committed_wrong_fold(field):
    """The second layer's codeword folded with a wrong first challenge and
    committed honestly: every path verifies, and only the check that the
    first layer's refold reappears in the second layer's leaf can fail."""
    scheme, honest, fields = _layered_honest(field, [2, 2])
    code, entries = scheme.code, honest.transcript.entries
    opening = scheme.openings[fields["oracles"][0].commitment]
    r0 = entries["basefold/r0"].result() + field.one
    r1 = entries["basefold/r1"].result()
    folded_once: Word = code.fold(opening.word, r0, level=4)
    bad_word: Word = code.fold(folded_once, r1, level=3)
    bad_tree = scheme.merkle_commit(bad_word, 2)
    proto = BasefoldEval(scheme, rep=4)
    queries = proto.query_positions(entries["basefold/queries"].result())

    def tamper(label, val):
        if label == "basefold/pi2":
            return bad_tree.root
        if label == "basefold/answers":
            return tuple(
                (top, (code.fiber_at(bad_word, j, 4), bad_tree.open(j)))
                for (top, _), j in zip(
                    val, (proto._walk(q)[1][0] for q in queries), strict=True
                )
            )
        return val

    assert not _replay_with_tamper(honest, field, scheme, fields, tamper)


def test_layer_schedule_checks_the_value_entering_each_layer(field):
    """The second layer's codeword changed inside one queried fiber, at the
    walk's offset, and compensated at another offset so that the fiber still
    refolds to the honest value: the paths verify and the walk lands on the
    base codeword, so only the check that the first layer's refold reappears
    in the second layer's leaf can fail."""
    scheme, honest, fields = _layered_honest(field, [2, 2])
    code, entries = scheme.code, honest.transcript.entries
    rs = [entries[f"basefold/r{s}"].result() for s in range(4)]
    opening = scheme.openings[fields["oracles"][0].commitment]
    folded_once: Word = code.fold(opening.word, rs[0], level=4)
    word: Word = code.fold(folded_once, rs[1], level=3)
    proto = BasefoldEval(scheme, rep=4)
    queries = proto.query_positions(entries["basefold/queries"].result())
    leaf, offset = proto._walk(queries[0])[1]
    other = (offset + 1) % 4

    def refold_weight(t):
        unit = FieldVector(field, 4)
        unit[t] = field.one
        return code.fold_fibers(unit, rs[2:], 2, [leaf])[0]

    # The refold is linear in the fiber: moving `offset` by one and `other` by
    # -w_offset / w_other leaves it unchanged.
    word[4 * leaf + offset] = word[4 * leaf + offset] + field.one
    word[4 * leaf + other] = (
        word[4 * leaf + other] - refold_weight(offset) * refold_weight(other).inverse()
    )
    tree = scheme.merkle_commit(word, 2)

    def tamper(label, val):
        if label == "basefold/pi2":
            return tree.root
        if label == "basefold/answers":
            return tuple(
                (top, (code.fiber_at(word, j, 4), tree.open(j)))
                for (top, _), j in zip(
                    val, (proto._walk(q)[1][0] for q in queries), strict=True
                )
            )
        return val

    assert not _replay_with_tamper(honest, field, scheme, fields, tamper)


class _TwoClaims(Relation):
    """Evaluation claims on one oracle at two points (synthetic)."""

    name = "two-claims"
    fields = ("oracles", "points", "values")


class _SplitClaims(Protocol):  # _TwoClaims -> its two Relation_Eval claims
    reduces_from = _TwoClaims
    reduces_to = (Relation_Eval,)

    async def prove(self, prover, statements):
        return self._go(statements)

    async def verify(self, verifier, statements):
        return self._go(statements)

    def _go(self, statements):
        (s,) = statements
        return [
            self.reduce(statements, oracles=s.oracles, point=p, value=v)
            for p, v in zip(s.points, s.values, strict=True)
        ]


def test_layer_schedule_fiat_shamir_with_two_claims(field):
    """Two claims on one commitment, opened one after the other: the second
    claim's challenges are drawn after the first claim's answers, so the
    chain absorbs a layer's fibers, vectors over the field."""
    scheme, f = _setup(field, num_vars=6, k0=4, c=4, d=4, schedule=[2, 2])
    oracle, opening = scheme.commit(f)
    (p0, v0), (p1, v1) = _claim(f, field), _claim(f, field)
    p1 = {var: value + field.one for var, value in p1.items()}
    v1 = f.evaluate(p1, in_place=False).constant()

    def fresh() -> IOP:
        iop = IOP(domain=field, fiat_shamir=True)
        iop.register(_TwoClaims, _SplitClaims())
        iop.register(Relation_Eval, BasefoldEval(scheme, rep=4))
        return iop

    def statement(values):
        return Statement(
            _TwoClaims(), oracles=(oracle,), points=[p0, p1], values=values
        )

    prover = fresh()
    prover.prover.witnesses[oracle.commitment] = opening
    proof = prover.prove(statement([v0, v1]))
    answers = [label for label, _ in proof.messages if label.endswith("/answers")]
    assert len(answers) == 2
    assert fresh().verify(statement([v0, v1]), proof)
    assert not fresh().verify(statement([v0, v1 + field.one]), proof)
