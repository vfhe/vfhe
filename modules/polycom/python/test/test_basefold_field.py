# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Tests for the basefold PCS over a field: the same scheme and Eval protocol
as over R_q, run on FieldVector-backed tables and the field RS code. Honest
runs accept (interactive, Fiat-Shamir, and prove/verify on an argument
string), false claims and tampered messages are rejected, the compiled
sumcheck -> basefold chain works, and every whole-codeword step stays a
vector. Each case runs over both field implementations that have a transform
-- an extension field and a pseudo-Mersenne one -- since they differ in the
element and vector types the protocol carries as well as in the code."""

import pytest
from vfhe.arith import Field, FieldVector, PseudoMersenneField
from vfhe.piop import (
    IOP,
    MLE,
    MLE_Variable,
    Proof,
    Relation_Eval,
    Relation_Sum,
    Statement,
    Sumcheck,
)
from vfhe.piop.merkle import DIGEST_LEN
from vfhe.polycom import Basefold, BasefoldEval, FieldFoldableRS

_PRIME, _W = 562949948178433, 5  # 50 bits, 2-adicity 20; x^2 - 5 irreducible


@pytest.fixture(params=["extension", "pseudo-mersenne"])
def field(request):
    if request.param == "extension":
        return Field(_PRIME, 2, _W)
    return PseudoMersenneField.generate(260, two_adicity=10)


def _setup(field, num_vars: int, k0: int, c: int, d: int):
    scheme = Basefold(FieldFoldableRS(field, k0=k0, c=c, d=d))
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


def _iop(field, scheme, commitment, opening, rep: int = 4, fiat_shamir=False) -> IOP:
    iop = IOP(domain=field, fiat_shamir=fiat_shamir)
    iop.register(Relation_Eval, BasefoldEval(scheme, rep=rep))
    if opening is not None:
        iop.prover.witnesses[commitment] = opening
    return iop


def test_commit_then_eval_accepts(field):
    scheme, f = _setup(field, num_vars=4, k0=4, c=4, d=2)
    com, opening = scheme.commit(f)
    assert isinstance(opening.word, FieldVector) and len(opening.word) == 64
    assert len(com.root) == DIGEST_LEN and len(opening.tree) == 32
    point, value = _claim(f, field)
    iop = _iop(field, scheme, com, opening)
    assert iop.run(Statement(Relation_Eval(), commitment=com, point=point, value=value))
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
    com, opening = scheme.commit(f)
    point, value = _claim(f, field)
    iop = _iop(field, scheme, com, opening)
    assert not iop.run(
        Statement(Relation_Eval(), commitment=com, point=point, value=value + field.one)
    )
    other_point, _ = _claim(f, field)
    other_point[f.variables[0]] = field.random_element(b"elsewhere")
    iop = _iop(field, scheme, com, opening)
    assert not iop.run(
        Statement(Relation_Eval(), commitment=com, point=other_point, value=value)
    )


def test_fiat_shamir_deterministic_and_prove_verify(field):
    from vfhe.piop import element_digest

    scheme, f = _setup(field, num_vars=4, k0=4, c=4, d=2)
    com, opening = scheme.commit(f)
    point, value = _claim(f, field)
    stmt_fields = {"commitment": com, "point": point, "value": value}
    runs = []
    for _ in range(2):
        iop = _iop(field, scheme, com, opening, fiat_shamir=True)
        assert iop.run(Statement(Relation_Eval(), **stmt_fields))
        t = iop.transcript
        runs.append([(lbl, element_digest(t.entries[lbl].result())) for lbl in t.order])
    assert runs[0] == runs[1]

    prover_iop = _iop(field, scheme, com, opening, fiat_shamir=True)
    proof = prover_iop.prove(Statement(Relation_Eval(), **stmt_fields))
    checker = _iop(field, scheme, com, None, fiat_shamir=True)
    assert not checker.prover.witnesses
    assert checker.verify(Statement(Relation_Eval(), **stmt_fields), proof)
    wrong = Statement(
        Relation_Eval(), commitment=com, point=point, value=value + field.one
    )
    assert not _iop(field, scheme, com, None, fiat_shamir=True).verify(wrong, proof)
    assert not _iop(field, scheme, com, None, fiat_shamir=True).verify(
        Statement(Relation_Eval(), **stmt_fields), Proof(proof.messages[:-1])
    )


def test_commit_once_evaluate_many_and_open(field):
    scheme, f = _setup(field, num_vars=4, k0=4, c=4, d=2)
    com, opening = scheme.commit(f)
    for i in range(3):
        point = {
            var: field.random_element(bytes([i, j]))
            for j, var in enumerate(f.variables)
        }
        value = f.evaluate(point, in_place=False).constant()
        iop = _iop(field, scheme, com, opening)
        assert iop.run(
            Statement(Relation_Eval(), commitment=com, point=point, value=value)
        )
        assert scheme.openings[com] is opening
    assert scheme.open(com, f)
    other = MLE(field=field, variables=f.variables, evaluations=f.table + field.one)
    assert not scheme.open(com, other)


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
    com, opening = scheme.commit(f)
    point, value = _claim(f, field)
    assert _iop(field, scheme, com, opening).run(
        Statement(Relation_Eval(), commitment=com, point=point, value=value)
    )
    scheme, f = _setup(field, num_vars=5, k0=4, c=4, d=3)
    com, opening = scheme.commit(f)
    point, value = _claim(f, field)
    assert _iop(field, scheme, com, opening).run(
        Statement(Relation_Eval(), commitment=com, point=point, value=value)
    )


def test_sumcheck_then_basefold_pipeline(field):
    scheme, f = _setup(field, num_vars=3, k0=2, c=8, d=2)
    scheme.commit(f)
    total = f.table.sum()

    def pipeline_iop() -> IOP:
        iop = IOP(domain=field)
        iop.register(Relation_Sum, Sumcheck())
        iop.register(Relation_Eval, BasefoldEval(scheme, rep=4))
        return iop

    iop = pipeline_iop()
    assert iop.run(Statement(Relation_Sum(), oracles=[f], value=total))
    assert "basefold/0/g0" in iop.transcript.order
    assert not pipeline_iop().run(
        Statement(Relation_Sum(), oracles=[f], value=total + field.one)
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
    com, opening = scheme.commit(f)
    point, value = _claim(f, field)
    honest = _iop(field, scheme, com, opening)
    fields = {"commitment": com, "point": point, "value": value}
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
    opening = scheme.openings[fields["commitment"]]
    d = scheme.code.d
    wrong_r = honest.transcript.entries["basefold/r0"].result() + field.one
    bad_word = scheme.code.fold(opening.word, wrong_r, level=d)
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
