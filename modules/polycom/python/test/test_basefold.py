# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Tests for the basefold PCS over R_q with Merkle-committed codewords:
commit once / evaluate many (the commitment is a succinct root, reused
across single-use IOPs), honest runs accept — standalone on a
commitment-carrying Relation_Eval and chained from sumcheck, whose
oracle-carrying eval claims BasefoldEval resolves to commitments — false
claims are rejected in-round, and every tamperable surface is caught: a
round root, the base table, the commitment root, and an authentication
answer. Dishonest-prover cases replay an honest transcript against a
verifier-only run, since an in-process prover is honest by construction."""

import pytest
from vfhe.arith import Polynomial, Ring
from vfhe.piop import (
    IOP,
    MLE,
    Merkle,
    MLE_Variable,
    Proof,
    Relation_Eval,
    Relation_Sum,
    Statement,
    Sumcheck,
)
from vfhe.piop.merkle import DIGEST_LEN
from vfhe.polycom import (
    Basefold,
    BasefoldCommitment,
    BasefoldEval,
    FoldableRS,
)
from vfhe.polycom.basefold import pair_digest


def _setup(num_vars: int, k0: int, c: int, d: int, prime_size=None):
    ring = Ring(1024, prime_size=prime_size or [49], split_degree=4)
    scheme = Basefold(FoldableRS(ring, k0=k0, c=c, d=d))
    variables = [MLE_Variable(f"x{i}") for i in range(num_vars)]
    f = MLE(
        ring=ring,
        variables=variables,
        evaluations=[ring.random_element() for _ in range(1 << num_vars)],
    )
    return ring, scheme, f


def _claim(f: MLE, ring: Ring):
    """A random evaluation point and f's true value there."""
    point = dict(
        zip(f.variables, [ring.random_exceptional() for _ in f.variables], strict=True)
    )
    value = f.evaluate(point, in_place=False).constant()
    return point, value


def _iop(ring: Ring, scheme: Basefold, commitment, opening, rep: int = 4) -> IOP:
    iop = IOP(domain=ring)
    iop.register(Relation_Eval, BasefoldEval(scheme, rep=rep))
    if opening is not None:
        iop.prover.witnesses[commitment] = opening
    return iop


def test_basefold_fiat_shamir_deterministic():
    # End-to-end FS: challenges and query coins derived from the transcript
    # chain (seeded with the commitment-carrying root statement); two runs
    # of the same claim produce byte-identical transcripts.
    from vfhe.piop import element_digest

    ring, scheme, f = _setup(num_vars=4, k0=4, c=4, d=2)
    com, opening = scheme.commit(f)
    point, value = _claim(f, ring)
    runs = []
    for _ in range(2):
        iop = IOP(domain=ring, fiat_shamir=True)
        iop.register(Relation_Eval, BasefoldEval(scheme, rep=4))
        iop.prover.witnesses[com] = opening
        stmt = Statement(Relation_Eval(), commitment=com, point=point, value=value)
        assert iop.run(stmt)
        t = iop.transcript
        runs.append(
            [(label, element_digest(t.entries[label].result())) for label in t.order]
        )
    assert runs[0] == runs[1]


def test_basefold_prove_then_verify_proof():
    # The non-interactive shape: the prover alone emits an argument string,
    # and a second IOP — no prover, no witnesses, no codeword, just the
    # commitment-carrying statement — checks it.
    ring, scheme, f = _setup(num_vars=4, k0=4, c=4, d=2)
    com, opening = scheme.commit(f)
    point, value = _claim(f, ring)

    def fresh() -> IOP:
        iop = IOP(domain=ring, fiat_shamir=True)
        iop.register(Relation_Eval, BasefoldEval(scheme, rep=4))
        return iop

    prover_iop = fresh()
    prover_iop.prover.witnesses[com] = opening
    stmt = Statement(Relation_Eval(), commitment=com, point=point, value=value)
    proof = prover_iop.prove(stmt)
    assert len(proof) == len(prover_iop.transcript.order) - len(
        prover_iop.transcript.derived
    )

    checker = fresh()
    assert not checker.prover.witnesses
    assert checker.verify(stmt, proof)

    # A false claim over the same proof, and a false proof over the claim.
    wrong = Statement(Relation_Eval(), commitment=com, point=point, value=value + 1)
    assert not fresh().verify(wrong, proof)
    assert not fresh().verify(stmt, Proof(proof.messages[:-1]))


def test_basefold_commit_then_eval_accepts():
    ring, scheme, f = _setup(num_vars=4, k0=4, c=4, d=2)
    com, opening = scheme.commit(f)
    point, value = _claim(f, ring)
    iop = _iop(ring, scheme, com, opening)
    stmt = Statement(Relation_Eval(), commitment=com, point=point, value=value)
    assert iop.run(stmt)
    # Canonical write order: the rounds publish roots (pi_k), the base table
    # goes in the clear, and only then does the query phase run — the
    # verifier cannot spot-check a root without the prover's answers.
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
    # The round message is a root, not a codeword — and the query message
    # is the raw bit-string challenge, expanded into positions by the
    # protocol on both sides.
    assert isinstance(iop.transcript.entries["basefold/pi1"].result(), bytes)
    assert isinstance(iop.transcript.entries["basefold/queries"].result(), bytes)
    # The shared opening survives the prover's in-place folds untouched.
    assert f.num_vars == 4


def test_commitment_is_a_succinct_root():
    # The point of the Merkle layer: the commitment's size is one digest,
    # whatever the codeword length — it used to be the whole codeword.
    _, small, f_small = _setup(num_vars=4, k0=4, c=4, d=2)  # n_d = 64
    _, big, f_big = _setup(num_vars=4, k0=4, c=16, d=2)  # n_d = 256
    com_small, _ = small.commit(f_small)
    com_big, _ = big.commit(f_big)
    assert len(com_small.root) == len(com_big.root) == DIGEST_LEN
    # And the tree commits to pairs, so it has half a codeword's leaves.
    assert len(small.openings[com_small].tree) == small.code.n_d // 2


def test_basefold_commit_once_evaluate_many():
    # The PCS amortization: one commitment (one encode, one tree), several
    # evaluation claims at different points and different moments (separate
    # single-use IOPs), all against the same commitment and opening.
    ring, scheme, f = _setup(num_vars=4, k0=4, c=4, d=2)
    com, opening = scheme.commit(f)
    for _ in range(3):
        point, value = _claim(f, ring)
        iop = _iop(ring, scheme, com, opening)
        stmt = Statement(Relation_Eval(), commitment=com, point=point, value=value)
        assert iop.run(stmt)
        assert stmt.commitment is com  # no re-commitment anywhere
        assert scheme.openings[com] is opening  # nor re-encoding


def test_basefold_rejects_false_claim():
    ring, scheme, f = _setup(num_vars=4, k0=4, c=4, d=2)
    com, opening = scheme.commit(f)
    point, _ = _claim(f, ring)
    iop = _iop(ring, scheme, com, opening)
    wrong = ring.random_element()
    assert not iop.run(
        Statement(Relation_Eval(), commitment=com, point=point, value=wrong)
    )


def test_basefold_opening_witness_is_required_and_typed():
    ring, scheme, f = _setup(num_vars=4, k0=4, c=4, d=2)
    com, _ = scheme.commit(f)
    point, value = _claim(f, ring)
    stmt = {"commitment": com, "point": point, "value": value}

    # commit() records the opening on the scheme and BasefoldEval falls back
    # to it, so "missing" means neither the prover nor the scheme holds it
    # (e.g. a prover that did not run commit).
    del scheme.openings[com]
    iop = _iop(ring, scheme, com, opening=None)  # witness never installed
    with pytest.raises(LookupError, match=r"prover\.witnesses"):
        iop.run(Statement(Relation_Eval(), **stmt))

    # The polynomial alone is no longer enough: the codeword and its tree
    # are part of the opening now.
    iop = _iop(ring, scheme, com, opening=f)
    with pytest.raises(TypeError, match="BasefoldOpening"):
        iop.run(Statement(Relation_Eval(), **stmt))


def test_basefold_open_algorithm():
    ring, scheme, f = _setup(num_vars=4, k0=4, c=4, d=2)
    com, _ = scheme.commit(f)
    assert scheme.open(com, f)
    other = MLE(
        ring=ring,
        variables=f.variables,
        evaluations=[ring.random_element() for _ in range(16)],
    )
    assert not scheme.open(com, other)


def test_basefold_all_variables_folded():
    # kappa = 0: the base code has a single-element message.
    ring, scheme, f = _setup(num_vars=2, k0=1, c=16, d=2)
    com, opening = scheme.commit(f)
    point, value = _claim(f, ring)
    iop = _iop(ring, scheme, com, opening)
    assert iop.run(Statement(Relation_Eval(), commitment=com, point=point, value=value))


def test_basefold_multi_prime_ring():
    ring, scheme, f = _setup(num_vars=3, k0=2, c=8, d=2, prime_size=[49, 49])
    com, opening = scheme.commit(f)
    point, value = _claim(f, ring)
    iop = _iop(ring, scheme, com, opening)
    assert iop.run(Statement(Relation_Eval(), commitment=com, point=point, value=value))


def test_basefold_deeper_code():
    # d = 3: three folds, three roots (commitment + pi1 + pi2), and a query
    # walk three levels deep.
    ring, scheme, f = _setup(num_vars=4, k0=2, c=8, d=3)
    com, opening = scheme.commit(f)
    point, value = _claim(f, ring)
    iop = _iop(ring, scheme, com, opening)
    assert iop.run(Statement(Relation_Eval(), commitment=com, point=point, value=value))
    assert "basefold/pi2" in iop.transcript.order


def _replay_with_tamper(honest: IOP, ring, scheme, stmt_fields, tamper):
    """A verifier-only run over the honest transcript with one label's value
    replaced: recorded challenges and query positions are returned as-is by
    `challenge` / `challenge_bits`, so this replays the interaction exactly
    with a single dishonest message."""
    iop = _iop(ring, scheme, None, None)
    for label in honest.transcript.order:
        value = honest.transcript.entries[label].result()
        iop.transcript.write(label, tamper(label, value))
    stmt = Statement(Relation_Eval(), **stmt_fields)
    return iop.loop.run_until_complete(iop.verifier.verify(stmt))


def _honest_run(num_vars=4, k0=4, c=4, d=2):
    ring, scheme, f = _setup(num_vars=num_vars, k0=k0, c=c, d=d)
    com, opening = scheme.commit(f)
    point, value = _claim(f, ring)
    honest = _iop(ring, scheme, com, opening)
    fields = {"commitment": com, "point": point, "value": value}
    assert honest.run(Statement(Relation_Eval(), **fields))
    return ring, scheme, honest, fields


def test_basefold_rejects_tampered_round_root():
    ring, scheme, honest, fields = _honest_run()

    def tamper(label: str, val):
        if label == "basefold/pi1":
            return bytes(DIGEST_LEN)  # a root nothing opens to
        return val

    assert not _replay_with_tamper(honest, ring, scheme, fields, tamper)


def test_basefold_rejects_tampered_base_table():
    ring, scheme, honest, fields = _honest_run()
    one = Polynomial(ring).from_array([1])

    def tamper(label: str, val):
        if label == "basefold/h0":
            return MLE(
                ring=ring,
                variables=val.variables,
                evaluations=[p + one for p in val.table],
            )
        return val

    assert not _replay_with_tamper(honest, ring, scheme, fields, tamper)


def test_basefold_rejects_tampered_commitment():
    # The commitment is instance data: a statement carrying a root the
    # answers do not open under fails the very first path check.
    ring, scheme, honest, fields = _honest_run()
    fields["commitment"] = BasefoldCommitment(
        fields["commitment"].variables, bytes(DIGEST_LEN)
    )
    assert not _replay_with_tamper(honest, ring, scheme, fields, lambda _, v: v)


def test_basefold_rejects_tampered_answer():
    # Tampering an authenticated pair breaks its path against the root it
    # was committed under — this is what the Merkle layer buys.
    ring, scheme, honest, fields = _honest_run()
    one = Polynomial(ring).from_array([1])

    def tamper(label: str, val):
        if label == "basefold/answers":
            first, *rest = val
            (lo, hi), path = first[0]
            return ((((lo + one, hi), path), *first[1:]), *rest)
        return val

    assert not _replay_with_tamper(honest, ring, scheme, fields, tamper)


def test_basefold_rejects_answer_from_the_wrong_position():
    # A pair that is genuinely in the tree, but not at the queried position:
    # the path is checked against the index the verifier chose itself, which
    # is why MerklePath deliberately does not carry one.
    ring, scheme, honest, fields = _honest_run()
    opening = scheme.openings[fields["commitment"]]
    proto = BasefoldEval(scheme, rep=4)
    queries = proto.query_positions(
        honest.transcript.entries["basefold/queries"].result()
    )
    other = 1 if queries[0] != 1 else 2

    def tamper(label: str, val):
        if label == "basefold/answers":
            first, *rest = val
            moved = (scheme.code.pair_at(opening.word, other), opening.tree.open(other))
            return (((moved, *first[1:])), *rest)
        return val

    assert not _replay_with_tamper(honest, ring, scheme, fields, tamper)


def test_pair_leaves_authenticate_both_fold_operands():
    # One leaf, one path, both operands of a fold check: the leaf digest is
    # over the pair, so the tree has n/2 leaves and Merkle.verify accepts
    # the pair as a unit.
    _, scheme, f = _setup(num_vars=4, k0=4, c=4, d=2)
    com, opening = scheme.commit(f)
    tree = opening.tree
    assert len(tree) == len(opening.word) // 2 and tree.root == com.root
    for i in (0, 3, len(tree) - 1):
        pair = scheme.code.pair_at(opening.word, i)
        assert Merkle.verify(com.root, i, tree.open(i), pair, hash=pair_digest)
        # The pair is ordered: swapping the two entries is a different leaf.
        assert not Merkle.verify(
            com.root, i, tree.open(i), (pair[1], pair[0]), hash=pair_digest
        )


def test_sumcheck_then_basefold_pipeline():
    # The compiled chain: Relation_Sum --Sumcheck--> Relation_Eval
    # --BasefoldEval. The eval claim carries only the oracle (sumcheck
    # inherits it); BasefoldEval resolves the commitment from the scheme's
    # records — committed before the run, before any challenge fixes the
    # evaluation point — and installs its opening as the witness itself.
    ring, scheme, f = _setup(num_vars=3, k0=2, c=8, d=2)
    scheme.commit(f)
    total = None
    for p in f.table:
        total = p.copy() if total is None else total + p

    def pipeline_iop() -> IOP:
        iop = IOP(domain=ring)
        iop.register(Relation_Sum, Sumcheck())
        iop.register(Relation_Eval, BasefoldEval(scheme, rep=4))
        return iop

    iop = pipeline_iop()
    assert iop.run(Statement(Relation_Sum(), oracles=[f], value=total))
    # DAG: sumcheck's eval claim is child /0, discharged in place — no
    # bridge level in the paths.
    assert "basefold/0/g0" in iop.transcript.order
    assert iop.transcript.order.index("basefold/0/g0") > iop.transcript.order.index(
        "sumcheck/r2"
    )

    iop = pipeline_iop()
    wrong = total + Polynomial(ring).from_array([1])
    assert not iop.run(Statement(Relation_Sum(), oracles=[f], value=wrong))


def test_basefold_requires_a_committed_oracle():
    # An oracle-carrying eval claim whose oracle was never committed cannot
    # be resolved to a commitment.
    ring, scheme, f = _setup(num_vars=3, k0=2, c=8, d=2)
    total = None
    for p in f.table:
        total = p.copy() if total is None else total + p
    iop = IOP(domain=ring)
    iop.register(Relation_Sum, Sumcheck())
    iop.register(Relation_Eval, BasefoldEval(scheme, rep=4))
    with pytest.raises(LookupError, match="never committed"):
        iop.run(Statement(Relation_Sum(), oracles=[f], value=total))


def test_replay_harness_accepts_an_untampered_transcript():
    # The positive control for every tamper test above: replaying the honest
    # transcript unchanged must accept, or those tests would pass vacuously.
    ring, scheme, honest, fields = _honest_run()
    assert _replay_with_tamper(honest, ring, scheme, fields, lambda _label, v: v)


def test_basefold_rejects_a_consistently_committed_wrong_fold():
    # The strongest liar the Merkle layer has to survive: every path is
    # valid, because the prover really did build a tree over the codeword it
    # answers from — but it folded with the wrong challenge. Only the
    # fold-consistency check (not the path check) can catch this.
    ring, scheme, honest, fields = _honest_run()
    opening = scheme.openings[fields["commitment"]]
    d = scheme.code.d
    r0 = honest.transcript.entries["basefold/r0"].result()
    wrong_r = r0 + Polynomial(ring).from_array([1])
    bad_word = scheme.code.fold(opening.word, wrong_r, level=d)
    bad_tree = scheme.merkle_commit(bad_word)
    queries = BasefoldEval(scheme, rep=4).query_positions(
        honest.transcript.entries["basefold/queries"].result()
    )

    def tamper(label: str, val):
        if label == "basefold/pi1":
            return bad_tree.root  # a real root, of the wrong codeword
        if label == "basefold/answers":
            # Keep the honest level-d answers; re-answer level 1 from the
            # bad tree, so every path verifies against the root above it.
            return tuple(
                (top, (scheme.code.pair_at(bad_word, j), bad_tree.open(j)))
                for (top, (_, _)), j in zip(val, (q // 2 for q in queries), strict=True)
            )
        return val

    assert not _replay_with_tamper(honest, ring, scheme, fields, tamper)


def test_query_positions_expand_deterministically_and_distinctly():
    # The sampler is a pure function of the seed: same positions on both
    # sides; in range; and rejection-sampled so their projections to the
    # level-0 codeword are pairwise distinct — a repeated bottom position
    # would re-run the same final fold check and add no soundness.
    _, scheme, _f = _setup(num_vars=4, k0=4, c=4, d=2)
    proto = BasefoldEval(scheme, rep=scheme.code.n0 - 1)  # worst legal case
    seed = bytes(range(32))
    positions = proto.query_positions(seed)
    assert positions == proto.query_positions(seed)  # deterministic
    assert proto.query_positions(bytes(32)) != positions  # seed-dependent
    assert len(positions) == scheme.code.n0 - 1
    d = scheme.code.d
    bases = [q >> (d - 1) for q in positions]
    assert all(0 <= q < scheme.code.n_d // 2 for q in positions)
    assert len(set(bases)) == len(bases)  # distinct in the smallest codeword


def test_rep_must_fit_the_base_codeword():
    # The distinct positions live in the level-0 codeword, so there must be
    # more of it than queries — otherwise rejection sampling cannot finish.
    _, scheme, _f = _setup(num_vars=4, k0=4, c=4, d=2)  # n0 = 16
    with pytest.raises(ValueError, match="base codeword"):
        BasefoldEval(scheme, rep=scheme.code.n0)
    with pytest.raises(ValueError, match="base codeword"):
        BasefoldEval(scheme)  # the default rep = 32 exceeds n0 = 16 too
