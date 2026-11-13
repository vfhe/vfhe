# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for `IOP.and_composition` and `IOP.batched_call` (piop.md §5).

A toy protocol composes several sub-protocols, each over a residue field
of a ring: in round j a sub-protocol sends its previous challenge plus j and then asks for a
challenge, and the verifier checks the message. The tests check:

- the transcript order (each round's messages in sub-protocol order, then
  one seed);
- that every sub-protocol receives the challenge derived from the round's
  seed in its own field;
- acceptance interactively and as a non-interactive argument;
- rejection of a tampered message and of two sub-protocols' messages
  swapped;
- that a sub-protocol yielding in the middle of its turn does not change
  the chain;
- sub-protocols of different lengths and a composition inside one;
- that a failing sub-protocol stops the others, leaving no task behind;
- `batched_call`, and a round where the sub-protocols disagree on what
  they reach.
"""

import asyncio

import pytest
from vfhe.arith import Ring
from vfhe.piop import (
    IOP,
    Proof,
    Protocol,
    Rejection,
    Relation,
    Statement,
)

RING = Ring(64, prime_size=[28, 50], split_degree=4)
FIELDS = [RING.residue_field(0), RING.residue_field(1)]


class _Flows(Relation):
    """`spec` = ((rounds, field index), ...), one entry per sub-protocol."""

    name = "parts"
    fields = ("spec",)

    def check(self, statement):
        return True


class _Toy(Protocol):
    reduces_from = _Flows
    reduces_to = ()

    def __init__(self, yield_mid=False, lie=None, crash=None, nest=None):
        self.yield_mid = yield_mid
        self.lie = lie  # (part, round): the prover sends a wrong message there
        self.crash = crash  # (part, round): the prover part raises there
        self.nest = nest  # part index that runs two parts of its own
        #: party name -> per part, the challenges it received
        self.received: dict[str, list[list]] = {}

    async def _prove_part(self, iop, label, part, rounds, field):
        received = self.received["prover"][part]
        previous = field.one
        for j in range(rounds):
            message = previous + j
            if self.lie == (part, j):
                message = message + 1
            if self.crash == (part, j):
                raise ValueError("prover part crashed")
            if self.yield_mid:
                await asyncio.sleep(0)
            iop.transcript.write(f"{label}/m{j}", message)
            if self.yield_mid:
                await asyncio.sleep(0)
            previous = await iop.verifier.challenge(f"{label}/c{j}", field)
            received.append(previous)

    async def _verify_part(self, iop, label, part, rounds, field):
        received = self.received["verifier"][part]
        previous = field.one
        for j in range(rounds):
            message = await iop.transcript.read(f"{label}/m{j}")
            if not (message == previous + j):
                raise Rejection(f"{label}: wrong message in round {j}")
            previous = await iop.verifier.challenge(f"{label}/c{j}", field)
            received.append(previous)

    def _parts(self, party, iop, label, spec, half):
        self.received[party] = [[] for _ in spec]
        parts = []
        for i, (rounds, f) in enumerate(spec):
            part = half(iop, f"{label}/f{i}", i, rounds, FIELDS[f])
            if self.nest == i:
                part = self._nested(iop, f"{label}/f{i}", part, half)
            parts.append(part)
        return parts

    async def _nested(self, iop, label, part, half):
        await part
        inner = [half(iop, f"{label}/inner{j}", j, 2, FIELDS[j]) for j in range(2)]
        saved = self.received
        self.received = {k: [[], []] for k in ("prover", "verifier")}
        try:
            await iop.and_composition(f"{label}/inner", inner)
        finally:
            self.received = saved

    async def prove(self, prover, statements):
        (s,) = statements
        label = f"toy{s.path}"
        parts = self._parts("prover", prover.iop, label, s.spec, self._prove_part)
        await prover.iop.and_composition(label, parts)
        return []

    async def verify(self, verifier, statements):
        (s,) = statements
        label = f"toy{s.path}"
        parts = self._parts("verifier", verifier.iop, label, s.spec, self._verify_part)
        await verifier.iop.and_composition(label, parts)
        return []


def _iop(protocol, fiat_shamir=False) -> IOP:
    iop = IOP(domain=RING, fiat_shamir=fiat_shamir)
    iop.register(_Flows, protocol)
    return iop


def _statement(spec) -> Statement:
    return Statement(_Flows(), spec=tuple(spec))


def _accepts(spec, **kwargs) -> bool:
    """Interactive run, and a proof checked by a separate verifier."""
    protocol = _Toy(**kwargs)
    interactive = _iop(protocol).run(_statement(spec))
    proof = _iop(_Toy(**kwargs), True).prove(_statement(spec))
    return interactive and _iop(_Toy(**kwargs), True).verify(_statement(spec), proof)


def _pending(iop: IOP) -> list:
    return [task for task in asyncio.all_tasks(iop.loop) if not task.done()]


def test_each_round_is_the_messages_then_one_seed():
    iop = _iop(_Toy(), True)
    iop.prove(_statement([(2, 0), (2, 1), (2, 0)]))
    assert iop.transcript.order == [
        "toy/f0/m0",
        "toy/f1/m0",
        "toy/f2/m0",
        "toy/round0",
        "toy/f0/m1",
        "toy/f1/m1",
        "toy/f2/m1",
        "toy/round1",
    ]


def test_every_subprotocol_receives_the_seed_read_in_its_field():
    protocol = _Toy()
    iop = _iop(protocol)
    assert iop.run(_statement([(2, 0), (2, 1), (2, 0)]))
    for j in range(2):
        seed = iop.transcript.entries[f"toy/round{j}"].result()
        for party in ("prover", "verifier"):
            parts = protocol.received[party]
            assert parts[0][j] == parts[2][j] == FIELDS[0].random_exceptional(seed=seed)
            assert parts[1][j] == FIELDS[1].random_exceptional(seed=seed)


def test_honest_subprotocols_accept():
    assert _accepts([(3, 0), (3, 1)])


def test_a_wrong_message_in_any_subprotocol_is_rejected():
    for part in range(2):
        for j in range(3):
            assert not _iop(_Toy(lie=(part, j))).run(_statement([(3, 0), (3, 1)]))


def test_tampered_and_reordered_proofs_are_rejected():
    spec = [(2, 0), (2, 1)]
    proof = _iop(_Toy(), True).prove(_statement(spec))
    assert _iop(_Toy(), True).verify(_statement(spec), proof)
    messages = list(proof.messages)
    label, value = messages[2]
    tampered = Proof([*messages[:2], (label, value + 1), *messages[3:]])
    assert not _iop(_Toy(), True).verify(_statement(spec), tampered)
    # The two parts' round-0 messages in the other order.
    swapped = Proof([messages[1], messages[0], *messages[2:]])
    assert not _iop(_Toy(), True).verify(_statement(spec), swapped)


def test_a_subprotocol_yielding_mid_turn_leaves_the_chain_unchanged():
    spec = [(3, 0), (3, 1), (3, 0)]
    plain = _iop(_Toy(), True)
    plain_proof = plain.prove(_statement(spec))
    yielding = _iop(_Toy(yield_mid=True), True)
    assert yielding.prove(_statement(spec)).digest() == plain_proof.digest()
    assert yielding.transcript.state() == plain.transcript.state()
    assert _accepts(spec, yield_mid=True)


def test_subprotocols_of_different_lengths():
    spec = [(3, 0), (1, 1), (2, 0)]
    assert _accepts(spec)
    iop = _iop(_Toy(), True)
    iop.prove(_statement(spec))
    # Part 1 leaves after round 0, part 2 after round 1.
    assert iop.transcript.order == [
        "toy/f0/m0",
        "toy/f1/m0",
        "toy/f2/m0",
        "toy/round0",
        "toy/f0/m1",
        "toy/f2/m1",
        "toy/round1",
        "toy/f0/m2",
        "toy/round2",
    ]


def test_a_composition_inside_a_subprotocol():
    spec = [(2, 0), (2, 1)]
    assert _accepts(spec, nest=0)
    iop = _iop(_Toy(nest=0), True)
    iop.prove(_statement(spec))
    # The inner composition runs during part 0's turn, after its own rounds.
    order = iop.transcript.order
    assert order.index("toy/f0/inner/round0") > order.index("toy/round1")


def test_a_rejecting_subprotocol_stops_the_others():
    iop = _iop(_Toy(lie=(0, 1)))
    assert not iop.run(_statement([(3, 0), (3, 1), (3, 0)]))
    assert _pending(iop) == []


def test_a_crashing_prover_subprotocol_is_raised():
    iop = _iop(_Toy(crash=(1, 1)))
    with pytest.raises(ValueError, match="prover part crashed"):
        iop.run(_statement([(3, 0), (3, 1)]))
    assert _pending(iop) == []


def test_batched_call_runs_one_kernel_over_every_subprotocol():
    iop = IOP(domain=RING)
    calls = []

    def kernel(operands):
        calls.append(list(operands))
        return [10 * x for x in operands]

    async def part(x):
        first = await iop.batched_call("scale", x, kernel)
        second = await iop.batched_call("scale", first + 1, kernel)
        return second

    async def main():
        return await iop.and_composition("c", [part(1), part(2), part(3)])

    assert iop.loop.run_until_complete(main()) == [110, 210, 310]
    assert calls == [[1, 2, 3], [11, 21, 31]]
    # Outside a composition: the kernel on the one operand.
    assert iop.loop.run_until_complete(iop.batched_call("scale", 4, kernel)) == 40
    assert iop.transcript.order == []


def test_subprotocols_reaching_different_steps_raise():
    iop = IOP(domain=RING)

    async def by_batched_call():
        return await iop.batched_call("scale", 1, lambda xs: xs)

    async def by_challenge():
        return await iop.verifier.challenge("x", FIELDS[0])

    async def main():
        return await asyncio.wait_for(
            iop.and_composition("m", [by_batched_call(), by_challenge()]), 5
        )

    with pytest.raises(RuntimeError, match="same batched call"):
        iop.loop.run_until_complete(main())
    assert _pending(iop) == []
