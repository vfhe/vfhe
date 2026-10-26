# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""AND-composition with shared challenges (piop.md §5): the machinery behind
`IOP.and_composition` and `IOP.batched_call`.

`await iop.and_composition(label, subprotocols)` runs several sub-protocols,
one coroutine each on the calling party's side, as their AND-composition:
the composed protocol proves that every sub-protocol's statement holds, and
in each round every sub-protocol sends its messages, then the verifier sends
one challenge, which every sub-protocol receives.

This is the conjunction case of [CDS94, Thm. 8]. There, a proof that a
monotone combination of statements holds runs one public-coin protocol per
statement and splits the verifier's challenge among them with a secret
sharing scheme for the dual combination; for a conjunction the dual
combination needs any one participant, and with Shamir's scheme, which
[CDS94] uses for threshold combinations, every share is the challenge itself.
So every protocol answers the verifier's challenge unchanged: the AND-proof
construction of Σ-protocols, stated and proved in [BS23, §19.7.1,
Thm. 19.18]. [CDS94] states its construction for three-round protocols and
notes that it generalizes to any number of rounds; here the protocols are
multi-round and public-coin, with one shared challenge per round.

It is neither of the framework's other two compositions: the Composer
(`compose`) builds a protocol from layers run one after another, a
sequential composition [KP23, Thm. 5], and the driver's bundles reduce
products of relations, the parallel composition of [KP23, Thm. 6], whose
protocols draw their own challenges.

A sub-protocol is ordinary protocol code: it writes and reads the transcript
as usual and draws its challenges with
`await iop.verifier.challenge(label, domain)`, which, inside an
AND-composition, waits for the other sub-protocols instead of drawing at
once. Nothing in a sub-protocol's code says it is composed, so an existing
protocol's half runs unchanged as one.

**Rounds.** Round j is the j-th challenge request of every sub-protocol still
running, whatever its label: requests are matched by position, not by name.
A sub-protocol that returns leaves the composition, and the others go on
without it. One seed is drawn per round, as the bit-string challenge
``{label}/round{j}`` of the transcript, once every sub-protocol has written
its round-j messages. Each sub-protocol receives the challenge
``domain.random_exceptional(seed=seed)`` for the domain it asked in, so
sub-protocols over one domain receive one value. A protocol may pass its own
`challenge_sampler(seed, domain)` instead, for instance to read one ring
challenge in a different residue field per sub-protocol. A request for a
bit-string challenge (`challenge_bits`) receives bytes expanded from the
same seed.

**Order.** The sub-protocols take turns, in the order they are listed: one
runs until it requests a challenge or returns, and then the next one runs.
So the transcript order depends only on the sub-protocols' code, never on
scheduling, which Fiat-Shamir needs (the prover's write order is the chain
the verifier recomputes, and a stored proof is read back in that order).

**Soundness.** Each sub-protocol's statement must be fixed before the
composition starts (a precondition of `IOP.and_composition`, not checked).
Then when one of the statements is false, it is false before any challenge
is drawn, and accepting requires that sub-protocol to accept, so the
soundness error of the composition is the largest of the sub-protocols'
errors -- for each, computed with the challenge
set it actually receives (round by round as well, as [BCFW25] proves for
parallel copies of an interactive oracle reduction). Sharing the challenges
gives no amplification: composing copies of one protocol here never lowers
its error.

**Rules for sub-protocols.**

- A sub-protocol must not wait for something that only another one produces:
  it holds the turn while it runs, so such a wait never ends. This is the
  independence that the composition assumes.
- A protocol called inside a sub-protocol shares its challenges with the
  other sub-protocols by position too, as the sub-protocol's own code does.
- An AND-composition inside a sub-protocol draws its seeds directly, during
  that sub-protocol's turn; its rounds are not rounds of the outer one.
- With fresh randomness (an interactive run), a sub-protocol's challenge is
  still derived from the round's seed.

`await iop.batched_call(step, operand, kernel)` is the other synchronization
point: every sub-protocol hands in an operand, and `kernel` is called once on
all of them (one batched kernel in place of a call per sub-protocol); it
writes nothing to the transcript. Outside an AND-composition it is the
kernel on the one operand.
"""

from __future__ import annotations

import asyncio
import contextvars
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from .iop import IOP

#: The AND-composition and sub-protocol index of the running task: each
#: sub-protocol is a task with its own copy of the context, so the binding
#: follows it into every protocol it calls.
_current_subprotocol: contextvars.ContextVar[tuple[ANDComposition, int] | None] = (
    contextvars.ContextVar("vfhe_piop_and_composition", default=None)
)


def current_subprotocol(iop: IOP) -> tuple[ANDComposition, int] | None:
    """The AND-composition on `iop`, and the sub-protocol index in it, that
    the running task belongs to; None outside one."""
    bound = _current_subprotocol.get()
    if bound is None or bound[0].iop is not iop:
        return None
    return bound


class _Request(NamedTuple):
    """What a sub-protocol waits on at the end of its turn. `kind` is
    `challenge` (`label` the challenge's, `argument` its domain), `bits`
    (`argument` the byte count) or `batched_call` (`label` the step,
    `argument` this sub-protocol's operand, `kernel` the function run once
    on every sub-protocol's operand)."""

    kind: str
    label: str
    argument: Any
    kernel: Any = None


class ANDComposition:
    """The state of one `IOP.and_composition` call: the turn order, the
    requests of the current round, and each sub-protocol's received
    challenges."""

    def __init__(
        self,
        iop: IOP,
        label: str,
        challenge_sampler: Callable[[bytes, Any], Any] | None,
    ):
        self.iop = iop
        self.label = label
        self.challenge_sampler = challenge_sampler
        self.rounds = 0
        self._live: list[int] = []
        self._turns: list[asyncio.Future] = []
        # Per sub-protocol, the request it is waiting on in the current round,
        # and the value the round gave it.
        self._requests: dict[int, _Request] = {}
        self._values: dict[int, Any] = {}
        # Per sub-protocol, label -> challenge, so that a challenge requested
        # twice by one sub-protocol is one value, as `Verifier.challenge`
        # records it outside a composition.
        self._received: list[dict[str, Any]] = []
        # The first failure of a sub-protocol, which `run` re-raises.
        self._failure: BaseException | None = None

    def _pass_turn(self, index: int) -> None:
        """Hand the turn to the next running sub-protocol after `index`; past
        the last one, close the round and start the next with the first."""
        later = [j for j in self._live if j > index]
        if later:
            self._turns[later[0]].set_result(None)
            return
        if not self._requests:
            return  # every sub-protocol has returned
        self._close_round()
        self._turns[self._live[0]].set_result(None)

    def _close_round(self) -> None:
        """Answer the round's requests: run the batched call, or draw the
        round's seed and give every sub-protocol its value from it: a
        challenge from `challenge_sampler` (by default the domain's
        `random_exceptional`), and a bit-string challenge as bytes expanded
        from ``b"bits"`` followed by the seed."""
        requests, self._requests = self._requests, {}
        kinds = {request.kind for request in requests.values()}
        if "batched_call" in kinds:
            steps = {request.label for request in requests.values()}
            if kinds != {"batched_call"} or len(steps) != 1:
                raise RuntimeError(
                    f"{self.label}: the sub-protocols do not all reach the same "
                    "batched call"
                )
            order = sorted(requests)
            kernel = requests[order[0]].kernel
            results = list(kernel([requests[i].argument for i in order]))
            if len(results) != len(order):
                raise RuntimeError(
                    f"{self.label}: batched call {requests[order[0]].label!r} "
                    f"returned {len(results)} results for {len(order)} "
                    "sub-protocols"
                )
            for i, result in zip(order, results, strict=True):
                self._values[i] = result
            return
        from .fs import expand_bytes  # fs imports party, which imports this module

        seed = self.iop.verifier._recorded_bits(f"{self.label}/round{self.rounds}", 256)
        self.rounds += 1
        sampler = self.challenge_sampler
        for i, request in requests.items():
            if request.kind == "bits":
                self._values[i] = expand_bytes(b"bits" + seed, request.argument)
            elif sampler is None:
                self._values[i] = request.argument.random_exceptional(seed=seed)
            else:
                self._values[i] = sampler(seed, request.argument)

    async def _request(self, index: int, request: _Request) -> Any:
        """Wait for the round to close, holding no turn meanwhile."""
        self._requests[index] = request
        self._turns[index] = self.iop.loop.create_future()
        self._pass_turn(index)
        await self._turns[index]
        return self._values.pop(index)

    async def challenge(self, index: int, label: str, domain) -> Any:
        received = self._received[index]
        if label not in received:
            received[label] = await self._request(
                index, _Request("challenge", label, domain)
            )
        return received[label]

    async def challenge_bits(self, index: int, label: str, nbytes: int) -> bytes:
        received = self._received[index]
        if label not in received:
            received[label] = await self._request(
                index, _Request("bits", label, nbytes)
            )
        return received[label]

    async def batched_call(self, index: int, step: str, operand, kernel) -> Any:
        return await self._request(
            index, _Request("batched_call", step, operand, kernel)
        )

    async def run(self, subprotocols: list[Coroutine]) -> list:
        n = len(subprotocols)
        self._live = list(range(n))
        self._turns = [self.iop.loop.create_future() for _ in range(n)]
        self._received = [{} for _ in range(n)]

        async def run_subprotocol(index: int, subprotocol: Coroutine) -> Any:
            _current_subprotocol.set((self, index))
            try:
                await self._turns[index]
            except asyncio.CancelledError:
                subprotocol.close()  # never started
                raise
            try:
                result = await subprotocol
            except BaseException as error:
                # A failing sub-protocol keeps the turn: one resumed now would
                # run until it is cancelled below, possibly in the middle of a
                # transcript read.
                if self._failure is None:
                    self._failure = error
                    for turn in self._turns:
                        turn.cancel()
                raise
            self._live.remove(index)
            self._pass_turn(index)
            return result

        tasks = [
            self.iop.loop.create_task(run_subprotocol(i, subprotocol))
            for i, subprotocol in enumerate(subprotocols)
        ]
        if tasks:
            self._turns[0].set_result(None)
        try:
            pending = set(tasks)
            while pending:
                _, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_EXCEPTION
                )
                if self._failure is not None:
                    raise self._failure
            return [task.result() for task in tasks]
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
