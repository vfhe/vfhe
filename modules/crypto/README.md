<!-- SPDX-FileCopyrightText: 2026 Antonio Guimarães -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# vfhe.crypto

The basic cryptographic primitives every other module builds on: randomness
and a vector commitment.

## Randomness

Two independent generators, both in `c/src/prng.c`, exposed to Python as two
singletons in `python/src/vfhe/crypto/prng.py`:

- `entropy` (`EntropyPRNG`) — the unpredictable stream. Seeds from RDRAND or
  `/dev/urandom` and expands with BLAKE3, which is why this is the one module
  that depends on BLAKE3. `bytes(n)`, `below(bound)`, `normal(sigma)`, and the
  raw entry points behind them.
- `seeded` (`SeededPRNG`) — `below(bound, context, seed)` /
  `below_many(count, ...)` / `bytes(n, ...)`: values uniform in `[0, bound)`
  that are a pure function of `(context, seed)`, so both parties of a protocol
  can recompute them. `context` is a domain-separation tag — a fixed string
  literal per call site.

This is the library's only randomness: nothing else in the tree draws from
`secrets`, `random`, or the OS directly. Use `entropy` where a value must be
unguessable and `seeded` where it must be reproducible from a transcript.

- `c/src/aes_rng.c`: an AES-NI (VAES where the engine has it) CTR keystream,
  used in place of BLAKE3 where the CPU offers AES and the build is not
  portable. Called by `prng.c`, never by a caller directly.
- `c/src/normal.c`: the Box-Muller draw, beside the bytes it consumes.

`prng.c` also carries the deterministic-seed override a probabilistic test
needs, reachable as `entropy.deterministic(seed)` — a context manager that
pins the stream and restores hardware entropy on exit. It affects only the
entropy-backed generator; a seeded draw is already a pure function of its
arguments. Production never calls it.

## Vector commitment

- `c/src/merkle.c` and `python/src/vfhe/crypto/merkle.py`: `Merkle`, a binary
  Merkle tree over BLAKE3 — a commitment to a list of arbitrary Python objects
  (`root`, `open(index)`, static `verify(root, index, path, leaf)`;
  `MerklePath` is the sibling list of one opening, without the index, which
  the verifier supplies itself). The only requirement on a leaf type is a
  `.hash()` method returning its digest (`vfhe.arith.FieldElement` has one); a
  `hash=` callable supplies it for types that do not
  (`hash=Polynomial.get_hash`). Leaf hashing is the Python layer's only
  per-leaf work — the tree itself is built in C.

  It is a primitive, not a protocol: `verify` is a local deterministic check
  with no challenges and no rounds. `vfhe.polycom`'s basefold PCS is its main
  consumer.
