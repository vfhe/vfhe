<!-- SPDX-FileCopyrightText: 2026 The vFHE Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# vfhe.crypto

The basic cryptographic primitives every other module builds on: randomness,
hashing and a vector commitment.

The randomness is two kinds: `Random`, the source a scheme's samplers draw
from, the unpredictable entropy stream by default or a stream a seed
reproduces, and `seeded_below_many`, values that are a pure function of
`(context, seed)` and their index. Use `Random` where a value must be
unguessable and `seeded_below_many` where another party must recompute it from
a transcript; `context` is a domain-separation tag, a fixed string literal per
call site. This is the library's only randomness: nothing else in the tree
draws from `secrets`, `random`, or the OS directly. The entropy stream may be
drawn from on any number of threads at once: each thread keeps a pool of its
own, and a forked child discards the pool it inherits. A failing seed source
exits the process; a predictable key is never returned.

Every concern is one unit at every level: a header in `c/include/`, its source
in `c/src/`, the cdef in `python/cdef/` listing what Python calls of it, and a
subpackage of `vfhe.crypto`, imported on its own as util's are:
`from vfhe.crypto.random import Random`, `from vfhe.crypto.merkle import Merkle`.

## Layout

- `python/src/vfhe/crypto/random/`: `Random` (`bytes(n)`, `below(bound)`,
  `below_many(count, bound)`), the source a scheme's samplers draw from: the
  entropy stream without a seed, a replayable stream with one, so
  `MLWE_Scheme(..., random=Random(seed))` is a reproducible scheme; and
  `seeded_below_many(count, bound, context, seed)`, value `i` located by
  its index.
- `python/src/vfhe/crypto/hash/`: `DIGEST_LEN`, `digest` (BLAKE3 of a byte
  string; encoding a value as bytes is the caller's) and the chain
  `h_i = H(h_{i-1} | part_i)` as the accumulator `DigestChain`.
- `python/src/vfhe/crypto/merkle/`: `Merkle`, the binary tree over leaf
  digests packed end to end (`root`, `open(index)`, static
  `verify(root, index, path, digest)`), an opening being the sibling digests
  packed in one `bytes`; and `join_digests`, which hashes several packed
  digest arrays together index by index, so one tree commits to all of them.
- `python/bench/`: the module's benchmarks, `hash` and `randomness`,
  installed as `vfhe.crypto.bench` and timed by `python -m vfhe.benchmark`.
- `python/cdef/random.h`, `seeded.h`, `hash.h`, `merkle.h`: what Python may
  call of each concern.
- `c/include/`, `c/src/`: `random` (the source every sampler takes: `NULL`
  for the entropy stream, seeded from RDRAND or `getentropy` and expanded by
  the `stream` kernel, or a seeded BLAKE3 XOF), `seeded` (the indexed
  sampler over BLAKE3), `normal` (Box-Muller, from the stream or from a
  secret seed), `expand` (AES-128 in counter mode, AES-NI/VAES or table-based
  on the portable engine, and the seed expander built on it: values below a
  bound from a public seed, identically on every engine), `hash` (one-shot
  digests, the batch across the SIMD lanes, the incremental hasher, the
  digest chain) and `merkle` (the tree).
- `c/src/kernels/`: the pieces with an implementation per instruction set,
  one small header each: `aes128.h` (the keystream), `filter.h` (the words
  of a keystream run kept below a bound), `seed.h` (the entropy stream's
  32-byte seed) and `stream.h` (that seed expanded into the stream), with
  one source per family: `portable/` (a table AES, one loop, `getentropy`,
  a BLAKE3 XOF), `x86_64/` (AES-NI with VAES, AVX-512, RDRAND, AES-CTR), and
  `aarch64/` when NEON kernels exist. A file whose instructions the engine
  lacks compiles to nothing; `gates.h` beside them is the gates they
  test. One helper sits beside the sources: `below.h`, the mask every
  sampler below a bound starts from; zeroing seeds and keys once used is
  util's `wipe.h`.

## Dependencies

`crypto` includes `util`'s headers and nothing above it: the ISA gates come
from `x86_64.h`, allocation from `alloc.h` and zeroing from `wipe.h`. `arith` draws its samples and
digests here; `piop` and `polycom` their transcripts and commitments.
