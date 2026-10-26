<!-- SPDX-FileCopyrightText: 2026 The vFHE Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until 1.0.0, minor
versions may contain breaking changes.

## [Unreleased]

### Added

- Python 3.15 support, tested in CI, built as wheels and declared in the
  classifiers.
- Add per-module benchmarks, `make bench`: Python benchmarks of a module's
  API in `modules/<mod>/python/bench/` and C benchmarks of its kernels in
  `modules/<mod>/c/bench/`, timed per engine and thread limit into a results
  file that `tools/bench/compare.py` compares with another. The complete
  depth of `make test` runs every benchmark once.
- `RNSRing.crt_components(elements, index)`, `from_crt_components(vectors)`
  and `residue_field(index)`: the CRT components of ring elements at one RNS
  prime p -- their residues modulo the maximal ideals `P_s = (p, X^d - w_s)`
  of `R_q` containing p, all carried into the residue field `R_q/P_0` by the
  isomorphisms the automorphisms `X -> X^(x_s)` induce -- as one extension
  field vector, and back (`polynomial_RNS_crt_components`,
  `polynomial_RNS_from_crt_components` and `rns_crt_maps_new` in C).
- `BasefoldEval` opens several evaluation claims together, at any points
  and on any of its scheme's commitments: one sumcheck reduces them to a
  common point and one basefold run opens a random linear combination of
  the polynomials there ([CCCFGS26, §7, Figs. 3-4]). The sumcheck is skipped
  when the points coincide, the combination when there is one polynomial.
  `soundness_error` takes the bundle's `claims` and `polynomials`.
- `FieldFoldableRS.residue_code(code, index)`: the code a foldable code over
  `R_q` induces on the residue field at one of its primes -- the code whose
  codewords are the CRT components there of the ring code's codewords, with
  the same tables over the prime subfield; folding commutes with taking
  components.
- `FoldableRS(..., twist_field="field")`: the general code over `R_q` with
  its twists in the residue fields `F_(p^d)` of the ring's primes rather than
  in `F_p` -- one `FieldFoldableRS` per prime (`residue_codes`), whose
  codewords are the CRT components of the ring code's. The distance bound
  runs on `log2 p^d` bits; a twist multiply costs an `F_(p^d)` product.
  `"prime"`, the integer tables, stays the default.
- `FieldFoldableRS.encode_batch(messages, count)`, `fold_batch(words,
  challenges, level)` and `fold_fibers_batch(values, challenges, level,
  indices)`: several codewords encoded or folded in one call, each fold with
  its own challenge.
- `FieldVector.fold_twisted_blocks(untwist, r)`: `fold_twisted` of
  several blocks sharing one table, block `b` with its own element `r[b]`, in
  one pass (`field_vec_fold_twisted_blocks` in C).
- `FieldFoldableRS(..., root_field=)` (`ROOT_FIELDS`): where the general
  code's base transform may take its root of unity -- `"prime"`, `F_p` only
  (the default), or `"field"`, the extension when `F_p` has none.
- `RNSRing.crt_components(..., element_major=True)`: the components of each
  element together, entry `b * k + s`.
- `vfhe.polycom.BatchFoldEval` ([CCCFGS26, §7]): a second Eval protocol for
  basefold commitments over `R_q`. The claims are carried to the CRT
  components at each prime and proved there, over the residue field, by the
  common-point sumcheck, a random linear combination of the components and
  one basefold run over the residue code. The primes run in lockstep, one
  shared ring challenge per round read at every prime as its CRT component
  0. The ring commitments serve every prime, and the runs share one set of
  query positions, the ring leaves answering every prime's first layer.
  Each later fold layer is committed by one Merkle tree for all primes,
  answered with every prime's fiber and one path.
  `soundness_error` reports one prime's error over the smallest residue
  field: a false bundle fails at a prime fixed before any challenge, so no
  factor for the number of primes is needed.
- `RNSRing.crt_linear_combination(parts, index, coefficients)`
  (`polynomial_RNS_crt_linear_combination` in C): a linear combination,
  with coefficients in the residue field, of the CRT components of several
  arrays of ring elements, computed as inner products of the elements'
  mul-domain words with one table per array and output coefficient, without
  forming the components (the precomputation of HELIOPOLIS [ACGS24],
  ePrint 2023/1949, App. C). BatchFold forms its combinations with it.
- `vfhe.crypto.join_digests(parts)` (`merkle_join_digests` in C): leaf
  digests joining several packed digest arrays index by index, so one
  Merkle tree commits to several codewords of one length.
- `IOP.and_composition(label, subprotocols, challenge_sampler=None)`: runs
  sub-protocols as their AND-composition, one challenge per round shared by
  all of them. They take turns in a fixed order, so the transcript order
  depends only on the protocols; an existing protocol's half runs unchanged
  as a sub-protocol. `IOP.batched_call(step, operand, kernel)` hands every
  sub-protocol's operand to one kernel call.
- `Ring.random_exceptional(seed=None)`, `Polynomial.sample_exceptional(
  seed=None)` and `Field.random_exceptional(seed=None, bound=None)`: the
  exceptional samplers take an optional seed, and the field's an optional
  bound on its coefficients. Both draw their coefficients uniformly with
  `vfhe.crypto.seeded` (or `entropy` without a seed), so a seeded ring
  element and the bounded field element from the same seed have the same
  coefficients.
- `Verifier.challenge(label, domain)`: a challenge from the exceptional set
  of `domain` rather than of the IOP's domain (the Fiat-Shamir verifier
  derives it the same way).
- `Protocol.bundle_key(oracle)`: what a bundle protocol groups the claims on
  an oracle by -- the oracle by default; `BasefoldEval` returns its scheme,
  so the driver hands it every pending claim on the oracles committed under
  it, parking each until no other statement can produce another.
- `vfhe.piop.relations.Relation_Lookup` / `vfhe.piop.protocols.Lookup` and
  `Relation_Range` / `RangeDecomposition`: the lookup argument (offline
  memory checking over four grand products, two degree-3 zerochecks over
  virtual oracles) and the range check over packed `R_q` polynomials that
  reduces to it through the base-`beta` digits of the coefficients, as in
  [CCCFGS26]'s appendix; `modules/piop/range.md` is the design note.
- `vfhe.crypto.digest`: `chain_digest`, the chained digest of
  already-digested parts, and `bytes_digest`, `int_digest`, `str_digest`,
  the canonical digests of primitives every `digest()` method is built
  from; `DigestChain`, the same chain as an accumulator, which absorbs a
  run of equal-length parts packed in one buffer in a single native call
  (`hash_chain` in C).
- `RNSPolynomial.digest_sequence(polys)`: the BLAKE3 digest of the rows of
  several polynomials as one message (`polynomial_RNS_hash_update` in C).
- `MLE.digest` covers the entries with one digest of the whole table --
  `FieldVector.hash()` over a field, `digest_sequence` over a ring -- rather
  than one per entry.
- `Basefold.commit(f, prover=prover)` installs on `prover` the table of the
  committed oracle and the opening of its commitment.
- `Basefold(code, schedule)`: a layer schedule groups the code's `d` folds
  into layers of `schedule[i]` folds, positive counts summing to `d`. Only
  the codeword each layer starts from is committed, by a Merkle tree whose
  leaves are its fibers of `2^schedule[i]` adjacent symbols, so a schedule
  of fewer layers builds fewer trees and opens fewer paths per query. The
  commitment depends on `schedule[0]`; the default `[1] * d` is the scheme
  as before, with the same commitments and transcripts.
  `Basefold.merkle_commit(word, folds)` builds the tree over `2^folds`-symbol
  fibers. `Basefold.soundness_error` is the [ZCF24] bound, stated for the
  default schedule.
- `FieldFoldableRS.fold_fibers(values, challenges, level, indices)` and
  `FoldableRS.fold_fibers`: fibers of `2^s` adjacent symbols, end to end,
  folded `s = len(challenges)` times to the one symbol each determines --
  over a field as whole-vector operations. With them, on both codes,
  `fiber_at(word, i, size)`, `join_fibers(fibers, size)` and
  `leaf_digests(word, size)` for fibers of `size` symbols.
- `FieldVector.digest()`: a 32-byte digest of the vector, its `hash()` under
  a tag of its own, so a vector can appear in a transcript entry or a
  statement field.
- Oracles committed under different commitment schemes, or under one scheme
  with different parameters (two basefold codes), in one IOP: a
  `CommittedOracle` carries its `scheme`, a scheme's evaluation protocol
  declares it as `Protocol.scheme`, and the registry keys the protocol by
  it, so each claim on a committed oracle reaches the protocol of the
  scheme it was committed under. A claim with no entry for its scheme
  falls back to the entry of its kind, then to the default; a claim on a
  plain oracle is refused (`TypeError`) once a scheme is registered for its
  relation.
- `Statement` refuses a bare table among its `oracles` (a `TypeError`
  naming `TableOracle`), and `VirtualOracle` a bare table as a constituent.

- Add indexed relations to `vfhe.piop`: a relation declares `index_fields`,
  its index is then a mapping over exactly those names, and the oracles
  among its values (`Relation.index_oracles`, `statement_oracles`) are
  oracles of every statement of the relation -- read by the parking rule,
  the Composer's rules and routing, and the prover view, and digested in the
  declared order (`Relation.index_digest`). `Relation.head_index` may name
  an index field, and a statement then derives its head from that entry.
  `SparseMLE.restrict` slices a sparse table at 0/1 constants;
  `VirtualOracle.restriction` reads one member of a family of tables packed
  into one master oracle over slot and table variables, and `resolve_table`
  resolves it to a slice. New in `vfhe.piop.sparse`: `FamilyShape`, the
  shape of such a family; `Relation_SpEval`, the abstract sparse-evaluation
  relation "the sparse master is what the dense index oracle encodes", every
  encoding being a subclass with its own `compute_head` and a protocol
  registered against its `Start`; `dense_relation`, deriving from a relation
  with sparse index oracles the relation whose index holds the dense oracle,
  the shape and the realization; `dense_layering`, its two-layer protocol.
  `Relation_Circuit` is now that relation for circuits: the wiring predicates
  committed through one dense index oracle, GKR running in the first layer.
- Add sequential composition to `vfhe.piop`: a statement may carry a `head`,
  an implicit oracle (`Implicit`) its relation computes from the statement's
  other oracles (`Relation.head_index`, `compute_head`); `with_start` derives
  the start relation `R.Start` (the relation with its head bound, plus
  evaluation claims on it) and `R.start(statement, claims)`; `compose` /
  `Composed` build a relation's protocol from a layered reduction, computing
  the heads deepest layer first and starting each headed statement with the
  claims on its head. `Relation_SumDef` / `SumDefStart` are the sum
  definition `L(z) = sum_b G(z, b)` and its composable protocol;
  `Relation_Layer`, `Relation_Equal` / `OutputTest` and `GKR_Start` are the
  pieces of GKR as a composition.
- Speed up `Polynomial.get_coeff_matrix` / `from_coeff_matrix` on an
  unsplit ring: a row is now read with `ffi.unpack` and written with one
  packed `ffi.memmove` instead of element by element. `BFV_Scheme.encode` /
  `decode` at N=4096 go from 1.9 / 2.1 ms to 0.27 / 0.29 ms.
- Add `Polynomial.convert_base(ring, out, exact)`: a value re-expressed over
  another base, which need not contain this one -- the general move
  `base_extend` is the special case of. The default is the fast conversion
  already behind `base_extend`, which writes `x + u * M`; `exact=True` removes
  that overflow, for about a tenth again of the cost, for a value the caller
  keeps clear of the ends of its range. Existing conversions are unchanged.
- Add `Polynomial.rescale_to_power_of_two(k)`: `round(2^k * c / q)` for every
  coefficient, on centered representatives, as `N` machine words. It is the
  rescale for a target modulus that is not a product of base primes -- how an
  element leaves for a scheme that works at a power of two -- which
  `round_division` and `scaled_lift` between them cannot express. Exact, with
  no floating point anywhere in the rounding.
- Add `FieldVector.sample_random_at(seed, indices)`: the seeded elements at
  scattered positions of the sequence `sample_random` fills from, in one call.
- Add `Field.digest_elements(elements)`: the digest of a few elements without
  building a vector to hold them.
- Add `FieldVector.padding_unit`, the width `view`'s `start` and `length` must
  be multiples of -- stated by `view` before, but not obtainable.
- Add `FieldVector.fma_interleave(a, b, c_even, c_odd)`: `a + b * c_even` and
  `a + b * c_odd` written straight into one vector of twice the length.
- Add `FieldVector.fold_twisted(untwist, r)`: a vector of adjacent pairs
  folded to half its length in one call, `out[i] = (w[2i] + w[2i+1]) / 2 +
  (w[2i] - w[2i+1]) * untwist[i] * r` -- the inverse of `lift_twisted` at
  `untwist = 1 / (2 t)`, with two extension products per pair.
- Add `layout=` to `ExtensionFieldNTT.pack` / `unpack`: a caller says
  whether it holds a batch by blocks (the default, as before) or interleaved,
  and the conversion to and from the plan's own layout follows from that and
  the domain.
- Add `FieldVector.lift_twisted(a, b, twist, out=None)`: `a + b * t` and
  `a - b * t` interleaved into one vector of twice the length, as one kernel
  call that forms the product once per pair and reads a table shorter than `a`
  cyclically rather than tiled. The inverse move to `fold_twisted`.
- Add prime-subfield tables to the extension field's vectors: `mul`, `fma`,
  `fold_twisted` and `lift_twisted` take a vector over the degree-1 field with
  the same prime and multiply plane-wise (`field_vec_mul_plane` /
  `field_vec_fma_plane`: `d` prime-field products per element in place of the
  extension product).
- Add `mod_eltwise_inner_product_addto` and
  `polynomial_matrix_mul_addto_RNS_polynomial` (C, `arith.h`): element-wise
  inner products with one reduction per element, and the product `a * b^T` of
  matrices of ring elements. Also `polynomial_new_RNS_polynomial_rows`, a
  polynomial with only its mask's rows allocated.
- Add `MerklePath.from_bytes(packed)`, and `FoldableRS.fold_pairs` /
  `leaf_digests_of` for a set of positions rather than one.
- Add `entropy.below_many(count, bound)` (`vfhe.crypto`; native
  `generate_uniform_below`): values exactly uniform in `[0, bound)` by
  rejection sampling, so a draw reveals only how many values were redrawn.
- Add `prng_normal_seeded` (C): `generate_normal_random`'s rounded draw as a
  function of a seed, for parallel kernels that must not depend on their
  thread count.
- Add seed expansion (`expand.c`): `prng_expand_key` derives a key from a seed
  and a label, and `prng_expand_below` / `prng_expand_below32` expand it into
  values below a bound by AES-128 in counter mode (`prng_aes128_ctr`), the
  same sequence on every engine. It replaces `aes_rng.c` as the entropy
  stream's AES-NI expander. Also `Blake3Stream` (`hash_stream_*`),
  incremental BLAKE3.
- Add native multithreading under one library-wide limit:
  `vfhe.util.parallel.set_num_threads(n)` / `num_threads()` (C `vfhe_set_num_threads`
  / `vfhe_num_threads`), defaulting to `VFHE_NUM_THREADS` or else 1 (single
  threaded). Operations taking
  `n_threads` use up to the limit by default and never exceed it: single
  operations (key switch, automorphism, multiplication, relinearization,
  round division and rescale, domain conversions) split over the ring's
  primes from N = 4096 up, and batches over their items.
  `vfhe.util.parallel.local_num_threads(n)` (C `vfhe_set_local_num_threads` /
  `vfhe_local_num_threads`) limits the threads of the calls one thread makes,
  and a parallel call inside another gets only its share of the outer call's
  threads (`vfhe_share_for`), so nesting never goes past the limit. Results
  do not depend on the thread count. The native loop is `vfhe_parallel_for`
  (util's `parallel.h`), with helper threads kept asleep between calls. The complex-FFT
  batch, which always used 8 threads, follows the limit too. A CKKS rotation
  at N = 2^14 over 17 primes runs about 6x faster on 96 cores than on one.
- Add a memory pool for the buffers the library allocates and frees
  repeatedly (`vfhe.util.memory_pool`; C `mempool_*`, in util's `memory_pool.h`): a released RNS
  polynomial row, field or pseudo-Mersenne vector, or complex polynomial is
  kept by the releasing thread and handed out again for the next request of
  its size. At N = 2^14 the CKKS batch calls went from thousands of minor page
  faults per call to none. `set_capacity(n_bytes | "auto" | None)` bounds what
  it keeps: `"auto"` (the default) at most the library's peak use, so a
  process keeps its peak working set resident; `0` turns it off.
  `release_all()` hands back what it keeps, and `statistics()` reports it
  (`peak_outstanding_bytes` may miss a short-lived peak by a few MiB per
  thread). `set_wipe_on_release(True)` zeroes every buffer as it is released
  (off by default). `VFHE_MEMPOOL_CAPACITY` and
  `VFHE_MEMPOOL_WIPE_ON_RELEASE=1` set the defaults. A sanitized build
  bypasses the pool.
- Add serialization, `vfhe.util.io`: the type classes `Encoder`, `Decoder` and
  `Codec`. `codec.encode(obj, f)` writes an object to a binary stream and
  `codec.decode(f)` reads one back, consuming exactly the bytes its encoder
  wrote (`decode_strict` also refuses trailing bytes), so codecs compose: `ListCodec`, `DictCodec`, `TupleCodec` and
  `OptionalCodec` build one codec from others, and `IntCodec`, `FloatCodec`,
  `BoolCodec`, `StrCodec` and `BytesCodec` cover plain values. A vfhe codec is
  built from the codecs of what its objects hold --
  `KeySwitchKeyCodec(SampleCodec(MLWESchemeCodec(scheme), packing="tight"))` --
  and streams a key of any size one sample at a time, its scheme written once.
  A ring or scheme codec
  built with the caller's objects hands them back where the stream's
  `parameters` match, so a loaded ciphertext refers to the scheme holding the
  keys; one built with none rebuilds them from the stream, once.
  `PolynomialCodec(rings, packing, domain)` and `SampleCodec(schemes, seeded,
  packing, domain)` choose tight or word rows, the rows' domain, and whether a
  fresh sample is written as its 32-byte mask seed, which halves it. A stream
  names its row packing and seed expansion, and a reader refuses ones it does
  not know. Every residue is checked against its prime on load. The format is
  `vfhe.util.io.protobuf.ObjectMessageCodec`, a length-prefixed protobuf message per
  object, with the schemas under `proto/vfhe/<module>/io/v1/`, checked by
  `buf`. Codecs cover RNS rings and polynomials, MLWE and MGSW schemes,
  samples, key-switch keys, MGSW ciphertexts, LWE samples, the secret keys
  (`MLWEKeyCodec`, `LWEKeyCodec`), the CKKS and BFV schemes, CKKS
  ciphertexts, `SAB_Key`, `CGGI16_Key` and `CKKS_LinearTransform`.
- Give rings and schemes value equality: `RNSRing`, `MLWE_Scheme` and its
  subclasses, and `MGSW_Scheme` compare and hash by `parameters`, a frozen
  value of what keys and ciphertexts depend on (degree, split degree and primes
  for a ring; class, ring chains, rank and a subclass's own parameters for a
  scheme). Two objects over the same primes are one ring, whichever holds it;
  `vfhe.util.io` binds a stored scheme to the caller's by the same value.
- Add `MLWE_Scheme.key_gen(sigma_err, distribution="ternary")`: a key with
  every coefficient uniform in `{-1, 0, 1}`.
- Add `MLWE(..., obj=)`: adopt a native sample.
- Add a documentation site, `make docs`, published from `main` to GitHub Pages:
  the guides, the Python API from the docstrings and the C API from the
  Doxygen blocks, built with warnings as errors.
- Add `MLWE.from_components(scheme, a, b, lvl=None)`: the sample with mask
  `a` and body `b`, copied from the polynomials (which may be temporaries, in
  either domain), over a level of the scheme or another of its rings.
- Add `MLWE.mod_reduce(ring | lvl)`: a ciphertext reduced into a smaller
  level in place, the value kept rather than divided -- CKKS's level drop.
- Add `vfhe.util.kernels`: compile your own C kernels *into* the library. Your
  sources and vfhe's are one translation-unit set and one link with link-time
  optimisation, so a loop of yours calling `add_modq` inlines it exactly as
  vfhe's own loops do, and the process still holds one copy of the library.
  `kernels.ExtensionBuilder(sources, declarations).build()` does it in place: the engine it builds
  carries the prebuilt engine's module name and goes first on `sys.path`, so
  `vfhe.util.bindings` imports it in place of the prebuilt one. It must run
  before anything imports vfhe's C -- a loaded extension cannot be replaced, so
  a late call raises. `python -m vfhe.util.kernels` builds the same artefact and
  exits, so putting its directory on `PYTHONPATH` removes the ordering rule and
  lets an image ship without a compiler. Builds are cached under
  `$VFHE_BUILD_DIR`, else `~/.cache/vfhe`, and ninja rebuilds only what a change
  actually reached.
- Add a `kernels` extra: `pip install vfhe[kernels]` pulls in meson and ninja, which
  `vfhe.util.kernels` drives. Only a custom build needs it -- the default path loads
  a prebuilt engine -- and building without it now says so instead of failing
  with `FileNotFoundError`.
- Ship the library's C in wheels and sdists as `vfhe/_source/kernels-src.tar.gz`,
  which is what `vfhe.util.kernels` compiles against: the sources, the public
  headers and the cdefs, and nothing else. It adds about 350 KB to a wheel.
- Add `MLWE.multiply_by_monomial(exponent, minus_one=False)`: the sample
  times `X^exponent` (or `X^exponent - 1`), any integer exponent, keeping its
  ring (special primes included), level, type (a CKKS ciphertext keeps its
  `delta`) and representation. And `MGSW.multiply_by_monomial`, the same on
  every row, which turns an MGSW of `m` into one of `m * X^exponent`.
- Add hoisted automorphisms: `MLWE_Scheme.automorphisms(c, gens, ksks)`
  applies several automorphisms to one ciphertext, decomposing it against the
  gadget once for every key switch (both gadgets): 16 rotations at N = 2^14
  are about 1.5x faster. Natively `mlwe_hoist` /
  `mlwe_automorphism_RNSc_GHS_hoisted`, and the automorphism of a fully split
  ring's NTT representation, `polynomial_RNS_automorphism_index` /
  `polynomial_RNS_permute`.
- Add `MLWE_Scheme.automorphism_batch(cts, gens, ksks)`, one automorphism
  each on several ciphertexts in either domain, and
  `MLWE_Scheme.automorphism_sum(cts, gens, ksks)`: `sum_i
  Aut_gens[i](cts[i])` with a single division by the special primes for all
  the key switches.
- Add the automorphism multiplexer (the B-to-1 mux-rotate of [CHKS25]):
  `MLWE_Scheme.automorphism_mux(c, gens, ksks)` computes `sum_d b_d Aut_d(c)`
  for bits `b_d` encrypted in the keys (`gen_automorphism_mux_keys`),
  decomposing `c` once and dividing by the special primes once;
  `automorphism_mux_batch` runs many. Also `gadget_mul_addto_automorphism`.
- Add `MLWE_Scheme.linear_combination(cts, coefficients)` /
  `linear_combinations(cts, rows)`: ciphertexts combined with plaintext
  coefficients in one pass, for one or several rows of coefficients, and
  `mlwe.PlaintextMatrix`, such rows prepared once for reuse.
- Add `MLWE_Scheme.multiply_batch(lhs, rhs, ksk, n_threads, *, lvl=None)`:
  several products in one native call, each also round-divided to level
  `lvl` when given; and `round_division_batch(cts, lvl)`.
- Add `MLWE_Scheme.trace_pack(vec, ksk, n_threads=0)` (`mlwe_trace_pack`):
  the trace packing of [CDKS21] with each level's difference halved before
  its automorphism, after [LY26]'s reverse homomorphic trace. The packed
  values keep their scale, and each level's key-switch noise is added once
  instead of being doubled by every level above it: at N = 256, 5 instead of
  12 bits of noise packing 256 samples. Each level's packings run in
  parallel.
- Add `MLWE_Scheme.normalized_trace(c, ksk, n=1)` (`mlwe_normalized_trace`):
  the normalized trace onto the subring `Z[X^(N/n)]` [AP13], each level
  halved before its automorphism as in [LY26] -- the coefficients at the
  multiples of `N / n` at their scale and zero elsewhere, with the noise of
  `log2(N / n)` key switches. It takes the ascending automorphism keys of
  `trace_pack`; `trace` keeps its factor `N` and its keys.
- Add ring switching [GHPS12] between `R_N` and `R_n` when one dimension
  divides the other, `R_n` being `Z[X^k]` inside `R_N`:
  `MLWE_Scheme.ring_switch(c, ksk)` takes a sample of a scheme of the other
  dimension to the level of this one with the same primes, and
  `gen_ring_switch_key(key_out, key_in, lvl, gadget_params=None,
  n_threads=0, hybrid=True)` makes its key, between any two ranks. Down, the
  sample is read as one of rank `r k` over `R_n` and decrypts to the
  coefficients `k m` of the message; up, a list of up to `n / N` samples is
  interleaved coefficient by coefficient. The noise is one key switch's.
  Native `mlwe_ring_switch`, `mlwe_project_subring`, `mlwe_embed_subring`;
  `MLWE_Scheme.level_with_primes(ring)` finds the level over a ring's primes.
- Add LWE extraction and the packing key switch to `MLWE_Scheme`:
  `extract_lwe(c, idx)`, `gen_packing_ksk(key_out, lwe_key, lvl=0,
  gadget_params=None, n_threads=0)` and `packing_keyswitch(extracted,
  packing_key, lvl=0)`, which takes samples of another ring dimension over
  the same primes (matched by value) and keys of either gadget.
- Add MGSW internal products, automorphisms and noiseless keys:
  `MGSW.internal_product(other)` (also `MGSW * MGSW`, native
  `mgsw_internal_product`), an MGSW of the product of the messages;
  `MGSW_Scheme.automorphism(c, g, aut_key, relin_key)` (`mgsw_automorphism`),
  the MGSW of `m(X^g)`; and `MGSW_Scheme.trivial(msg)` (`mgsw_trivial`), the
  gadget matrix times `msg`. Without special primes the keys are of level 0
  (pair them with the radix gadget); with special primes the product and
  automorphism keys come from a scheme with one more special prime, and a
  product adds 5-7 bits of noise to the next external product, against 39 for
  the radix route.
- Add `CMUX_batch(in1, in2, selector, n_threads=0)` (`vfhe.mlwe`): the
  multiplexers of many pairs by one MGSW selector in one native call
  (`mgsw_CMUX_batch`), in parallel, with the results left canonical.
- Add `mgsw_shift_sequences` (C, `mlwe.h`): shifts sequences of samples by an
  encrypted amount `d <= 3` given as one-hot MGSW keys, `y_i = x_(i+w-d)`.
- Generate keys on several threads: every key generator samples through
  `MLWE_Scheme.sample_scaled` (`mlwe_RNS_sample_scaled_batch`) and takes
  `n_threads` -- the `gen_*` key-switch keys, `MGSW_Scheme.encrypt`, the CKKS
  and BFV rotation and conjugation keys, and the bootstrapping keys of
  `CGGI16`, `GP25` and `SHIP`. Each sample's mask comes from a seed, kept as
  `MLWE.seed`, and its noise from a secret seed (`prng_normal_seeded`), all
  drawn in order on the calling thread, so the keys do not depend on the
  thread count. `MGSW_Scheme.encrypt_constants(values, key)` encrypts
  constant polynomials as one batch, which is what a bootstrapping key is. A
  CGGI16 bootstrapping key at N = 2048 for n = 918 is about 4.5x faster on
  eight threads than on one.
- Add `fhe.BFV_Scheme`: BFV over `vfhe.mlwe`, with the plaintext modulus taken
  from the ciphertext ring's lowest primes so that `Delta = q/t` is exact.
  Batched encoding into `N` slots (two rows of `N/2`), encrypt/decrypt,
  plaintext and ciphertext multiplication with relinearization, slot rotation
  and conjugation, and modulus switching.
- Add `ring=` and `scale=` to `CKKS_Scheme.encode`: the ring the plaintext
  lives in (default: level 0's) and the factor the values are scaled by
  (default: `scaling_factor`), e.g. for a ciphertext further down the chain.
  A contiguous buffer of complex doubles, such as a numpy `complex128` array,
  is read in place.
- Add `CKKS_Scheme.encode_batch(values, ring=, scale=, n_threads=)`: many
  vectors encoded in one native call
  (`complex_values_ifft_scale_round_to_RNS_batch`, also
  `ComplexRing.values_ifft_scale_round_to_RNS_batch`). `encode` is one such
  call, so the two agree bit for bit.
- Add `CKKS_Scheme.multiply_plain(ciphertext, plaintext, scale)`: a plaintext
  product without the rescale, with `delta` multiplied by the plaintext's
  `scale` (default: `scaling_factor`; 1 for an unscaled plaintext), so
  several products can be summed before one rescale. `ciphertext *
  plaintext` is this followed by `rescale`.
- Add `CKKS_Scheme.linear_combination(cts, coefficients, scale)` /
  `linear_combinations(cts, rows)`, the `MLWE_Scheme` ones with `delta`
  tracked, and `CKKS_Scheme.conjugate` / `gen_conjugation_key`.
- Add `CKKS_Scheme.rotation_generator(k)`: the Galois element `5^k mod 2N`
  of a rotation of the slots by `k` steps.
- Add `CKKS_Scheme.multiply_batch(lhs, rhs, ksk, n_threads, *, lvl=None)`,
  which rescales each product to `lvl` in the same native call when given,
  and `rescale_batch(cts)`.
- Add `CKKS_Scheme.product(cts)`: the product of ciphertexts as a balanced
  tree, `ceil(log2(n))` levels deep.
- Add `fhe.CKKS_LinearTransform`: a slot matrix, given by its diagonals and
  acting on every block of `n` slots, encoded once for a level (through
  `encode_batch`) and applied by baby-step giant-step with hoisted baby
  rotations and `automorphism_sum` for the giant steps; `rotations` lists the
  keys it needs, and `apply` runs in parallel. `slot_to_coeff` /
  `coeff_to_slot` build SlotToCoeff and its inverse, for any `n` dividing
  `N/2`.
- Add a native CGGI16 blind rotation (`cggi16_blind_rotate`, on the
  library's threads) with an optionally unfolded loop [ZYL+17, BMMP18]:
  `CGGI16(scheme, gsw_ell=None, unfolding=1, gadget_params=None)` consumes
  `unfolding` input coefficients per step for one gadget decomposition, with
  `2^u - 1` MGSW keys per step of `u` coefficients, which pays where the
  synchronization between steps dominates (many threads).
  `functional_bootstrap_batch(outs, tv, inputs, bk, torus_base)` runs one
  rotation per thread (`cggi16_blind_rotate_batch`). `gadget_params` is the
  bootstrapping key's gadget (the RNS one by default). `generate_bootstrap_key`
  raises `ValueError` for a non-binary input key, and `CGGI16_Key` records `n`
  and `unfolding`.
- Add the SHIP bootstrap [CHKS25] (`vfhe.fhe.SHIP`, `SHIP_Key`): bootstraps
  `N/2` real slots using only the `ceil(log2(h + 1))` levels of a product
  tree, `h` the input key's Hamming weight. `SHIP.sample_input_key` draws the
  sparse key (one coefficient per public window) and `generate_bootstrap_key`
  the keys; `bootstrap` (also `half_bootstrap`, `blind_rotate`) runs natively
  on the library's threads. Inputs come at the SlotToCoeff level at
  `input_scale`, and outputs leave at `output_lvl` and `output_scale`, by
  default the scale that comes back to `input_scale` when every level in
  between multiplies and rescales, so bootstrap, multiply, bootstrap needs no
  rescaling of its own. At the paper's LL13 parameters (N = 2^13): about 5.1
  bits of worst-slot precision. Native encoders `ship_encode_pi` /
  `ship_encode_pt0`.
- Add `FoldableRS.relative_distance(security_bits)` /
  `FieldFoldableRS.relative_distance(security_bits)`: a lower bound on the
  code's relative minimum distance, which is the `delta`
  `Basefold.soundness_error` takes -- the exact Reed-Solomon value for the
  `"rs"` instantiation, [ZCF24, Thm. 2]'s bound for the general code.
- Add `polycom.foldable_relative_distance(k0, c, d, field_bits,
  security_bits, bound)`, the general code's distance bound as a pure
  function. `bound="cccfgs26"` (the default) is the tightening of [ZCF24,
  Thm. 2] proven in [CCCFGS26] (*Batch, Pack, and Prove*, ASIACRYPT 2026, to
  appear) for this code; `"zcf24"` is the original, validated against that
  paper's Table 1. `polycom.DISTANCE_BOUNDS` lists them, and
  `relative_distance` on both codes takes `bound` too.
- Add `instantiation=` and `seed=` to `FoldableRS` / `FieldFoldableRS`, with
  `polycom.INSTANTIATIONS` and `polycom.DEFAULT_SEED`; and `twists_odd`, the
  odd-position twists `-T`, derived from `twists` on request.
- Add `twist_field="field" | "prime"` to `FieldFoldableRS`
  (`polycom.TWIST_FIELDS`): the general code's twists drawn from the whole
  field (the default) or from its prime subfield, with `relative_distance` on
  the twists' field. Prime twists make every lift and fold plane-wise, for
  more queries at the same soundness.
- Add `polycom.batch_inverse(values, p)`: modular inverses by Montgomery's
  trick, which is how `FoldableRS` now builds its fold tables (one
  exponentiation per level and prime rather than one per entry).

### Changed

- **BREAKING**: `Verifier.challenge` and `Verifier.challenge_bits` are
  coroutines: write `await iop.verifier.challenge(label)`. Inside an
  AND-composition a challenge waits for the other sub-protocols; outside one
  it returns at once, with the same values and transcripts as before.
- **BREAKING**: `Field.exceptional_from_seed(seed)` and
  `fs.ring_exceptional_from_seed(ring, seed)` are removed; use
  `random_exceptional(seed=seed)` on the field or ring. The Fiat-Shamir
  verifier draws through it. Field challenges keep their values; ring
  challenges are now drawn by `vfhe.crypto.seeded`'s rejection sampler,
  which makes their coefficients exactly uniform (the previous reduction of
  128-bit words was slightly biased), so ring transcripts change.
- The ring's exceptional set is the elements whose CRT components at
  every prime are all one element with `split_degree` integer coefficients
  below the smallest prime (size `p_min^split_degree`, as before), sampled
  natively (`polynomial_RNS_sample_exceptional`), with or without a seed.
  Evaluation points made of challenges are then one point in every slot.
  The sampler used to copy a random slot, sampled independently modulo each
  prime, into every slot -- a larger set whose differences are not all
  invertible. Ring challenges and transcripts change.
- `BatchFoldEval`'s common-point sumcheck is skipped for claims at one
  point made of exceptional elements, and for distinct such points its
  prover runs on one combined table per polynomial instead of one per CRT
  component, with the same messages.
- `Transcript.read` shields the entry it awaits, so cancelling a reader
  leaves the entry for its writer.
- **BREAKING**: `FieldFoldableRS`'s general code takes the root of its base
  transform in `F_p` only, unless `root_field="field"`. Over an extension
  field whose `p - 1` lacks the 2-adicity for the base length, the base is
  now a Vandermonde product on seeded points instead of a transform on
  roots in the extension -- another code, so its codewords and commitments
  change; pass `root_field="field"` for the previous one.
- `BasefoldEval` receives, as one bundle, the claims on all the oracles
  committed under its scheme rather than those on one oracle, and opens a
  bundle of several claims with the batched opening rather than one run per
  claim; transcripts of a single claim are unchanged.
- `BasefoldEval`'s verifier checks a layer's leaves together: one word of
  all the queried leaves, digested in one call and folded with
  `fold_fibers`, then compared with the values the layer above produced as
  one vector. A leaf of the wrong shape or over the wrong domain is now a
  rejection rather than an exception.
- **BREAKING**: `vfhe.piop` is restructured. The framework is one file per
  concern (`oracle`, `virtual`, `relation`, `protocol`, `transcript`,
  `party`, `fs`, `iop`, `composer`, `digest`) and is what `vfhe.piop`
  exports; the relation toolbox is `vfhe.piop.relations` (`Relation_Sum`,
  `Relation_SumProd`, `Relation_Zero`, `Relation_Equal`, `Relation_SumDef`,
  `Relation_CircuitSparseIndex`, `Relation_Circuit`, `Relation_Layer`,
  `Relation_SpEval`, `Relation_Lookup`, `Relation_Range`, `FamilyShape`,
  `dense_relation`, `wiring_predicates`; `Relation_Eval` stays in the
  framework, in `vfhe.piop.evaluation` with `with_start`, since the head
  machinery is written against it) and the protocols are
  `vfhe.piop.protocols` (`Sumcheck`, `SumcheckProd`, `VirtualEval`,
  `SumDefStart`, `OutputTest`, `GKR`, `GKR_Start`, `Lookup`,
  `RangeDecomposition`, `dense_layering`); neither is re-exported by
  `vfhe.piop`, and the modules `vfhe.piop.piop`, `.sumcheck`, `.circuit`,
  `.sparse`, `.lookup` and `.range` are gone. The sumcheck round-message
  kernels are functions in `vfhe.piop.protocols.sumcheck.rounds`
  (`round_evals`, `prod_round_evals`, their native, whole-vector and
  pure-Python paths, `interpolate_evals`), no longer static methods of the
  protocols, and `exceptional_set_size` is `vfhe.piop.exceptional_set_size`;
  the prover's form of a virtual oracle is
  `vfhe.piop.protocols.sumcheck.prover_view.prover_view(oracle, witnesses,
  strategy)`, no longer `VirtualOracle.prover_view`, and the `strategy`
  switch is an attribute of the sumcheck protocols. `Implicit` is
  `ImplicitOracle`; `Party` takes its IOP (`Party(iop)`). The driver, the
  registry and the Composer route a claim by `Relation.subject(statement)`,
  the oracle a statement is a claim about (`None` by default,
  `Relation_Eval`'s one oracle), so any relation that declares a subject is
  bundled, parked and discharged per oracle kind as evaluation claims are;
  `Relation_Eval.subject` is an instance method. The helpers
  protocols share have public names (`hypercube`, `hypercube_sum`,
  `aligned_tables`, `value_of`, `map_oracles`, `table_like`,
  `child_statement`, `expanded_tables`, `hypercube_index`).
- **BREAKING**: A statement names oracles, never tables. `vfhe.piop.oracle`
  has the oracle classes: `TableOracle(table, public=...)` wraps an arith
  `MLE` or `SparseMLE` (kind `public` for a table the verifier holds,
  `plain` otherwise); `CommittedOracle(variables, commitment, scheme)` is
  the verifier's view of a committed polynomial -- the commitment, the
  scheme it was committed under and the variables, no table; the prover's
  table lives in `Prover.witnesses[oracle]`, where `resolve_table` finds it; `ImplicitOracle`
  is the handle. Every oracle has `kind`, `query(point)` (the ideal
  decider's one query) and `digest()`; `oracle_kind` is gone. `MLE` and
  `SparseMLE` no longer take or carry `public`, and `MLE.eq` returns an
  unmarked table. `Basefold.commit(f)` returns `(CommittedOracle,
  BasefoldOpening)`, no longer tags the table, and `Basefold.commitments`
  is gone; `Relation_Eval` has no `commitment` field -- a compiled claim
  names the committed oracle -- and `BasefoldEval` refuses a claim on any
  other oracle. `RangeDecomposition` takes the ring of its claims as a
  constructor parameter. `Relation_SumProd.check` decides an instance whose
  oracles disagree on their variables as not in the relation.
- **BREAKING**: `digest()` is a method of every value a transcript or a
  statement absorbs: `Statement`, the oracles and `Proof` in `vfhe.piop`;
  `MLE`, `SparseMLE`, `MLE_Variable`, `Polynomial` and `FieldElement` in
  `vfhe.arith`; `MerklePath` in `vfhe.crypto`. `element_digest` digests
  primitives, containers, classes and protobuf messages itself and calls
  `digest()` on everything else; the chain is `vfhe.crypto.chain_digest`,
  seeded differently, so every Fiat-Shamir proof differs from those of
  `821f951`.
- **BREAKING**: `Relation_Circuit` with public wiring predicates is now
  `Relation_CircuitSparseIndex`, built by `indexed(circuit, domain)`: its
  index is the mapping `("circuit", "predicates")`, the predicates built
  once by the indexer and read into each layer body through a renaming map;
  `GKR` / `GKR_Start` are registered against it and its `Start`, and
  `GKR.layers` no longer needs the IOP's domain. `Relation_Circuit` is the
  dense-index relation derived from it, proven by `compose(dense_layering)`
  registered against it and its `Start`. The `.index` of `Relation_Layer`
  (`("circuit", "layer", "lin", "xmult")`), `Relation_Lookup` (`("beta",)`)
  and `Relation_Range` (`("bound", "chunks")`) is a mapping over those
  names. The root digest records a committed oracle by its commitment, so
  every Fiat-Shamir proof whose root has a declared index or a committed
  oracle differs from those of `9a8e0f8`. `resolve_table` returns a sparse table as it is and
  resolves restrictions; `prover_view` classifies a constituent as sparse by
  its resolved table.
- **BREAKING**: `Relation_Circuit`'s instance is `oracles = (out, w_in)`,
  the output first and the relation's head, instead of `oracles = (w_in,)`
  with a separate `output` field; its constructor's parameter is `index`.
  `GKR` is now a composed protocol (the output test, then one
  `Relation_Layer` statement per gate layer); `GKR.layers(root, iop)` is an
  instance method and users register `OutputTest` for `Relation_Equal` and
  `SumDefStart` for `Relation_Layer.Start` beside it. The messages are
  unchanged; transcript labels carry statement paths and differ.
- **BREAKING**: The CKKS encoder and decoder (`ComplexRing`) run on a new
  negacyclic FFT kernel (`fft_new_plan` / `fft_forward` / `fft_inverse` /
  `fft_pointwise_mul`, native `FFT_Plan`) in place of the special-root
  transforms; results agree to rounding. `ComplexRing(N)` no longer takes
  `special_rous` and gains `plan` and `slot_positions`
  (`complex_slot_positions`); the native
  `complex_polys_ifft_scale_round_to_RNS_batch` takes `(plan, slots)` in
  place of `(n_complex, log_prec, gs_ws)` and no longer overwrites its input.
  Building a `CKKS_Scheme` at N = 2^16 takes milliseconds instead of seconds.
- Keep `Polynomial.automorphism` of an NTT-domain element on a fully split ring
  in the NTT domain (a reordering of the transform's points) instead of
  converting the element to coefficients first.
- **BREAKING**: Move util's packages under `vfhe.util`: `vfhe.util.bindings`,
  `vfhe.util.kernels` and `vfhe.util.io`, plus the internal `vfhe.util._engine`
  and `vfhe.util._cpu`. The kernels command is `python -m vfhe.util.kernels`.
  `vfhe.info` and `python -m vfhe.info` stay where they were.
- **BREAKING**: Make `python -m vfhe.info` print JSON, with plain values: `version`
  (`null` in a source tree), `engine`, the loaded engine's name (`null` when none
  loads, with the reason on stderr), `engine_file`, its path, and `runs`, which
  maps every engine to whether this CPU runs it (`true`, `false`, or `null` when
  the CPU cannot say).
- **BREAKING**: Prime selection follows the ring's splitting. A partially
  split ring keeps primes `1 mod 2N / split_degree` with an odd multiplier,
  so that its residues are fields of size `p^split_degree` [CCCFGS25,
  Sec. 3.1]; a fully split ring now also takes full-size primes with an even
  multiplier instead of falling back to a smaller prime (at N = 2^11 and 16
  bits, 40961 instead of 12289). Rings that had a full-size prime keep their
  primes. `RNSRing.gen_prime` takes `split_degree` (keyword, required) and
  raises `ValueError` when no prime is left, and a ring given its primes
  (`primes=` or `mask=`) with `split_degree > 1` raises `ValueError` for a
  prime with an even multiplier, which silently gave residues of size `p`.
- Make `Ring(..., mask=...)` and `RNSRing.quotient_ring(mask=...)` raise when
  the mask names primes outside the pool they are given, instead of dropping
  them silently. `RNSRing.union` builds the ring over two rings' primes.
- Make `Multiprecision.compute_crt_consts` raise for moduli its single-digit
  Barrett step cannot serve (primes wider than 52 bits) rather than returning
  constants that reconstruct incorrectly.
- Route `ExtensionFieldNTT.pack` / `unpack`'s transpose through a tile buffer
  with an odd stride, so both its memory sides are contiguous runs; at `2^22`
  elements over `F_(p^4)` it takes 107 ms to about 50.
- Let `FieldVector.query` also accept a buffer of unsigned 64-bit indices, handed
  to the gather as it stands; the kernel now range-checks them itself.
- Make `FieldVector.hash_elements` / `hash_fibers` return a read-only `memoryview`
  of the kernel's buffer rather than a copy. It compares, slices and converts
  like `bytes`; `bytes(digests)` is the copy, and is what hashing or keying by
  them needs.
- Let `Merkle.from_digests` accept any bytes-like and keep it rather than
  copying, so a mutable buffer stays connected -- `commit()` re-reads it.
- Ask the kernel for huge pages for allocations of 8 MiB and up.
  [`USAGE.md`](https://github.com/vfhe/vfhe/blob/main/docs/USAGE.md) covers the allocator settings that go with it.
- **BREAKING**: Gadget parameters are one object. `GadgetParams(log_base=None,
  digits=None, primes_per_digit=1)` (`vfhe.mlwe`; `log_base=None` is the RNS
  gadget) replaces the `radix_log_base` argument of `gadget_scalars`,
  `gen_ksk`, `gen_rlk`, `gen_ksk_for_level`, `gen_rlk_for_level`,
  `gen_ksk_automorphism(_set)`, `gen_ksk_trace`, `MLWE_Set` and
  `MGSW_Scheme` (`gadget_params=`, which the other key generators, `GP25`,
  `CGGI16` and `SHIP` take too); `MLWE_Set` and `MGSW_Scheme` expose
  `gadget_params` in place of `log_base` / `radix_log_base`. It adds two
  decompositions. `primes_per_digit=a` groups the RNS digits, the `dnum` of
  hybrid key switching [HK20]: the digit of a group of `a` primes is the value
  modulo their product `Q_J`, lifted to every prime of the key by the fast
  base conversion (`polynomial_RNSc_lifted_combination`), for about `a` times
  fewer keys and products (a key switch about 1.6x faster with groups of 4 at
  N = 2^14 over eight primes); the special modulus `P` has to cover a group,
  which is not checked, and it does not combine with the radix gadget.
  `digits` keeps only the top digits of each residue of the radix gadget, the
  approximate decomposition, rounding the residue first; it holds over one
  prime only and is refused at a level of more than one. In C, `GadgetParams
  {log_base, digits, primes_per_digit, balanced}`, passed by pointer,
  replaces the `log_base` argument of `gadget_mul_addto_polynomial`,
  `gadget_mul_subto_polynomial`, `mlwe_new_RNS_ks_key` and the `mgsw_*`
  products, and `RNS_MLWE_KS_Key.gadget_params` replaces `log_base`;
  `GadgetParams.native(balanced)` builds it.
- **BREAKING**: Gadget digits are balanced by default (RNS and radix;
  `balanced=False` restores the old outputs). With `MLWE_Scheme(...,
  balanced=True)`, which `CKKS_Scheme` and `BFV_Scheme` pass through and the
  scheme's keys and `MGSW_Scheme` follow (unless given `balanced`), the RNS
  gadget decomposes into centered residues in `(-p_j/2, p_j/2]`, and the
  radix gadget into base-`2^w` digits of the centered residue in
  `[-2^(w-1), 2^(w-1))` (the top one taking the carry; `w = 1` keeps the
  unsigned digits). The keys are the same and the digits' second moment is a
  quarter, so every gadget product adds about one bit less noise standard
  deviation at the same cost. Outputs change; what they decrypt to does not.
  With balanced RNS digits, `MLWE_Scheme.automorphisms` gives the same
  ciphertexts as `automorphism`. `vfhe.util.io` records the choice, and the C
  `GadgetParams` carries it (`balanced`).
- **BREAKING**: The division by the special primes after a gadget product is
  the ModDown of hybrid key switching (`mlwe_mod_down`), the fast base
  conversion of the value modulo `P`, in place of a rounding division by one
  special prime at a time, whose cost grew with the square of their number:
  about 3x faster at four special primes, for an added error of at most half
  their number. Outputs change; what they decrypt to does not. The exact
  rounding division (`mlwe_round_division`: rescale, level moves,
  decryption) is unchanged.
- **BREAKING**: Fresh samples and keys are drawn from seeds.
  `MLWE_Scheme.sample` expands a sample's mask from a fresh 32-byte seed by
  default (`prng_expand_below`) and keeps it as `MLWE.seed`, so `vfhe.util.io` can
  store the sample as its seed; `sample(..., seeded=False)` or
  `MLWE_Scheme.seeded_encryption = False` draws the mask directly. Key
  generation samples the same way and leaves its keys in the NTT domain,
  where they are used. Distributions are unchanged, but a pinned stream
  (`entropy.deterministic`) gives other samples and keys than before.
- **BREAKING**: `MLWE_Scheme(ring, special_primes=k)` derives every level
  down to one prime by default (`max_lvl` defaults to
  `len(ring.primes) - k`): with no special primes the one-prime level used to
  be missing, so such a scheme now has one more level. `max_lvl` outside
  `1 .. len(ring.primes) - k` raises `ValueError`.
- **BREAKING**: Rename the phase `b - <a, s>` to the linear decryption, the
  scheme-neutral term: `MLWE_Scheme.phase` and `LWE.phase` become
  `linear_decrypt`, and the native `mlwe_RNS_phase` / `lwe_phase` become
  `mlwe_RNS_linear_decrypt` / `lwe_linear_decrypt`. The old names are
  removed.
- Make `MLWE_Scheme.linear_decrypt` reuse a key over a wider ring instead of
  rebuilding it for the ciphertext's ring, and takes `ring=` to compute it
  modulo a quotient only; `MLWE_Key.at_ring(ring)` builds the key over
  another ring once and keeps it. `Polynomial.mod_reduce` keeps the element's
  domain.
- **BREAKING**: Rename the installed `engine.h` to `x86_64.h`, holding only
  what its name says: `VFHE_HAVE_X86_64`, `VFHE_HAVE_AVX512F` and
  `VFHE_HAVE_AVX512IFMA`. The cipher and RNG instructions it also carried
  (`VFHE_HAVE_AESNI`, `VFHE_HAVE_VAES`, `VFHE_HAVE_RDRAND`) are crypto's alone
  and moved into that module, unexported -- no other module read them. Include
  `<x86_64.h>` where you included `<engine.h>`.
- **BREAKING**: Move the index and modulus helpers (`array_reduce_mod_N`,
  `array_mod_switch_from_2k`, `double2int`, `int_rev`, `bit_rev`) from
  `util.h` to `arith.h`, and what is left of that header is now `alloc.h`:
  `util` is allocation and nothing else. Every caller of the moved helpers
  already compiled against `arith.h`; include `<alloc.h>` for the rest.
- **BREAKING**: Rename `vfhe.engine` to `vfhe.util.bindings`: import `from vfhe.util.bindings
  import ffi, lib`. `vfhe.util._engine` is instead the catalogue -- an `Engine` enum
  of every engine this build knows, and which of them this CPU can run. The two
  names were one letter apart while doing unrelated jobs, and only the
  catalogue is answerable before anything compiled is mapped, which is what
  `vfhe.util.kernels` and the test runner need.
- Drop `setuptools` as a runtime dependency. Nothing imports it on any
  path that still exists; it served the old runtime-compilation code.
- **BREAKING**: Replace two build options with one: `-Dfuzz=<engine>` is what asks
  for a fuzzing build, since a session instruments exactly one ISA and there is
  no such thing as fuzzing without naming it. The old `-Dfuzz=true` boolean and
  `-Dfuzz_engine=<name>` string could disagree, and passing the engine while
  forgetting the boolean configured an ordinary wheel build, silently.
- Move the CPU and cache probes to pure Python in `vfhe.util._cpu`. It is private, so
  no import path a caller uses changes.
- Make `tools/test/unit/run.sh` read the built engines off meson's test list, keep
  those this CPU runs, and select their tests by name; meson runs every test
  directly. Under
  `EMULATE=1` the emulated test setup launches each test through SDE as its
  `exe_wrapper`, once `tools/test/sde/fetch.sh` has fetched it.
- Build engines with link-time optimisation. The shipped extension was
  already an LTO build, but the static archive beside it was not, so C linked
  against it paid a call where the extension inlined -- measured at 2870 ns per
  call against 530 ns. LTO is not a build option: inlining across the boundary
  is the point of the feature that needs it.
- Change every copyright header to `The vFHE Authors`, in
  place of the individual and institutional lines they replaced.
- Configure ruff and pyright in `ruff.toml` and `pyrightconfig.json`
  rather than in `pyproject.toml`, which each tool reads by default and which
  keeps packaging metadata separate from tool settings.
- Make the version a static `version` in `pyproject.toml`; the build has no
  version logic left: `project()` declares none, and `setuptools-scm` is no
  longer a build dependency. It no longer comes from `git describe` -- a
  release edits the line and tags it -- so sdists need no frozen `.version`.
- **BREAKING**: Name cffi declaration files `.h` instead of `.cdef`, so
  editors and `clang-format` treat them as the C they contain. The library's
  own move with them, and `vfhe.util.kernels` reads a caller's `.h` as
  declarations.
- `MLWE_Scheme.gen_ksk(..., hybrid=False)` (also `gen_ksk_for_level` and
  `gen_ring_switch_key`): a key-switch key in the level's own ring rather than
  its special ring, so the key switch divides by no special prime (BV). For
  an output key that must stay at a small modulus, such as a sparse one; pair
  it with a small radix, `GadgetParams(log_base=...)`.
- **BREAKING**: `mlwe_full_packing_keyswitch(out, in, size, ksk)` no longer
  takes the unused `lvl`; it takes samples of another ring dimension over the
  same primes (matched by value) and decomposes through the key's gadget, so
  it takes radix keys.
- `mgsw_CMUX_to_coeff` and `mgsw_NCMUX_to_coeff` keep the external product
  canonical, skipping a transform pair per call.
- Let `CKKS_Scheme.encode` take any number of values dividing `N/2` and pack
  fewer than `N/2` sparsely (repeated across the slots, so the plaintext is a
  polynomial in `X^(N/2n)`); `decode(..., slots=n)` reads them back.
- `CKKS_Scheme.decode` reconstructs the coefficients natively
  (`polynomial_RNSc_to_centered_doubles`, exact up to the doubles' rounding
  for any value) instead of a Python big-integer CRT per coefficient: about
  100x faster at N = 2^14 over 8 primes, with the same values.
- Make `CKKS_Scheme.decrypt(ciphertext, key, *, message_bound=None, drop=True)`
  return the plaintext over the fewest primes that hold it, the lowest level
  whose modulus exceeds `2 * delta * message_bound`, computing the linear
  decryption over those primes only (about 8x faster to decrypt and decode a
  level-0 ciphertext at N = 2^16 over 9 primes). By default `message_bound` is
  what the smallest level with at least two bits above the scaling factor
  holds at the scheme's scale; `drop=False` keeps every prime.
- **BREAKING**: Rework GP25 [GP25]. `GP25(scheme, gsw_ell=None,
  gadget_params=None, bits_per_layer=1)` rotates over
  `scheme`, the rotation key's over `R_N`, and bootstraps every coefficient
  of a sample over `R_n`, for any `n` and `N`. Three keys take part: the
  output key, dense over `R_n`, which inputs arrive under (at the lowest level
  of its scheme) and outputs leave under, so bootstraps compose; the input
  key, a sparse key of the same scheme (`GP25.sample_input_key(scheme, h,
  gap_bits, sigma_err)` draws one until every gap, `GP25.gaps`, fits in
  `gap_bits`), which inputs are switched to by a Hamming-weight-reducing key
  generated for one level only, so the sparse key only has to be secure at
  that modulus; and the rotation key. Level 0 of `scheme` must have the
  primes of a level of the output key's scheme, where outputs land.
  `generate_bootstrap_key(input_key, rotation_key, output_key, h, gap_bits,
  ternary=True, hw_reducing_lvl=None, hw_reducing_hybrid=True,
  hw_reducing_gadget_params=None, n_threads=0)` makes the `SAB_Key` (`gaps`,
  `signs`, `gap_bits`, `bits_per_layer`, `n`, `h`, `automorphism_key`,
  `hw_reducing_key`, `hw_reducing_lvl`, `output_lvl`, `trace_repack_key` and
  `output_switch_key`); input keys may have any rank, and `ternary=False`
  takes a binary one. `output_key=None` switches nothing:
  inputs come under the input key and `repack(blind_rotate(...))` leaves the
  result over `R_N` under the rotation key, and
  `generate_repacking_key(rotation_key, n)` makes only the keys `rotate` and
  `repack` read.
  `test_vector(table)` takes exact integers mod q. `bootstrap(rlwe_in, tv,
  key)` is `blind_rotate` (the weight-reducing switch, the switch to `Z_2N`
  and `blind_rotate_exponents`), `repack` and the output switch. The blind
  rotation is one native call on the library's threads, `gp25_blind_rotate`,
  built from `GP25.rotate(acc, keys, automorphism_key)` (`gp25_rotate`) and
  `GP25.multiply_by_signed_monomials`. Repacking goes through `trace_pack`
  and a ring switch, for any `n`, `N` and rank, with 1-3 MiB of keys against
  36-576 MiB for the packing key switch it replaces at N = 1024.
  `bits_per_layer=2` is BatchBoot's multi-bit rotation [LWZ+26]: a pair of
  rotation bits of value `d` takes three MGSW keys, of `[d = 1]`, `[d = 2]`
  and `[d = 3]`, while the pair's power `p` has `4 p <= n`
  (`GP25.rotation_layers`), for 1.5 times the MGSW keys, about 0.8 bit more
  rotation noise with the RNS gadget, and a rotation about 1.4x faster;
  `encrypt_rotations(values, n_bits, n, key)` makes such keys. See Removed
  for the previous surface.
- **BREAKING**: Make `FoldableRS` and `FieldFoldableRS` build the general
  foldable code of [ZCF24, Def. 5] by default -- seeded twist tables at every
  level above a Reed-Solomon base code -- rather than a Reed-Solomon code on
  roots of unity at every level. The same constructor call gives a different
  code, with different codewords and commitments, and its distance is the
  paper's recursive bound rather than the exact Reed-Solomon one, so security
  parameters derived from the old `relative_distance` do not carry over. The
  previous code is `instantiation="rs"`. What the change buys: no 2-adicity
  condition on the codeword length (`FieldFoldableRS` needed
  `2 n_d | p - 1`, `FoldableRS` needed `n_d | N/split_degree`); only the base
  length `n0` wants a transform, and does without one where the field has
  none.
- Make `FoldableRS.twists` / `FieldFoldableRS.twists` the general code's
  twist table `T` (the odd position uses `-T`, as in the paper), and the
  field code's are lists of elements rather than integers (a twist of the
  general code need not lie in the prime subfield). `twists2_inv` is gone:
  the fold's `1 / (2 T)` is held privately.
- Make `relative_distance` a method, not a property: the general code's bound
  depends on the security parameter.
- Make the general code over an extension field encode the base messages of a
  level-`l` message with one batched transform (`ExtensionField.ntt_plan(n0)`,
  in the base domain or, on a field with too little 2-adicity, the extension
  one) where it ran a Horner scheme of `k0` passes; at `k0 = 128` and a `2^20`
  codeword over `F_(p^4)` the encode is 12x faster (1.69 s to 0.14 s,
  avx512ifma), and a field whose only roots of unity lie outside `F_p` now
  has a transform as its base encoder rather than a Vandermonde product.
- Make the general code's lifts `lift_twisted` calls alternating between two
  buffers, in place of `fma_interleave` over a tiled and negated table into a
  fresh destination per level: at a `2^22` codeword over `F_(p^4)` the lifts
  of an 11-level encode go from 724 ms to 363 ms and the commit from 1.16 s to
  0.88 s (avx512ifma); with `twist_field="prime"` the commit is 0.66 s.

### Removed

- **BREAKING**: Python 3.10, past its end of life on 2026-10-04; the CI
  toolchain pins `filelock` 4.2.1, whose advisories have no fix on 3.10.
- **BREAKING**: `ImplicitOracle`, `ImplicitEval` and the `implicit` entry of
  the protocol registry. An oracle defined by a hypercube sum is a
  `Relation_SumDef` statement headed at that oracle inside a layered
  reduction; `IOP.protocol_for` refuses an evaluation claim on an implicit
  oracle, which only the Composer consumes. The driver's parking rule now
  concerns committed and plain oracles only.
- **BREAKING**: Reduce `vfhe.util.bindings` to `ffi` and `lib`.
  `active_engine()` and `runnable_engines()` are gone, since the loaded engine
  is internal knowledge; `set_num_threads` and `num_threads` moved to
  `vfhe.util.parallel` (see Added); the "a faster engine could run" warning is gone —
  only a pin could trigger it, and `python -m vfhe.info` reports what else
  this CPU runs.
- **BREAKING**: Remove `vfhe_engine_active()` from the C, and with it
  `build_info.c`, util's `python/cdef/engine.h` and the `-DVFHE_ENGINE_NAME`
  every engine was compiled with. It reported which engine a binary was, for a
  "you could be faster" hint that became pure Python; nothing has called it
  since. `python -m vfhe.info` reports the loaded engine.
- **BREAKING**: Remove `cache_root` from `vfhe.util.kernels`. `ExtensionBuilder.compile()`
  and `build()` return the directory they produced, which is the path a
  caller actually needs; nothing in the tree or the docs ever asked where the
  cache root was.
- **BREAKING**: Remove `libvfhe` and its `LibVFHE` class from
  `vfhe.util.bindings`. Nothing read them: `multithreaded` and `num_threads` were never
  consulted, since the C that threads takes its count as a call argument. Use
  `from vfhe.util.bindings import ffi, lib`, which is what the wrappers already do.
- **BREAKING**: Remove the CPU probe from C: `vfhe_cpu.h`, `cpu_probe.c` and
  the `vfhe_cpu` cdef. What a host supports is `vfhe.util._cpu`'s
  answer, in pure Python, because it has to be answerable *before* an engine
  loads -- a probe inside the engine cannot say whether that engine may load.
- **BREAKING**: Remove `rebind_state` from `vfhe.arith`, and
  `vfhe.arith.state` no longer offers `register_rebind` / `rebind`. They
  existed only to rebuild state after the hot swap, which is gone; the `reset`
  half stays, and is what empties caches between tests.
- **BREAKING**: Remove `vfhe.dynamic_extensions`, replaced by `vfhe.util.kernels`
  (see Added). The hot swap it was built around -- `update_cffi_references`,
  `register_reinitializer`, `REINITIALIZATION_REGISTRY` -- goes with it:
  CPython cannot unload an extension, so a swap could never reach objects a
  caller already held, which in a cryptography library is a wrong answer rather
  than an untidiness.
- **BREAKING**: Stop installing a standalone C library: the
  `libvfhe_<engine>` shared libraries, their pkg-config files and the public
  headers under `<prefix>/include/vfhe` are gone. Nothing linked them -- the
  engine extension carries the code -- and vFHE is a Python library with C
  underneath, not a C library with Python bindings. `vfhe.util.kernels` is how C
  reaches these kernels.
- **BREAKING**: Stop shipping `libvfhe_<engine>.a` and
  `engine-<engine>.json`. Nothing links the archive any more, and the JSON's
  contents moved into the generated `_vfhe_flags` table, so meson's engine
  catalog reaches Python by one route instead of two.
- **BREAKING**: `ComplexRing`'s `special_rous` argument, `gen_special_rous`,
  `gen_special_rous_hp`, `CT_rous` and `GS_rous`, and the native special-root
  transforms `CT_NR` / `GS_RN` with their tables (`load_rous_CT`,
  `load_rous_GS`, `bit_reverse_array`), replaced by the FFT kernel.
- `mpmath` is no longer a dependency: its only use was building the old
  transform's root tables.
- **BREAKING**: `array_reduce_mod_N` and `array_mod_switch_from_2k` (C,
  `util.h`), whose only uses were the biased uniform sampling fixed below.
- **BREAKING**: `MLWE_Scheme.phase`, `LWE.phase`, `mlwe_RNS_phase` and
  `lwe_phase`; use `linear_decrypt` (see Changed).
- **BREAKING**: `MLWE_Scheme.full_packing_keyswitch_scaled` and
  `mlwe_full_packing_keyswitch_scaled`: `trace_pack` packs the same samples
  without the factor and with less noise.
- **BREAKING**: The unimplemented LWE key switch (`LWE_KS_Key`,
  `lwe_new_KS_key`, `lwe_keyswitch`) and the undefined
  `mlwe_RNSc_extract_lwe`, from `mlwe.h`.
- **BREAKING**: GP25's previous surface (see the rework in Changed):
  `GP25(..., debug_key=, threads=, trace_repack=)` (repacking always goes
  through the trace), `mod_switch` (from `vfhe.fhe`),
  `generate_sparse_ternary_key` (use
  `sample_input_key` and `generate_bootstrap_key`), `encrypt_bits`,
  `sab_rlwe_bootstrap` (use `bootstrap`), `sab_blind_rotate`, `setup_tv_xb`,
  `sab_LUT_packing` (use `test_vector`), `sparse_mul`, `RGSW_monomial_mul`,
  `sub_a`, the `CMUX` / `NCMUX` / `mul_by_xai` / `mul_by_xai_minus_1`
  wrappers, `get_noise` / `get_noise_stderr`, `extract_lwe_key` (use
  `MLWE_Key.extract_lwe_key`), `rlwe_extract_lwe`, `gen_packing_ksk` and
  `packing_keyswitch` (now `MLWE_Scheme.extract_lwe`, `gen_packing_ksk` and
  `packing_keyswitch`), the `SAB_Key` fields `s`, `s_sign`, `r_prec` and
  `b_prec`, and the native `gp25_RGSW_monomial_mul(_mt)` and
  `gp25_sub_a_mt`.
- **BREAKING**: `CGGI16_Key.b_prec`, which nothing read.

### Fixed

- Correct `Polynomial.base_extend`'s documented contract: it returns a value
  congruent to this element modulo the ring's modulus, not the element
  itself. Behaviour is unchanged; the docstring claimed exactness the fast
  base conversion behind it never provided.
- Fix uniform sampling below a prime (`Polynomial.sample_uniform`, the masks
  of MLWE and LWE samples, `gen_sparse_ternary_array_modq`) being biased:
  random words were mapped onto the modulus by a floating-point scaling or a
  masked reduction, which lost the low bits of a prime above 2^53 and
  favoured some residues of a prime well below its power of two. Values are
  now drawn by rejection (`generate_uniform_below`) and are exactly uniform.
- Fix `Ring(N, prime_size=[...])` searching forever when no prime of a
  requested size is `1 mod 2N / split_degree` with an odd multiplier (17 bits
  at N = 2^13, for one): it now takes a smaller prime, a fully split ring one
  with an even multiplier first (65537 there), and raises `ValueError` when
  none is left.
- Fix `Ring(N, primes=[...])`, one of the documented ways to give the
  modulus, failing unless `prime_size=None` was passed too.
- Fix RNS `Polynomial.from_bigint_array` with fewer than N values reading past
  its buffers: the remaining coefficients were heap contents instead of zero.
- Fix `arith_scalar_new` on an RNS ring whose primes do not start the base
  copying one value per prime when the scaling kernels index the scalar by
  base row: they read past the copy, and `arith_scale_by` and
  `MLWE.multiply_scalar` scaled the ring's higher rows by whatever lay there.
- Fix the native RNS ring handles stopping being shared after 256 distinct
  rings: past that, every lookup allocated a new handle that was never freed
  (one per sample allocated), and two handles for one ring no longer compared
  equal. The table now grows, under a lock.
- Fix CKKS encoding at a scale that takes a coefficient to 2^63 or beyond
  (`encode` and the native rounding paths): the rounding went through an
  `int64_t` and wrapped silently. A value that large is reduced exactly now,
  as the integer the double holds, so any scale works; values below 2^63 take
  the same path as before. A NaN or infinite value to round (a non-finite
  input or scale) now raises `ValueError`; the native
  `complex_poly_round_to_RNS` and `complex_*_ifft_scale_round_to_RNS_batch`
  return `false` for it.
- Fix `ComplexRing(N)` crashing for `N < 8` on avx512ifma, and
  `complex_poly_scale_double` scaling nothing below `N = 4` there.
- Fix `ComplexPolynomial` assignment, `from_array` and `*=` rejecting
  `numbers.Complex` / `numbers.Real` values (e.g. a subclass of `complex`, or
  a `Fraction`) other than the exact builtin types.
- Fix `FieldVector.view` of a view starting at the parent's buffers rather than at
  the view: the nested view's planes skipped the outer offset, so it read and
  wrote the wrong elements. Both vector implementations.
- Fix `Multiprecision.from_polynomial` leaking its result. `free_mp_polynomial`
  was never exposed, so no caller could release one; the returned handle now
  owns its native allocation.
- Fix multiprecision reconstruction (`Multiprecision.from_polynomial`) reading
  rows of the shared RNS base that the polynomial does not own, which crashed
  as soon as a second ring existed in the process, and pairing its CRT
  constants with the wrong primes. It now follows the polynomial's own prime
  mask.
- Fix multiprecision reconstruction returning values that were not reduced
  modulo q. The Barrett constants were sized from the prime width, which put
  the quotient's digit read out of range whenever the primes were narrower
  than a base-2^52 digit; they now come from the moduli themselves, and the
  reconstruction reduces each residue before accumulating, so the number of
  primes no longer bounds what can be reconstructed.
- Make the entropy-backed generators (`generate_random_bytes`,
  `generate_normal_random`, `entropy`) safe to call from several threads at
  once: concurrent callers could be served the same bytes of the shared 1 KiB
  pool, and each thread now keeps a pool of its own. A forked child discards
  the pool it inherits instead of serving it again after its parent, and
  pinned threads (`entropy.deterministic`) never take the same seed.
- Fix `MLWE_Scheme(ring, special_primes=n)` dropping the special primes for
  any ring but the first one built over its RNS base. The mask was derived
  from the working primes' *count*, which is their base index only when the
  ring starts at index 0; every other ring got a special ring equal to its
  level 0, so key switching silently ran BV instead of the GHS hybrid, at
  about a prime's worth of extra noise.
- Fix `MLWE_Scheme(ring, special_primes=k)` for `k >= 2`, and for a ring of
  one prime: the default level chain was one level too long, and the
  constructor raised `ValueError` (`IndexError` for one prime).
- Fix MGSW keys of a level whose primes do not come first in the MGSW ring
  (a special prime with a lower index in the RNS base than the level's): the
  gadget covered the ring's first primes rather than the level's, so every
  external product and CMUX by a key encrypting anything but 0 returned
  garbage. The keys of nested chains with the special primes last are
  unchanged.
- Fix `MGSW.external_product` and `CMUX` returning a level-0 sample whatever
  the input's level; below level 0 it decrypted to garbage. The products now
  keep the input's level and type, and raise `ValueError` when the MGSW was
  encrypted for another level (`MGSW.check_level`,
  `MGSW_Scheme.gadget_size`).
- Fix `MLWE_Scheme.sample` at a level holding primes the key's ring does not
  (a level that shares no prime with level 0): the sample was drawn with no
  key on those primes and did not decrypt. It now uses the key over the
  sample's ring, as decryption already did.
- Fix `mlwe_round_division` leaving the sample on its old ring: the
  components were divided into the destination ring, but `->ring` still named
  the source, so native code that allocates or key-switches from a sample
  after a rescale worked in the wrong ring.
- Fix `MLWE.new_like()` (behind `copy`, `+`, `-` and plaintext products)
  allocating at the level's defaults rather than like its input: a sample over
  a special ring came back over `rings[-1]`, and an unrelinearized product
  lost every component past the scheme's rank. With neither `lvl` nor `ring`
  given, the result now has the input's ring, rank and `is_extended`.
- Fix `CKKS_Scheme.multiply` (inherited from `MLWE_Scheme`) returning the
  first operand's `delta` instead of the product of both, so a product taken
  with it decoded at the wrong scale; `*` was right.

## [0.0.3] - 2026-09-10

### Added

- Add engine selection at import. An install carries one extension per
  instruction-set level its architecture supports (`portable`, `avx512ifma`,
  `neon`) and loads the best one this CPU can run, asking a separate probe
  extension so choosing an engine never loads one. `VFHE_ENGINE=<name>` pins
  the choice and refuses a CPU that cannot run it, and
  `vfhe_engine_active()` reports it.
- Add wheels for manylinux x86_64/aarch64 and macOS arm64, published
  alongside the sdist with the same provenance attestation, so most installs
  need no compiler.
- Add arm64 Linux to the tested platforms.
- Add `python -m vfhe.info`, which prints the version, the selected engine,
  anything faster this CPU could have run, and the platform — the whole
  environment a bug report needs.
- Add PEP 561 typing markers to every subpackage, so a type checker resolves
  `vfhe.*` against an install.
- Add `vfhe.crypto.entropy` and `vfhe.crypto.seeded`, a typed Python surface
  over the C generators: unpredictable bytes, values below a bound, Gaussian
  draws, and the seeded `(context, seed)` sampler a transcript recomputes,
  plus `entropy.deterministic(seed)` for tests. They are the library's only
  randomness — nothing in the tree draws from `secrets`, `random`, or the OS
  directly any more.
- Add the metadata a scanner and a redistributor need to every wheel:
  - CycloneDX SBOM fragments at `.dist-info/sboms/`, the location PEP 770
    standardises — the vendored BLAKE3 sources, plus pedigree entries for code
    adapted from MOSFHET, Intel HEXL, and a modular-inverse routine whose
    upstream licence is unstated. A metadata scan cannot see C compiled into
    an extension, and install tools copy that directory, so a scanner reading
    an installed wheel can.
  - The Apache-2.0 licence text and `NOTICE` at `.dist-info/licenses/`, so a
    redistributor receives the attribution the licence requires for the C
    vFHE ships.

### Changed

- **BREAKING**: Move the multilinear-extension layer from `vfhe.piop` to
  `vfhe.arith` and the Merkle tree from `vfhe.piop` to `vfhe.crypto`, which
  is now a Python-facing module. Import `MLE`, `SparseMLE`, `MLE_Basis` and
  `MLE_Variable` from `vfhe.arith`, and `Merkle` / `MerklePath` from
  `vfhe.crypto`; `vfhe.piop` no longer re-exports them. Neither depended on
  anything PIOP-specific.
- **BREAKING**: Split `vfhe.misc` into `vfhe.engine` (the native handle) and
  `vfhe.dynamic_extensions` (runtime C compilation), and move randomness into
  a new internal `crypto` module. Import `from vfhe.engine import ffi, lib`
  in place of `vfhe.misc.libvfhe`, and `vfhe.dynamic_extensions` in place of
  `vfhe.misc.dynamic_extensions`.
- Change `vfhe.dynamic_extensions` to compile only your files and link them
  against the shipped `libvfhe_<engine>.a`, rather than rebuilding the whole
  library from source. A runtime compile drops from the full kernel set to
  your snippet, and an install no longer carries C sources or build
  machinery.
- Change the build backend to meson (`meson-python`). `setup.py` and
  `MANIFEST.in` are gone, and an sdist vendors the BLAKE3 sources with a
  frozen version, so building one needs neither git nor submodules.
- Verify a release's provenance after publishing as well as before: every
  file the index serves is downloaded and checked against the one workflow
  allowed to have signed it.
- Change the security contact to <security@vfhe.ai> and the conduct contact
  to <conduct@vfhe.ai>.

### Removed

- **BREAKING**: Remove the `VFHE_PORTABLE` and `VFHE_TUNED` build knobs.
  Every build now carries every engine its architecture supports and picks
  one at import.
- Remove the CycloneDX SBOM asset from releases. It described one CI runner's
  dependency resolution, which no user reproduces; the vendored-C record it
  uniquely held now ships inside every wheel.

### Fixed

- Fix a crash in the AES-NI/VAES keystream when asked for 256 bytes or more
  into a buffer that is not 64-byte aligned: it wrote through aligned vector
  stores while its callers' signatures promised a plain `uint8_t *`. Every
  in-tree caller happened to pass aligned memory, so it only surfaced once
  the generators were reachable from Python.
- Fix `vfhe.dynamic_extensions`:
  - It failed from an installed package, because the C sources it recompiles
    were never installed and a clean Python 3.12 or later lacks the
    setuptools cffi compiles through. Installs now ship the build inputs as
    `vfhe/_source`.
  - User-supplied compiler and linker flags replaced the build plan's instead
    of extending them, reinitialization failures degraded to warnings,
    repeated compiles accumulated `sys.path` entries, and `XDG_CACHE_HOME`
    was ignored.
- Fix an async multilinear-polynomial evaluation being collected mid-flight,
  because the event loop holds tasks only weakly and the spawned task was
  never referenced.
- Fix the generated protobuf bindings:
  - `import vfhe.circuit` raised `VersionError` on protobuf 6.x, which the
    declared floor allowed, because the bindings are generated against 7.35.1
    and protobuf refuses a runtime older than its gencode. The floor now
    states what they need (`protobuf>=7.35.1,<8`).
  - They installed as implicit namespace packages, so another distribution
    shipping a `_vfhe_proto` directory would have merged into the same
    namespace instead of conflicting.

## [0.0.2] - 2026-07-22

### Added

- Add `AUTHORS.md`, `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  and this changelog.
- Add Python 3.14 support, tested in CI and declared in the classifiers.
- Add coverage reporting for Python and C, informational only.
- Add nightly and per-pull-request fuzzing (ClusterFuzzLite).
- Attach a CycloneDX SBOM and Sigstore build provenance to each GitHub
  Release (verifiable with `gh attestation verify`), and publish to PyPI via
  Trusted Publishing.

### Changed

- Move the development guide from the README to `docs/DEVELOPMENT.md`. The
  README now targets users and renders cleanly on PyPI.
- Rework CI end to end: parallel required checks behind one gate, SHA-pinned
  actions, hardened permissions, and an sdist install-and-smoke check.
- Raise the minimum `cffi` to 2.1 at runtime and `setuptools-scm` to 10.2 at
  build time.

### Fixed

- Fix `ntt_new_proc` looping forever searching for a primitive root of unity
  with certain prime and ring-size combinations. The search is now
  deterministic and always terminates.
- Fix a module compiled at runtime auto-tuning independently of the loaded
  engine, so a portable process could load AVX-512 kernels and crash. A
  custom build now inherits the loaded engine's mode.

## [0.0.1] - 2026-07-08

### Added

- Publish the initial pre-release on PyPI: RNS polynomial arithmetic with
  incomplete NTTs (`arith`), LWE / Module-LWE and MGSW (`mlwe`), CKKS with
  CGGI16 and GP25 bootstrapping (`fhe`), layered GKR circuits (`circuit`),
  and an AVX-512 or portable native engine.
- Distribute as an sdist, which builds against the host CPU at install time.

[Unreleased]: https://github.com/vfhe/vfhe/compare/0.0.3...HEAD
[0.0.3]: https://github.com/vfhe/vfhe/compare/0.0.2...0.0.3
[0.0.2]: https://github.com/vfhe/vfhe/compare/0.0.1...0.0.2
[0.0.1]: https://github.com/vfhe/vfhe/releases/tag/0.0.1
