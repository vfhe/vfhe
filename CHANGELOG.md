<!-- SPDX-FileCopyrightText: 2026 Alin-Petru Roșu -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until 1.0.0, minor
versions may contain breaking changes.

## [Unreleased]

### Added

- Speed up `Polynomial.get_coeff_matrix` / `from_coeff_matrix` on an
  unsplit ring: a row is now read with `ffi.unpack` and written with one
  packed `ffi.memmove` instead of element by element. `BFV_Scheme.encode` /
  `decode` at N=4096 go from 1.9 / 2.1 ms to 0.27 / 0.29 ms.
- Correct `Polynomial.base_extend`'s documented contract: it returns a value
  congruent to this element modulo the ring's modulus, not the element
  itself. Behaviour is unchanged; the docstring claimed exactness the fast
  base conversion behind it never provided.
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
- Add `FieldVector.fold_twisted(twist2_inv, twist, r)`: a vector of
  `(P(x), P(-x))` pairs folded to half its length in one call.
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
- Add `dynamic_extensions.compile(reuse=True)`: hand the process over to an
  earlier build of the same inputs in `output_dir`, when one is there and
  still newer than the `libvfhe_<engine>.a` it links, and compile only
  otherwise. The same inputs means the same registered sources and
  declarations, the same `extra_compile_args` / `extra_link_args`, the same
  compiler at the same version, and the same engine -- all of which the
  module's name now carries, so one set of sources built under two sets of
  flags gives two modules and neither answers for the other. Reuse is the
  library's to offer because that name is: a caller naming the module for
  itself either rebuilds every time or loads a module built for other flags
  or another ABI.
- Add native multithreading under one library-wide limit:
  `vfhe.engine.set_num_threads(n)` / `num_threads()` (C `vfhe_set_num_threads`
  / `vfhe_num_threads`), defaulting to `VFHE_NUM_THREADS` or else 1 (single
  threaded) and kept across `dynamic_extensions` reloads. Operations taking
  `n_threads` use up to the limit by default and never exceed it: single
  operations (key switch, automorphism, multiplication, relinearization,
  round division and rescale, domain conversions) split over the ring's
  primes from N = 4096 up, and batches over their items.
  `vfhe.engine.local_num_threads(n)` (C `vfhe_set_local_num_threads` /
  `vfhe_local_num_threads`) limits the threads of the calls one thread makes,
  and a parallel call inside another gets only its share of the outer call's
  threads (`vfhe_share_for`), so nesting never goes past the limit. Results
  do not depend on the thread count. The native loop is `vfhe_parallel_for`
  (util), with helper threads kept asleep between calls. The complex-FFT
  batch, which always used 8 threads, follows the limit too. A CKKS rotation
  at N = 2^14 over 17 primes runs about 6x faster on 96 cores than on one.
- Add a memory pool for the buffers the library allocates and frees
  repeatedly (`vfhe.engine.memory_pool`; C `mempool_*`): a released RNS
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
  `VFHE_MEMPOOL_WIPE_ON_RELEASE=1` set the defaults, and both settings survive
  `dynamic_extensions` reloads. A sanitized build bypasses the pool.
- Add serialization, `vfhe.io`: `Serializer.dump` / `dumps` write one value
  per stream -- a vfhe object, `None`, a JSON scalar, or lists, tuples and
  dicts of values -- and `load` / `loads` read it back, binding each scheme to
  the caller's with the same primes and parameters (`schemes=`, `rings=`).
  Shared objects such as rings and schemes are written once. Codecs cover RNS
  rings and polynomials, MLWE and MGSW schemes, samples, key-switch keys, MGSW
  ciphertexts, LWE samples and keys, the CKKS and BFV schemes, CKKS
  ciphertexts, `SAB_Key`, `CGGI16_Key` and `CKKS_LinearTransform`; `register`
  adds a `Codec`. Secret keys are written only by `dump_secret` /
  `dumps_secret`. `Options` (presets in `PROFILES`: `"default"`, `"compact"`,
  `"fast"`): `seeded` (default) stores a fresh sample or key as its 32-byte
  mask seed and its body, halving its size; `packing="tight"` stores each
  residue in exactly the bits its prime needs; `domain` picks the rows'
  domain; `checksum` and `validate` (both default) check records and residues
  on load.
- Add `MLWE_Scheme.key_gen(sigma_err, distribution="ternary")`: a key with
  every coefficient uniform in `{-1, 0, 1}`.
- Add `MLWE(..., obj=)`: adopt a native sample.
- Add `MLWE.from_components(scheme, a, b, lvl=None)`: the sample with mask
  `a` and body `b`, copied from the polynomials (which may be temporaries, in
  either domain), over a level of the scheme or another of its rings.
- Add `MLWE.mod_reduce(ring | lvl)`: a ciphertext reduced into a smaller
  level in place, the value kept rather than divided -- CKKS's level drop.
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

- **Breaking:** The CKKS encoder and decoder (`ComplexRing`) run on a new
  negacyclic FFT kernel (`fft_new_plan` / `fft_forward` / `fft_inverse` /
  `fft_pointwise_mul`, native `FFT_Plan`) in place of the special-root
  transforms; results agree to rounding. `ComplexRing(N)` no longer takes
  `special_rous` and gains `plan` and `slot_positions`
  (`complex_slot_positions`); the native
  `complex_polys_ifft_scale_round_to_RNS_batch` takes `(plan, slots)` in
  place of `(n_complex, log_prec, gs_ws)` and no longer overwrites its input.
  Building a `CKKS_Scheme` at N = 2^16 takes milliseconds instead of seconds.
- `Polynomial.automorphism` of an NTT-domain element on a fully split ring
  stays in the NTT domain (a reordering of the transform's points) instead of
  converting the element to coefficients first.
- **Breaking:** Prime selection follows the ring's splitting. A partially
  split ring keeps primes `1 mod 2N / split_degree` with an odd multiplier,
  so that its residues are fields of size `p^split_degree` [CCCFGS25,
  Sec. 3.1]; a fully split ring now also takes full-size primes with an even
  multiplier instead of falling back to a smaller prime (at N = 2^11 and 16
  bits, 40961 instead of 12289). Rings that had a full-size prime keep their
  primes. `RNSRing.gen_prime` takes `split_degree` (keyword, required) and
  raises `ValueError` when no prime is left, and a ring given its primes
  (`primes=` or `mask=`) with `split_degree > 1` raises `ValueError` for a
  prime with an even multiplier, which silently gave residues of size `p`.
- `Ring(..., mask=...)` and `RNSRing.quotient_ring(mask=...)` now raise when
  the mask names primes outside the pool they are given, instead of dropping
  them silently. `RNSRing.union` builds the ring over two rings' primes.
- `Multiprecision.compute_crt_consts` raises for moduli its single-digit
  Barrett step cannot serve (primes wider than 52 bits) rather than returning
  constants that reconstruct incorrectly.
- `ExtensionFieldNTT.pack` / `unpack`'s transpose goes through a tile buffer
  with an odd stride, so both its memory sides are contiguous runs; at `2^22`
  elements over `F_(p^4)` it takes 107 ms to about 50.
- `FieldVector.query` also accepts a buffer of unsigned 64-bit indices, handed
  to the gather as it stands; the kernel now range-checks them itself.
- `FieldVector.hash_elements` / `hash_fibers` return a read-only `memoryview`
  of the kernel's buffer rather than a copy. It compares, slices and converts
  like `bytes`; `bytes(digests)` is the copy, and is what hashing or keying by
  them needs.
- `Merkle.from_digests` accepts any bytes-like and keeps it rather than
  copying, so a mutable buffer stays connected -- `commit()` re-reads it.
- `dynamic_extensions` names a compiled module after the flags it was built
  with and the compiler that built it, version included, as well as its
  sources and the active engine. Existing caches are invalidated, which is the
  intended effect.
- `dynamic_extensions.update_cffi_references` now runs the
  `REINITIALIZATION_REGISTRY` itself. Installing the new handles and rebuilding
  the state bound to the old ones are one operation -- a reinitializer reaches
  the library through `vfhe.engine`, so it must run after the handles move --
  and callers no longer iterate the registry after calling it.
- Allocations of 8 MiB and up ask the kernel for huge pages.
  [`USAGE.md`](docs/USAGE.md) covers the allocator settings that go with it.
- **Breaking:** Gadget parameters are one object. `GadgetParams(log_base=None,
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
- **Breaking:** Gadget digits are balanced by default (RNS and radix;
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
  ciphertexts as `automorphism`. `vfhe.io` records the choice, and the C
  `GadgetParams` carries it (`balanced`).
- **Breaking:** The division by the special primes after a gadget product is
  the ModDown of hybrid key switching (`mlwe_mod_down`), the fast base
  conversion of the value modulo `P`, in place of a rounding division by one
  special prime at a time, whose cost grew with the square of their number:
  about 3x faster at four special primes, for an added error of at most half
  their number. Outputs change; what they decrypt to does not. The exact
  rounding division (`mlwe_round_division`: rescale, level moves,
  decryption) is unchanged.
- **Breaking:** Fresh samples and keys are drawn from seeds.
  `MLWE_Scheme.sample` expands a sample's mask from a fresh 32-byte seed by
  default (`prng_expand_below`) and keeps it as `MLWE.seed`, so `vfhe.io` can
  store the sample as its seed; `sample(..., seeded=False)` or
  `MLWE_Scheme.seeded_encryption = False` draws the mask directly. Key
  generation samples the same way and leaves its keys in the NTT domain,
  where they are used. Distributions are unchanged, but a pinned stream
  (`entropy.deterministic`) gives other samples and keys than before.
- **Breaking:** `MLWE_Scheme(ring, special_primes=k)` derives every level
  down to one prime by default (`max_lvl` defaults to
  `len(ring.primes) - k`): with no special primes the one-prime level used to
  be missing, so such a scheme now has one more level. `max_lvl` outside
  `1 .. len(ring.primes) - k` raises `ValueError`.
- **Breaking:** Rename the phase `b - <a, s>` to the linear decryption, the
  scheme-neutral term: `MLWE_Scheme.phase` and `LWE.phase` become
  `linear_decrypt`, and the native `mlwe_RNS_phase` / `lwe_phase` become
  `mlwe_RNS_linear_decrypt` / `lwe_linear_decrypt`. The old names are
  removed.
- `MLWE_Scheme.linear_decrypt` reuses a key over a wider ring instead of
  rebuilding it for the ciphertext's ring, and takes `ring=` to compute it
  modulo a quotient only; `MLWE_Key.at_ring(ring)` builds the key over
  another ring once and keeps it. `Polynomial.mod_reduce` keeps the element's
  domain.
- `MLWE_Scheme.gen_ksk(..., hybrid=False)` (also `gen_ksk_for_level` and
  `gen_ring_switch_key`): a key-switch key in the level's own ring rather than
  its special ring, so the key switch divides by no special prime (BV). For
  an output key that must stay at a small modulus, such as a sparse one; pair
  it with a small radix, `GadgetParams(log_base=...)`.
- **Breaking:** `mlwe_full_packing_keyswitch(out, in, size, ksk)` no longer
  takes the unused `lvl`; it takes samples of another ring dimension over the
  same primes (matched by value) and decomposes through the key's gadget, so
  it takes radix keys.
- `mgsw_CMUX_to_coeff` and `mgsw_NCMUX_to_coeff` keep the external product
  canonical, skipping a transform pair per call.
- `CKKS_Scheme.encode` takes any number of values dividing `N/2` and packs
  fewer than `N/2` sparsely (repeated across the slots, so the plaintext is a
  polynomial in `X^(N/2n)`); `decode(..., slots=n)` reads them back.
- `CKKS_Scheme.decode` reconstructs the coefficients natively
  (`polynomial_RNSc_to_centered_doubles`, exact up to the doubles' rounding
  for any value) instead of a Python big-integer CRT per coefficient: about
  100x faster at N = 2^14 over 8 primes, with the same values.
- `CKKS_Scheme.decrypt(ciphertext, key, *, message_bound=None, drop=True)`
  returns the plaintext over the fewest primes that hold it, the lowest level
  whose modulus exceeds `2 * delta * message_bound`, computing the linear
  decryption over those primes only (about 8x faster to decrypt and decode a
  level-0 ciphertext at N = 2^16 over 9 primes). By default `message_bound` is
  what the smallest level with at least two bits above the scaling factor
  holds at the scheme's scale; `drop=False` keeps every prime.
- **Breaking:** Rework GP25 [GP25]. `GP25(scheme, gsw_ell=None,
  gadget_params=None, trace_repack=True, bits_per_layer=1)` rotates over
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
  `hw_reducing_key`, `hw_reducing_lvl`, `output_lvl`, and `packing_key` or
  `trace_repack_key` and `output_switch_key`); input keys may have any rank,
  and `ternary=False` takes a binary one. `output_key=None` switches nothing:
  inputs come under the input key and `repack(blind_rotate(...))` leaves the
  result over `R_N` under the rotation key, and
  `generate_repacking_key(rotation_key, n)` makes only the keys `rotate` and
  `repack` read.
  `test_vector(table)` takes exact integers mod q. `bootstrap(rlwe_in, tv,
  key)` is `blind_rotate` (the weight-reducing switch, the switch to `Z_2N`
  and `blind_rotate_exponents`), `repack` and the output switch. The blind
  rotation is one native call on the library's threads, `gp25_blind_rotate`,
  built from `GP25.rotate(acc, keys, automorphism_key)` (`gp25_rotate`) and
  `GP25.multiply_by_signed_monomials`. Repacking now goes through
  `trace_pack` and a ring switch by default, for any `n`, `N` and rank, with
  1-3 MiB of keys against 36-576 MiB for the packing key switch
  (`trace_repack=False`) at N = 1024. `bits_per_layer=2` is BatchBoot's
  multi-bit rotation [LWZ+26]: a pair of rotation bits of value `d` takes
  three MGSW keys, of `[d = 1]`, `[d = 2]` and `[d = 3]`, while the pair's
  power `p` has `4 p <= n` (`GP25.rotation_layers`), for 1.5 times the MGSW
  keys, about 0.8 bit more rotation noise with the RNS gadget, and a rotation
  about 1.4x faster; `encrypt_rotations(values, n_bits, n, key)` makes such
  keys. See Removed for the previous surface.
- **Breaking:** `FoldableRS` and `FieldFoldableRS` now build the general
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
- `FoldableRS.twists` / `FieldFoldableRS.twists` are the general code's
  twist table `T` (the odd position uses `-T`, as in the paper), and the
  field code's are lists of elements rather than integers (a twist of the
  general code need not lie in the prime subfield). `twists2_inv` is gone:
  the fold's `1 / (2 T)` is held privately.
- `relative_distance` is a method, not a property: the general code's bound
  depends on the security parameter.
- The general code over an extension field encodes the base messages of a
  level-`l` message with one batched transform (`ExtensionField.ntt_plan(n0)`,
  in the base domain or, on a field with too little 2-adicity, the extension
  one) where it ran a Horner scheme of `k0` passes; at `k0 = 128` and a `2^20`
  codeword over `F_(p^4)` the encode is 12x faster (1.69 s to 0.14 s,
  avx512ifma), and a field whose only roots of unity lie outside `F_p` now
  has a transform as its base encoder rather than a Vandermonde product.
- The general code's lifts are `lift_twisted` calls alternating between two
  buffers, in place of `fma_interleave` over a tiled and negated table into a
  fresh destination per level: at a `2^22` codeword over `F_(p^4)` the lifts
  of an 11-level encode go from 724 ms to 363 ms and the commit from 1.16 s to
  0.88 s (avx512ifma); with `twist_field="prime"` the commit is 0.66 s.

### Removed

- **Breaking:** `ComplexRing`'s `special_rous` argument, `gen_special_rous`,
  `gen_special_rous_hp`, `CT_rous` and `GS_rous`, and the native special-root
  transforms `CT_NR` / `GS_RN` with their tables (`load_rous_CT`,
  `load_rous_GS`, `bit_reverse_array`), replaced by the FFT kernel.
- `mpmath` is no longer a dependency: its only use was building the old
  transform's root tables.
- **Breaking:** `array_reduce_mod_N` and `array_mod_switch_from_2k` (C,
  `util.h`), whose only uses were the biased uniform sampling fixed below.
- **Breaking:** `MLWE_Scheme.phase`, `LWE.phase`, `mlwe_RNS_phase` and
  `lwe_phase`; use `linear_decrypt` (see Changed).
- **Breaking:** `MLWE_Scheme.full_packing_keyswitch_scaled` and
  `mlwe_full_packing_keyswitch_scaled`: `trace_pack` packs the same samples
  without the factor and with less noise.
- **Breaking:** The unimplemented LWE key switch (`LWE_KS_Key`,
  `lwe_new_KS_key`, `lwe_keyswitch`) and the undefined
  `mlwe_RNSc_extract_lwe`, from `mlwe.h`.
- **Breaking:** GP25's previous surface (see the rework in Changed):
  `GP25(..., debug_key=, threads=)`, `generate_sparse_ternary_key` (use
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

### Fixed

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
- `FieldVector.view` of a view started at the parent's buffers rather than at
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
