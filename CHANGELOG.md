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

- Generate keys on several threads. Every key generator samples through
  `MLWE_Scheme.sample_scaled` (`mlwe_RNS_sample_scaled_batch`, on
  `vfhe_parallel_for`): `gen_ksk`, `gen_rlk`, `gen_ksk_automorphism(_set)`,
  `gen_ksk_trace`, `MGSW_Scheme.encrypt`, `CGGI16.generate_bootstrap_key`,
  GP25's `generate_sparse_ternary_key` and `gen_packing_ksk`, and the CKKS and
  BFV rotation and conjugation keys, each of which takes `n_threads` (0, the
  default: the library limit, `vfhe.engine.set_num_threads`). A sample's mask
  is expanded from a seed, kept as `MLWE.seed` as `sample` keeps it, and its
  noise from a secret seed of its own (`prng_normal_seeded`); every seed is
  drawn in order on the calling thread, so the keys do not depend on the
  thread count, and under `entropy.deterministic` any count gives the same
  keys. A bootstrapping key is one batch, its constants folded into the gadget
  scalars, and `gen_ksk_automorphism_set` draws a level's keys in one batch.
  At N = 2048 over five 42-bit primes, a CGGI bootstrapping key for n = 918
  takes 1.05 s on one thread and 0.23 s on eight at rank 1, and 4.3 s and
  0.87 s at rank 4, where it took 2.9 s and 11.2 s before (i7-11850H,
  avx512ifma).
- Add `MGSW_Scheme.encrypt_constants(values, key)`: MGSW encryptions of
  constant polynomials as one batch, which is what a bootstrapping key is.
- Add `prng_normal_seeded` (C): `generate_normal_random`'s rounded draw as a
  function of a seed, for parallel kernels that must not depend on their
  thread count.
- Add `MLWE(..., obj=)`: adopt a native sample.
- Speed up `Polynomial.get_coeff_matrix` / `from_coeff_matrix` on an
  unsplit ring: a row is now read with `ffi.unpack` and written with one
  packed `ffi.memmove` instead of element by element. `BFV_Scheme.encode` /
  `decode` at N=4096 go from 1.9 / 2.1 ms to 0.27 / 0.29 ms; CKKS `decode`
  reads its coefficients through the same path.
- Correct `Polynomial.base_extend`'s documented contract: it returns a value
  congruent to this element modulo the ring's modulus, not the element
  itself. Behaviour is unchanged; the docstring claimed exactness the fast
  base conversion behind it never provided.
- Add a documentation site, `make docs`, published from `main` to GitHub Pages:
  the guides, the Python API from the docstrings and the C API from the
  Doxygen blocks, built with warnings as errors.
- Add `Polynomial.convert_base(ring, out, exact)`: a value re-expressed over
  another base, which need not contain this one -- the general move
  `base_extend` is the special case of. The default is the fast conversion
  already behind `base_extend`, which writes `x + u * M`; `exact=True` removes
  that overflow, for about a tenth again of the cost, for a value the caller
  keeps clear of the ends of its range. Existing conversions are unchanged.
- Add `fhe.BFV_Scheme`: BFV over `vfhe.mlwe`, with the plaintext modulus taken
  from the ciphertext ring's lowest primes so that `Delta = q/t` is exact.
  Batched encoding into `N` slots (two rows of `N/2`), encrypt/decrypt,
  plaintext and ciphertext multiplication with relinearization, slot rotation
  and conjugation, and modulus switching.
- Add `FieldVector.sample_random_at(seed, indices)`: the seeded elements at
  scattered positions of the sequence `sample_random` fills from, in one call.
- Add `Field.digest_elements(elements)`: the digest of a few elements without
  building a vector to hold them.
- Add `FieldVector.padding_unit`, the width `view`'s `start` and `length` must
  be multiples of -- stated by `view` before, but not obtainable.
- Add `MerklePath.from_bytes(packed)`, and `FoldableRS.fold_pairs` /
  `leaf_digests_of` for a set of positions rather than one.
- Add `FieldVector.fma_interleave(a, b, c_even, c_odd)`: `a + b * c_even` and
  `a + b * c_odd` written straight into one vector of twice the length.
- Add `FieldVector.fold_twisted(twist2_inv, twist, r)`: a vector of
  `(P(x), P(-x))` pairs folded to half its length in one call.
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
- Add `twist_field="field" | "prime"` to `FieldFoldableRS` (`polycom.TWIST_FIELDS`):
  the general code's twists drawn from the whole field (the default) or from
  its prime subfield, with `relative_distance` on the twists' field. Prime
  twists make every lift and fold plane-wise, for more queries at the same
  soundness.
- Add `polycom.batch_inverse(values, p)`: modular inverses by Montgomery's
  trick, which is how `FoldableRS` now builds its fold tables (one
  exponentiation per level and prime rather than one per entry).
- Add `Polynomial.rescale_to_power_of_two(k)`: `round(2^k * c / q)` for every
  coefficient, on centered representatives, as `N` machine words. It is the
  rescale for a target modulus that is not a product of base primes -- how an
  element leaves for a scheme that works at a power of two -- which
  `round_division` and `scaled_lift` between them cannot express. Exact, with
  no floating point anywhere in the rounding.
- Add `ring=` and `scale=` to `CKKS_Scheme.encode`: the ring the plaintext
  lives in (default: level 0's) and the factor the values are scaled by
  (default: `scaling_factor`), e.g. for a ciphertext further down the chain.
- Add `CKKS_Scheme.multiply_plain(ciphertext, plaintext, scale)`: a plaintext
  product without the rescale, with `delta` multiplied by the plaintext's
  `scale` (default: `scaling_factor`; 1 for an unscaled plaintext), so
  several products can be summed before one rescale. `ciphertext *
  plaintext` is this followed by `rescale`.
- Add hoisted automorphisms: `MLWE_Scheme.automorphisms(c, gens, ksks)`
  applies several automorphisms to one ciphertext, decomposing it against the
  gadget once and reusing the digits for every key switch (each automorphism
  permutes them instead of recomputing them). Both gadgets are supported. 16
  rotations at N=2^14 over nine primes: 1.4-1.6x faster on avx512ifma, 2.4x
  on portable. Natively, `mlwe_hoist` / `mlwe_automorphism_RNSc_GHS_hoisted`,
  and the automorphism on the NTT representation of a fully split ring,
  `polynomial_RNS_automorphism_index` / `polynomial_RNS_permute`.
- Add `fhe.CKKS_LinearTransform`: a slot matrix, given by its diagonals and
  acting on every block of `n` slots, encoded once for a level and applied by
  baby-step giant-step with hoisted baby rotations; `rotations` lists the keys
  it needs, and `apply` runs the rotations and products in parallel.
  `slot_to_coeff` / `coeff_to_slot` build SlotToCoeff and its inverse, for
  any `n` dividing `N/2`. Full-packing SlotToCoeff at N=2^13: 0.57 s on one
  thread, 0.18 s on eight (avx512ifma).
- Add `CKKS_Scheme.linear_combination(cts, coefficients, scale)` /
  `MLWE_Scheme.linear_combination`: ciphertexts combined with plaintext
  coefficients in one pass, and `linear_combinations(cts, rows)` for several
  rows of coefficients at once; `MLWE_Scheme.automorphism_batch`, one
  automorphism each on several ciphertexts; and `CKKS_Scheme.conjugate` /
  `gen_conjugation_key`.
- Add `CKKS_Scheme.product(cts)`: the product of ciphertexts as a balanced
  tree, `ceil(log2(n))` levels deep.
- Let `CKKS_Scheme.encode` take any number of values dividing `N/2` and pack
  fewer than `N/2` sparsely (repeated across the slots, so the plaintext is a
  polynomial in `X^(N/2n)`); `decode(..., slots=n)` reads them back.
- Add `MLWE.mod_reduce(ring | lvl)`: a ciphertext reduced into a smaller
  level in place, the value kept rather than divided -- CKKS's level drop.
- Add a library-wide limit on parallelism: `lib.vfhe_set_num_threads(n)` /
  `lib.vfhe_num_threads()`, defaulting to `VFHE_NUM_THREADS` or else 1 (single
  threaded). Every parallel operation stays within it, and those taking `n_threads` (`automorphisms`,
  `automorphism_batch`, `linear_combinations`, `CKKS_LinearTransform.apply`)
  use all of it by default; gp25's `threads` is capped by it, so gp25 too is
  single-threaded until the limit is raised. The threads are
  native (`vfhe_parallel_for`, in util's `parallel.h`), and a parallel operation started
  inside another runs on its caller's thread. The complex-FFT batch, which
  always used 8 threads, now follows the limit too.
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

### Changed

- Key generation leaves its keys in the NTT domain: `MGSW_Scheme.encrypt` and
  the key-switch key generators return samples in NTT form, which every
  consumer converted them to anyway, without the round trip through the
  coefficient domain each sample used to make. The values are unchanged in
  distribution; a pinned stream gives other keys than before.
- Speed up `Polynomial.get_coeff_matrix` / `from_coeff_matrix` on an
  unsplit ring: a row is now read with `ffi.unpack` and written with one
  packed `ffi.memmove` instead of element by element. `BFV_Scheme.encode` /
  `decode` at N=4096 go from 1.9 / 2.1 ms to 0.27 / 0.29 ms; CKKS `decode`
  reads its coefficients through the same path.
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
- Rename the phase `b - <a, s>` to the linear decryption, the scheme-neutral
  term: `MLWE_Scheme.phase` and `LWE.phase` become `linear_decrypt`, and the
  native `mlwe_RNS_phase` / `lwe_phase` become `mlwe_RNS_linear_decrypt` /
  `lwe_linear_decrypt`. The old names are removed.
- Make `CKKS_Scheme.decode` reconstruct the coefficients natively
  (`polynomial_RNSc_to_centered_doubles`: mixed-radix digits in modular
  arithmetic, read in floating point only at the end, so exact up to the
  doubles' rounding for any value) instead of a Python big-integer CRT per
  coefficient: 550 -> 5.2 ms at N=2^14 over 8 primes, the same values to the
  bit. The values are handed to Python as C `double _Complex` pairs, which
  builds the list ten times faster than per element. On avx512ifma the
  conversion of a single-prime plaintext to doubles, and encode's rounding of
  doubles to integers, use the AVX-512 conversions (0.37 -> 0.017 ms for the
  former at N=2^16).
- Make `CKKS_Scheme.decrypt` return the plaintext over the fewest primes that
  hold it -- the last level for a ciphertext at the scheme's scale, higher for
  one at a larger `delta` -- computing the linear decryption over those primes
  only, so decrypting and decoding a level-0 ciphertext at N=2^16 over 9 primes
  takes 2.65 ms instead of 21.9. `message_bound=` sets the bound on the slots
  it assumes; `drop=False` keeps every prime.
- Make `MLWE_Scheme.linear_decrypt` reuse a key over a wider ring instead of
  rebuilding it for the ciphertext's ring, and takes `ring=` to compute it
  modulo a quotient only; `MLWE_Key.at_ring(ring)` builds the key over another ring
  once and keeps it. `Polynomial.mod_reduce` keeps the element's domain.
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
  length `n0` wants a transform, and does without one where the field has none.
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
  codeword over `F_(p^4)` the encode is 12x faster (1.69 s to 0.14 s, avx512ifma),
  and a field whose only roots of unity lie outside `F_p` now has a transform
  as its base encoder rather than a Vandermonde product.
- Make the general code's lifts `lift_twisted` calls alternating between two
  buffers, in place of `fma_interleave` over a tiled and negated table into a
  fresh destination per level: at a `2^22` codeword over `F_(p^4)` the lifts of
  an 11-level encode go from 724 ms to 363 ms and the commit from 1.16 s to
  0.88 s (avx512ifma); with `twist_field="prime"` the commit is 0.66 s.
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
- Make `Ring(..., mask=...)` and `RNSRing.quotient_ring(mask=...)` raise when
  the mask names primes outside the pool they are given, instead of dropping
  them silently. `RNSRing.union` builds the ring over two rings' primes.
- Make `Multiprecision.compute_crt_consts` raise for moduli its single-digit
  Barrett step cannot serve (primes wider than 52 bits) rather than returning
  constants that reconstruct incorrectly.
- Ask the kernel for huge pages for allocations of 8 MiB and up.
  [`USAGE.md`](https://github.com/vfhe/vfhe/blob/main/docs/USAGE.md) covers the allocator settings that go with it.
- Decompose the RNS gadget into balanced digits by default: centered
  residues, in `(-p_j/2, p_j/2]`, rather than residues in `[0, p_j)`.
  `MLWE_Scheme(..., balanced=True)` is the option, which `BFV_Scheme` and
  `CKKS_Scheme` pass through and `MGSW_Scheme` follows unless given its own
  `balanced`. The two digits agree modulo `p_j`, so the decomposition stays
  exact and the keys are the same, and the centered one has a quarter of the
  second moment, which is what the gadget products' noise grows with. Key
  switches, automorphisms (hoisted ones included), relinearization, MGSW
  external products and CMUX, the CGGI16 blind rotation and the LWE packing
  key switch, all over the RNS gadget, add about one bit less noise standard
  deviation at the same cost: at N = 2048 over five
  42-bit primes, a GHS key switch's goes from 2^7.4 to 2^6.4, and a BV
  external product's from 2^49.9 to 2^49.0. Their output ciphertexts change;
  what they decrypt to does not, and `balanced=False` gives the previous
  outputs bit for bit. With balanced digits, `MLWE_Scheme.automorphisms`
  gives the same ciphertexts as `automorphism`: negating a centered digit
  gives the centered digit of the negation. The radix gadget
  (`radix_log_base`) is unchanged. `vfhe.util.io` records the choice with schemes,
  MGSW schemes and key-switch keys. In C, a key-switch key carries the choice
  (`mlwe_new_RNS_ks_key` takes `balanced`), and `gadget_mul_*`,
  `gadget_decompose*`, `mgsw_*`, `gp25_*` and `cggi16_blind_rotate*` take it
  as an argument. The digit is the new `polynomial_RNSc_mod_reduce_lifted_centered`,
  one fused pass per prime at the cost of the `[0, p_j)` lift, which
  `polynomial_RNSc_mod_reduce_lifted` still gives.

### Removed

- **BREAKING**: Reduce `vfhe.util.bindings` to `ffi` and `lib`.
  `active_engine()` and `runnable_engines()` are gone, since the loaded engine
  is internal knowledge; `set_num_threads`
  and `num_threads` wrapped `lib.vfhe_set_num_threads` / `lib.vfhe_num_threads`,
  which callers use directly; the "a faster engine could run" warning is gone —
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

### Fixed

- Fix `arith_scalar_new` on an RNS ring whose primes do not start the base
  copying one value per prime when the scaling kernels index the scalar by
  base row: they read past the copy, and `arith_scale_by` and
  `MLWE.multiply_scalar` scaled the ring's higher rows by whatever lay there.
- Make the entropy-backed generators (`generate_random_bytes`,
  `generate_normal_random`, `generate_uniform_below`, `entropy`) safe to call
  from several threads at once. The 1 KiB pool and its index were one
  process-wide buffer, so concurrent callers could be served the same bytes;
  each thread now keeps a pool of its own. A forked child discards the pool it
  inherits, which it used to serve again after its parent. The
  deterministic-seed counter is advanced atomically, so pinned threads never
  take the same seed, and setting or clearing the override discards every
  thread's pool. Draws of 512 bytes or more are unchanged.
- Correct `Polynomial.base_extend`'s documented contract: it returns a value
  congruent to this element modulo the ring's modulus, not the element
  itself. Behaviour is unchanged; the docstring claimed exactness the fast
  base conversion behind it never provided.
- Fix `CKKS_Scheme.multiply` (inherited from `MLWE_Scheme`) returning the
  first operand's `delta` instead of the product of both, so a product taken
  with it decoded at the wrong scale; `*` was right.
- Fix `ComplexRing(N)` crashing for `N < 8` on avx512ifma, and
  `complex_poly_scale_double` scaling nothing below `N = 4` there: the
  vectorized transforms need a full vector of values, and their table loader
  under-flowed for shorter lengths. Those lengths now run the scalar
  transforms, which every engine compiles.
- Fix `mlwe_round_division` leaving the sample on its old ring: the
  components were divided into the destination ring, but `->ring` still named
  the source, so native code that allocates or key-switches from a sample
  after a rescale worked in the wrong ring.
- Fix `MLWE.new_like()` (behind `copy`, `+`, `-` and plaintext products)
  allocating at the level's defaults rather than like its input: a sample over
  a special ring came back over `rings[-1]`, and an unrelinearized product
  lost every component past the scheme's rank. With neither `lvl` nor `ring`
  given, the result now has the input's ring, rank and `is_extended`.
- Fix the native RNS ring handles stopping being shared after 256 distinct
  rings: past that, every lookup allocated a new handle that was never freed
  (one per sample allocated), and two handles for one ring no longer compared
  equal. The table now grows, under a lock.
- Let `ComplexPolynomial` assignment, `from_array` and `*=` accept any
  `numbers.Complex` / `numbers.Real` (e.g. a subclass of `complex`, or a
  `Fraction`) instead of only the exact builtin types.
- Fix `FieldVector.view` of a view starting at the parent's buffers rather than at
  the view: the nested view's planes skipped the outer offset, so it read and
  wrote the wrong elements. Both vector implementations.
- Fix `Multiprecision.from_polynomial` leaking its result. `free_mp_polynomial`
  was never exposed, so no caller could release one; the returned handle now
  owns its native allocation.
- Fix `MLWE_Scheme(ring, special_primes=n)` dropping the special primes for
  any ring but the first one built over its RNS base. The mask was derived
  from the working primes' *count*, which is their base index only when the
  ring starts at index 0; every other ring got a special ring equal to its
  level 0, so key switching silently ran BV instead of the GHS hybrid, at
  about a prime's worth of extra noise.
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
