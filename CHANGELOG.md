<!-- SPDX-FileCopyrightText: 2026 The vfhe authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until 1.0.0, minor
versions may contain breaking changes.

## [Unreleased]

## [0.0.4] - 2026-11-13

### Added

- The execution tape, `vfhe.util.trace`: while a trace is enabled, every
  ring operation the native code makes over an RNS ring leaves an event
  with its kind, operands, parameters and output, cut into operations by
  the public calls that made them; it can be bounded, drained by another
  thread and verified against digests of its values. The `vfhe.vfhe`
  package reads it: `computation` turns a trace into the steps of a
  homomorphic computation (multiplications with their tensor products, key
  switches and digits, rescales, plaintext-linear steps), `witness` applies
  an arithmetization and checks every bound and relation on the recorded
  values, and `cccfgs26` is the arithmetization of [CCCFGS26, App. A.5].
- Native multithreading under one library-wide limit, `vfhe.util.parallel`:
  `set_num_threads`, `num_threads` and `local_num_threads`, defaulting to
  `VFHE_NUM_THREADS` or 1. Key switches, automorphisms, products, rescales,
  domain conversions, key generation, the proof-side kernels and every
  batch call run on those threads, with results independent of their
  number.
- Batch calls that do many operations in one native call: `sample_batch`,
  `encrypt_batch`, `encode_batch`, `multiply_batch`, `relinearize_batch`,
  `round_division_batch`, `rescale_batch`, `rational_rescale_batch`,
  `automorphism_batch`, `automorphism_sum`, `CMUX_batch`,
  `functional_bootstrap_batch` and `encrypt_constants`.
- A memory pool, `vfhe.util.memory_pool`, keeping the buffers the library
  releases for the next request of the same size, bounded by `set_capacity`
  (`"auto"`, bytes or `0`), with `release_all`, `statistics` and
  `set_wipe_on_release`; `VFHE_MEMPOOL_CAPACITY` and
  `VFHE_MEMPOOL_WIPE_ON_RELEASE` set the defaults.
- Serialization, `vfhe.util.io`: `Encoder`, `Decoder` and `Codec`, the
  plain-value codecs and combinators, and codecs for rings, polynomials,
  MLWE, MGSW, CKKS and BFV schemes, samples, keys and key-switch keys,
  `CKKS_LinearTransform` and the bootstrapping keys, streaming a key one
  sample at a time with a fresh sample written as its mask seed. The format
  is a length-prefixed protobuf message per object, with schemas under
  `proto/`.
- `vfhe.util.kernels`: compile your own C kernels into the library as one
  link-time-optimised engine, `kernels.ExtensionBuilder(sources,
  declarations).build()`, cached under `~/.cache/vfhe`; the `kernels` extra
  pulls in meson and ninja, and wheels ship the library's C sources for it.
- `fhe.BFV_Scheme`: BFV over `vfhe.mlwe` with batched encoding,
  plaintext and ciphertext multiplication, rotation, conjugation and modulus
  switching.
- Bootstrapping: the SHIP bootstrap [CHKS25] (`fhe.SHIP`), a native CGGI16
  blind rotation with an optionally unfolded loop, and the GP25 rework
  described under Changed.
- `fhe.CKKS_LinearTransform`: a slot matrix by its diagonals applied by
  baby-step giant-step with hoisted rotations, with `slot_to_coeff` and
  `coeff_to_slot`.
- `CKKS_Scheme`: `encode(ring=, scale=)` also from a complex buffer in
  place, sparse packing of fewer than `N/2` values, `multiply_plain`,
  `linear_combination(s)`, `conjugate`, `rotation_generator`, `product`,
  and `decrypt(message_bound=, drop=)` over the fewest primes that hold the
  plaintext.
- `MLWE_Scheme`: hoisted automorphisms (`automorphisms`, `automorphism_sum`,
  `automorphism_mux` of [CHKS25]), `linear_combination(s)` with
  `PlaintextMatrix` or an `IntegerMatrix`, `trace_pack` [CDKS21, LY26],
  `normalized_trace` [AP13], ring switching [GHPS12], LWE extraction and the
  packing key switch, `key_gen(distribution="ternary")`, `gen_ksk(...,
  hybrid=False)`, `multiply_by_monomial`, `from_components`, `mod_reduce`,
  `level_with_primes`, and `linear_decrypt` of an extended product.
- MGSW internal products, automorphisms and noiseless encryptions
  (`MGSW.internal_product`, `MGSW_Scheme.automorphism`, `trivial`), and
  `mgsw_shift_sequences` in C.
- Lookup and range arguments in `vfhe.piop`: `Relation_Lookup` with
  `Lookup` and `LogUpGKR` [Hab22, PH23] (the default), `Relation_Range` with
  `RangeDecomposition(max_digit_bits)` and `Relation_ResidueRange` with
  `ResidueRangeDecomposition`, the range PIOP of [CCCFGS26] over packed
  `R_q` polynomials and over residue oracles; `modules/piop/range.md` is the
  design note.
- `vfhe.piop` composition: indexed relations, sequential composition
  (`compose`, statement heads, `Relation_SumDef`, GKR as a composition),
  AND-composition (`IOP.and_composition`, `IOP.batched_call`), bundle
  protocols (`Protocol.bundle_key`), oracles over their own domain
  (`Oracle.domain`), several commitment schemes in one IOP,
  `ClosedFormOracle`, `VirtualOracle.partial_evaluation`, and oracles sent
  as messages (`OracleMessages`, `BasefoldMessages`).
- Basefold: batched openings of several claims on any of a scheme's
  commitments, layer schedules, `BatchFoldEval` over `R_q` through CRT
  components [CCCFGS26, §7], the parameter estimator `basefold_parameters`,
  `relative_distance(security_bits)` and `foldable_relative_distance` with
  the `"cccfgs26"` bound, the general foldable code's `instantiation`,
  `seed`, `twist_field` and `root_field`, `truncated`, `residue_code`,
  batch encoding and folding, and fibers of adjacent symbols.
- Native kernels on fields and vectors: `Field.lagrange_interpolate`,
  `FieldVector.reindex`, the `fold_*` inner products and product sums,
  `fold_twisted`, `lift_twisted`, `fma_interleave`, prime-subfield tables,
  `set_uint64`, `to_uint64`, `set_radix_decomposition`, `coefficients`,
  `sample_random_at`, `digest_elements`, `digest`, native `inverse`,
  `pmf_inverse`, `histogram`, `count_occurrences`, `crt_lift_mod_2_128`
  and `ExtensionFieldNTT.pack(zero_padding=, layout=)`.
- Rings: `IntegerMatrix` with `integer_linear_combinations`, `RNSScalar`,
  CRT components (`crt_components`, `from_crt_components`, `residue_field`,
  `crt_linear_combination`, `from_diagonal`), `lift_mod_2_128`,
  `to_uint64` and `from_uint64` on polynomials and tables,
  `convert_base`, `rescale_to_power_of_two`, `gen_primes_near`,
  `Ring.digest`, `MLE.linear_combination`, `digest_sequence`, and the
  generic operations `decompose_digit`, `round_division_approximate`,
  `scaled_lift` and `scale_subto`.
- Native sumcheck rounds for products of any number of ring tables.
- `Merkle.open_many` and `verify_many`, `polycom.LayerAnswers`,
  `vfhe.crypto.join_digests`, `MerklePath.from_bytes`, `chain_digest` and
  `DigestChain`, seeded exceptional samplers (`random_exceptional(seed=)`),
  `entropy.below_many`, seed expansion by AES-128 in counter mode and
  `prng_normal_seeded` in C.
- Value equality of rings and schemes by their parameters.
- Per-module benchmarks, `make bench`, with `tools/bench/compare.py`;
  a documentation site, `make docs`, published to GitHub Pages; Python 3.15;
  and `docs/USAGE.md` on threads, processes and memory.

### Changed

- **BREAKING (proof format)**: Fiat-Shamir proofs change. The chain is
  seeded with the root statement's digest and the registry digest, so a
  proof is bound to the verifier's configuration; every value a transcript
  absorbs has a `digest()` method; answers of an opening are layer-major
  (`LayerAnswers`); field tables sent as messages travel as vectors; ring
  challenges are drawn exactly uniformly from an exceptional set whose
  points are one element in every slot; and the general foldable code of
  [ZCF24, Def. 5] is the default (`instantiation="rs"` is the previous
  one), with twists in the residue fields on a split ring and the base
  transform's root in `F_p`.
- **BREAKING**: `vfhe.piop` is restructured: the framework is one file per
  concern and is what `vfhe.piop` exports, relations are
  `vfhe.piop.relations`, protocols `vfhe.piop.protocols`, with nothing
  re-exported. A statement names oracles, never tables (`TableOracle`,
  `CommittedOracle`, `ImplicitOracle`); `Verifier.challenge` and
  `challenge_bits` are coroutines; `Relation_Circuit` is an indexed
  relation and `GKR` a composed protocol; `Relation_Range(bound)` takes the
  bound alone and `RangeDecomposition` the digit parameters.
- **BREAKING**: `vfhe.util` holds the tooling: `vfhe.engine` is
  `vfhe.util.bindings` (`ffi` and `lib` alone), the catalogue is
  `vfhe.util._engine`, `vfhe.util.kernels` replaces `vfhe.dynamic_extensions`,
  the kernels command is `python -m vfhe.util.kernels`, and `python -m
  vfhe.info` prints JSON.
- **BREAKING**: Gadget parameters are one object, `GadgetParams(log_base,
  digits, primes_per_digit)`, replacing `radix_log_base` everywhere, with
  grouped RNS digits (the `dnum` of [HK20]) and approximate decomposition;
  digits are balanced by default; the division by the special primes is the
  ModDown of hybrid key switching. Outputs change; what they decrypt to does
  not.
- **BREAKING**: Fresh samples and keys are drawn from seeds, kept as
  `MLWE.seed`, which `vfhe.util.io` stores in place of the mask; keys are
  left in the NTT domain. Distributions are unchanged.
- **BREAKING**: `MLWE_Scheme(ring, special_primes=k)` derives every level
  down to one prime by default; `phase` is `linear_decrypt`; prime selection
  follows the ring's splitting (a partially split ring takes primes with an
  odd multiplier, `gen_prime` takes `split_degree`); `Ring(mask=)` and
  `quotient_ring(mask=)` raise for primes outside their pool.
- **BREAKING**: GP25 [GP25] is reworked: `GP25(scheme, gsw_ell, gadget_params,
  bits_per_layer)` bootstraps samples of any `n` and `N` between an input
  key, a sparse key switched to by a Hamming-weight-reducing key, and an
  output key, with repacking through `trace_pack` and a ring switch;
  `bits_per_layer=2` is the multi-bit rotation of [LWZ+26].
- **BREAKING**: The CKKS encoder runs on a negacyclic FFT kernel;
  `ComplexRing(N)` takes no `special_rous`, and building a scheme at
  N = 2^16 takes milliseconds.
- **BREAKING (C)**: `engine.h` is `x86_64.h`; `util.h` is `alloc.h` with the
  index and modulus helpers moved to `arith.h`; cffi declaration files are
  `.h`; `polynomial_RNS_get_natural` and `set_natural` take the row order;
  `mlwe_full_packing_keyswitch` takes no `lvl`; `-Dfuzz=<engine>` replaces
  two build options. Engines are built with link-time optimisation.
- **BREAKING**: `Merkle.from_digests` reads its buffer once and keeps no
  reference; `commit(digests)` recommits.
- Faster: `Lookup` bundles claims of one table into one argument; batched
  openings touch only the block a 0/1-suffixed point selects; the ring's
  `get_polynomial`, `get_coeff_matrix` and `from_coeff_matrix`, CKKS
  `decode` and `decrypt`, field vector inversion, `expand_table`, the
  general code's encode and lifts, and `ExtensionFieldNTT.pack` are one
  native pass or several times faster; large allocations ask for huge
  pages; `RNSRing` equality returns at once for one object.
- The version is a static line in `pyproject.toml`; ruff and pyright are
  configured in their own files; `setuptools` is no longer a runtime
  dependency; every copyright header reads `The vfhe authors`.
- The library is written `vfhe` everywhere, in prose as in code, as its
  authors voted; `vFHE` is gone from the documentation, the metadata and
  the copyright line.

### Removed

- **BREAKING**: Python 3.10, past its end of life.
- **BREAKING**: `vfhe.dynamic_extensions` and the hot swap, `rebind_state`,
  the standalone C library (`libvfhe_<engine>` shared and static libraries,
  pkg-config files, installed headers, `engine-<engine>.json`),
  `vfhe_engine_active` and the C CPU probe, `libvfhe` / `LibVFHE`,
  `active_engine`, `runnable_engines` and `cache_root`.
- **BREAKING**: `ImplicitOracle`, `ImplicitEval` and the `implicit` registry
  entry: an oracle defined by a sum is a `Relation_SumDef` statement.
- **BREAKING**: `ComplexRing`'s special-root transforms and `mpmath`;
  `array_reduce_mod_N` and `array_mod_switch_from_2k`; `phase` and its
  native forms; `full_packing_keyswitch_scaled`; the unimplemented LWE key
  switch; GP25's previous surface (`sab_rlwe_bootstrap`, `encrypt_bits`,
  `generate_sparse_ternary_key` and the rest, see the rework); and
  `CGGI16_Key.b_prec`.

### Fixed

- Uniform sampling below a prime was biased: masks, keys and
  `sample_uniform` now draw by rejection and are exactly uniform.
- `MLWE_Scheme(ring, special_primes=n)` dropped the special primes for every
  ring but the first over its base, so key switching ran BV with a prime's
  worth of extra noise; `special_primes >= 2` and one-prime rings raised.
- MGSW keys of a level whose primes do not come first in the ring gave
  garbage from every external product and CMUX; `external_product` and
  `CMUX` returned level-0 samples whatever the input's level.
- `MLWE_Scheme.sample` at a level sharing no prime with level 0 did not
  decrypt; `mlwe_round_division` left a sample on its old ring;
  `MLWE.new_like` lost the ring, rank or extension of its input; `ct + p`
  and `ct - p` returned an uninitialized mask; `CKKS_Scheme.multiply`
  returned the wrong `delta`.
- CKKS encoding at a scale reaching 2^63 wrapped silently and now reduces
  exactly; a non-finite value raises `ValueError`; `ComplexRing(N)` crashed
  for `N < 8` on avx512ifma.
- `to_uint64` and `from_uint64` on a ring listing its primes out of the
  base's order permuted the rows, so range proofs rejected honest claims;
  `from_coeff_matrix` read past a short row; `from_bigint_array` with fewer
  than `N` values read heap contents; the product of a polynomial by a
  large or numpy integer raised.
- The entropy stream served several threads the same bytes and a forked
  child its parent's; a forked child could hang on its first ring lookup;
  native ring handles stopped being shared after 256 rings.
- Multiprecision reconstruction crashed with a second ring, paired constants
  with the wrong primes, returned unreduced values and leaked; `ExtensionField`
  accepted a composite modulus; `FieldVector.view` of a view read the wrong
  elements; `field_vec_inv` wrote before finding a zero.
- `Basefold.commit` with a prover kept every proof's opening on the scheme;
  `BasefoldEval`, `RangeDecomposition` and `Lookup` drew challenges in the
  IOP's domain instead of their own; `Lookup` raised on tables of fewer than
  eight entries; `Division.remainder` was wrong when the divisor exceeded
  the quotient's modulus, and `cccfgs26` rejected honest computations.
- `Ring(N, prime_size=[...])` searched forever when no prime of a size was
  left, and `Ring(N, primes=[...])` failed without `prime_size=None`.
- The documented meaning of `delta` in `soundness_error` and the contract of
  `base_extend` were corrected; behaviour is unchanged.

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
    vfhe ships.

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

[Unreleased]: https://github.com/vfhe/vfhe/compare/0.0.4...HEAD
[0.0.4]: https://github.com/vfhe/vfhe/compare/0.0.3...0.0.4
[0.0.3]: https://github.com/vfhe/vfhe/compare/0.0.2...0.0.3
[0.0.2]: https://github.com/vfhe/vfhe/compare/0.0.1...0.0.2
[0.0.1]: https://github.com/vfhe/vfhe/releases/tag/0.0.1
