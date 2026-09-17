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
  security_bits)`, that bound as a pure function, validated against the
  paper's Table 1.
- Add `instantiation=` and `seed=` to `FoldableRS` / `FieldFoldableRS`, with
  `polycom.INSTANTIATIONS` and `polycom.DEFAULT_SEED`; and `twists_odd`, the
  second twist table, next to `twists`.

- Add `Polynomial.rescale_to_power_of_two(k)`: `round(2^k * c / q)` for every
  coefficient, on centered representatives, as `N` machine words. It is the
  rescale for a target modulus that is not a product of base primes -- how an
  element leaves for a scheme that works at a power of two -- which
  `round_division` and `scaled_lift` between them cannot express. Exact, with
  no floating point anywhere in the rounding.

### Changed

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
  length `n0` wants a transform, and does without one where the field has none.
- `FoldableRS.twists` / `FieldFoldableRS.twists` are the general code's
  even-position table `T`, and the field code's are lists of elements rather
  than integers (a twist of the general code need not lie in the prime
  subfield). `twists2_inv` is gone: the fold's tables are `1 / (T - T')` and
  `-T'`, held privately.
- `relative_distance` is a method, not a property: the general code's bound
  depends on the security parameter.

### Fixed

- `FieldVector.view` of a view started at the parent's buffers rather than at
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

### Changed

- `FieldVector.query` also accepts a buffer of unsigned 64-bit indices, handed
  to the gather as it stands; the kernel now range-checks them itself.
- `FieldVector.hash_elements` / `hash_fibers` return a read-only `memoryview`
  of the kernel's buffer rather than a copy. It compares, slices and converts
  like `bytes`; `bytes(digests)` is the copy, and is what hashing or keying by
  them needs.
- `Merkle.from_digests` accepts any bytes-like and keeps it rather than
  copying, so a mutable buffer stays connected -- `commit()` re-reads it.

- `Ring(..., mask=...)` and `RNSRing.quotient_ring(mask=...)` now raise when
  the mask names primes outside the pool they are given, instead of dropping
  them silently. `RNSRing.union` builds the ring over two rings' primes.
- `Multiprecision.compute_crt_consts` raises for moduli its single-digit
  Barrett step cannot serve (primes wider than 52 bits) rather than returning
  constants that reconstruct incorrectly.

- Allocations of 8 MiB and up ask the kernel for huge pages.
  [`USAGE.md`](docs/USAGE.md) covers the allocator settings that go with it.

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
