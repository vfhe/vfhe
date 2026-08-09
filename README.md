<!-- SPDX-FileCopyrightText: 2026 Alin-Petru Roșu -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# vfhe

[![PyPI](https://img.shields.io/pypi/v/vfhe)](https://pypi.org/project/vfhe/)
[![Python versions](https://img.shields.io/pypi/pyversions/vfhe)](https://pypi.org/project/vfhe/)
[![CI](https://github.com/vfhe/vfhe/actions/workflows/ci-postsubmit.yml/badge.svg)](https://github.com/vfhe/vfhe/actions/workflows/ci-postsubmit.yml)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/vfhe/vfhe/badge)](https://securityscorecards.dev/viewer/?uri=github.com/vfhe/vfhe)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/13712/badge)](https://www.bestpractices.dev/projects/13712)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/vfhe/vfhe/blob/main/LICENSE)

The vFHE library: a library for Zero-Knowledge Proofs, (verifiable) Fully Homomorphic Encryption, and related techniques.

> ❗ **Warning:** This is a pre-release version of the library and is subject to breaking changes. ❗

---

## Modules

Each module is a self-contained folder under `modules/`. A module is
**Python-facing** (ships a `python/` package + a `cdef` exposing its C to
Python) or **internal C-only** (contributes compiled kernels used by other
modules, no Python symbols).

| Module | Kind | What it provides |
|---|---|---|
| `arith` | Python-facing | RNS polynomial arithmetic over `Z_q[X]/(X^N+1)`: incomplete NTTs, complex FFTs, general multiprecision, and basic number theory procedures |
| `misc` | Python-facing | The native handle (`ffi`/`lib`/`libvfhe`) plus the internal C utilities: BLAKE3-seeded PRNG, AES-CTR RNG, aligned allocation, mod-switching helpers |
| `mlwe` | Python-facing | LWE / Module-LWE and MGSW: key generation, encryption, key-switching, arithmetic, and ring morphisms |
| `fhe` | Python-facing | Schemes on top of `mlwe`: CKKS (encode/encrypt/rescale/rotate/multiply), CGGI16 functional bootstrap, GP25 sparse-amortized bootstrap |
| `piop` | Python-facing | Sketch of IOP prover/verifier framework (currently under development) |
| `circuit` | Python-facing | Arithmetic circuits (protobuf wire format) and their polynomial export to `arith` (currently under development) |
| `compiler`, `polycom`, `snark`, `vfhe` | placeholder | reserved for the compiler frontend, polynomial commitments, the SNARK layer, and the top-level assembly (currently under development) |

## Installing

Supported platforms: Linux and macOS (POSIX only), on x86_64 and arm64.

```bash
pip install vfhe
```

vFHE ships one **engine** per instruction-set level it has kernels for —
today the **portable** one that runs on any CPU, and **avx512ifma** for
x86_64 with AVX-512 IFMA — and **picks one when you import it**: on a
machine with AVX-512 IFMA you get the SIMD kernels, AES-NI, and BLAKE3
SIMD; anywhere else, the portable engine. An install carries every engine
its architecture can run, so it is safe to move between machines of that
architecture. Wheels cover Linux x86_64/arm64 and macOS arm64; elsewhere
pip builds the same install from the sdist, which needs a C compiler
(clang/gcc) and bundles the BLAKE3 sources, so no submodules are required.

> **Choosing the compiler** (sdist builds only). The build uses your
> interpreter's compiler by default (no configuration needed). To pick a
> specific one (e.g. with several installed), set the standard `CC`
> environment variable:
>
> ```bash
> CC=gcc-14 pip install --no-binary vfhe vfhe
> ```

> **What did I get?** `python -m vfhe.info` prints the version, the engine
> this CPU selected, anything faster it could have run, and the platform —
> the whole environment a bug report needs.
>
> **Pinning the engine.** `VFHE_ENGINE=<name>` overrides the CPU probe at
> import and refuses if this CPU cannot run that engine. Pinning a slower
> engine than the CPU deserves gets a one-time hint (silence it with the
> usual `warnings` filters).

### Verifying a release

Each release artifact — sdist and wheels — is signed with Sigstore build
provenance. To verify one came from this repository's release workflow (needs
the [GitHub CLI](https://cli.github.com)):

```bash
pip download --no-deps --no-binary :all: vfhe==<version>   # fetch the sdist
gh attestation verify vfhe-<version>.tar.gz --repo vfhe/vfhe
```

Expected output confirms the attestation and shows the signer identity: the
`.github/workflows/release-pypi.yml` workflow of `vfhe/vfhe`, issued by GitHub
Actions' OIDC (`https://token.actions.githubusercontent.com`). A mismatch, or
no attestation, means the artifact is not a genuine release. PyPI also records
these attestations at `https://pypi.org/project/vfhe/#files`.

---

## Usage

A walkthrough of encrypted computation with CKKS is in
[`USAGE.md`](https://github.com/vfhe/vfhe/blob/main/USAGE.md);
[`smoke/ckks.py`](https://github.com/vfhe/vfhe/blob/main/smoke/ckks.py) is its
runnable form. More examples will be added as the library continues to be developed.

---

## Development

The development guide, covering the repository layout, the build system, testing,
coverage, and CI, is in
[`DEVELOPMENT.md`](https://github.com/vfhe/vfhe/blob/main/DEVELOPMENT.md).
The avx512ifma engine is tested even without AVX-512 IFMA hardware:
`make test ENGINE=avx512ifma` runs the suites on it, natively on IFMA machines and under the
[Intel Software Development Emulator](https://www.intel.com/content/www/us/en/download/684897/intel-software-development-emulator.html)
elsewhere (x86_64 Linux; CI runs it on every pull request).
Contribution expectations are in
[`CONTRIBUTING.md`](https://github.com/vfhe/vfhe/blob/main/CONTRIBUTING.md).

---

## Authors

See [`AUTHORS.md`](https://github.com/vfhe/vfhe/blob/main/AUTHORS.md); how the
project is run is in
[`GOVERNANCE.md`](https://github.com/vfhe/vfhe/blob/main/GOVERNANCE.md).

Maintainers can be reached at <maintainers@vfhe.ai>.

---

## Citation

If you use vFHE in academic work, please cite the software using
[`CITATION.cff`](https://github.com/vfhe/vfhe/blob/main/CITATION.cff) (GitHub renders a **"Cite this repository"**
button from it) and the archived release DOI where relevant. (When a paper is
published, we'll add it as the preferred citation.) In BibTeX form:

```bibtex
@software{The_vFHE_Library,
  author = {Cascudo, Ignacio and Costache, Anamaria and Cozzo, Daniele and
            Dubois, Adrien and Fiore, Dario and Guimarães, Antonio and
            Köstler, Robin and Osadnik, Michał and Roșu, Alin-Petru and
            Soria-Vazquez, Eduardo},
  license = {Apache-2.0},
  title = {{The vFHE Library}},
  url = {https://github.com/vfhe/vfhe}
}
```

<!-- TODO(release): once a release is archived on Zenodo (enable the
GitHub-Zenodo integration, then cut a GitHub release), add the DOI badge:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
-->

## Acknowledgements

This work is supported by: Grants JDC2024-055789-I and Excelencia María de Maeztu (CEX2024-001471-M) funded by MICIU/AEI/10.13039/501100011033 and ESF+; the PICOCRYPT project that has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (Grant agreement No. 101001283); the ERC Proof of Concept grant VERIFHE (GA: 101287502). Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.
