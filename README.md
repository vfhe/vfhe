# vfhe

[![PyPI](https://img.shields.io/pypi/v/vfhe)](https://pypi.org/project/vfhe/)
[![Python versions](https://img.shields.io/badge/python-3.10--3.14-blue)](https://pypi.org/project/vfhe/)
[![CI](https://github.com/vfhe/vfhe/actions/workflows/ci.yml/badge.svg)](https://github.com/vfhe/vfhe/actions/workflows/ci.yml)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/vfhe/vfhe/badge)](https://securityscorecards.dev/viewer/?uri=github.com/vfhe/vfhe)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/vfhe/vfhe/blob/main/LICENSE)

The VFHE library: a library for Zero-Knowledge Proofs, (verifiable) Fully Homomorphic Encryption, and related techniques.

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
| `circuit` | Python-facing | Layered GKR arithmetic circuits (protobuf wire format) and their polynomial export to `arith` |
| `compiler`, `polycom`, `snark`, `vfhe` | placeholder | reserved for the compiler frontend, polynomial commitments, the SNARK layer, and the top-level assembly |

## Installing

```bash
pip install vfhe
```

VFHE ships **as an sdist only, with no pre-built wheels**, so every install compiles
from source and **tunes to your CPU**. On an x86 machine with AVX-512 IFMA this
enables `-march=native`, the SIMD kernels, AES-NI, and BLAKE3 SIMD; on any other
CPU it builds the portable engine (and prints a notice). The result targets **this**
machine. You need a C compiler (clang/gcc); the sdist bundles the BLAKE3 sources,
so no submodules are required.

> **Choosing the compiler.** The build uses your interpreter's compiler by
> default (no configuration needed). To pick a specific one (e.g. with several
> installed), set the standard `CC` environment variable; it steers both
> the CPU detection and the compile, in lockstep:
>
> ```bash
> CC=gcc-14 pip install vfhe
> ```

> **Forcing a portable build.** If you build in one place and run in another
> (Docker image built on a big CI box, run on a smaller node), set
> `VFHE_PORTABLE=1` to skip CPU detection and build the portable engine that runs
> on any CPU; `-march=native` would otherwise bake in the *build* host's
> features and crash elsewhere. When the portable engine ends up on a CPU that
> *does* support AVX-512 IFMA, VFHE prints a one-time hint to rebuild tuned
> (silence it with the usual `warnings` filters).

---

## Development

The development guide, covering the repository layout, the build system, testing,
coverage, and CI, is in
[`docs/DEVELOPMENT.md`](https://github.com/vfhe/vfhe/blob/main/docs/DEVELOPMENT.md).
Contribution expectations are in
[`CONTRIBUTING.md`](https://github.com/vfhe/vfhe/blob/main/CONTRIBUTING.md).

---

## Authors

See [`AUTHORS.md`](https://github.com/vfhe/vfhe/blob/main/AUTHORS.md). Authors are sorted alphabetically by surname,
following [mathematical tradition](https://www.ams.org/profession/leaders/culture/JointResearchandItsPublicationfinal.pdf).

Maintainers can be reached at <maintainers@vfhe.ai>.

---

## Citation

If you use VFHE in academic work, please cite the software using
[`CITATION.cff`](https://github.com/vfhe/vfhe/blob/main/CITATION.cff) (GitHub renders a **"Cite this repository"**
button from it) and the archived release DOI where relevant. (When a paper is
published, we'll add it as the preferred citation.)

Once a release is archived on [Zenodo](https://zenodo.org) (enable the
GitHub-Zenodo integration, then cut a GitHub release), add the DOI badge here:

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```
