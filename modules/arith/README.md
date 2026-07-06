# vfhe.arith

The lattice-crypto arithmetic engine: modular arithmetic, NTT, RNS/CRT
polynomials, tower operations, CKKS-style encoding, and multi-precision
integer arithmetic. Portable C11 baseline with an opt-in AVX-512 IFMA fast
path selected at compile time (`VFHE_SIMD=1`; see the root README).

## Layer stack

Each directory under `c/src/` is one concern; each layer depends only on the
layers below it. Public headers mirror the layers under `c/include/arith/`
(umbrella: `c/include/arith.h`).

```
encode/   complex FFT + threaded batch encode pipeline           (cfft.h)
mp/       base-2^52 big-integer polynomials + CRT bridge         (mp.h)
tower/    limb sets, base conversion, rescaling, scaled lifting  (tower.h)
vec/      RNS scalar vectors (no polynomial structure)           (zqvec.h)
poly/     RNS polynomials: the central, domain-tagged data type  (poly.h)
ring/     ring context = RNS pool x layout x mul strategy        (ring.h)
ntt/      per-prime negacyclic NTT plans (FFTW-style)            (ntt.h)
zq/       per-prime Z_q contexts + element-wise kernel vtable    (zq.h)
nt/       number theory on plain integers                        (nt.h)
```

Key design points:

- **One vtable, zero copy-pasted dispatch.** A `zq_ctx` selects its kernel
  table (`zq_ops`) once at creation based on the prime's bit tier
  (32 / 50 / 64) or the portable build. Every element-wise call site goes
  through `zq_arr_*`; adding a backend (e.g. NEON) is one new file exporting
  one table.
- **Domain is data.** `rns_poly` carries its representation domain
  (`VFHE_COEFF` / `VFHE_EVAL`) next to its limb mask. Operations declare the
  domain they need and return `VFHE_ERR_DOMAIN` instead of computing garbage;
  the check is one integer compare per call. Status codes live in
  `arith/error.h` — the engine never asserts on user-reachable paths.
- **One multiplication kernel.** `poly_mul` / `_addto` / `_subto` / `_into`
  funnel into a single accumulate-mode kernel (`vfhe_acc`), and the
  split-degree cross-term logic exists exactly once (`ring/ring_mul.c`).
- **Hashing is quarantined.** `poly/poly_digest.c` is the only file that
  includes BLAKE3; drop it to build arithmetic without the dependency.

## C / Python boundary

- **C API** = `c/include/arith/*.h`. Structs are visible to C consumers (this
  is a C library; other modules compile into the same LTO'd extension), but
  all derived fields are documented read-only.
- **Python ABI** = `python/cdef/arith.cdef`, a strict subset: opaque handles
  plus accessor functions (`poly_mask`, `poly_domain`, `poly_limb_data`,
  `mp_scalar_digit`, ...). It is identical for portable and SIMD builds; the
  build-dependent `mp_vector_t` never crosses the boundary by value.
- **Python owns zero arithmetic semantics**: `vfhe.arith` adds identity and
  routing (`RingRegistry`), big-int conveniences (prime schedules, CRT
  reconstruction), and operator sugar. The polynomial's `domain` and mask are
  *read* from the native handle, never shadowed.

## Testing

`c/test/` holds one targeted Unity suite per layer -- `test_nt`, `test_zq`,
`test_ntt`, `test_poly`, `test_tower`, `test_mp`, `test_vec` -- sharing
`test_arith_support.h` (deterministic LCG, ring builder, schoolbook oracle).
`python/test/test_arith.py` (pytest) is the characterization suite (Python big
ints as the exact oracle for ring multiplication, CRT, tower ops, the MP
bridge); `test_arith_stress.py` adds randomized stress. `c/fuzz/fuzz_poly.c` is
a libFuzzer harness for the load/transform surface (built by
`scripts/build_fuzzers.py`, run in the nightly CI). Run everything with
`python scripts/run_c_tests.py` (add `VFHE_SANITIZE=address,undefined` for
ASan/UBSan) and `pytest`.

## Runtime CPU dispatch (planned)

The AVX-512 IFMA path is selected at compile time today (`VFHE_SIMD=1`), which
yields an AVX-512-only wheel. A single "fat" wheel that autodetects IFMA at run
time is a bounded change: backend selection already funnels through
`zq_ctx_init` / `ntt_plan_init`, and `base`'s `cpu_detect()` supplies the
feature query. What remains is to build the SIMD kernels with per-function
`__attribute__((target(...)))` so the AVX-512 and portable code coexist in the
one CFFI translation unit, and to keep `mp_vector_t` at its portable width so
the Python cdef is unaffected. It must be verified on x86 (the CI SIMD job +
nightly sanitizers cover that runner); it is intentionally not enabled from an
arm64-only dev box.

## Circuit bridge

`vfhe.circuit.export` lowers GKR circuits to ring elements of this module:
MLE tables (wire values, wiring predicates) load through
`Polynomial.from_array` / `poly_from_int_array` as coefficients. The
dependency points one way (`circuit -> arith`, lazily imported); nothing in
this module knows circuits exist.

Depends on `base`, `rng`, and the vendored BLAKE3.
