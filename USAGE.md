<!-- SPDX-FileCopyrightText: 2026 Alin-Petru Roșu -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Using vFHE

> **Warning:** vFHE is pre-release and not audited; do not use it in
> production. See
> [SECURITY.md](https://github.com/vfhe/vfhe/blob/main/SECURITY.md).

Install with `pip install vfhe`; the install carries both engines and picks
the right one for your CPU at import. Compiler selection (sdist builds) and
engine pinning are covered in the
[README](https://github.com/vfhe/vfhe/blob/main/README.md#installing).

## CKKS in five steps

CKKS computes on encrypted vectors of complex numbers; results are approximate
to the configured scale.

```python
from vfhe.arith import Ring
from vfhe.fhe import CKKS_Scheme

# 1. A scheme over Z_q[X]/(X^256 + 1): 128 complex slots, scale 2^49.
scheme = CKKS_Scheme(
    Ring(256, 300, split_degree=1), scaling_factor=2**49, special_primes=0
)

# 2. Keys: a sparse secret key, and the relinearization key multiplication needs.
key = scheme.key_gen_sparse(32, 3.2)
secret = key.poly[0]
scheme.rlk = scheme.gen_ksk(key, [-(secret * secret)])

# 3. Encode and encrypt two vectors (padded to the 128 slots).
a = [1.5 + 0.5j, -2.0 + 1.0j] + [0j] * 126
b = [0.5 - 0.5j, 1.0 + 0.0j] + [0j] * 126
ct_a = scheme.encrypt(scheme.encode(a), key)
ct_b = scheme.encrypt(scheme.encode(b), key)

# 4. Compute under encryption.
ct_sum = ct_a + ct_b
ct_prod = ct_a * ct_b  # relinearizes and rescales via scheme.rlk

# 5. Decrypt and decode.
sum_result = scheme.decode(scheme.decrypt(ct_sum, key))
prod_result = scheme.decode(
    scheme.decrypt(ct_prod, key), scaling_factor=ct_prod.delta
)
```

[`smoke/ckks.py`](https://github.com/vfhe/vfhe/blob/main/smoke/ckks.py) is the
runnable, CI-verified form of this walkthrough; it checks every result against
plaintext arithmetic.

## Going further

- The module surfaces (`vfhe.arith`, `vfhe.mlwe`, `vfhe.fhe`, `vfhe.circuit`,
  ...) are listed in the
  [README](https://github.com/vfhe/vfhe/blob/main/README.md#modules); each
  module's public API is its `__init__` re-exports.
- Runtime C extensions: `vfhe.misc.dynamic_extensions` compiles user C files
  into the loaded engine; `vfhe_create_headers` generates the umbrella header
  to include. See the module docstrings.
- Reproducible randomness (tests only):
  `vfhe.misc.libvfhe.lib.vfhe_prng_set_deterministic_seed(seed)`.
