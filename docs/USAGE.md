# Using vFHE

For developers computing on encrypted data with vFHE. It walks one CKKS
example end to end; the module surfaces are listed in the
[README](https://github.com/vfhe/vfhe/blob/main/README.md#modules).

> **Warning:** vFHE is pre-release and not audited; do not use it in
> production. See
> [SECURITY.md](https://github.com/vfhe/vfhe/blob/main/docs/SECURITY.md).

## Prerequisites

- **Python 3.10 or later**
- **vFHE installed** — `pip install vfhe`, covered in the
  [README](https://github.com/vfhe/vfhe/blob/main/README.md#installing)

## Example: CKKS in five steps

CKKS computes on encrypted vectors of complex numbers, and results are
approximate to the configured scale. Save this as `ckks_example.py`:

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

print(f"a + b -> {sum_result[0]:.4f}  (plaintext {a[0] + b[0]})")
print(f"a * b -> {prod_result[0]:.4f}  (plaintext {a[0] * b[0]})")
```

## Verify it works

```bash
python ckks_example.py
```

Both encrypted results match the plaintext arithmetic to a few decimals:

```text
a + b -> 2.0000-0.0000j  (plaintext (2+0j))
a * b -> 1.0000-0.5000j  (plaintext (1-0.5j))
```

The trailing digits differ per run, because CKKS carries encryption noise.
A result wrong in the first decimals means the parameters are too small for
the computation, not that the install is broken.

## Next steps

- Raise `Ring`'s degree and modulus bit-count for deeper computations, at the
  cost of speed and ciphertext size.
- Read each module's `__init__` re-exports for its public API, listed in the
  [README](https://github.com/vfhe/vfhe/blob/main/README.md#modules).
- Compile your own C against the installed library with
  `vfhe.dynamic_extensions`.

Additional examples will be added as the library evolves.
