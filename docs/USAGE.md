# Using vFHE

For developers computing on encrypted data with vFHE. It walks one CKKS
example end to end; the module surfaces are listed in the
[README](https://github.com/vfhe/vfhe/blob/main/README.md#modules).

> **Warning:** vFHE is pre-release and not audited; do not use it in
> production. See
> [SECURITY.md](https://github.com/vfhe/vfhe/blob/main/docs/SECURITY.md).

## Prerequisites

- **Python 3.11 or later**
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
prod_result = scheme.decode(scheme.decrypt(ct_prod, key), scaling_factor=ct_prod.delta)

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
the computation.

## Record what ran

For verifiable computation, vFHE can record every ring operation a
computation makes, with a copy of each result, and name the values an
arithmetization needs. With the example's scheme and ciphertexts:

```python
from vfhe.util import trace
from vfhe.vfhe import cccfgs26, witness

with trace.recording() as t:
    product = ct_a * ct_b
operations = t.operations()

for op in operations:
    print(op.name, [e.kind for e in op.all_events()][:6])

w = witness.of(operations, cccfgs26)
print([e.name for e in w.entries])
w.check()
```

`trace.recording` records the operations made inside the block; while it
records, the computation takes a slower path through the generic ring
interface, so nothing is left out. `witness.of` reads the recorded product
as the paper's witness (`d0`, `d1`, `d2`, the digits `w_ks`, the quotients
and remainders of the rescale) and `check` evaluates the relations between
them on the recorded values. `witness.range_classes(w)` groups the values
that carry a bound, and `witness.coefficients(w, "w_ks")` gives one
class's values as signed coefficient vectors, the input of a range check.
The tape is one: a second `recording` resumes
it, and a value it already holds, such as a key, is not recorded again, so
a witness built from a later piece is given the earlier values
(`witness.of(..., known=)`). A long computation is read in pieces that way,
and with `trace.enable(limit=...)` a consumer thread drains the tape while
the computation runs.

## Threads, processes and memory

vFHE runs on one thread unless told otherwise. The limit is one setting for
the whole library: `VFHE_NUM_THREADS` in the environment, or
`vfhe.util.parallel.set_num_threads(n)` at run time, and every operation that
can use several threads stays within it. `local_num_threads(n)` lowers it for
the calls one thread makes inside a `with` block. The threads are native and
kept in a pool between calls; Python only makes one call per operation.

Native calls release the GIL, so Python threads may call vFHE at the same
time. They gain little on small operations, whose time is mostly the Python
around the call; raising the limit is what puts more cores on one operation.

A process may `fork` after building vFHE objects, and the child may use them
and the library as usual:

- the child starts its own thread pool on its first parallel call;
- the memory pool keeps, in the child, only the buffers the forking thread had
  retained;
- the entropy stream discards what the child inherited, so parent and child
  never draw the same bytes;
- rings, keys, polynomials and tables are the child's own copies.

Fork while no other thread of the process is inside a vFHE call, and with no
trace enabled.

vFHE does not choose where memory is placed. On a machine with several NUMA
nodes, `numactl --interleave=all` spreads large tables over every node's
memory, and `numactl --cpunodebind=N --membind=N` keeps a process on one
node. The memory pool keeps freed buffers per thread for reuse, up to the
peak in use; `VFHE_MEMPOOL_CAPACITY` bounds it in bytes, and `0` turns it
off.

## Next steps

- Raise `Ring`'s degree and modulus bit-count for deeper computations, at the
  cost of speed and ciphertext size.
- Read each module's `__init__` re-exports for its public API, listed in the
  [README](https://github.com/vfhe/vfhe/blob/main/README.md#modules).
- Compile your own C kernels into the library with `vfhe.util.kernels`, so the
  compiler inlines vfhe's primitives into your loops. It needs a C compiler, and
  meson and ninja from `pip install vfhe[kernels]`.

Additional examples will be added as the library evolves.
