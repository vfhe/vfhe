# vfhe.misc

The native boundary plus the shared internal C utilities.

- `python/src/.../libvfhe.py`: the handle to the compiled extension: it
  re-exports `ffi` / `lib` from `_vfhe_native` and a `libvfhe` singleton. Every
  other Python module reaches C through `from vfhe.misc.libvfhe import ffi, lib`.
- `c/src/`: utilities the kernels share: a BLAKE3-seeded pseudo-random
  generator and AES-CTR stream (`prng.c`, `aes_rng.c`), aligned allocation, and
  small numeric helpers (`misc.c`, `misc_tp.c`).

`misc` has no cdef of its own; its C is internal engine support compiled into
`_vfhe_native`, and its Python side only exposes the native handle.
