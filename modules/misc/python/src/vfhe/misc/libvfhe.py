# SPDX-License-Identifier: Apache-2.0
"""cffi handle to the native library.

The C sources are compiled into the single ``_vfhe_native`` CFFI extension
(see native/build_ffi.py); this module re-exports its ``ffi`` / ``lib`` plus a
``libvfhe`` singleton that the wrappers use as ``libvfhe.lib``.
"""

import warnings

from _vfhe_native import ffi, lib


class LibVFHE:
    def __init__(self) -> None:
        self.lib = lib
        self.ffi = ffi
        self.multithreaded = False
        self.num_threads = 1


# Singleton instance
libvfhe = LibVFHE()


def _warn_if_leaving_performance_on_the_table(native=lib) -> None:
    """Hint (once) when the portable build runs on a CPU that could go faster.

    vfhe installs from source and normally tunes to the CPU; a portable build on
    an AVX-512-IFMA machine means it was forced (VFHE_PORTABLE=1) or the compiler
    did not detect the feature. Silence with the usual warning filters or
    ``PYTHONWARNINGS=ignore``. ``native`` is injectable for testing.
    """
    if native.vfhe_build_is_portable() and native.vfhe_cpu_has_avx512ifma():
        warnings.warn(
            "vfhe is running the portable build, but this CPU supports AVX-512 "
            "IFMA. For a faster, CPU-tuned build, reinstall without VFHE_PORTABLE:\n"
            "    pip install --force-reinstall --no-cache-dir vfhe",
            RuntimeWarning,
            stacklevel=2,
        )


_warn_if_leaving_performance_on_the_table()
