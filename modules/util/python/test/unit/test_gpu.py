# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The CUDA extension: absent, present without a device, or usable."""

import pytest
from vfhe.util import gpu


def test_the_extension_is_handed_out_only_with_a_device() -> None:
    if not gpu.available():
        with pytest.raises(RuntimeError, match="no CUDA device"):
            gpu.extension()
        pytest.skip("no CUDA device")
    ffi, lib = gpu.extension()
    assert lib.vfhe_gpu_device_count() > 0
    assert ffi.string(lib.vfhe_gpu_error_string(0)) == b"no error"
    gpu.check(0)
    with pytest.raises(RuntimeError):
        gpu.check(1)
