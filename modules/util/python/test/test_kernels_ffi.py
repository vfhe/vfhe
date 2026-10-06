# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""`vfhe.util.kernels._ffi`: the cffi wrapper, as a function and as meson's script."""

import runpy
import sys

from vfhe.util.kernels import _ffi


def test_the_wrapper_binds_every_cdef_against_the_headers(tmp_path) -> None:
    """meson and ExtensionBuilder both build their engine's module from this."""
    header = tmp_path / "mine.h"
    header.write_text("int mine(void);")
    out = tmp_path / "m.c"
    _ffi.emit("m", out, [header], [header])
    text = out.read_text()
    assert "PyInit_m" in text
    assert '#include "mine.h"' in text
    assert "_cffi_f_mine" in text


def test_the_module_runs_as_mesons_script(monkeypatch, tmp_path) -> None:
    header = tmp_path / "f.h"
    header.write_text("int f(void);")
    out = tmp_path / "m.c"
    monkeypatch.setattr(
        sys, "argv", ["_ffi.py", "m", str(out), str(header), "--", str(header)]
    )
    runpy.run_path(_ffi.__file__, run_name="__main__")
    assert "PyInit_m" in out.read_text()
