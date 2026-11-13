# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The engine calls its own functions directly, not through the PLT.

A shared object's exported functions may be replaced at load time, so unless
the build rules that out GCC reaches each of them through the PLT and inlines
none of them across files. This reads the loaded engine's disassembly for calls
through the PLT to functions the engine itself defines.
"""

import importlib
import re
import shutil
import subprocess
import sys

import pytest
from vfhe.util._engine import Engine

# Callees reached through the PLT all the same, and why.
_EXPECTED = (
    # BLAKE3's SIMD kernels are assembly, outside link-time optimization.
    re.compile(r"blake3_\w+_(sse2|sse41|avx2|avx512)"),
    # mlwe_add_RNS_sample has the same body, and GCC folds it into a jump here.
    re.compile(r"mlwe_add_RNSc_sample"),
)

_FUNCTION = re.compile(r"[0-9a-f]+ <(.+)>:")
_PLT_CALL = re.compile(r"\s(?:call|jmp)\s+[0-9a-f]+ <(.+)@plt>")


def _own_calls_through_plt(path: str, nm: str, objdump: str) -> list[str]:
    """Every ``caller -> callee`` in `path` whose callee `path` defines and reaches through the PLT."""
    symbols = subprocess.run(  # noqa: S603 - binutils from PATH, the loaded engine
        [nm, "-D", "--defined-only", path], capture_output=True, text=True, check=True
    )
    defined = {line.split()[-1] for line in symbols.stdout.splitlines()}
    disassembly = subprocess.run(  # noqa: S603 - as above
        [objdump, "-d", "--no-show-raw-insn", "-j", ".text", path],
        capture_output=True,
        text=True,
        check=True,
    )
    calls = []
    caller = None
    for line in disassembly.stdout.splitlines():
        if function := _FUNCTION.fullmatch(line):
            caller = function[1]
        elif (
            (call := _PLT_CALL.search(line))
            and call[1] in defined
            and not any(expected.fullmatch(call[1]) for expected in _EXPECTED)
        ):
            calls.append(f"{caller} -> {call[1]}")
    return calls


@pytest.mark.skipif(sys.platform != "linux", reason="reads an ELF engine")
def test_own_functions_are_not_called_through_the_plt() -> None:
    nm, objdump = shutil.which("nm"), shutil.which("objdump")
    if nm is None or objdump is None:
        pytest.skip("needs binutils' nm and objdump")
    path = importlib.import_module(Engine.select().module).__file__
    assert path is not None
    calls = _own_calls_through_plt(path, nm, objdump)
    assert not calls, f"{len(calls)} calls through the PLT, first: {calls[:10]}"
