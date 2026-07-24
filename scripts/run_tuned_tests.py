#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the test suites against this architecture's optimised engine.

x86-64: builds the tuned AVX-512 engine (VFHE_TUNED=1) and runs it natively
when this CPU has AVX-512 IFMA, else under the Intel Software Development
Emulator posing as an Ice Lake CPU (Linux only: SDE cannot instrument
emulated processes, so Rosetta/QEMU hosts cannot run it; it is downloaded
sha256-pinned into .cache/sde/ on first use). CI passes --emulate so the
merge gate always exercises the one path every runner can take.
arm64: no tuned engine exists yet; errors until its kernels land
(--if-supported succeeds doing nothing instead, for the pre-wired CI jobs).

Usage:
    python scripts/run_tuned_tests.py [--suite c|fast|complete] [--emulate]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import shlex
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from _common import ROOT, discovery, error, log

SDE_VERSION = "10.8.0"
SDE_TARBALL = f"sde-external-{SDE_VERSION}-2026-03-15-lin.tar.xz"
SDE_SHA256 = "50b320cd226acef7a491f5b321fc1be3c3c7984f9e27a456e64894b5b0979dd3"
# Intel first; the pinned sha256 makes the mirror equally trustworthy.
SDE_URLS = (
    f"https://downloadmirror.intel.com/915934/{SDE_TARBALL}",
    f"https://github.com/petarpetrovt/setup-sde/releases/download/binaries/{SDE_TARBALL}",
)
# Ice Lake: AVX-512 F/IFMA/DQ/VL + AES-NI + RDRND, exactly the tuned target.
SDE_CHIP = "-icl"


def host_has_native_ifma() -> bool:
    """The packaged build's own CPU probe, loaded by file path (the package
    is not importable before the extension exists)."""
    path = ROOT / "modules/misc/python/src/vfhe/misc/_native_compiler.py"
    spec = importlib.util.spec_from_file_location("_vfhe_native_compiler", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.host_has_avx512ifma()


def ensure_sde() -> Path | None:
    """The cached sde64 launcher, downloading and unpacking SDE if absent."""
    cache = ROOT / ".cache" / "sde"
    sde64 = cache / SDE_TARBALL.removesuffix(".tar.xz") / "sde64"
    if sde64.exists():
        return sde64

    cache.mkdir(parents=True, exist_ok=True)
    tarball = cache / SDE_TARBALL
    for url in SDE_URLS:
        log(f"[sde] downloading {url}")
        try:
            urllib.request.urlretrieve(url, tarball)
        except OSError as exc:
            log(f"[sde] download failed: {exc}")
            continue
        if hashlib.sha256(tarball.read_bytes()).hexdigest() == SDE_SHA256:
            break
        log("[sde] sha256 mismatch, trying the next source")
    else:
        error(f"could not fetch {SDE_TARBALL} with sha256 {SDE_SHA256}")
        return None

    with tarfile.open(tarball) as tar:
        tar.extractall(cache, filter="data")
    tarball.unlink()
    return sde64 if sde64.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("c", "fast", "complete"),
        default="fast",
        help="c: C unit tests only; fast adds the fast Python suite; "
        "complete adds the heavy @complete tests (default: fast)",
    )
    parser.add_argument(
        "--emulate",
        action="store_true",
        help="use SDE even on AVX-512 IFMA hardware (CI does, for one "
        "deterministic path across the runner fleet)",
    )
    parser.add_argument(
        "--if-supported",
        action="store_true",
        help="succeed doing nothing where no tuned engine exists, instead of "
        "failing (the pre-wired CI jobs for future architectures)",
    )
    args = parser.parse_args()

    if not discovery.host_has_tuned_engine():
        message = (
            "no tuned engine exists for this architecture yet; its builds "
            "use the portable engine (make test). Once its kernels land "
            "(discovery.TUNED_ARCHES), this command runs them natively here."
        )
        if args.if_supported:
            log(f"[tuned] {message}")
            return 0
        error(message)
        return 2
    if not discovery.is_x86_host():
        error(
            "discovery.TUNED_ARCHES declares this architecture, but this "
            "script only implements the x86-64 runner; add the native branch."
        )
        return 2

    if not args.emulate and host_has_native_ifma():
        wrapper: list[str] = []
        log("[tuned] native AVX-512 IFMA detected: running on hardware, no emulator")
    elif sys.platform != "linux":
        error(
            "this CPU lacks AVX-512 IFMA, and its stand-in, Intel SDE, needs "
            "an x86-64 Linux host (it cannot instrument emulated processes). "
            "Use CI or an x86-64 Linux machine."
        )
        return 2
    else:
        sde64 = ensure_sde()
        if sde64 is None:
            return 1
        wrapper = [str(sde64), SDE_CHIP, "--"]

    how = " ".join(wrapper) or "natively"
    log(f"[tuned] C tests, {how}")
    c_tests = subprocess.run(
        [
            sys.executable,
            "scripts/run_c_tests.py",
            "--tuned",
            f"--wrapper={shlex.join(wrapper)}",
        ],
        cwd=ROOT,
    )
    if c_tests.returncode != 0 or args.suite == "c":
        return c_tests.returncode

    # VFHE_TUNED also reaches pytest: the dynamic_extensions test recompiles at
    # runtime and must pick the same engine.
    env = {**os.environ, "VFHE_TUNED": "1"}
    log("[tuned] building the tuned extension (VFHE_TUNED=1)")
    for step in ("packaging/generate_protos.py", "packaging/build_ffi.py"):
        if subprocess.run([sys.executable, step], cwd=ROOT, env=env).returncode != 0:
            error(f"{step} failed")
            return 1

    log(f"[tuned] {args.suite} Python suite, {how}")
    pytest_cmd = [*wrapper, sys.executable, "-m", "pytest", "-q"]
    if args.suite == "complete":
        pytest_cmd.append("--complete")
    return subprocess.run(pytest_cmd, cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
