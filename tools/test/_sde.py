# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Intel SDE: download, verify, cache, and wrap.

The pin — version, tarball, sha256, mirrors — is .sde.json at the repo root,
which CI also keys its cache on. Run as a script to fetch it eagerly (the
setup-sde action does, so the download lands in its own step).
"""

import hashlib
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))  # the shared parts live one level up

import _engines  # noqa: E402  (the parts; TOOLS above)
from _common import ROOT, error, host_supports, log  # noqa: E402

SDE = json.loads((ROOT / ".sde.json").read_text())
CACHE_DIR = ROOT / ".cache" / "sde"
YAMA = Path("/proc/sys/kernel/yama/ptrace_scope")


def download_pinned_tarball(into: Path) -> bool:
    """Fetch from the first mirror whose bytes match the pinned sha256."""
    for url in SDE["urls"]:
        log(f"[sde] downloading {url}")
        try:
            with (
                urllib.request.urlopen(url, timeout=60) as response,  # noqa: S310
                into.open("wb") as out,
            ):
                shutil.copyfileobj(response, out)
        except OSError as exc:
            log(f"[sde] download failed: {exc}")
            continue

        if hashlib.sha256(into.read_bytes()).hexdigest() == SDE["sha256"]:
            return True
        log("[sde] sha256 mismatch, trying the next source")

    into.unlink(missing_ok=True)  # never leave an unverified download
    error(f"could not fetch {SDE['tarball']} with sha256 {SDE['sha256']}")
    return False


def unpack(tarball: Path, into: Path) -> None:
    with tarfile.open(tarball) as tar:
        tar.extractall(into, filter="data")
    tarball.unlink()


def cached_launcher() -> Path | None:
    """The sde64 launcher, fetching and unpacking the pin if it is absent."""
    launcher = CACHE_DIR / SDE["tarball"].removesuffix(".tar.xz") / "sde64"
    if launcher.exists():
        return launcher

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tarball = CACHE_DIR / SDE["tarball"]
    if not download_pinned_tarball(tarball):
        return None
    unpack(tarball, CACHE_DIR)

    if not launcher.exists():
        error(f"no sde64 at {launcher} after unpacking {SDE['tarball']}")
        return None
    return launcher


def warn_if_ptrace_is_restricted() -> None:
    """SDE detaches from the compilers a test spawns, which needs ptrace."""
    if YAMA.exists() and YAMA.read_text().strip() != "0":
        log(
            "[sde] kernel.yama.ptrace_scope is on: tests that spawn compilers "
            "may fail under SDE. Fix: sudo sysctl -w kernel.yama.ptrace_scope=0"
        )


def resolve_wrapper(engine: _engines.Engine, emulate: bool) -> list[str] | None:
    """How to run this engine's binaries here: [] natively, an emulator prefix
    when this CPU lacks its ISA, None when neither is possible."""
    if not emulate and host_supports(engine.requires):
        log(f"[{engine.name}] running on hardware")
        return []

    if engine.emulator is None:
        error(f"{engine.name} names no emulator in tools/_engines.py")
        return None
    tool, *target = engine.emulator
    if tool != "sde":
        error(f"no runner for emulator '{tool}'; add it here")
        return None

    if sys.platform != "linux":
        error(
            f"this CPU cannot run {engine.name}, and its stand-in, Intel SDE, "
            "needs an x86_64 Linux host (it cannot instrument emulated "
            "processes). Use CI or an x86_64 Linux machine."
        )
        return None

    launcher = cached_launcher()
    if launcher is None:
        return None
    warn_if_ptrace_is_restricted()

    # -no-follow-child: the compilers dynamic_extensions spawns run natively
    # (SDE's follow-execve breaks them); only the loading process needs
    # emulation, and modules dlopen back into it, so they stay emulated.
    return [str(launcher), *target, "-no-follow-child", "--"]


if __name__ == "__main__":
    raise SystemExit(0 if cached_launcher() else 1)
