# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Describes the machine a benchmark run executes on, and whether its timings can be trusted.

The verdict concerns wall-clock timing only. A machine is untrusted when one
of these holds:

- it is WSL, where the Windows host owns scheduling and the clock;
- it is battery powered, so sustained runs throttle;
- a CPU quota limits its control group, so threaded runs stall once the
  quota is spent;
- its clock is held at the minimum: the ``powersave`` governor under a
  driver other than ``intel_pstate`` or ``amd-pstate``;
- a hypervisor took 1% or more of the CPU time (steal) while it was sampled;
- other work kept at least one CPU busy while it was sampled.

The last two are sampled for one second at the start of a run and again at
the end. Turbo, SMT, the governor, transparent huge pages, NUMA nodes and
virtualization are recorded as facts and do not change the verdict. Off
Linux nothing can be checked, and the verdict is ``unverified``.
"""

from __future__ import annotations

import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

__all__ = ["conditions", "facts", "machine_tag", "sample", "verdict"]

STEAL_LIMIT = 0.01
BUSY_CPUS_LIMIT = 1.0
SAMPLE_SECONDS = 1.0

_SYS_CPU = Path("/sys/devices/system/cpu")
_GENERIC_PRODUCT_NAMES = {
    "",
    "system product name",
    "default string",
    "to be filled by o.e.m.",
    "none",
}


def _read(path: str | Path) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _is_wsl() -> bool:
    return bool(
        re.search(r"microsoft|wsl", _read("/proc/version") or "", re.IGNORECASE)
    )


def _cpu_flags() -> set[str]:
    for line in (_read("/proc/cpuinfo") or "").splitlines():
        name, _, value = line.partition(":")
        if name.strip() == "flags":
            return set(value.split())
    return set()


def _cpu_model() -> str:
    for line in (_read("/proc/cpuinfo") or "").splitlines():
        name, _, value = line.partition(":")
        if name.strip() == "model name":
            return value.strip()
    return platform.processor() or platform.machine()


def _on_battery_hardware() -> bool:
    supplies = Path("/sys/class/power_supply")
    return any(_read(s / "type") == "Battery" for s in supplies.glob("*"))


def _cpu_quota() -> str | None:
    """Returns the CPU quota limiting this process's control group, or None."""
    for line in (_read("/proc/self/cgroup") or "").splitlines():
        _, controllers, path = line.split(":", 2)
        if controllers == "":
            # cgroup v2: the quota may sit on any ancestor.
            group = Path("/sys/fs/cgroup" + path)
            for directory in (group, *group.parents):
                quota = _read(directory / "cpu.max")
                if quota is not None and not quota.startswith("max"):
                    return quota
                if directory == Path("/sys/fs/cgroup"):
                    break
        elif "cpu" in controllers.split(","):
            quota = _read(f"/sys/fs/cgroup/{controllers}{path}/cpu.cfs_quota_us")
            if quota is not None and quota != "-1":
                return quota
    return None


def _governor() -> tuple[str | None, str | None]:
    """Returns the first CPU's frequency governor and scaling driver."""
    cpufreq = _SYS_CPU / "cpu0" / "cpufreq"
    return _read(cpufreq / "scaling_governor"), _read(cpufreq / "scaling_driver")


def _turbo() -> bool | None:
    no_turbo = _read(_SYS_CPU / "intel_pstate" / "no_turbo")
    if no_turbo is not None:
        return no_turbo == "0"
    boost = _read(_SYS_CPU / "cpufreq" / "boost")
    return None if boost is None else boost == "1"


def _transparent_huge_pages() -> str | None:
    setting = _read("/sys/kernel/mm/transparent_hugepage/enabled")
    if setting is None:
        return None
    # The active choice is the bracketed one: "always [madvise] never".
    return setting[setting.find("[") + 1 : setting.find("]")] or None


def _cpu_times() -> list[int] | None:
    """Returns the machine-wide CPU time counters of /proc/stat, in clock ticks."""
    first = (_read("/proc/stat") or "").splitlines()[:1]
    if not first or not first[0].startswith("cpu "):
        return None
    return [int(value) for value in first[0].split()[1:]]


def sample(seconds: float = SAMPLE_SECONDS) -> dict[str, float] | None:
    """Samples the machine while this process sleeps.

    Returns the share of CPU time a hypervisor took (``steal``) and how many
    CPUs other work kept busy (``busy_cpus``), or None off Linux.
    """
    before = _cpu_times()
    if before is None:
        return None
    time.sleep(seconds)
    after = _cpu_times()
    if after is None:
        return None
    # user nice system idle iowait irq softirq steal; guest time is in user.
    delta = [b - a for a, b in zip(before[:8], after[:8], strict=True)]
    total = sum(delta)
    if total <= 0:
        return None
    idle, steal = delta[3] + delta[4], delta[7]
    busy = (total - idle - steal) / total * (os.cpu_count() or 1)
    return {"steal": steal / total, "busy_cpus": busy}


def conditions(sampled: dict[str, float] | None, when: str) -> list[str]:
    """Returns why the machine is untrusted at this moment; empty when nothing is wrong.

    ``sampled`` is what `sample` returned at that moment, and ``when`` names
    the moment in the reasons.
    """
    if not _is_linux():
        return []
    reasons = []
    if _is_wsl():
        reasons.append("WSL: the Windows host owns scheduling and the clock")
    if _on_battery_hardware():
        reasons.append("battery powered: sustained runs throttle")
    quota = _cpu_quota()
    if quota is not None:
        reasons.append(f"CPU quota {quota} on the control group: threaded runs stall")
    governor, driver = _governor()
    if governor == "powersave" and not re.match(
        r"intel_pstate|amd-pstate", driver or ""
    ):
        reasons.append(
            f"powersave governor under {driver}: the clock is held at its minimum"
        )
    if sampled is not None:
        if sampled["steal"] >= STEAL_LIMIT:
            reasons.append(f"hypervisor steal {sampled['steal']:.1%} at the {when}")
        if sampled["busy_cpus"] >= BUSY_CPUS_LIMIT:
            reasons.append(
                f"other work kept {sampled['busy_cpus']:.1f} CPUs busy at the {when}"
            )
    return reasons


def verdict(reasons: list[str]) -> str:
    """Returns ``trusted``, ``untrusted``, or ``unverified`` off Linux."""
    if not _is_linux():
        return "unverified"
    return "untrusted" if reasons else "trusted"


def facts() -> dict[str, Any]:
    """Returns what is recorded about the machine without affecting the verdict."""
    governor, driver = _governor()
    nodes = list(Path("/sys/devices/system/node").glob("node[0-9]*"))
    smt = _read(_SYS_CPU / "smt" / "active")
    return {
        "cpu": _cpu_model(),
        "cpus": os.cpu_count(),
        "system": platform.platform(),
        "virtualized": "hypervisor" in _cpu_flags() if _is_linux() else None,
        "governor": governor,
        "scaling_driver": driver,
        "turbo": _turbo(),
        "smt": None if smt is None else smt == "1",
        "numa_nodes": len(nodes) or None,
        "transparent_huge_pages": _transparent_huge_pages(),
        "load": round(os.getloadavg()[0], 2),
    }


def _slug(text: str, length: int = 32) -> str:
    text = re.sub(r"\(R\)|\(TM\)|CPU|Processor|@.*", " ", text)
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-").lower()[:length].strip("-")


def machine_tag() -> str:
    """Returns a short, file-name-safe name for this machine.

    The firmware's product name when it is specific (a cloud instance reports
    its type there), ``wsl2`` under WSL, else the CPU model, else the host name.
    """
    for path in (
        "/sys/devices/virtual/dmi/id/product_name",
        "/sys/class/dmi/id/product_name",
    ):
        name = _read(path)
        if name is not None and name.lower() not in _GENERIC_PRODUCT_NAMES:
            return _slug(name)
    if _is_wsl():
        return "wsl2"
    return _slug(_cpu_model()) or _slug(platform.node()) or "machine"
