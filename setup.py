# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

from setuptools import setup

HERE = Path(__file__).parent
MODULES_DIR = HERE / "modules"
GENERATED = HERE / ".generated"

sys.path.insert(0, str(HERE / "packaging"))
import discovery  # noqa: E402  (needs the sys.path insert above)

# Vendored native deps live in git submodules; without them the build fails deep
# in the compiler. Fail early with a clear message instead. The required files
# come from packaging/discovery.py, the same registry the build compiles from.
_missing = discovery.vendored_missing(HERE)
if _missing:
    raise SystemExit(
        "Missing vendored sources (submodules not checked out?):\n  "
        + "\n  ".join(str(m.relative_to(HERE)) for m in _missing)
        + "\nRun:  git submodule update --init --recursive"
    )

package_dir: dict[str, str] = {}
packages: list[str] = []

# Map each vfhe.* package onto its physical modules/*/python/src location.
for src in sorted(MODULES_DIR.glob("*/python/src")):
    for init in sorted(src.glob("vfhe/**/__init__.py")):
        pkg = ".".join(init.parent.relative_to(src).parts)
        package_dir[pkg] = str(init.parent.relative_to(HERE))
        packages.append(pkg)

# Proto bindings must exist before packages are collected below.
import generate_protos  # noqa: E402  (same packaging/ path as discovery above)

generate_protos.generate_all()

# Pick up the just-generated _vfhe_proto packages.
for init in sorted(GENERATED.glob("_vfhe_proto/**/__init__.py")):
    pkg = ".".join(init.parent.relative_to(GENERATED).parts)
    package_dir[pkg] = str(init.parent.relative_to(HERE))
    packages.append(pkg)

setup(
    packages=packages,
    package_dir=package_dir,
    cffi_modules=["packaging/build_ffi.py:ffibuilder"],
)
