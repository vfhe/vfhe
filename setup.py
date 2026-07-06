# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

from setuptools import setup

HERE = Path(__file__).parent
SCRIPTS = HERE / "scripts"
MODULES_DIR = HERE / "modules"
GENERATED = HERE / ".generated"

# Vendored native deps live in git submodules. BLAKE3's C sources are compiled
# into the extension, so a non-recursive clone (or an sdist built without the
# submodule checked out) leaves external/blake3 empty and the build fails deep
# in the compiler. Fail early with a clear message instead.
if not (HERE / "external" / "blake3" / "c" / "blake3.c").exists():
    raise SystemExit(
        "external/blake3 is empty (BLAKE3 submodule not checked out).\n"
        "Run:  git submodule update --init --recursive"
    )

package_dir: dict[str, str] = {}
packages: list[str] = []

# Discovery of modules
for src in sorted(MODULES_DIR.glob("*/python/src")):
    for init in sorted(src.glob("vfhe/**/__init__.py")):
        pkg = ".".join(init.parent.relative_to(src).parts)
        package_dir[pkg] = str(init.parent.relative_to(HERE))
        packages.append(pkg)

# Generation of protos
sys.path.insert(0, str(SCRIPTS))
import gen_proto

gen_proto.generate_all()

# Discovery of generated modules
for init in sorted(GENERATED.glob("_vfhe_proto/**/__init__.py")):
    pkg = ".".join(init.parent.relative_to(GENERATED).parts)
    package_dir[pkg] = str(init.parent.relative_to(HERE))
    packages.append(pkg)

setup(
    packages=packages,
    package_dir=package_dir,
    cffi_modules=["native/build_ffi.py:ffibuilder"],
)
