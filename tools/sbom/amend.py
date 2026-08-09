#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Add the vendored native components to a CycloneDX SBOM.

cyclonedx-py reads the Python environment and cannot see C compiled into the
extension. The vendored BLAKE3 ships in every artifact, so it is appended
here: version from the header it is compiled from, pinned commit from the
submodule gitlink. Nothing is hand-maintained; a submodule bump flows through.
The amended document is schema-validated before it is written.

Usage: python tools/sbom/amend.py <sbom.json>
(`make sbom` generates the document and runs this over it.)
"""

import json
import re
import subprocess
import sys
from pathlib import Path

from cyclonedx.schema import SchemaVersion
from cyclonedx.validation.json import JsonStrictValidator

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))  # the shared parts live one level up

from _common import ROOT, find_tool  # noqa: E402  (parts; TOOLS above)

BLAKE3_REF = "vendored:blake3"


def blake3_component(root: Path) -> dict:
    header = (root / "external" / "blake3" / "c" / "blake3.h").read_text()
    match = re.search(r'#define BLAKE3_VERSION_STRING "([^"]+)"', header)
    if match is None:
        raise SystemExit("BLAKE3_VERSION_STRING not found in blake3.h")
    version = match.group(1)
    gitlink = subprocess.run(
        [find_tool("git"), "ls-tree", "HEAD", "external/blake3"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    if not gitlink.stdout:
        raise SystemExit("external/blake3 not found in the HEAD tree")
    commit = gitlink.stdout.split()[2]
    return {
        "bom-ref": BLAKE3_REF,
        "type": "library",
        "name": "BLAKE3",
        "version": version,
        "description": (
            "Vendored C sources, compiled into the _vfhe_native extension (see NOTICE)."
        ),
        "scope": "required",
        "licenses": [
            {"expression": "CC0-1.0 OR Apache-2.0 OR Apache-2.0 WITH LLVM-exception"}
        ],
        "purl": f"pkg:github/blake3-team/blake3@{version}",
        "externalReferences": [
            {"type": "vcs", "url": "https://github.com/BLAKE3-team/BLAKE3"}
        ],
        "properties": [{"name": "vfhe:submodule-commit", "value": commit}],
    }


def main() -> None:
    path = Path(sys.argv[1])
    sbom = json.loads(path.read_text())
    components = sbom.setdefault("components", [])
    if any(c.get("bom-ref") == BLAKE3_REF for c in components):
        raise SystemExit("SBOM already amended; run once per fresh document")
    vfhe_refs = [c["bom-ref"] for c in components if c.get("name") == "vfhe"]
    if not vfhe_refs:
        raise SystemExit("no vfhe component in the SBOM; was the venv empty?")

    components.append(blake3_component(ROOT))
    # The graph too, not just the list: BLAKE3 hangs off vfhe.
    dependencies = sbom.setdefault("dependencies", [])
    entry = next((d for d in dependencies if d["ref"] == vfhe_refs[0]), None)
    if entry is None:
        entry = {"ref": vfhe_refs[0]}
        dependencies.append(entry)
    entry.setdefault("dependsOn", []).append(BLAKE3_REF)
    dependencies.append({"ref": BLAKE3_REF})

    result = json.dumps(sbom, indent=2) + "\n"
    error = JsonStrictValidator(
        SchemaVersion.from_version(sbom["specVersion"])
    ).validate_str(result)
    if error:
        raise SystemExit(f"amended SBOM is schema-invalid: {error}")
    path.write_text(result)


if __name__ == "__main__":
    main()
