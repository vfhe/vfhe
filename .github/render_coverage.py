#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fill the coverage summary template from coverage.py and gcovr JSON reports.

Usage:
    python .github/render_coverage.py coverage-python.json coverage-c.json \
        .github/coverage-comment.md <artifact-url>
"""

import json
import sys
from pathlib import Path


def _cell(covered: int, total: int) -> str:
    if not total:
        return "n/a"
    return f"{100 * covered / total:.1f}% ({covered}/{total})"


def main() -> int:
    python_totals = json.loads(Path(sys.argv[1]).read_text())["totals"]
    c_totals = json.loads(Path(sys.argv[2]).read_text())
    template = Path(sys.argv[3]).read_text()
    print(
        template.format(
            py_lines=_cell(
                python_totals["covered_lines"], python_totals["num_statements"]
            ),
            py_branches=_cell(
                python_totals["covered_branches"], python_totals["num_branches"]
            ),
            c_lines=_cell(c_totals["line_covered"], c_totals["line_total"]),
            c_branches=_cell(c_totals["branch_covered"], c_totals["branch_total"]),
            artifact_url=sys.argv[4],
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
