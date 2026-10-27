<!-- SPDX-FileCopyrightText: 2026 The vFHE Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Benchmark baselines

One results file per reference machine, named after the machine as the runner
tags it (`r8i.metal-96xl.jsonl`): the reference run that
[`docs/PERFORMANCE.md`](../../../docs/PERFORMANCE.md) was generated from, in
the format `tools/bench/run.py` writes, without the host name.

Compare a run of your own on the same machine against it:

```bash
tools/bench/compare.py tools/bench/baselines/r8i.metal-96xl.jsonl results/bench/<your run>.jsonl
```

A new reference run replaces the file and the page together, from the same
results files:

```bash
tools/bench/report.py baseline <run files>
tools/bench/report.py page <run files>
```

Only a trusted run of a committed tree is accepted. The repository's history
keeps the earlier baselines.
