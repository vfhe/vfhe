<!-- SPDX-FileCopyrightText: 2026 The vFHE Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Testing

For contributors adding or debugging a test. It states how the test tree is
organised and how a run selects, skips, or emulates what it runs.

Tests of a module live **with** that module, and a tool's tests live with the
tool (`tools/release/notes/test.sh`, run as a pre-commit hook). What is
cross-cutting lives here.

| where | what | run by |
|---|---|---|
| `modules/<m>/c/test/*.c` | C unit tests (Unity) | `make test SUITES=c` |
| `modules/<m>/python/test/` | the Python suites | `make test SUITES=fast,complete` |
| `modules/<m>/c/fuzz/` | fuzz harnesses | CI's nightly, built by `.clusterfuzzlite/` |
| `test/smoke/cases/` | an **installed distribution** answering for itself | `make smoke` |
| `test/unit/` | the runner and the capability guard meson runs every test through | `make test` |

## Two axes, carried by the test's name

Every test is named `<engine>-c-<stem>` or `<engine>-pytest-<depth>`, and that
name **is** the selection API:

```sh
make test                        # every built engine, default depth
make test ENGINE=portable        # -> meson test 'portable-c-*' 'portable-pytest-complete'
make test SUITES=c,fast          # -> meson test '*-c-*' '*-pytest-fast'
```

Each selected cell reports its own line, and the tally closes the run:

```text
 1/4 vfhe:portable-c-test_mod             OK    0.01s
 2/4 vfhe:portable-c-test_number_theory   OK    0.01s
 3/4 vfhe:portable-c-test_cpu_probe       OK    0.01s
 4/4 vfhe:portable-c-test_ntt             OK    0.02s

Ok:                4
Fail:              0
```

A `Skip` count names engines this CPU cannot execute, and the run still
passes. Anything under `Fail` is a real failure, with the details in
`build/meson-logs/testlog.txt`.

Names, not `meson test --suite`: a `--suite` that matches nothing exits **0**,
so a typo would report success having run no tests. A name glob that matches
nothing exits 1.

## What runs, and what is skipped

meson never runs a test binary directly. It runs `require_cpu.sh`, which asks
the built `vfhe-cpu` probe whether this machine has the capability the engine
declares in [`meson.build`](https://github.com/vfhe/vfhe/blob/main/meson.build)'s engine list:

- **has it** → `exec` the real test;
- **lacks it** → exit **77**, meson's SKIP convention, so the run stays green;
- **cannot judge it** → exit 1, because a mistyped capability must not look
  like a CPU that merely lacks the feature.

Ask the probe directly to see which case a machine is in:

```sh
build/test/vfhe-cpu neon        # exit 0 -> this CPU has it
build/test/vfhe-cpu nonsense    # exit 2 -> unknown name, so the guard fails the test
```

`EMULATE=1` selects `meson test --setup <engine>_emulated`, which hands SDE's
flags to the guard; `tools/sde/fetch.sh` puts the launcher in `$VFHE_EMULATOR`.
So no configure waits on a download, and a missing launcher fails a test rather
than silently running kernels the CPU cannot execute. The guard runs SDE, rather
than an exe_wrapper above it, because SDE does not instrument what its program
execs.

## What belongs in a smoke test

`test/smoke/cases/` asserts things **only an installed distribution can answer** — the
version is real, every declared engine shipped, a runtime-compiled extension
links. tox gives them a venv holding only the distribution and runs them from a
temp directory, so the source tree is unreachable.

A helper both a smoke test and an in-tree suite want is a sign the smoke test
is testing the tree.
