<!-- SPDX-License-Identifier: Apache-2.0 -->

# Contributing to VFHE

This page covers what CI expects of a pull request;
the [development guide](docs/DEVELOPMENT.md) covers how the project is built and laid out.

By contributing you agree that your work is licensed under
[Apache-2.0](LICENSE), and you accept the [Code of Conduct](CODE_OF_CONDUCT.md).

For security problems, do not open an issue; see the
[security policy](SECURITY.md).

## Bugs and ideas

Report bugs as [GitHub issues](https://github.com/vfhe/vfhe/issues) with the
version or commit, your platform and compiler, and something we can reproduce.
For a large change, open an issue to discuss it before writing the code; this
avoids building changes that will not be accepted.

## Setting up

```bash
git clone --recursive https://github.com/vfhe/vfhe vfhe && cd vfhe
python3 -m venv .venv && source .venv/bin/activate
make deps
```

`make deps` installs the dev dependencies and the `pre-commit` hooks: ruff and
clang-format run on the files you touch, using the exact versions CI uses, so
you never fail a build on formatting. See [Prerequisites](docs/DEVELOPMENT.md#prerequisites-development) if the
build fails.

## Before you open a pull request

```bash
make test-fast    # build, C unit tests, and the fast Python suite
```

CI runs the complete suite across Linux and macOS on Python 3.10-3.14; the fast
suite catches most breakage. `make test` runs the complete suite locally,
including the heavy FHE bootstraps.

## What CI will check

| Check | What it wants |
|---|---|
| Lint | ruff (lint and format), pyright, clang-format |
| Tests | C unit tests, then the fast and complete Python suites |
| Sdist | the package still builds and installs from source |
| CodeQL, Scorecard, dependency review | static analysis and supply-chain hygiene |

`CI OK` is the single required check; it turns green only when everything above
passes.

### About coverage

Coverage is reported on every pull request by `github-code-quality[bot]`, which
compares your branch against `main`. **It does not block a merge**; treat it as
information for the reviewer.

Still, if you add code, add tests for it. When something genuinely cannot be
tested (unreachable defensive branches, allocation-failure paths,
platform-specific fallbacks), mark it so the report is not misleading:

```python
if impossible_state:  # pragma: no cover
    raise AssertionError("unreachable")
```

```c
if (rc != 0) /* GCOVR_EXCL_LINE */
    abort();
```

## Commits

Subject in the imperative, 50 characters or fewer, capitalised, no trailing
period. Body wrapped at 72, explaining *what* and *why* rather than how. Enable
the template once:

```bash
git config commit.template .gitmessage
```

This is a recommendation; no tooling enforces it.

## Adding a module

`modules/<name>/` is self-contained. The development guide has the
[full layout and the steps](docs/DEVELOPMENT.md#adding-a-new-module); the build discovers
sources automatically, so there is no central list to update.
