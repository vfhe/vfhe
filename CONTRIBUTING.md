<!-- SPDX-FileCopyrightText: 2026 Alin-Petru Roșu -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Contributing to vFHE

This page covers what the project asks of a pull request — all of it
enforced by CI, so nothing surprises you at review. The [development
guide](https://github.com/vfhe/vfhe/blob/main/DEVELOPMENT.md) covers how
the project is built and laid out.

By contributing you agree that your work is licensed under
[Apache-2.0](https://github.com/vfhe/vfhe/blob/main/LICENSE), and you accept
the [Code of
Conduct](https://github.com/vfhe/vfhe/blob/main/CODE_OF_CONDUCT.md).

For security problems, do not open an issue; see the
[security policy](https://github.com/vfhe/vfhe/blob/main/SECURITY.md).

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

`make deps` also installs the pre-commit hooks. They are the definition of
every formatter and validator this project runs — `make lint` and CI both
execute them — so a commit tells you what CI would, and `make format` fixes
what it can. Prerequisites and everything
about the build live in the
[development guide](https://github.com/vfhe/vfhe/blob/main/DEVELOPMENT.md#prerequisites-development).

## Before you open a pull request

```bash
make test SUITES=c,fast    # C tests + the fast Python suite, on every engine your machine runs for free
```

CI runs the complete suite across Linux and macOS on Python 3.10-3.14; the fast
suite catches most breakage. `make test` runs the complete suite locally,
including the heavy FHE bootstraps.

## What CI will check

| Check | What it wants |
|---|---|
| Lint | `make lint`: every pre-commit hook over the whole tree (ruff, clang-format, zizmor, actionlint, shellcheck, codespell, markdownlint, REUSE, the config validators), then pyright. It runs on documentation changes too |
| C, Python | the C tests, then the fast Python suite, on every engine |
| Sanitized | the same suites over an ASan+UBSan build, portable engine; the C suites on macOS, both halves on Linux |
| Distribution Build, Smoke Tests | the sdist still builds, installs, and works from source |
| Code Scanning | CodeQL, one leg per engine |
| Secret Scan, Dependency Review, DCO | history hygiene, the license allow-list, the [sign-off](#sign-off-dco) |

One check is required, `CI OK`, and it turns green only when every job above
succeeded. A skip counts as a failure unless the change classification is what
caused it, so a docs-only pull request passes by running the checks that are not
about code, while a job that vanishes from the workflow — or a cancelled run —
fails the gate instead of passing vacuously.

### About coverage

Coverage is reported on every pull request as a comment with per-line HTML
reports in the run's `coverage` artifact. **It does not block a merge**; treat
it as information for the reviewer.

**Testing policy.** Any change that adds or alters functionality MUST add or
update automated tests covering it — a new kernel, a new Python API, a bug fix
(which gets a test that fails before it and passes after), or a behavioural
change. Pure refactors, docs, and CI changes are exempt. Reviewers hold pull
requests to this. When something genuinely cannot be tested (unreachable
defensive branches, allocation-failure paths, platform-specific fallbacks),
mark it so the report is not misleading:

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

This style is a recommendation; no tooling enforces it. The sign-off below
**is** enforced.

### Sign-off (DCO)

Every commit must be signed off (`git commit -s`), certifying the
[Developer Certificate of Origin](https://developercertificate.org): that you
have the right to submit the code under Apache-2.0. Use your real name and the
commit author email; CI rejects unsigned commits. To fix a series:
`git rebase --signoff <base>` and force-push.

If you wrote the contribution as part of employment or a funded project, the
rights may belong to your employer or funder; by signing off you assert you
are authorised to contribute it. When unsure, check your institution's IP
policy first.

## Adding a module

Modules are a central component of the library's design, so we encourage you to
reach out before writing code, let alone opening a pull request — an issue, or
email <maintainers@vfhe.ai>.

`modules/<name>/` is self-contained. The development guide has the
[full layout and the steps](https://github.com/vfhe/vfhe/blob/main/DEVELOPMENT.md#adding-a-new-module);
the build discovers sources automatically, so there is no central list to
update.
