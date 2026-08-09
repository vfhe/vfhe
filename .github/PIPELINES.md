<!-- SPDX-FileCopyrightText: 2026 Alin-Petru Roșu -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Pipelines

Every file is one of three kinds:

| kind | starts from | examples |
|---|---|---|
| **entry point** | an event | `ci-presubmit`, `ci-postsubmit`, `hardware-tests`, `published-smoke`, `release-*` |
| **stage** | `workflow_call` | `static-checks`, `smoke-tests`, `distribution-build`, `release-artifacts`, `hardware-smoke-tests`, `alert-channel` |
| **action** | a step inside a job | `test-c`, `test-python`, `check-workflow-status`, `verify-provenance` |

Three files are dual. `code-scanning` is a stage that also runs weekly on its
own; `distribution-build` and `hardware-smoke-tests` are stages that keep a
manual trigger.

An **engine** is one ISA build of the kernels, and an installed package picks
one at import from what the CPU supports. A **leg** is one run of one suite for
one engine, on one platform and interpreter.

## Pull request — `ci-presubmit.yml`

| stage | jobs | notes |
|---|---|---|
| prepare | `changes` | sets `code=true` if the diff touches anything that is not `*.md` |
| checks | `static-checks`, `dco`, `code-scanning`\*, `dependency-review`\* | |
| test | `c-tests`\* → `python-tests`\*, `sanitized-tests`\* | a subset of the suites; sanitized legs add ASan+UBSan |
| coverage | `coverage-summary` → `coverage-comment` | needs both test jobs; informational |
| fuzz | `fuzz-changed-code`\* | a ClusterFuzzLite session against the changed code |
| dist | `distribution-build`\* → `smoke-tests` | sdist and wheels at the interpreter bounds, then an install of the sdist |
| gate | `CI OK` | all gating jobs above |

\* Runs if `code=true`.

### Key decisions

- `changes` classifies the diff to keep a documentation-only pull request out of
  the suites.
- The gate runs a subset of the suites, on the interpreter bounds only, to keep
  it short. Main runs the rest.
- Every engine runs natively or under emulation, which keeps every required
  check reproducible off this repository's own hardware.
- Sanitized legs run beside the fuzz build to catch a memory fault in the
  kernels before a merge.
- The gate builds the sdist and wheels at the interpreter bounds, putting a
  packaging break on the pull request that introduces it.
- `CI OK` collapses the run into one required name and rejects any skip that
  `code=false` did not authorise.

## Push to main — `ci-postsubmit.yml`, `hardware-tests.yml`

| stage | jobs | notes |
|---|---|---|
| checks | `static-checks`, `code-scanning` | |
| test | `c-tests` → `python-tests` | every platform and interpreter, whole suites |
| test | `sanitized-tests` | a subset, on the platforms that run the engine natively |
| coverage | `coverage-summary` | the comment job belongs to a pull request |
| dist | `artifacts` | sdist, wheels and SBOM, at the version setuptools-scm derives (`0.0.3.dev22+g036c735`) |
| smoke | `smoke-tests` | installs that sdist and tests it |
| alert | `alert-channel` | on any failure |

`hardware-tests.yml` runs on the same event, for pushes that touch more than
`*.md`, as ordered steps on the self-hosted runner: the C suites natively, the
whole Python suite, both sanitized, then a smoke test.

### Key decisions

- `artifacts` runs the release build on every commit, exercising the SBOM step
  a tag would otherwise reach first.
- `hardware-tests.yml` is separate to keep pull requests off that runner and to
  stop other work queueing behind it.
- The pipeline omits the presubmit jobs that need a pull request. `batch-fuzzing`
  covers main nightly in place of `fuzz-changed-code`.

## Release — `release-pypi.yml`, `release-testpypi.yml`

| stage | jobs | notes |
|---|---|---|
| validate | `validate` | the ref is a tag, the tag is bare semver, the commit is an ancestor of `origin/main`, `CI Postsubmit` succeeded for it, and `Hardware Tests` succeeded if it ran |
| dist | `artifacts` | `release-artifacts` → `distribution-build` (sdist, wheels) and the SBOM |
| publish | `publish` | attest, verify the bundle offline, upload to PyPI, create the GitHub Release |
| prove | `smoke-tests`, `verify-provenance` | installed from the index; every file it serves is checked against a pinned signer |
| alert | `alert-channel` | on any failure |

### Key decisions

- Candidates and releases go to TestPyPI, while only releases go to PyPI,
  keeping the version list one-to-one with tags and GitHub releases.
- The attestation is verified offline before the upload to stop a broken bundle
  while publication is still avoidable. PyPI accepts each version once.
- `verify-provenance` downloads what the index serves to prove those files are
  the signed ones.
- `published-smoke` is dispatched with the version to cover the portable and
  hardware-dependent paths after the release.

## Nightly and weekly — `published-smoke.yml`, and the schedules

| workflow | when | what it does |
|---|---|---|
| `batch-fuzzing` | nightly | one ClusterFuzzLite session per engine, each followed by a corpus prune |
| `published-smoke` | nightly | installs what an index serves and runs the smoke suite against it |
| `code-scanning` | weekly | CodeQL, whose query set grows over time |
| `scorecard-analysis` | weekly | the OpenSSF supply-chain posture score |

`published-smoke` calls `smoke-tests` on every hosted platform and
`hardware-smoke-tests` on the self-hosted runner.

### Key decisions

- The check repeats nightly to catch a dependency set `pip` resolves differently
  from the one the release shipped against.
- A schedule is the only available trigger. `GITHUB_TOKEN` suppresses the events
  a workflow emits, so a release cannot start this one.
- Hosted legs pin `VFHE_ENGINE=portable` to fix the engine on a mixed fleet. The
  self-hosted leg leaves it unpinned to test engine selection.

## Conventions

- Sanitizers follow the engine: each runs where its ISA is native, over a
  subset of the suites.
- Coverage is informational. One C leg and one Python leg per engine are
  instrumented; every other leg builds the release configuration (`-O3`, LTO).
- An action provisions its own environment, and the job supplies the checkout.
- A job waiting for an offline self-hosted runner stays queued rather than
  failing, so no alert fires. The symptom is a run that never finishes.
- A tag names its own version; elsewhere setuptools-scm derives one, and a pull
  request appends `.pr<number>` to the local segment.
- `status.sh` and `provenance.sh` read their input rather than fetching it, and
  `.github/tests/ci_scripts.py` replays both against recorded responses.

## Adding an engine

The matrix rows need to be specified in:

- `ci-presubmit.yml` — one `c-tests` row and one `python-tests` row
- `ci-postsubmit.yml` — the same two, plus a `sanitized-tests` row where a
  hosted runner has the ISA
- `code-scanning.yml` — one `analyze-c` row
- `batch-fuzzing.yml` — one row, naming the runner that executes the ISA
- `hardware-tests.yml`, `hardware-smoke-tests.yml` — only for an engine that
  needs the project's own machine

## Adding an interpreter

The version lists live in:

- `ci-presubmit.yml`, `ci-postsubmit.yml` — `python-tests` rows
- `smoke-tests.yml` — the `python` list
- `distribution-build.yml` — the `interpreters` default, and the bounds
  `ci-presubmit.yml` passes
