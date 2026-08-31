# Workflows

For contributors reading or changing CI. It records why the pipelines
have the shape they have; the workflow files state what runs.

## The model

- A `workflow_call`-only file is a **stage** with two or more subscribers;
  a step sequence shared inside jobs is an **action** with two or more call
  sites.
- Separate jobs of related workflows into two files only when they must not
  share the same lifetime. `hardware-test.yml` is separate from
  `ci-postsubmit.yml` because silicon takes one job at a time: inside the
  pipeline it would hold every push's `CI Postsubmit OK` behind the silicon
  queue, while separate the release waits only for hardware runs that exist.
  This is to prevent blocking the release due to an offline runner.
- A `*-ok` job exists only where a status check is needed:
  `CI Presubmit OK` for the branch ruleset, `CI Postsubmit OK` for the
  release wait. Since status check names are not unique across workflows,
  each carries its workflow's name.
- Only unattended runs (schedule, push) end in an alert job; a pull request
  or a dispatch already assumes a watcher.

## Key decisions

### The merge gate — `ci-presubmit.yml`

- Every required check runs on hosted runners or a deterministic emulator,
  so anyone can reproduce it without this repository's hardware.
- A reduced set of fuzzing, sanitizing, testing, and packaging runs on the
  pull request; the complete set is deferred to main. This keeps
  pull-request time short, and with no continuous delivery or deployment a
  red main is acceptable, blocking the next manual release.

### Push to main — `ci-postsubmit.yml`, `hardware-test.yml`

- `release-build` runs on every commit on `main`, stamped `0+g<commit>`
- `hardware-test.yml` skips documentation-only pushes

### Release — `release-pypi.yml`, `release-testpypi.yml`

- Release workflows wait for main's status checks on the commit, so a
  release started mid-build blocks until an answer exists. An offline runner
  leaves a check queued forever; the validating job's one-hour timeout turns
  that into a visible failure.
- Release candidates go only to TestPyPI, keeping PyPI one-to-one with tags.
- The attestation is verified before upload, while publication is still
  avoidable, and again after, against what the index actually serves.
- A release never re-tests source: it asserts what a tag can get wrong, and
  it reads main's already-proven status as a precondition it fails on, never
  as work it skips.

### Nightly and weekly

- The health check repeats nightly because `pip` resolves the dependency set
  anew, not as the release shipped it.
- Smoke jobs explicitly decide on the engine to use per architecture —
  portable on GitHub x86, whose fleet is mixed, neon on arm64, where it is
  baseline. The self-hosted run leaves the decision to the picker between
  portable or optimised engine.
- Fuzzing progresses through a corpus that every nightly session resumes
  and grows; a pull request only fuzzes its own changes briefly.
- SAST re-scans weekly: each run rebuilds its database
  and pulls the current query packs, so unchanged code can yield new
  findings.

### The self-hosted AVX-512 IFMA runner

The project added a self-hosted runner, labelled
`self-hosted, Linux, X64, avx512ifma`, because GitHub's hosted fleet is
mixed: whether a runner offers AVX-512 IFMA varies per job, so any hosted
result for that engine is nondeterministic. This runner answers
deterministically as `avx512ifma` is antively supported.

Three workflows name the label, and the runner serialises them one job at a
time. Because an offline runner queues jobs instead of failing them, the
pull-request gate schedules nothing on it to minimize development friction.
Only the release and post-submit workflows depend on it.

## Conventions

- The merge method is rebase, and its cost is accepted: rebase mints new
  SHAs, so a pull request's green status never belongs to the commits that
  ship, and a bisect can land on a commit CI never built. Squash would make
  commit and status one-to-one, at the price of the per-commit history.
- Filenames lead with the concern (`ci-`, `release-`, `scan-`, `test-`).
- A job calling a reusable workflow takes the callee's basename and `name:`.
- A reusable workflow's `concurrency.group` starts with its own literal name,
  never `${{ github.workflow }}`: inside a called workflow that expands to the
  *caller*, so the group becomes the caller's own and GitHub cancels the run as
  a deadlock.
- CI holds one secret: `ZULIP_API_KEY`, read only by `alert.yml`, which
  never executes repository code.
- Secret scanning is a platform setting, not a job: push protection rejects
  a credential before the commit exists, which no workflow can.
- No `${{ }}` reaches a `run:` block: every expression travels through
  `env:` and is used quoted. `zizmor --persona=pedantic` enforces it.

### Frozen names

Renaming one of these introduces breaking changes unless synchronized.

| identifier | frozen by |
|---|---|
| `CI Presubmit OK` job name | the branch ruleset |
| `DCO` (the [DCO app](https://github.com/apps/dco)'s check) | the branch ruleset |
| `CI Postsubmit OK` job name | both release workflows |
| `Test Hardware` job-name prefix | both release workflows wait on the prefix; renamed, the wait matches nothing and passes silently |
| `ci-postsubmit.yml` | the README badge |
| every workflow `name:` | the Zulip topic — a rename recreated an append-only thread |
| `release-pypi.yml`, `release-testpypi.yml` | the Trusted Publisher registrations and every published attestation |

## Running the checks locally

Every CI check is a `make` target, so the two cannot disagree about what a
check means:

| CI does | you do |
|---|---|
| lint | `make lint` |
| the C and Python suites | `make test`, `make test ENGINE=all SUITES=c,fast` |
| the sanitized jobs | `make test VFHE_SANITIZE=address,undefined` |
| the coverage table | `make test VFHE_COVERAGE=true` |
| the emulated jobs | `make test EMULATE=1` |
| the version a build carries | `make version` |
| the release notes the publish job posts | `make release-notes CHANGELOG_VERSION=x.y.z` |
| the installed package | `make smoke`, `make smoke REQUIREMENT=vfhe==<version>` |

## Releasing

- **TestPyPI** — dispatch **Release TestPyPI** from `main` with a `version`
  such as `0.0.2rc1`.
- **PyPI** — push a bare-semver tag (`0.1.0`, no leading `v`), then dispatch
  **Release PyPI** from that tag. A GitHub Release with the sdist, wheels
  and build provenance is created alongside.
- Read `make release-notes CHANGELOG_VERSION=x.y.z` before tagging: the
  publish job feeds that text to the GitHub Release, and a version the
  changelog never documents fails loudly.
