# Contributing to vFHE

For anyone opening an issue or a pull request against vFHE.

The [development
guide](https://github.com/vfhe/vfhe/blob/main/docs/DEVELOPMENT.md) covers how
the project is built and organized.

By contributing you agree that your work is licensed under
[Apache-2.0](https://github.com/vfhe/vfhe/blob/main/LICENSE), and you accept
the [Code of
Conduct](https://github.com/vfhe/vfhe/blob/main/docs/CODE_OF_CONDUCT.md).

For security problems, do not open an issue; see the
[security policy](https://github.com/vfhe/vfhe/blob/main/docs/SECURITY.md).

## Bugs and ideas

Report bugs as [GitHub issues](https://github.com/vfhe/vfhe/issues).
For a large change, open an issue to discuss it before writing the code; this
avoids building changes that will not be accepted.

Modules are a central component of the library's design, so reach out
before creating one. The
[development guide](https://github.com/vfhe/vfhe/blob/main/docs/DEVELOPMENT.md#adding-a-module)
has the layout and the steps.

## Before you open a pull request

Run `make test` and let it finish: its default is the **complete** suite. The
merge gate runs only the fast depth — the complete depth, heavy FHE bootstraps
included, runs after the merge — so a failure only the complete suite catches
would land on `main` instead of on your pull request.

### Testing policy

Any change that adds or alters functionality MUST add or
update automated tests covering it. Pure refactors, docs, and CI changes are
exempt. When something genuinely cannot be tested (unreachable defensive branches, allocation-failure paths, platform-specific fallbacks), mark it so the report is
not misleading:

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

This style is a recommendation; no tooling enforces it.

### Sign-off (DCO)

Every commit must be signed off (`git commit -s`), certifying the
[Developer Certificate of Origin](https://developercertificate.org).
The [DCO app](https://github.com/apps/dco) checks every pull request,
and the `dco-signoff` pre-commit hook refuses the commit locally.
To fix a series of unsigned commits: `git rebase --signoff <base>` and force-push.

If you wrote the contribution as part of employment or a funded project, the
rights may belong to your employer or funder; by signing off you assert you
are authorised to contribute it. When unsure, check your institution's IP
policy first.

## What CI will check

Most CI checks can be performed locally via `make`; see
[`WORKFLOWS.md`](https://github.com/vfhe/vfhe/blob/main/docs/WORKFLOWS.md) for how the pipelines are organised.

### About coverage

Coverage is reported on every pull request by
[Codecov](https://about.codecov.io), which combines the uploads from every engine
and platform into one figure. **It does not block a merge**; treat it as
information for the reviewer.
