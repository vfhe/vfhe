<!-- SPDX-FileCopyrightText: 2026 Alin-Petru Roșu -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Governance

How vFHE is run: who holds which role today, who decides what, and how the
project continues when someone leaves. Credit for the work itself is
[AUTHORS.md](https://github.com/vfhe/vfhe/blob/main/AUTHORS.md)'s business;
what a pull request must satisfy is in
[CONTRIBUTING.md](https://github.com/vfhe/vfhe/blob/main/CONTRIBUTING.md).

vFHE is both an open-source library and an academic collaboration, so authority
over the cryptography sits differently from authority over the packaging around
it. This document records that split.

## Roles

- **Contributors** open issues and pull requests. No affiliation is required.
  Contributions are made under Apache-2.0 with a [DCO
  sign-off](https://github.com/vfhe/vfhe/blob/main/CONTRIBUTING.md#sign-off-dco);
  there is no CLA.
- **Authors** designed and built the library's cryptography. Changes to the
  algorithms and the C kernels are reviewed by the authors of the code in
  question.
- **Maintainers** run the project day to day: review and merge, cut releases,
  triage security reports, and administer the repository and its
  infrastructure. They also act as the Community Moderators of the
  [Code of Conduct](https://github.com/vfhe/vfhe/blob/main/CODE_OF_CONDUCT.md)
  (<conduct@vfhe.ai>) until the steering committee is established.

[`.github/CODEOWNERS`](https://github.com/vfhe/vfhe/blob/main/.github/CODEOWNERS)
encodes the split: kernel changes request review from the author who owns
that code, packaging and CI from the maintainers.

## Decisions

**Code.** Every change lands as a pull request carrying at least one approving
review from someone other than its author, with the `CI OK` check green. This
binds maintainers too; nobody merges their own work unreviewed. On
cryptographic code, an author's objection blocks the change until it is
resolved.

**Design and scope.** A new module, a change to default parameters, or anything
altering the public API is discussed before it is written — in an issue, or with
the maintainers at <maintainers@vfhe.ai>. Day-to-day design calls are made by
the maintainers by consensus, in consultation with the authors of the area
concerned. Where consensus cannot be reached, the decision escalates to the
VERIFHE project lead, Dario Fiore, whose call is final in this initial
period; pending that ruling, the status quo stands.

**Strategic direction.** In this initial period, strategic decisions are
aligned with the plan and goals established by the VERIFHE project
([GA: 101287502](https://github.com/vfhe/vfhe/blob/main/README.md#acknowledgements)),
which funds this phase of development, and are taken with the involvement
of the authors. A steering committee will be established upon the first
release; its membership will be listed here.

**Everything else** — tooling, CI, documentation, release timing — is a
maintainer decision, made in the open through the same review process.

## Maintainers

| Maintainer | GitHub |
|---|---|
| Antonio Guimarães | [@antoniocgj](https://github.com/antoniocgj) |
| Alin-Petru Roșu | [@rosualinpetru](https://github.com/rosualinpetru) |

Maintainers are appointed by consensus of the current maintainers, from
contributors with a sustained, identifiable track record in the project. A maintainer may step down
at any time; one unreachable for six months may be retired by the others, and
their access removed.

The project keeps **at least two** maintainers. Should the count fall below
two, appointing a replacement takes precedence over other work.

## Access and continuity

The project must survive any one person becoming unavailable. Every sensitive
resource — the GitHub organization, the PyPI project, the release environment,
and the `vfhe.ai` domain with its mailboxes — is administered by **at least two
maintainers**, each able to act alone.

Access is scoped to what a role requires and revoked when no longer
needed. No maintainer holds an irreplaceable credential. Releases publish through
Trusted Publishing (short-lived OIDC tokens, no stored API keys) and are signed
with ephemeral Sigstore keys, so there is no signing key to inherit or lose.
When a maintainer departs, the remaining ones revoke their access and confirm
the two-holder rule still holds.

## Changing this document

By pull request, approved by the maintainers, like everything else.
