# Governance

vFHE is an open-source library built by an academic collaboration. This
document records who holds authority over what: the authors over the
cryptography, the maintainers over everything around it.

## Roles

- **Contributors** open issues and pull requests. No affiliation is required.
  Contributions are made under Apache-2.0 with a [DCO
  sign-off](https://github.com/vfhe/vfhe/blob/main/docs/CONTRIBUTING.md#sign-off-dco);
  there is no CLA.
- **Authors** designed and built the library's cryptography or had significant
  contributions to its original shaping. Changes are subject to review of the owners
  of the code in question.
- **Maintainers** run the project's day to day lifecycle: review and merge, cut releases,
  triage security reports, and administer the repository and its
  infrastructure. They also act as the Community Moderators of the
  [Code of Conduct](https://github.com/vfhe/vfhe/blob/main/docs/CODE_OF_CONDUCT.md)
  (<conduct@vfhe.ai>).

[`.github/CODEOWNERS`](https://github.com/vfhe/vfhe/blob/main/.github/CODEOWNERS)
encodes the people responsible for each part of the codebase.

## Maintainers

| Maintainer | GitHub |
|---|---|
| Antonio Guimarães | [@antoniocgj](https://github.com/antoniocgj) |
| Alin-Petru Roșu | [@rosualinpetru](https://github.com/rosualinpetru) |

Maintainers are appointed by consensus of the current maintainers, from
contributors with a sustained, identifiable track record in the project. A maintainer may step down
at any time; one unreachable for six months may be retired by the others.

The project keeps **at least two** maintainers. If the count falls below
two, appointing a replacement takes precedence over other work.

## Steering committee

The steering committee sets strategic direction and takes the final call on
decisions escalated to it. A report concerning a maintainer, or an appeal
against a moderation decision, goes to the steering committee.

The steering comitee will be established at the first release.

## Decisions

**Code.** Every change lands as a pull request carrying at least one approving
review from someone other than its author. This binds maintainers too such that
nobody merges their own work unreviewed. On cryptographic code, an author's review
can be solicited and objection from any source blocks the change until it is resolved.

**Design and scope.** A new module, a change to default parameters, or anything
altering the public API is discussed before it is written — in an issue, or with
the maintainers. Day-to-day design calls are made by the maintainers by consensus,
in consultation with the authors of the area concerned. Where consensus cannot be
reached, the decision escalates to the [steering committee](#steering-committee),
whose call is final.

**Strategic direction.** The [steering committee](#steering-committee)
sets it, following the plan and goals of the VERIFHE project
([GA: 101287502](https://github.com/vfhe/vfhe/blob/main/README.md#acknowledgements)),
and taking strategic decisions with the involvement of the authors.

**Everything else** — tooling, CI, documentation, release timing — is a
maintainer decision, made in the open through the same review process.

## Access and continuity

The project must survive any one person becoming unavailable. Every sensitive
resource — the GitHub organization, the PyPI project, the release environment,
and the `vfhe.ai` domain with its mailboxes — is administered by **at least two
maintainers**, each able to act alone.

Access is scoped to what a role requires and revoked when no longer
needed. No maintainer holds an irreplaceable credential.
When a maintainer departs, the remaining ones revoke their access and confirm
the two-holder rule still holds.

## Changing this document

This document changes by pull request, approved by the maintainers, like
everything else.
