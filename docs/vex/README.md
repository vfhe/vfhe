<!-- SPDX-License-Identifier: Apache-2.0 -->

# VEX feed

[`vfhe.openvex.json`](vfhe.openvex.json) is this project's
[OpenVEX](https://github.com/openvex) feed: the exploitability status of known
vulnerabilities in VFHE's dependencies. It lets downstream consumers suppress
alerts that do not affect them.

The feed is empty when no such vulnerability is outstanding.

## When a statement is added

When a scanner (Dependabot, `dependency-review`, a GHSA) reports a vulnerability
in a dependency, and review finds VFHE does not execute the vulnerable code
path, a `not_affected` statement is added here rather than treating the finding
as a release blocker (see the dependency policy in
[SECURITY.md](../../SECURITY.md)). A vulnerability that *does* affect VFHE is
fixed and released, not VEX-suppressed.

Each statement names the vulnerability, the affected product
(`pkg:pypi/vfhe`), a `status`, and — for `not_affected` — a `justification`
and an `impact_statement`. Edit the JSON by hand or with
[`vexctl`](https://github.com/openvex/vexctl); bump `version` and update
`timestamp` on every change.

Example (illustrative only; not a real advisory):

```json
{
  "vulnerability": { "name": "CVE-0000-00000" },
  "products": [{ "@id": "pkg:pypi/vfhe" }],
  "status": "not_affected",
  "justification": "vulnerable_code_not_in_execute_path",
  "impact_statement": "The affected function is never called by VFHE."
}
```
