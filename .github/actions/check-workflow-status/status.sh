#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# The status one workflow recorded for one commit, as a pass or a failure.
# Reads the workflow-runs JSON on stdin, so a decision can be replayed offline:
#
#   gh api "repos/$REPO/actions/runs?head_sha=$SHA" \
#     | status.sh "$SHA" "CI Postsubmit" required
#
# A run that concluded anything but success always fails. `required` fails when
# no run exists too; `optional` accepts that, for a workflow whose triggers let
# it legitimately skip a commit.
set -euo pipefail

USAGE="usage: status.sh <sha> <workflow> <required|optional>"
SHA=${1:?$USAGE}
WORKFLOW=${2:?$USAGE}
PRESENCE=${3:?$USAGE}

case "$PRESENCE" in
    required | optional) ;;
    *)
        echo "::error::presence is '$PRESENCE', not required or optional"
        exit 1
        ;;
esac

conclusion=$(jq -r --arg name "$WORKFLOW" '
    [.workflow_runs[] | select(.name == $name)] | first
    | select(. != null)
    | .conclusion // "in_progress"')

case "$conclusion" in
    success)
        echo "- $WORKFLOW: success"
        ;;
    "")
        if [ "$PRESENCE" = required ]; then
            echo "::error::$WORKFLOW never ran for $SHA"
            exit 1
        fi
        echo "- $WORKFLOW: did not run for this commit"
        ;;
    *)
        echo "::error::$WORKFLOW for $SHA is $conclusion"
        exit 1
        ;;
esac
