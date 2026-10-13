#!/bin/sh
# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
#
# Generates the protobuf bindings.
#   bindings.sh OUT PYTHON PROTOC_ARG...
set -eu

out=$1
python=$2
shift 2

mkdir -p "$out"
"$python" -m grpc_tools.protoc "$@" --python_out="$out" --pyi_out="$out"
find "$out" -type d -exec touch {}/__init__.py \;
