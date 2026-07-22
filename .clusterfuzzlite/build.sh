#!/bin/bash -eu
# SPDX-License-Identifier: Apache-2.0
#
# ClusterFuzzLite entry point; delegates to build_c_fuzz_tests_ci.py so new
# harnesses and modules need no change here.
python3 "$SRC/vfhe/.clusterfuzzlite/build_c_fuzz_tests_ci.py"
