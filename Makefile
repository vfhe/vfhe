# SPDX-License-Identifier: Apache-2.0
PYTHON ?= python3
CLANG_FORMAT ?= clang-format
SOURCES = modules native scripts setup.py

# Never write .pyc anywhere during the dev loop (no __pycache__ in the source).
export PYTHONDONTWRITEBYTECODE := 1

.PHONY: deps format proto build test wheel test-wheel clean

deps:     ## install the dev dependency group from pyproject (not the package; needs pip >= 25.1)
	$(PYTHON) -m pip install --group dev

format:   ## format all Python (ruff) and C (clang-format) sources in place
	$(PYTHON) -m ruff format $(SOURCES)
	$(PYTHON) -m ruff check --fix $(SOURCES)
	find modules \( -name '*.c' -o -name '*.h' \) -print0 | xargs -0 $(CLANG_FORMAT) -i

proto:    ## (re)generate protobuf bindings into .generated/_vfhe_proto/
	$(PYTHON) scripts/gen_proto.py

build: proto  ## generate proto bindings + compile the native lib/stub into .generated/
	$(PYTHON) native/build_ffi.py

test: build   ## build, then run the C (Unity) and Python (pytest) test suites
	$(PYTHON) scripts/run_c_tests.py
	$(PYTHON) -m pytest

wheel:    ## build a wheel for the current platform into dist/
	$(PYTHON) -m build --wheel

clean:    ## remove all generated/build artifacts and caches
	rm -rf .generated .cache build dist *.egg-info
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
