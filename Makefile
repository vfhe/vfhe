# SPDX-License-Identifier: Apache-2.0
PYTHON ?= python3
CLANG_FORMAT ?= clang-format
SOURCES = modules native scripts setup.py

# Never write .pyc anywhere during the dev loop (no __pycache__ in the source).
export PYTHONDONTWRITEBYTECODE := 1

.PHONY: deps format proto build test test-fast smoke sdist clean

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

test: build   ## build, then C tests + the COMPLETE Python suite (incl. heavy bootstraps)
	$(PYTHON) scripts/run_c_tests.py
	$(PYTHON) -m pytest --complete

test-fast: build  ## build, then C tests + the FAST Python suite (default; skips @complete)
	$(PYTHON) scripts/run_c_tests.py
	$(PYTHON) -m pytest

smoke: build  ## run the end-to-end CKKS smoke test (demo + integration check)
	$(PYTHON) scripts/smoke.py

sdist:    ## build the source distribution into dist/ (the only artifact released)
	$(PYTHON) -m build --sdist

clean:    ## remove all generated/build artifacts and caches
	rm -rf .generated .cache build dist *.egg-info
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
