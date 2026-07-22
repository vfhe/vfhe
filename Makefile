# SPDX-License-Identifier: Apache-2.0
PYTHON ?= python3
CLANG_FORMAT ?= clang-format
SOURCES = modules packaging scripts smoke .clusterfuzzlite .github setup.py

# Never write .pyc anywhere during the dev loop (no __pycache__ in the source).
export PYTHONDONTWRITEBYTECODE := 1

.PHONY: deps format proto build test test-fast fuzz-local sdist smoke clean

deps:     ## install the dev dependency group + git hooks (not the package; needs pip >= 25.1)
	$(PYTHON) -m pip install --group dev
	$(PYTHON) -m pre_commit install

format:   ## format all Python (ruff) and C (clang-format) sources in place
	$(PYTHON) -m ruff format $(SOURCES)
	$(PYTHON) -m ruff check --fix $(SOURCES)
	find modules \( -name '*.c' -o -name '*.h' \) -print0 | xargs -0 $(CLANG_FORMAT) -i

proto:    ## (re)generate protobuf bindings into .generated/_vfhe_proto/
	$(PYTHON) packaging/generate_protos.py

build: proto  ## generate proto bindings + compile the native lib/stub into .generated/
	$(PYTHON) packaging/build_ffi.py

test: build   ## build, then C tests + the complete Python suite (incl. heavy bootstraps)
	$(PYTHON) scripts/run_c_tests.py
	$(PYTHON) -m pytest --complete

test-fast: build  ## build, then C tests + the fast Python suite (default; skips @complete)
	$(PYTHON) scripts/run_c_tests.py
	$(PYTHON) -m pytest

fuzz-local:  ## build the c/fuzz harnesses with local clang and fuzz each for 60s
	$(PYTHON) scripts/run_c_fuzz_tests_local.py

sdist:    ## build the source distribution into dist/ (the only artifact released)
	rm -rf dist
	$(PYTHON) -m build --sdist

smoke: sdist  ## install the fresh sdist into a scratch venv and run every smoke test
	$(PYTHON) scripts/check_install.py dist/*.tar.gz
	$(PYTHON) scripts/run_smoke_tests.py --python .cache/install/venv/bin/python

clean:    ## remove all generated/build artifacts and caches
	rm -rf .generated .cache build dist *.egg-info
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
