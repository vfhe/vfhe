# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# The front end of the lifecycle: one target per stage, meson and tools/ behind
# them, and nothing CI does that you cannot do here. What varies between runs
# is a variable, never a target of its own. Targets are alphabetical.

# ------------------------------------------------------------------ knobs ---

PYTHON ?= python3
CLANG_FORMAT ?= clang-format
SOURCES = modules tools smoke .clusterfuzzlite .github
BUILD_DIR = build

# Build knobs, passed to meson: `make test VFHE_SANITIZE=address,undefined`.
VFHE_COVERAGE ?= 0
VFHE_SANITIZE ?= none

# Test axes: which engine, and how deep (see `test`).
ENGINE ?= all
SUITES ?= c,complete

# What to install or scan — a file, a glob, or a requirement — and the scratch
# environment it lands in. Unset means whatever `make sdist` left in dist/.
SOME_DIST = $(or $(DIST),dist/*.tar.gz)
VENV = .cache/install/venv

# An index is named, never typed as a URL: the two are spelled out once, here.
INDEX ?= pypi
INDEX_URL = $(strip $(if $(filter pypi,$(INDEX)),,$(if $(filter testpypi,$(INDEX)),\
	https://test.pypi.org/simple/,$(error INDEX is '$(INDEX)', not pypi or testpypi))))

SBOM = dist/sbom.json

# Never write .pyc anywhere during the dev loop (no __pycache__ in the source).
export PYTHONDONTWRITEBYTECODE := 1

.PHONY: build clean deps format help install lint release-notes sbom sdist \
	smoke test

# Bare `make` must inform, never mutate the environment.
.DEFAULT_GOAL := help

# --------------------------------------------------------------- coverage ---

# Coverage is not a command but a way of running the suites: VFHE_COVERAGE=1
# instruments the build (gcov instead of LTO, see `build`), makes whichever
# suites run report what they executed, and ends in the table below. It leaves
# build/ instrumented, so `make build` is what returns it to a release one.
#
# A leg is one measured run — this engine at this depth — written where CI's
# artifacts also unpack, so the union runs identically in both places. It holds
# everything that run measured, coverage.py's raw data included (COVERAGE_FILE),
# because the union re-combines those. Another engine, or another depth of the
# same engine, is another leg beside it: CI measures the C suites and the Python
# suites in the jobs that own them, and the union joins their legs per engine.
COMMA = ,
LEGS = .coverage/legs
LEG = $(LEGS)/coverage-$(ENGINE)-$(subst $(COMMA),+,$(SUITES))
# Locally the report's link is the directory holding it; CI passes its run URL.
REPORT_URL ?= .coverage

ifeq ($(VFHE_COVERAGE),1)
ifeq ($(ENGINE),all)
$(error measure one engine at a time: ENGINE=<name> VFHE_COVERAGE=1)
endif
MEASURE_ENV = COVERAGE_FILE=$(LEG)/data
MEASURE_ARGS = -- --cov=vfhe \
	--cov-report=json:$(LEG)/coverage-python.json \
	--cov-report=html:$(LEG)/coverage-python-html
# The C half is gcovr's, over the gcov data the run just left in build/. Its
# threshold 0 disables the suspicious-hits check: that check rejects counts over
# 2^32 as gcov bug gcc#68080, but our hot kernels legitimately exceed it, so it
# only ever fires falsely. Do not use --gcov-ignore-parse-errors instead: it
# rewrites the offending hits to 0.
MEASURE_C = mkdir -p $(LEG)/coverage-c-html && $(PYTHON) -m gcovr --root . \
	--filter 'modules/.*/c/src/' --gcov-suspicious-hits-threshold 0 \
	--merge-mode-functions=merge-use-line-max \
	--html-title "vFHE C coverage ($(ENGINE) engine)" \
	--json $(LEG)/coverage-c.json \
	--json-summary $(LEG)/coverage-c-summary.json \
	--html-details $(LEG)/coverage-c-html/index.html $(BUILD_DIR)
# The union of every leg measured so far, as one table: a line counts as covered
# when any engine covered it. CI's summary job runs these two tools over the legs
# its jobs uploaded — same tools, but nothing of its own to measure.
MEASURE_SUMMARY = $(PYTHON) tools/coverage/merge.py $(LEGS) .coverage/merged && \
	$(PYTHON) tools/coverage/render.py .coverage/merged/coverage.json \
		tools/coverage/summary.md.in "$(REPORT_URL)" | tee .coverage/summary.md
else
MEASURE_C = @: # nothing was measured
MEASURE_SUMMARY = @: # nothing to summarize
endif

# ---------------------------------------------------------------- targets ---

# Everything meson produces stays in build/, and the dev loop imports from
# there — extensions, archives, and the generated protobuf bindings alike.
# KERNELS=<engine> narrows the compile to that engine's kernels, which is what
# static analysis needs: it traces one preprocessor variant per run, and the
# other engines' kernels would be attributed to this engine's report.
build:    ## compile everything into build/ (KERNELS=<engine> for its kernels alone)
	meson setup $(BUILD_DIR) --reconfigure \
		-Db_coverage=$$([ "$(VFHE_COVERAGE)" = "1" ] && echo true || echo false) \
		-Db_sanitize=$(VFHE_SANITIZE)
	meson compile -C $(BUILD_DIR) $(if $(KERNELS),vfhe_$(KERNELS))

clean:    ## remove all generated/build artifacts and caches
	rm -rf .cache .coverage build dist *.egg-info
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

deps:     ## install the dev dependency group + git hooks (not the package)
	$(PYTHON) -m pip install --upgrade "pip>=25.1"  # fresh venvs bundle a pip too old for --group
	$(PYTHON) -m pip install --group dev
	$(PYTHON) -m pre_commit install

format:   ## format all Python (ruff) and C (clang-format) sources in place
	$(PYTHON) -m ruff format $(SOURCES)
	$(PYTHON) -m ruff check --fix $(SOURCES)
	find modules \( -name '*.c' -o -name '*.h' \) -print0 | xargs -0 $(CLANG_FORMAT) -i

# The menu derives from the `## description` each target line carries, so it
# can never drift; a target without one is silently absent from the list.
help:     ## list the targets
	@awk -F ':.*## ' '/^[a-z-]+:.*## /{printf "%-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# One scratch venv, a user's environment reproduced: `smoke` and `sbom` are the
# two things done to it. A requirement gets retried, because an index needs a
# moment to serve a release that was just published.
#   make install                                      the sdist built right here
#   make install DIST=vfhe==1.2.3rc1 INDEX=testpypi
install:  ## install DIST (default: a fresh sdist) into a scratch venv (INDEX=pypi|testpypi)
	$(if $(DIST),,$(MAKE) sdist)
	$(PYTHON) tools/install.py $(VENV) $(SOME_DIST) $(INDEX_URL)

# The hooks are the definition of every formatter and validator; this runs them
# over the whole tree, so CI and a commit cannot disagree about what they mean.
lint:     ## every static check CI runs: the pre-commit hooks, then pyright
	# pyright reads the protobuf bindings but cannot build them; configure only
	# if the tree is absent, so an instrumented build dir survives a lint.
	@[ -d $(BUILD_DIR) ] || meson setup $(BUILD_DIR)
	meson compile -C $(BUILD_DIR) proto-bindings
	$(PYTHON) -m pre_commit run --all-files --show-diff-on-failure
	$(PYTHON) -m pyright

# Read it before you tag: the publish job feeds this exact text to the GitHub
# Release, so a version the changelog never documents fails there instead.
release-notes:  ## print VERSION's section of CHANGELOG.md, compare link included
	@$(PYTHON) tools/release/extract_notes.py $(VERSION)

# What an install actually contains: cyclonedx-py scans the venv, then the
# vendored C that no Python-environment scan can see (BLAKE3) is appended from
# the submodule itself. Needs the release dependency group, which `make deps`
# installs.
sbom: install  ## CycloneDX SBOM of that installed package, beside the distribution
	mkdir -p dist
	$(PYTHON) -m cyclonedx_py environment $(VENV)/bin/python \
		--output-reproducible --output-format JSON --output-file $(SBOM)
	$(PYTHON) tools/sbom/amend.py $(SBOM)

sdist:    ## build the source distribution into a fresh dist/
	rm -rf dist
	$(PYTHON) -m build --sdist

smoke: install  ## run smoke/*.py against that installed package
	$(VENV)/bin/python tools/test/run_smoke.py

# One command, two axes: which engine and how deep. `all` is every engine this
# CPU runs natively — the ones needing an emulator are skipped by name, since
# an emulated suite costs 10-50x. EMULATE=1 runs the named engine on its
# emulator anyway (CI's one deterministic path); IF_SUPPORTED=1 succeeds doing
# nothing where this host cannot build it. The C suites are meson's: parallel,
# and sanitized through VFHE_SANITIZE.
#   make test                                  every engine, C + the heavy suite
#   make test SUITES=c,fast                    the same, quickly
#   make test ENGINE=avx512ifma EMULATE=1      one engine, under Intel SDE
test: build  ## SUITES on ENGINE (ENGINE=all|<name> SUITES=c,fast EMULATE=1 VFHE_COVERAGE=1)
	$(MEASURE_ENV) $(PYTHON) tools/test/run.py $(ENGINE) $(SUITES) \
		$(if $(filter 1,$(EMULATE)),--emulate) \
		$(if $(filter 1,$(IF_SUPPORTED)),--if-supported) $(MEASURE_ARGS)
	$(MEASURE_C)
	$(MEASURE_SUMMARY)
