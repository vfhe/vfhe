# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0

PYTHON ?= python3

# meson's b_coverage: true or false.
VFHE_COVERAGE ?= false
# meson's b_sanitize: none, address, address,undefined, ...
VFHE_SANITIZE ?= none
# meson's b_ndebug: if-release compiles the C asserts out of a release build,
# as a wheel is built; false keeps them. `make test` always keeps them.
VFHE_NDEBUG ?= if-release
# One engine's name, to compile its kernels alone.
KERNELS ?=
# Groups from pyproject's [dependency-groups], space separated.
DEPENDENCY_GROUPS ?= dev
# A heading in CHANGELOG.md, like 0.0.2 or Unreleased.
CHANGELOG_VERSION ?= Unreleased
# A local wheel or sdist to install, as a path or a glob. Wins over REQUIREMENT.
DIST ?=
# What pip installs from INDEX, like vfhe==1.2.3. Empty installs a local artefact.
REQUIREMENT ?=
# pypi or testpypi: the tox environment that installs.
INDEX ?= pypi
# Names of smoke test cases, space separated. Empty runs every one of them.
SMOKE_CASES ?=
# Benchmarks: modules (comma separated, empty for all), thread limits, the tier
# (headline or full), a glob on names, repetitions (empty for each benchmark's
# own), CPUs to pin to, and the label that names the results file.
MODULE ?=
THREADS ?= 1
TIER ?= headline
BENCH ?=
REPS ?=
PIN ?=
LABEL ?= run

# In precedence: the artefact DIST names, a requirement from INDEX, or the sdist.
PACKAGE = $(or $(wildcard $(DIST)),$(REQUIREMENT),$(wildcard dist/*.tar.gz))

export PYTHONDONTWRITEBYTECODE := 1

.PHONY: bench build clean deps dev-env docs format help lint release-notes sbom-embed sdist smoke spellcheck test wheel
.DEFAULT_GOAL := help

bench: build  ## time benchmarks into results/bench (MODULE=arith ENGINE=<name> THREADS=1,8 TIER=full BENCH='ntt*' PIN=0-31 LABEL=x)
	$(PYTHON) tools/bench/run.py --build-dir build --threads $(THREADS) --tier $(TIER) --label $(LABEL) \
		$(if $(MODULE),--module $(MODULE)) $(if $(ENGINE),--engine $(ENGINE)) \
		$(if $(BENCH),--bench '$(BENCH)') $(if $(REPS),--reps $(REPS)) $(if $(PIN),--pin $(PIN))

build:    ## compile everything into build/ (KERNELS=<engine> alone, VFHE_SANITIZE=address)
	meson setup build
	meson configure build -Db_coverage=$(VFHE_COVERAGE) -Db_sanitize=$(VFHE_SANITIZE) \
		-Db_ndebug=$(VFHE_NDEBUG)
	meson compile -C build $(if $(KERNELS),vfhe_$(KERNELS))

clean:    ## remove all generated/build artifacts and caches
	rm -rf .cache .tox build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

deps: export PIP_CONSTRAINT = ci-constraints-requirements.txt
deps:     ## install DEPENDENCY_GROUPS (default dev) into this environment
	$(PYTHON) -m pip install --upgrade "pip>=25.1"
	$(PYTHON) -m pip install $(addprefix --group ,$(DEPENDENCY_GROUPS))

dev-env: deps  ## everything a contributor needs: the dev group, then the git hooks
	$(PYTHON) -m pre_commit install

docs: build  ## generate the documentation site into build/docs/html (needs doxygen)
	mkdir -p build/docs/doxygen
	doxygen docs/site/api/Doxyfile
	$(PYTHON) -m sphinx -W --keep-going -q docs/site build/docs/html

format:   ## format all Python (ruff) and C (clang-format) sources in place
	find modules \( -name '*.c' -o -name '*.h' \) -print0 | xargs -0 clang-format -i
	$(PYTHON) -m ruff check --fix
	$(PYTHON) -m ruff format

help:     ## list the targets
	@awk -F ':.*## ' '/^[a-z-]+:.*## /{printf "%-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

lint:     ## every static check CI runs: the pre-commit hooks, then pyright
	meson setup build
	meson compile -C build proto-bindings
	$(PYTHON) -m pre_commit run --all-files --show-diff-on-failure
	$(PYTHON) -m pyright

release-notes:  ## print CHANGELOG_VERSION's section of CHANGELOG.md, compare link included
	@tools/release/notes/print.sh "$(CHANGELOG_VERSION)"

# meson-python cannot write .dist-info, so the fragments go in after the build.
# When mesonbuild/meson-python#843 is resolved, install them to {datadir}/vfhe/sboms/.
sbom-embed:  ## put every vendored dependency's SBOM in dist/*.whl, per PEP 770
	@PYTHON=$(PYTHON) tools/release/sbom/embed.sh dist/*.whl

sdist:    ## build the source distribution into a fresh dist/
	rm -rf dist
	$(PYTHON) -m build --sdist

smoke: $(if $(DIST)$(REQUIREMENT),,sdist)  ## run test/smoke against PACKAGE in a sandbox venv (SMOKE_CASES="info ckks")
	VFHE_DIST="$(PACKAGE)" $(PYTHON) -m tox run -e $(INDEX) --recreate -- $(SMOKE_CASES)

spellcheck:  ## codespell over the tree; manual-only, so no gate blocks on prose
	$(PYTHON) -m pre_commit run --hook-stage manual codespell --all-files

# The tests run with the C asserts compiled in; the build they depend on
# inherits this.
test: VFHE_NDEBUG = false
test: build  ## SUITES on ENGINE (ENGINE=all|<name> SUITES=c,fast EMULATE=1 VFHE_COVERAGE=true)
	tools/test/unit/run.sh build

wheel:    ## build this interpreter's wheel into a fresh dist/ (then `make sbom-embed`)
	rm -rf dist
# install_subdir copies the tree as it stands, so bytecode left by a bare
# `python -c` would ship. Clear it, then prove none reached the wheel; the
# check covers every package at once.
	find modules -type d -name __pycache__ -prune -exec rm -rf {} +
	$(PYTHON) -m build --wheel
	@if unzip -l dist/*.whl | grep -q '\.pyc'; then \
		echo 'bytecode reached the wheel' >&2; exit 1; \
	fi
