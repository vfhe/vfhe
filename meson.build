# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# The native build: every module's C into one extension plus one static
# archive PER ENGINE, and every C unit test as a meson test (so `meson test`
# runs them in parallel, with timeouts, sanitizers via -Db_sanitize, and
# emulators via --wrapper). Engine facts come from tools/_engines.py via
# engine_info.py — this file carries no engine flag literals, so the two
# cannot drift.

project(
    'vfhe',
    'c',
    version: run_command('tools/meson/print_version.py', check: true).stdout().strip(),
    license: 'Apache-2.0',
    meson_version: '>=1.3.0',
    default_options: ['c_std=gnu11', 'buildtype=release'],
)

fs = import('fs')
m_dep = meson.get_compiler('c').find_library('m', required: false)

# Running a helper script needs an interpreter; building an extension needs
# Python's headers. Keeping the two apart is what lets `-Dfuzz=true`
# configure with neither the headers nor the protobuf toolchain installed —
# a fuzzing container wants the instrumented kernels and nothing else.
fuzz = get_option('fuzz')
py = find_program('python3')
if not fuzz
    py_install = import('python').find_installation(pure: false)
endif

if not fs.exists('external/blake3/c/blake3.c')
    error(
        'Missing vendored BLAKE3 (submodule not checked out?). Run:  git submodule update --init --recursive',
    )
endif

# --- Engine facts (single truth: tools/_engines.py) --------------------------

# The engines this host can compile, best first — the registry decides, so
# adding one there adds its artifacts and its test suites here. The
# pure-Python `_vfhe_native` shim picks among them at import.
engine_info = files('tools/meson/engine_info.py')
engines = run_command(py, engine_info, 'names', check: true).stdout().strip().split('\n')

# Coverage swaps LTO for gcov instrumentation, as everywhere in the build.
#
# -fno-semantic-interposition is what lets LTO actually inline the reduction
# primitives. Without it GCC has to assume any default-visibility global in a
# shared object may be interposed at load time, so a call to one becomes a PLT
# call it cannot inline through -- which put a dynamic call per element inside
# every scalar kernel.
lto_args = get_option('b_coverage') ? [] : ['-flto', '-fno-semantic-interposition']

# --- What the modules contribute (filled by modules/*/meson.build) -----------

# Where the package's files land. A fuzzing build installs nothing (it only
# compiles one archive), so both are placeholders there: the modules' install
# rules are still declared, but nothing ever runs them — which is what keeps
# `if not fuzz` out of every module.
source_dir = fuzz ? 'nothing-is-installed' : py_install.get_install_dir() / 'vfhe' / '_source'
python_dir = fuzz ? 'nothing-is-installed' : py_install.get_install_dir()

lib_sources = files(
    'external/blake3/c/blake3.c',
    'external/blake3/c/blake3_dispatch.c',
    'external/blake3/c/blake3_portable.c',
)
lib_includes = [include_directories('external/blake3/c')]
header_files = []
cdef_files = []
c_test_files = []
proto_files = []
python_source_dirs = []

subdir('modules/arith')
subdir('modules/circuit')
subdir('modules/fhe')
subdir('modules/misc')
subdir('modules/mlwe')
subdir('modules/piop')
subdir('modules/polycom')

# --- The CPU probe ------------------------------------------------------------

# A few kilobytes, no engine flags, no library: the picker imports this to
# ask what the CPU can run, then imports exactly one engine.
if not fuzz
probe_ffi_c = custom_target(
    'ffi-c-probe',
    output: '_vfhe_cpu.c',
    command: [
        py,
        files('tools/meson/generate_ffi_c.py'),
        '_vfhe_cpu',
        '@OUTPUT@',
        probe_cdef,
        '--',
        probe_header,
    ],
    depend_files: probe_cdef + probe_header,
)

py_install.extension_module(
    '_vfhe_cpu',
    [probe_ffi_c, probe_sources],
    include_directories: probe_includes,
    install: true,
)
endif

# --- Unity (the C test framework) ---------------------------------------------

# Test-only, and deliberately absent from sdists: an sdist build configures
# without C tests rather than failing. A repo clone has it (submodules).
build_c_tests = fs.exists('external/unity/src/unity.c')
if build_c_tests
    unity_includes = include_directories('external/unity/src')
    unity_lib = static_library(
        'unity',
        files('external/unity/src/unity.c'),
        include_directories: unity_includes,
    )
else
    message('Unity absent (sdist or non-recursive clone): C unit tests not configured.')
endif

# --- The artifacts, once per engine -------------------------------------------

# Each engine gets an extension (LTO: kernels inline across module
# boundaries) and a static archive (plain objects: they link on user
# machines with whatever compiler is there, where LTO objects would demand
# a matching toolchain), plus the engine-<name>.json a matching user-module
# compile needs.
foreach engine : engines
    cflags = run_command(py, engine_info, engine, 'cflags', check: true).stdout().strip().split('\n')
    # Its own name, so vfhe_engine_active() answers per engine and
    # dynamic_extensions can find the matching archive.
    cflags += '-DVFHE_ENGINE_NAME="' + engine + '"'
    cflags += '-Wundef'
    engine_sources = []
    extra_out = run_command(py, engine_info, engine, 'extra-sources', check: true).stdout().strip()
    if extra_out != ''
        foreach source : extra_out.split('\n')
            engine_sources += files(source)
        endforeach
    endif

    # Named 'inline_*', not 'vfhe_*': this one is swallowed whole by the
    # extension and must not look like the installable archive beside it.
    lto_lib = static_library(
        'inline_' + engine,
        lib_sources + engine_sources,
        include_directories: lib_includes,
        c_args: cflags + lto_args,
        pic: true,
    )

    test_lib = static_library(
        'vfhe_' + engine,
        lib_sources + engine_sources,
        include_directories: lib_includes,
        c_args: cflags,
        pic: true,
        install: not fuzz,
        install_dir: source_dir / 'lib',
    )

    # A fuzzing container wants exactly that archive.
    if fuzz
        continue
    endif

    custom_target(
        'engine-json-' + engine,
        output: 'engine-' + engine + '.json',
        command: [py, engine_info, engine, 'json', '@OUTPUT@'],
        build_by_default: true,
        install: true,
        install_dir: source_dir / 'lib',
    )

    ffi_c = custom_target(
        'ffi-c-' + engine,
        output: '_vfhe_native_' + engine + '.c',
        command: [
            py,
            files('tools/meson/generate_ffi_c.py'),
            '_vfhe_native_' + engine,
            '@OUTPUT@',
            cdef_files,
            '--',
            header_files,
        ],
        depend_files: cdef_files + header_files,
    )

    py_install.extension_module(
        '_vfhe_native_' + engine,
        ffi_c,
        include_directories: lib_includes,
        c_args: cflags,  # the wrapper #includes engine-sensitive headers
        link_whole: lto_lib,
        link_args: lto_args,
        dependencies: m_dep,
        install: true,
    )

    # One test binary per C test file per engine. Every test joins three
    # suites — its engine, its depth, and the pair — because meson's
    # --suite flags union: only the pair name selects one exact cell.
    # Unity is linked into all of them (it defines no main, so plain
    # assert/main() tests are unaffected), which beats sniffing each file
    # for its include.
    foreach test_file : build_c_tests ? c_test_files : []
        name = fs.stem(test_file) + '-' + engine
        test_exe = executable(
            name,
            test_file,
            include_directories: lib_includes + [unity_includes],
            c_args: cflags,
            link_with: [test_lib, unity_lib],
            dependencies: m_dep,
            override_options: ['warning_level=2'],
        )
        test(name, test_exe, suite: [engine, 'c', engine + '-c'], timeout: 300)
    endforeach

    # The Python suites, same axes. The environment makes the build tree
    # self-sufficient: the extensions and bindings are here, the package and
    # the engine picker are in the modules' sources, and VFHE_ENGINE pins
    # which engine this leg loads.
    pytest_env = environment()
    pytest_env.append('PYTHONPATH', meson.current_build_dir(), separator: ':')
    foreach src : python_source_dirs
        pytest_env.append('PYTHONPATH', src, separator: ':')
    endforeach
    pytest_env.set('VFHE_ENGINE', engine)
    pytest_env.set('PYTHONDONTWRITEBYTECODE', '1')

    foreach depth : ['fast', 'complete']
        test(
            'pytest-' + depth + '-' + engine,
            py,
            args: ['-m', 'pytest', '-q'] + (depth == 'complete' ? ['--complete'] : []),
            env: pytest_env,
            suite: [engine, depth, engine + '-' + depth],
            workdir: meson.current_source_dir(),
            timeout: depth == 'complete' ? 3600 : 900,
        )
    endforeach
endforeach

# The picker's table of installed engines, generated from the same registry
# the archives came from, so the two can never disagree.
if not fuzz
    custom_target(
        'engine-table',
        output: '_vfhe_engines.py',
        command: [py, engine_info, 'table', '@OUTPUT@', engines],
        build_by_default: true,
        install: true,
        install_dir: py_install.get_install_dir(),
    )
endif

# Proto bindings: a build target (not an install script — meson-python wheels
# carry only what the install plan knows), so they regenerate exactly when a
# schema changes. One directory output, since protoc decides the file names
# from each schema's package path. The sdist ships schemas, never bindings.
if not fuzz
    custom_target(
        'proto-bindings',
        output: '_vfhe_proto',
        command: [py, files('tools/meson/generate_proto_bindings.py'), '@OUTPUT@'],
        depend_files: proto_files,
        build_by_default: true,
        install: true,
        install_dir: py_install.get_install_dir(),
    )
endif

meson.add_dist_script(py, files('tools/meson/prepare_sdist.py'), meson.project_version())
