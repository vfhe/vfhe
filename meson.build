# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0

project(
    'vfhe',
    'c',
    # files() resolves against this directory, and meson forbids module
    # methods (fs) during project().
    version: run_command(files('tools/release/version.sh'), check: true).stdout().strip(),
    license: 'Apache-2.0',
    meson_version: '>=1.4.0',
    default_options: ['c_std=gnu11', 'buildtype=release'],
)

fs = import('fs')
m_dep = meson.get_compiler('c').find_library('m', required: false)
py = find_program('python3')

# -Dfuzz=true configures without Python headers or protobuf. Its harnesses link
# as C++ because $LIB_FUZZING_ENGINE supplies main() and is C++.
fuzz = get_option('fuzz')
if fuzz
    add_languages('cpp', native: false)
else
    py_install = import('python').find_installation(pure: false)
endif

# Coverage swaps LTO for gcov instrumentation.
#
# -fno-semantic-interposition is what lets LTO actually inline the reduction
# primitives. Without it GCC has to assume any default-visibility global in a
# shared object may be interposed at load time, so a call to one becomes a PLT
# call it cannot inline through -- which put a dynamic call per element inside
# every scalar kernel.
lto_args = get_option('b_coverage') ? [] : ['-flto', '-fno-semantic-interposition']

# A fuzzing build installs nothing; the placeholder keeps install_dir defined.
python_dir = fuzz ? 'nothing-is-installed' : py_install.get_install_dir()
source_dir = python_dir / 'vfhe' / '_source'

# --- The parts ------------------------------------------------------------

# A vendored dependency is just a part that contributes sources: adding either
# is a directory, its meson.build declaring one `module` dict, and a path here.
parts = [
    'external/blake3',
    'modules/arith',
    'modules/circuit',
    'modules/crypto',
    'modules/fhe',
    'modules/mlwe',
    'modules/piop',
    'modules/polycom',
    'modules/util',
]

# subdir() shares this scope, so a part must never assign these names.
collected = {
    'sources': [],
    'includes': [],
    'simd_sources': [],
    'simd_flags_on': [],
    'simd_flags_off': [],
    'headers': [],
    'cdefs': [],
    'c_tests': [],
    'fuzz': [],
    'protos': [],
}
probe = {}
proto_roots = []
python_source_dirs = []

foreach here : parts
    module = {}
    subdir(here)

    foreach key : collected.keys()
        collected += {key: collected[key] + module.get(key, [])}
    endforeach
    probe += module.get('probe', {})
    has_python = fs.exists(here / 'python/src')
    if module.has_key('protos')
        proto_roots += meson.current_source_dir() / here / 'proto'
    endif
    if has_python
        python_source_dirs += meson.current_source_dir() / here / 'python/src'
    endif

    if not fuzz
        if module.has_key('headers')
            install_subdir(here / 'c/include', install_dir: source_dir / here / 'c')
        endif
        if module.has_key('cdefs')
            install_subdir(
                here / 'python/cdef',
                install_dir: source_dir / here / 'python',
                exclude_directories: ['__pycache__'],
            )
        endif
        if has_python
            # exclude_directories names paths, not globs: every package dir is
            # listed, so a developer's stale bytecode cannot reach the wheel.
            stale = ['__pycache__']
            foreach package : module.get('packages', [fs.name(here)])
                stale += package + '/__pycache__'
            endforeach
            install_subdir(
                here / 'python/src/vfhe',
                install_dir: python_dir,
                exclude_directories: stale,
            )
        endif
    endif
endforeach

# --- Engine facts -----------------------------------------------------------

# One entry per ISA level, best first. An engine is named after the capability
# cpu_probe.c must confirm; the fallback needs none.
engines = [
    {
        'name': 'avx512ifma',
        'arch': 'x86_64',
        'cargs': [
            '-mavx512f',
            '-mavx512ifma',
            '-mavx512dq',
            '-mavx512vl',
            '-mavx2',
            '-maes',
            '-mvaes',
            '-mrdrnd',
            # BMI2/ADX predate AVX512IFMA, so the engine's runtime gate already
            # implies them; they let GCC use MULX in the scalar reduction path.
            '-mbmi2',
            '-madx',
            '-funroll-all-loops',
        ],
        'sde': ['-icl'],
    },
    {'name': 'neon', 'arch': 'aarch64', 'cargs': []},
    {'name': 'portable', 'arch': 'any', 'cargs': ['-DPORTABLE_BUILD'], 'fallback': true},
]

engine_facts = []
foreach engine : engines
    if engine['arch'] == 'any' or engine['arch'] == host_machine.cpu_family()
        fallback = engine.get('fallback', false)
        engine_facts += [
            {
                'name': engine['name'],
                'capability': fallback ? '' : engine['name'],
                'cargs': engine['cargs'] + collected[fallback ? 'simd_flags_off' : 'simd_flags_on'],
                'extra_sources': fallback ? [] : collected['simd_sources'],
                'sde': engine.get('sde', []),
            },
        ]
    endif
endforeach

# A session fuzzes exactly one engine. Refuse kernels this machine cannot
# execute, or every input looks like a crash (Linux only, like the container).
if fuzz
    picked = []
    foreach fact : engine_facts
        if fact['name'] == get_option('fuzz_engine')
            picked = [fact]
        endif
    endforeach
    if picked.length() == 0
        error('-Dfuzz_engine must name one of the engines above, not "' + get_option('fuzz_engine') + '"')
    endif
    engine_facts = picked

    # /proc/cpuinfo spells x86 capabilities only; the probe treats Advanced
    # SIMD as arm64 baseline.
    capability = picked[0]['capability']
    if capability != '' and host_machine.cpu_family() == 'x86_64' and run_command('grep', '-qw', capability, '/proc/cpuinfo', check: false).returncode() != 0
        error('this machine has no ' + capability + ', which "' + get_option('fuzz_engine') + '" needs')
    endif
endif

# --- The CPU probe ----------------------------------------------------------

# No engine flags, no library: the picker asks it what the CPU can run, then
# imports exactly one engine.
if not fuzz
    probe_ffi_c = custom_target(
        'ffi-c-probe',
        output: '_vfhe_cpu.c',
        command: [
            py, files('tools/build/ffi_c.py'),
            '_vfhe_cpu', '@OUTPUT@',
            probe['cdef'], '--', probe['header'],
        ],
        depend_files: probe['cdef'] + probe['header'],
    )

    py_install.extension_module(
        '_vfhe_cpu',
        [probe_ffi_c, probe['sources']],
        include_directories: probe['includes'],
        install: true,
    )
endif

# --- The artifacts, once per engine -------------------------------------------

# Per engine: an extension whose LTO inlines kernels across modules, an archive
# of plain objects any compiler links, and the json describing both.
engine_builds = {}
foreach fact : engine_facts
    engine = fact['name']
    # Its own name, so vfhe_engine_active() answers per engine and
    # dynamic_extensions finds the matching archive.
    # -Wundef: the VFHE_HAVE_* configuration macros are tested with #if, so a
    # misspelt or missing one must be a warning rather than a silent 0.
    cflags = fact['cargs'] + ['-DVFHE_ENGINE_NAME="' + engine + '"', '-Wundef']
    engine_sources = collected['sources'] + fact['extra_sources']

    installed_lib = static_library(
        'vfhe_' + engine,
        engine_sources,
        include_directories: collected['includes'],
        c_args: cflags,
        pic: true,
        install: not fuzz,
        install_dir: source_dir / 'lib',
    )

    # Linked in this loop, so a harness cannot drift from the kernels it fuzzes.
    if fuzz
        foreach harness : collected['fuzz']
            executable(
                fs.stem(harness),
                harness,
                include_directories: collected['includes'],
                c_args: cflags,
                link_with: installed_lib,
                link_args: get_option('fuzz_link_args').split(),
                link_language: 'cpp',
                dependencies: m_dep,
            )
        endforeach
        continue
    endif

    # 'inline_', not 'vfhe_': the extension links this one whole, so it must
    # not look like the installable archive beside it.
    lto_lib = static_library(
        'inline_' + engine,
        engine_sources,
        include_directories: collected['includes'],
        c_args: cflags + lto_args,
        pic: true,
    )

    # A matching user-module compile needs these flags: the headers change
    # types under them.
    cargs_json = fact['cargs'].length() == 0 ? '[]' : '["' + '", "'.join(fact['cargs']) + '"]'
    sde_json = fact['sde'].length() == 0 ? '[]' : '["' + '", "'.join(fact['sde']) + '"]'
    # A coverage-instrumented archive needs its compiler's gcov runtime, which
    # only a link naming --coverage pulls in.
    link_json = get_option('b_coverage') ? '["--coverage"]' : '[]'
    configure_file(
        output: 'engine-' + engine + '.json',
        command: [
            'printf', '%s\n',
            '{',
            '  "engine": "' + engine + '",',
            '  "capability": ' + (fact['capability'] == '' ? 'null' : '"' + fact['capability'] + '"') + ',',
            '  "sde": ' + sde_json + ',',
            '  "cargs": ' + cargs_json + ',',
            '  "link_args": ' + link_json,
            '}',
        ],
        capture: true,
        install: true,
        install_dir: source_dir / 'lib',
    )

    ffi_c = custom_target(
        'ffi-c-' + engine,
        output: '_vfhe_native_' + engine + '.c',
        command: [
            py, files('tools/build/ffi_c.py'),
            '_vfhe_native_' + engine, '@OUTPUT@',
            collected['cdefs'], '--', collected['headers'],
        ],
        depend_files: collected['cdefs'] + collected['headers'],
    )

    py_install.extension_module(
        '_vfhe_native_' + engine,
        ffi_c,
        include_directories: collected['includes'],
        c_args: cflags,  # the wrapper #includes engine-sensitive headers
        link_whole: lto_lib,
        link_args: lto_args,
        dependencies: m_dep,
        install: true,
    )

    engine_builds += {
        engine: {'lib': installed_lib, 'cflags': cflags, 'capability': fact['capability']},
    }
endforeach

# --- What the install reads back ----------------------------------------------

if not fuzz
    # Generated from the same facts as the archives, so the two cannot disagree.
    table_rows = []
    foreach fact : engine_facts
        requires = fact['capability'] == '' ? 'None' : '\'' + fact['capability'] + '\''
        table_rows += '    (\'' + fact['name'] + '\', ' + requires + '),'
    endforeach
    configure_file(
        output: '_vfhe_engines.py',
        command: [
            'printf', '%s\n',
            '# Generated by meson from its engine facts - do not edit.',
            'ENGINES = [', table_rows, ']',
        ],
        capture: true,
        install: true,
        install_dir: python_dir,
    )

    # A build target, not an install script: wheels carry only what the install
    # plan knows, and bindings must regenerate when a schema changes. Every
    # part's proto/ is an import root (buf convention); protoc wants each -I as
    # an exact absolute prefix of each schema path and will not create its
    # output directory, hence the sh -c.
    proto_args = []
    foreach root : proto_roots
        proto_args += '-I' + root
    endforeach
    foreach schema : collected['protos']
        proto_args += schema.full_path()
    endforeach
    custom_target(
        'proto-bindings',
        output: '_vfhe_proto',
        command: [
            'sh', '-c', 'out=$1 && mkdir -p "$out" && shift && "$@" && find "$out" -type d -exec sh -c \'touch "$0/__init__.py"\' {} ";"', 'sh', '@OUTPUT@',
            py.full_path(), '-m', 'grpc_tools.protoc',
            proto_args, '--python_out=@OUTPUT@', '--pyi_out=@OUTPUT@',
        ],
        depend_files: collected['protos'],
        build_by_default: true,
        install: true,
        install_dir: python_dir,
    )
endif

# --- What the sdist carries ---------------------------------------------------

# `meson dist` archives every submodule and has no flag to skip one; NOTICE
# promises the test-only Unity stays out. `:?` refuses an unset variable.
meson.add_dist_script('sh', '-c', 'rm -rf "${MESON_DIST_ROOT:?}/external/unity"')

# An sdist has no git history for setuptools-scm; this frozen version answers
# for it and for every wheel built from it. project() prefers it.
meson.add_dist_script(
    'sh', '-c',
    'echo "$1" > "$MESON_DIST_ROOT/.version"',
    'sh', meson.project_version(),
)

# --- The tests ------------------------------------------------------------

# A fuzzing build has none: no probe to guard them, no interpreter to run them.
if not fuzz
    # files() refuses '..' and the Unity submodule has no meson.build of its
    # own, so the root builds it. A non-recursive clone (or the sdist, which
    # drops Unity) configures without C tests instead of failing.
    build_c_tests = fs.exists('external/unity/src/unity.c')
    if build_c_tests
        unity_includes = include_directories('external/unity/src')
        unity_lib = static_library(
            'unity',
            files('external/unity/src/unity.c'),
            include_directories: unity_includes,
        )
    else
        message('Unity absent: C unit tests not configured.')
    endif

    subdir('test')
endif
