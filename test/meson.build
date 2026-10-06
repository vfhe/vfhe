# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0

# --- Static libraries ---------------------------------------------------------

static_libs = {}
foreach engine_name, c_args : engines
    static_libs += {
        engine_name: static_library(
            'vfhe_' + engine_name,
            vfhe_c_sources,
            include_directories: vfhe_c_include_dirs,
            c_args: c_args,
            dependencies: [m_dep, thread_dep],
            build_by_default: false,
        ),
    }
endforeach

# --- Emulation ----------------------------------------------------------------

# The setup launches every test under Intel SDE, which tools/test/sde/fetch.sh fetches
# into .cache/sde before run.sh reconfigures; configure itself downloads nothing.
sde = find_program(
    'sde64',
    dirs: [meson.project_source_root() / '.cache/sde'],
    required: false,
)

if host_machine.cpu_family() == 'x86_64' and sde.found()
    add_test_setup(
        'avx512ifma_emulated',
        exe_wrapper: [sde, '-icl', '-no-follow-child', '--'],
        timeout_multiplier: 20,
        env: {'VFHE_SDE_FLAGS': '-icl'},
    )
endif

# --- Sanitizers ---------------------------------------------------------------

asan = 'address' in get_option('b_sanitize')

asan_options = 'detect_stack_use_after_return=0'
leak_check = ''
asan_runtime = ''

# LeakSanitizer and LD_PRELOAD are both Linux-only.
if asan and host_machine.system() == 'linux'
    leak_check = ':detect_leaks=1'
    probe = run_command(
        meson.get_compiler('c').cmd_array(),
        '-print-file-name=libasan.so',
        check: false,
    )

    if probe.stdout().strip().startswith('/')
        asan_runtime = probe.stdout().strip()
    endif
endif

# The C suites are pure C, so a leak there is ours. CPython allocates at import
# and keeps it, so leak detection is off under pytest.
asan_env = asan ? {'ASAN_OPTIONS': asan_options + leak_check} : {}
pytest_asan_env = asan ? {'ASAN_OPTIONS': asan_options + ':detect_leaks=0'} : {}
if asan_runtime != ''
    pytest_asan_env += {'LD_PRELOAD': asan_runtime}
endif

# --- Import path --------------------------------------------------------------

pythonpath = ':'.join(
    [
        meson.project_build_root(),
        fs.parent(proto_bindings.full_path()),
    ]
    + [
        meson.project_source_root() / 'modules/arith/python/src',
        meson.project_source_root() / 'modules/circuit/python/src',
        meson.project_source_root() / 'modules/crypto/python/src',
        meson.project_source_root() / 'modules/fhe/python/src',
        meson.project_source_root() / 'modules/mlwe/python/src',
        meson.project_source_root() / 'modules/piop/python/src',
        meson.project_source_root() / 'modules/polycom/python/src',
        meson.project_source_root() / 'modules/util/python/src',
    ],
)

# --- C suites -----------------------------------------------------------------

foreach engine_name, c_args : engines
    foreach test_file : vfhe_c_unit_tests
        test_name = engine_name + '-c-' + fs.stem(test_file)
        test_exe = executable(
            test_name,
            test_file,
            include_directories: vfhe_c_include_dirs,
            c_args: c_args,
            link_with: static_libs[engine_name],
            dependencies: unity_dep,
            build_by_default: false,
        )
        test(test_name, test_exe, env: asan_env)
    endforeach
endforeach

# --- Python suites ------------------------------------------------------------

coverage_dir = meson.project_build_root() / 'meson-logs'

# fast leaves out the heavy @pytest.mark.complete computations.
depths = {
    'fast': {'args': ['-m', 'not complete'], 'timeout': 900},
    'complete': {'args': [], 'timeout': 3600},
}

foreach engine_name : engines.keys()
    foreach depth, spec : depths
        test_name = engine_name + '-py-' + depth

        pytest_env = pytest_asan_env + {
            'PYTHONPATH': pythonpath,
            'VFHE_ENGINE': engine_name,
            'PYTHONDONTWRITEBYTECODE': '1',
        }

        cov_args = []
        if get_option('b_coverage')
            pytest_env += {'COVERAGE_FILE': coverage_dir / test_name + '.data'}
            cov_args = [
                '--cov=vfhe',
                '--cov-report=xml:' + coverage_dir / test_name + '.xml',
                '--cov-report=json:' + coverage_dir / test_name + '.json',
            ]
        endif

        test(
            test_name,
            py,
            args: ['-m', 'pytest', '-q'] + spec['args'] + cov_args,
            env: pytest_env,
            workdir: meson.project_source_root(),
            timeout: spec['timeout'],
        )
    endforeach
endforeach

# --- Fuzzing ------------------------------------------------------------------

if get_option('fuzz') == ''
    subdir_done()
endif

# $LIB_FUZZING_ENGINE is C++ and supplies main().
add_languages('cpp', native: false)

wanted = get_option('fuzz')
if not engines.has_key(wanted)
    error('this build has no engine "' + wanted + '"')
endif

fuzzers = []
foreach fuzzer : vfhe_c_fuzz_tests
    fuzzers += executable(
        fs.stem(fuzzer),
        fuzzer,
        include_directories: vfhe_c_include_dirs,
        c_args: engines[wanted],
        link_with: static_libs[wanted],
        link_args: get_option('fuzz_link_args').split(),
        link_language: 'cpp',
    )
endforeach

# build.sh asks for this name alone, so only the fuzzers build.
alias_target('fuzzers', fuzzers)
