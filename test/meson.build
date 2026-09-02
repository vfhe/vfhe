# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Which tests exist and how they run; the root builds the artefacts and hands
# them over in engine_builds. A test's NAME is the selection API (`meson test
# '<engine>-*'`), not --suite: a --suite matching nothing reports success.

# The picker's probe as a command: exit 0 supported, 1 absent, 2 unjudgeable.
cpu_probe = executable(
    'vfhe-cpu',
    probe['sources'],
    include_directories: probe['includes'],
    c_args: ['-DVFHE_CPU_MAIN'],
    install: false,
)

# Wraps every test: exit 77 (meson's SKIP) when this CPU lacks the capability.
guard = find_program('unit/require_cpu.sh')

# One setup per emulatable engine (setups do not compose). The flags go to the
# guard rather than an exe_wrapper, which would emulate the guard and leave the
# kernels it execs on the bare CPU. The launcher comes from the caller, so its
# absence fails a test rather than the build.
foreach fact : engine_facts
    if fact['sde'].length() > 0
        add_test_setup(
            fact['name'] + '_emulated',
            timeout_multiplier: 20,
            env: {'VFHE_SDE_FLAGS': ' '.join(fact['sde'])},
        )
    endif
endforeach

# ASan's leak checker counts CPython's own allocations, so it is off everywhere.
sanitized = 'address' in get_option('b_sanitize')
asan_env = sanitized ? {'ASAN_OPTIONS': 'detect_leaks=0'} : {}

# A non-instrumented interpreter dlopens the instrumented extension too late
# for ASan, so its runtime must load before the interpreter.
asan_runtime = ''
if sanitized and host_machine.system() == 'linux'
    # required: false, because run_command errors on a missing program even
    # with check: false.
    gcc = find_program('gcc', required: false)
    if gcc.found()
        found = run_command(gcc, '-print-file-name=libasan.so', check: false)
        # gcc echoes the bare name back when it has no such library.
        if found.returncode() == 0 and found.stdout().strip().startswith('/')
            asan_runtime = found.stdout().strip()
        endif
    endif
endif

# meson-logs already holds meson's own C coverage report, so both halves land
# in one directory for a single upload.
coverage = get_option('b_coverage')
reports = meson.project_build_root() + '/meson-logs'

pythonpath = ':'.join([meson.project_build_root()] + python_source_dirs)

foreach engine, built : engine_builds
    needs_cpu = [cpu_probe, built['capability'], '--']

    foreach test_file : build_c_tests ? collected['c_tests'] : []
        name = engine + '-c-' + fs.stem(test_file)
        test_exe = executable(
            name,
            test_file,
            include_directories: collected['includes'] + [unity_includes],
            c_args: built['cflags'],
            link_with: [built['lib'], unity_lib],
            dependencies: m_dep,
            override_options: ['warning_level=2'],
        )
        test(
            name,
            guard,
            args: needs_cpu + [test_exe],
            env: asan_env,
            depends: [cpu_probe, test_exe],
            timeout: 300,
        )
    endforeach

    # The fast depth filters; complete runs whatever is collected.
    foreach depth : ['fast', 'complete']
        name = engine + '-pytest-' + depth

        pytest_env = environment()
        pytest_env.set('PYTHONPATH', pythonpath)
        pytest_env.set('VFHE_ENGINE', engine)
        pytest_env.set('PYTHONDONTWRITEBYTECODE', '1')
        if sanitized
            pytest_env.set('ASAN_OPTIONS', 'detect_leaks=0')
            if asan_runtime != ''
                pytest_env.set('LD_PRELOAD', asan_runtime)
            endif
        endif

        # Named per test, so the two depths never overwrite each other's data.
        measure = []
        if coverage
            pytest_env.set('COVERAGE_FILE', reports + '/' + name + '.data')
            measure = [
                '--cov=vfhe',
                '--cov-report=xml:' + reports + '/' + name + '.xml',
                '--cov-report=json:' + reports + '/' + name + '.json',
            ]
        endif

        test(
            name,
            guard,
            args: needs_cpu + [py.full_path(), '-m', 'pytest', '-q']
            + (depth == 'fast' ? ['-m', 'not complete'] : []) + measure,
            env: pytest_env,
            depends: [cpu_probe],
            workdir: meson.project_source_root(),
            timeout: depth == 'complete' ? 3600 : 900,
        )
    endforeach
endforeach
