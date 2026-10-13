# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0

project(
    'vfhe',
    'c',
    license: 'Apache-2.0',
    meson_version: '>=1.4.0',
    default_options: ['c_std=gnu11', 'buildtype=release'],
)

fs = import('fs')
py = import('python').find_installation(pure: false)

# --- Proto --------------------------------------------------------------------

subdir('proto')

# --- Module declarations ------------------------------------------------------

vfhe_c_sources = []            # compiled into every engine
vfhe_c_include_dirs = []       # -I paths, private dirs too
vfhe_c_public_headers = []     # the wrapper's #include lines
vfhe_c_unit_tests = []         # c/test/unit suites
vfhe_c_fuzz_tests = []         # c/test/fuzz fuzzers
vfhe_python_cdefs = []         # what Python may call of the headers

# --- Modules ------------------------------------------------------------------

subdir('modules/arith')
subdir('modules/circuit')
subdir('modules/crypto')
subdir('modules/fhe')
subdir('modules/mlwe')
subdir('modules/piop')
subdir('modules/polycom')
subdir('modules/util')

# --- External -----------------------------------------------------------------

subdir('external/blake3')
subdir('external/unity')

# --- Kernels ------------------------------------------------------------------

kernel_sources = []
foreach src : vfhe_c_sources
    kernel_sources += fs.relative_to(src, meson.project_source_root())
endforeach

custom_target(
    'kernel-sources-snapshot',
    output: 'kernels-src.tar.gz',
    command: [
        files('tools/build/kernels/sources.sh'),
        meson.project_source_root(), '@OUTPUT@', kernel_sources,
    ],
    build_always_stale: true,
    install: true,
    install_dir: py.get_install_dir() / 'vfhe' / '_source',
)

# --- Engines ------------------------------------------------------------------

# The whole library, once per instruction set.
#
# Every engine's flags beyond its instruction set: -DVFHE_BUILDING_LIBRARY, so
# the library's own sources read the concrete mp_vector_t that a consumer sees
# opaque; -Wundef, so a misspelt VFHE_HAVE_* in an #if warns; on ELF, direct
# calls within the module.
common_c_args = ['-DVFHE_BUILDING_LIBRARY', '-Wundef']
if host_machine.system() == 'linux'
    common_c_args += ['-fno-semantic-interposition']
endif

engines = {}

# BLAKE3 enables its SIMD kernels from the architecture; these turn them off.
blake3_no_simd_flags = []

if host_machine.cpu_family() == 'x86_64'
    blake3_no_simd_flags = [
        '-DBLAKE3_NO_SSE2',
        '-DBLAKE3_NO_SSE41',
        '-DBLAKE3_NO_AVX2',
        '-DBLAKE3_NO_AVX512',
    ]
    engines += {
        'avx512ifma': [
            # x86_64.h reads __AVX512IFMA__ itself, so the -m flags say it all
            '-mavx512f',
            '-mavx512ifma',
            '-mavx512dq',
            '-mavx512vl',
            '-mavx2',
            '-maes',
            '-mvaes',
            '-mrdrnd',
            '-mbmi2',
            '-madx',
            '-funroll-all-loops',
        ] + common_c_args,
    }
endif

if host_machine.cpu_family() == 'aarch64'
    blake3_no_simd_flags = ['-DBLAKE3_USE_NEON=0']
    # Advanced SIMD is arm64 baseline, so this engine needs no ISA flag.
    engines += {'neon': common_c_args}
endif

engines += {
    'portable': [
        # x86_64.h reads VFHE_HAVE_X86_64 as 0 under it, on x86 too
        '-DPORTABLE_BUILD',
    ] + common_c_args + blake3_no_simd_flags,
}

# --- Engine flags -------------------------------------------------------------

conf = configuration_data({'AVX512IFMA': '', 'NEON': '', 'PORTABLE': ''})
foreach engine_name, c_args : engines
    conf.set(engine_name.to_upper(), ';'.join(c_args))
endforeach

configure_file(
    input: 'tools/build/engines/flags.py.in',
    output: '_vfhe_flags.py',
    configuration: conf,
    install: true,
    install_dir: py.get_install_dir(),
)

# --- Extensions ---------------------------------------------------------------

m_dep = meson.get_compiler('c').find_library('m', required: false)
thread_dep = dependency('threads')

foreach engine_name, c_args : engines
    ffi_c = custom_target(
        'ffi-c-' + engine_name,
        output: '_vfhe_native_' + engine_name + '.c',
        command: [
            py, files('modules/util/python/src/vfhe/util/kernels/_ffi.py'),
            '_vfhe_native_' + engine_name, '@OUTPUT@',
            vfhe_python_cdefs, '--', vfhe_c_public_headers,
        ],
        depend_files: vfhe_python_cdefs + vfhe_c_public_headers,
    )

    shared_library(
        '_vfhe_native_' + engine_name,
        vfhe_c_sources + [ffi_c],
        name_prefix: '',
        name_suffix: 'so',
        include_directories: vfhe_c_include_dirs,
        c_args: c_args,
        override_options: ['b_lundef=false', 'b_lto=true'],
        dependencies: [m_dep, thread_dep, py.dependency(embed: false)],
        install: true,
        install_dir: py.get_install_dir(),
    )
endforeach

# --- Tests --------------------------------------------------------------------

subdir('test')
