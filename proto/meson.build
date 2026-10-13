# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0

schemas = files('vfhe/circuit/gkr/v1/gkr.proto')

protoc_args = ['-I' + meson.current_source_dir()]
foreach schema : schemas
    protoc_args += schema.full_path()
endforeach

proto_bindings = custom_target(
    'proto-bindings',
    output: '_vfhe_proto',
    command: [files('bindings.sh'), '@OUTPUT@', py.full_path(), protoc_args],
    depend_files: schemas,
    install: true,
    install_dir: py.get_install_dir(),
)
