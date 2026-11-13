// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
// Python-facing ABI of the CUDA extension's runtime (util/c/src/cuda/gpu.cu,
// prototyped in util/c/include/gpu.h).

uint64_t vfhe_gpu_device_count(void);
const char *vfhe_gpu_error_string(int error);
