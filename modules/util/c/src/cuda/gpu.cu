// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

#include "gpu.h"

uint64_t vfhe_gpu_device_count(void)
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess)
        return 0;
    return (uint64_t)count;
}

const char *vfhe_gpu_error_string(int error) { return cudaGetErrorString((cudaError_t)error); }
