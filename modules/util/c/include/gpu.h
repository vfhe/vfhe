// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file gpu.h
 * @brief The CUDA extension's runtime: whether a device is usable, and what an error means.
 *
 * nvcc compiles every module's `.cu` sources into `_vfhe_cuda`, an extension
 * beside the engines, when the build finds a CUDA toolkit; the kernels are
 * declared in their own modules' headers. The CUDA runtime is linked
 * statically, so the extension loads on a machine with no driver;
 * vfhe_gpu_device_count() then returns 0 and every kernel returns an error.
 *
 * @note Thread-safe. The first call initialises the CUDA runtime, which takes
 *       a moment.
 */
#ifndef VFHE_GPU_H
#define VFHE_GPU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief Returns how many CUDA devices this machine can run kernels on.
     * @return The device count; 0 when the runtime finds no device or no
     *         driver.
     */
    uint64_t vfhe_gpu_device_count(void);

    /**
     * @brief Returns the CUDA runtime's description of an error code.
     * @param[in] error A code a kernel returned.
     * @return A string owned by the runtime, valid for the life of the
     *         process.
     */
    const char *vfhe_gpu_error_string(int error);

#ifdef __cplusplus
}
#endif

#endif // VFHE_GPU_H
