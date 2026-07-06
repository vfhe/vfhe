// SPDX-License-Identifier: Apache-2.0
/**
 * @file arith/error.h
 * @brief Status codes returned by fallible engine entry points.
 *
 * Every user-reachable operation that can fail returns an int status code
 * instead of asserting: a shared library must never abort the host process
 * (in particular the Python interpreter). `0` (::VFHE_OK) means success;
 * negative values are errors. Functions that cannot fail return `void`.
 */
#ifndef VFHE_ARITH_ERROR_H
#define VFHE_ARITH_ERROR_H

#ifdef __cplusplus
extern "C"
{
#endif

    /** Status codes for fallible engine operations. */
    typedef enum vfhe_status
    {
        VFHE_OK = 0,                  /**< Success. */
        VFHE_ERR_DOMAIN = -1,         /**< Operand is in the wrong domain (see ::vfhe_domain). */
        VFHE_ERR_NOT_INVERTIBLE = -2, /**< An element has no inverse (zero evaluation slot). */
        VFHE_ERR_UNSUPPORTED = -3,    /**< Operation not supported for this ring shape. */
        VFHE_ERR_ARG = -4,            /**< Invalid argument (bad mask, size, or alias). */
    } vfhe_status;

#ifdef __cplusplus
}
#endif

#endif // VFHE_ARITH_ERROR_H
