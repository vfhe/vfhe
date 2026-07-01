/* SPDX-License-Identifier: Apache-2.0 */
#include "crypto.h"

uint64_t crypto_sample(uint64_t seed) { return seed * CRYPTO_LCG_MULT + CRYPTO_LCG_INC; }
