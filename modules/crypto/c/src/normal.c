// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "crypto.h"

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double int2double(uint64_t x) { return ((double)x) / 18446744073709551616.0; }

double generate_normal_random(double sigma)
{
    uint64_t rnd[2];
    generate_random_bytes(16, (uint8_t *)rnd);
    return cos(2. * M_PI * int2double(rnd[0])) * sqrt(-2. * log(int2double(rnd[1]))) * sigma;
}
