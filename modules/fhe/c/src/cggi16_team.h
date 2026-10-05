// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
// The data-parallel driver of cggi16_blind_rotate. Not part of the library's API.
#pragma once
#include "zyl17.h"

// The rotation of `acc` (canonical, in place) on `threads` threads, every one of them working on
// every group; see cggi16_team.c. Same result as the sequential rotation, bit for bit.
void cggi16_blind_rotate_team(const UnfoldedRotation *R, RNSc_MLWE acc, uint64_t threads);
