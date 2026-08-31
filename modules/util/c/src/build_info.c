// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
// Which engine THIS binary is. The build names it (meson.build passes the
// registry's name); CPU capability is cpu_probe.c's business.

#include "util.h"

#ifndef VFHE_ENGINE_NAME
#error "the build must define VFHE_ENGINE_NAME (see meson.build)"
#endif

const char *vfhe_engine_active(void) { return VFHE_ENGINE_NAME; }
