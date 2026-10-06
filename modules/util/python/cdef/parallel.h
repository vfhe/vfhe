// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
// What Python may call of parallel.h, in cffi's cdef dialect.

uint64_t vfhe_num_threads(void);
void vfhe_set_num_threads(uint64_t n);
