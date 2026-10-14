// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
// What Python may call of memory_pool.h, in cffi's cdef dialect: the pool's
// capacity, wiping and accounting, and its allocator for the buffers Python
// owns.

#define MEMPOOL_CAPACITY_AUTO ...
#define MEMPOOL_CAPACITY_DEFAULT ...
typedef struct
{
    uint64_t retained_bytes;
    uint64_t outstanding_bytes;
    uint64_t peak_outstanding_bytes;
    uint64_t retention_limit_bytes;
    uint64_t hits;
    uint64_t misses;
} MempoolStatistics;
void *mempool_aligned_malloc_with_size_header(size_t bytes);
void mempool_free_with_size_header(void *ptr);
void mempool_set_capacity(uint64_t bytes);
uint64_t mempool_capacity(void);
void mempool_release_all(void);
void mempool_set_wipe_on_release(int enabled);
int mempool_wipe_on_release(void);
void mempool_statistics(MempoolStatistics *out);
