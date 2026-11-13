// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
// The seeded sampler: values below a bound that are a pure function of a
// context and a seed, so any value is reachable by its index.
//
// Rejection sampling, laid out so that a value's index alone locates its
// bytes: index i reads word i of attempt stream 0, then word i of streams 1,
// 2, ... until the masked word lands below `bound`. Letting one stream run
// on, the obvious loop, would make i's offset depend on how many rejections
// preceded it, and reaching index i would mean drawing every value before.
//
// Fixed offsets also make a run of values one seek of the XOF, which runs
// lane-parallel over whole blocks, where a word at a time costs a scalar
// compression each: the same bytes at about a thirtieth of the time. So each
// attempt draws the whole outstanding run at once and carries the rejected
// positions to the next; when few are left, they go back to a seek apiece.
#include "seeded.h"

#include <alloc.h>
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include <blake3.h>

#include "below.h"

// Attempt streams kept initialized across a call. A draw reaches attempt k
// only when k masked words before it landed above `bound`, under a half each
// time, so past a handful a stream is derived on the spot: one init on a
// path a sixteenth of the draws take at worst. Each hasher is ~2 KB of stack.
#define KEPT_STREAMS 4

// Values one bulk seek resolves: 32 KiB of stream per attempt, which stays
// in cache and bounds the scratch however long `count` is.
#define BULK_VALUES 4096

// Below this share of a run still unresolved, a seek per remaining position
// beats another pass over the whole run. A bulk seek costs a few ns a value,
// a single seek over a hundred, so the crossover sits near a thirtieth; a
// sixteenth stays on the safe side of it.
#define SPARSE_SHARE 16

// Runs up to this length work on the stack, which keeps drawing one element
// by its index free of an allocation.
#define STACK_VALUES 64

// Attempt `attempt`'s stream: the key-derivation hash of `context` over the
// seed and the attempt number, so the streams are independent of each other.
static void attempt_stream(blake3_hasher *hasher, const char *context, const uint8_t *seed,
                           uint64_t seed_len, uint64_t attempt)
{
    uint8_t tag[sizeof(uint64_t)];
    for (unsigned i = 0; i < sizeof(tag); i++)
        tag[i] = (uint8_t)(attempt >> (8 * i));
    blake3_hasher_init_derive_key(hasher, context);
    blake3_hasher_update(hasher, seed, seed_len);
    blake3_hasher_update(hasher, tag, sizeof(tag));
}

void seeded_below_from(uint64_t *out, uint64_t count, uint64_t start, uint64_t bound,
                       const char *context, const uint8_t *seed, uint64_t seed_len)
{
    assert(bound > 0); // otherwise no attempt ever succeeds
    if (count == 0)
        return;
    const uint64_t mask = below_mask(bound);

    blake3_hasher kept[KEPT_STREAMS];
    bool ready[KEPT_STREAMS] = {false};

    const uint64_t chunk = count < BULK_VALUES ? count : BULK_VALUES;
    uint8_t stack_stream[STACK_VALUES * sizeof(uint64_t)];
    uint64_t stack_pending[STACK_VALUES];
    uint8_t *stream = stack_stream;
    uint64_t *pending = stack_pending;
    if (chunk > STACK_VALUES)
    {
        stream = (uint8_t *)safe_malloc(chunk * sizeof(uint64_t));
        pending = (uint64_t *)safe_malloc(chunk * sizeof(uint64_t));
    }

    for (uint64_t base = 0; base < count; base += chunk)
    {
        const uint64_t run = (count - base) < chunk ? (count - base) : chunk;
        // The offsets within this run no attempt has settled yet.
        uint64_t left = run;
        for (uint64_t k = 0; k < run; k++)
            pending[k] = k;

        for (uint64_t attempt = 0; left > 0; attempt++)
        {
            blake3_hasher derived;
            const blake3_hasher *hasher = &derived;
            if (attempt < KEPT_STREAMS)
            {
                if (!ready[attempt])
                {
                    attempt_stream(&kept[attempt], context, seed, seed_len, attempt);
                    ready[attempt] = true;
                }
                hasher = &kept[attempt];
            }
            else
                attempt_stream(&derived, context, seed, seed_len, attempt);

            const bool bulk = left * SPARSE_SHARE >= run;
            if (bulk)
                blake3_hasher_finalize_seek(hasher, (start + base) * sizeof(uint64_t), stream,
                                            run * sizeof(uint64_t));

            uint64_t still = 0;
            for (uint64_t k = 0; k < left; k++)
            {
                const uint64_t at = pending[k];
                uint8_t word[sizeof(uint64_t)];
                if (bulk)
                    memcpy(word, stream + at * sizeof(uint64_t), sizeof(word));
                else
                    blake3_hasher_finalize_seek(hasher, (start + base + at) * sizeof(uint64_t),
                                                word, sizeof(word));
                uint64_t v;
                memcpy(&v, word, sizeof(v));
                v &= mask;
                if (v < bound)
                    out[base + at] = v;
                else
                    pending[still++] = at;
            }
            left = still;
        }
    }

    if (chunk > STACK_VALUES)
    {
        free(stream);
        free(pending);
    }
}

void seeded_below(uint64_t *out, uint64_t count, uint64_t bound, const char *context,
                  const uint8_t *seed, uint64_t seed_len)
{
    seeded_below_from(out, count, 0, bound, context, seed, seed_len);
}
