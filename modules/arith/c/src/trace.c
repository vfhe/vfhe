// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
#include <arith_trace.h>

#include "arith_internal.h"

#include <alloc.h>
#include <arith.h>
#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// The tape is two things under one lock:
//
// - the events, a growable array in sequence order, each owning its copies
//   (the value, the scalar, the label, the operand lists);
// - the version table, a hash map from an element's storage (the
//   RNS_Polynomial address) to the sequence number of the event that last
//   wrote it. Open addressing with linear probing, grown at half full,
//   entries removed by shifting the probe chain back so no tombstones are
//   needed.
//
// The lock is held for the whole of an open and of a close, the copy
// included, so events are numbered in the order operations finish,
// whichever threads run them. Freeing a value runs the forget hook, which
// takes the lock, so nothing is freed while it is held.

typedef struct
{
    const void *handle; // NULL marks an empty slot
    uint64_t version;
    uint64_t digest; // of the value as written, kept while verifying
    int digested;    // whether `digest` was taken (verify may start late)
} VersionEntry;

typedef struct
{
    ArithTraceEvent *events;
    uint8_t *open; // per event: appended by a begin whose end has not run
    uint64_t length, capacity;
    uint64_t published; // the events a reader may see: every one before the first open
    uint64_t waiters;   // producers waiting for room
    uint64_t next_sequence;
    VersionEntry *table;
    uint64_t slots, used; // slots is a power of two; used counts the entries
    pthread_mutex_t lock;
    uint64_t limit;         // the bound on the length, 0 for none
    pthread_cond_t changed; // signalled when the length or the bound changes
    uint64_t untraced;      // the writes verification found
} Tape;

void *arith_trace_active = NULL;
void *arith_trace_tape = NULL;
static pthread_mutex_t creation_lock = PTHREAD_MUTEX_INITIALIZER;
static int verify_state = 0;

static Tape *tape(void) { return (Tape *)arith_trace_tape; }

// ---- the version table ------------------------------------------------------

// The slot a handle hashes to: the probe starts here.
static uint64_t hash_slot(const Tape *t, const void *handle)
{
    uint64_t h = (uint64_t)(uintptr_t)handle;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    return h & (t->slots - 1);
}

// The entry of `handle`, or NULL when the table does not know it.
static VersionEntry *version_lookup(const Tape *t, const void *handle)
{
    if (t->slots == 0)
        return NULL;
    for (uint64_t i = hash_slot(t, handle);; i = (i + 1) & (t->slots - 1))
    {
        VersionEntry *v = &t->table[i];
        if (v->handle == NULL)
            return NULL;
        if (v->handle == handle)
            return v;
    }
}

static void version_store(Tape *t, const void *handle, uint64_t version);

// Doubles the table and re-inserts every entry, since the slots depend on
// the table's size.
static void version_table_grow(Tape *t)
{
    VersionEntry *old = t->table;
    const uint64_t old_slots = t->slots;
    t->slots = old_slots ? 2 * old_slots : 1024;
    t->table = (VersionEntry *)safe_malloc(t->slots * sizeof(VersionEntry));
    memset(t->table, 0, t->slots * sizeof(VersionEntry));
    t->used = 0;
    for (uint64_t i = 0; i < old_slots; i++)
        if (old[i].handle != NULL)
            version_store(t, old[i].handle, old[i].version);
    free(old);
}

// Records that `handle` now holds `version`, replacing what it held.
static void version_store(Tape *t, const void *handle, uint64_t version)
{
    if (2 * (t->used + 1) > t->slots)
        version_table_grow(t);
    for (uint64_t i = hash_slot(t, handle);; i = (i + 1) & (t->slots - 1))
    {
        VersionEntry *v = &t->table[i];
        if (v->handle == NULL || v->handle == handle)
        {
            if (v->handle == NULL)
                t->used++;
            v->handle = handle;
            v->version = version;
            return;
        }
    }
}

// Removes `handle` from the table. The entries after it in its probe chain
// move back over the hole, unless an entry's own slot lies between the hole
// and it, in which case it already sits as early as it may.
static void version_forget(Tape *t, const void *handle)
{
    if (t->slots == 0)
        return;
    uint64_t i = hash_slot(t, handle);
    for (;; i = (i + 1) & (t->slots - 1))
    {
        if (t->table[i].handle == NULL)
            return;
        if (t->table[i].handle == handle)
            break;
    }
    t->used--;
    uint64_t hole = i;
    for (uint64_t j = (hole + 1) & (t->slots - 1);; j = (j + 1) & (t->slots - 1))
    {
        if (t->table[j].handle == NULL)
            break;
        const uint64_t home = hash_slot(t, t->table[j].handle);
        const int between = hole < j ? (home > hole && home <= j) : (home > hole || home <= j);
        if (!between)
        {
            t->table[hole] = t->table[j];
            hole = j;
        }
    }
    t->table[hole].handle = NULL;
    t->table[hole].version = 0;
}

// ---- events -----------------------------------------------------------------

// Appends an empty event with the next sequence number, once the tape has
// room under its bound; waiting releases the lock, and stopping the
// recording ends the wait.
static ArithTraceEvent *event_append(Tape *t)
{
    while (t->limit != 0 && t->length >= t->limit && arith_trace_active != NULL)
    {
        t->waiters++;
        pthread_cond_wait(&t->changed, &t->lock);
        t->waiters--;
        pthread_cond_broadcast(&t->changed); // for a clear waiting on the waiters
    }
    if (t->length == t->capacity)
    {
        t->capacity = t->capacity ? 2 * t->capacity : 256;
        t->events =
            (ArithTraceEvent *)safe_realloc(t->events, t->capacity * sizeof(ArithTraceEvent));
        t->open = (uint8_t *)safe_realloc(t->open, t->capacity);
    }
    ArithTraceEvent *e = &t->events[t->length++];
    memset(e, 0, sizeof(*e));
    e->sequence = t->next_sequence++;
    t->open[t->length - 1] = 0;
    return e;
}

// Publishes every complete event up to the first still open one, and wakes
// the readers when something new became visible.
static void publish(Tape *t)
{
    const uint64_t before = t->published;
    while (t->published < t->length && !t->open[t->published])
        t->published++;
    if (t->published != before)
        pthread_cond_broadcast(&t->changed);
}

// A 64-bit digest of the words `element` holds, over its active rows. Not
// cryptographic: it tells a value from what a later write left in its
// place, which is all verification asks. The domain is not part of it: a
// change of domain label is not a write, and a conversion refreshes it.
static uint64_t digest_of(const ArithElement *element)
{
    RNS_Polynomial p = arith_rns_polynomial(element);
    const uint64_t N = p->base->N;
    uint64_t h = p->rns_mask;
    for (size_t i = 0; i < p->base->l; i++)
    {
        if (!(p->rns_mask & (1ULL << i)))
            continue;
        if (rns_row_is_narrow(p->base, i))
        {
            const uint32_t *row = rns_row32(p, i);
            for (uint64_t j = 0; j < N; j++)
            {
                h = (h ^ row[j]) * 0xff51afd7ed558ccdULL;
                h ^= h >> 33;
            }
        }
        else
        {
            const uint64_t *row = rns_row64(p, i);
            for (uint64_t j = 0; j < N; j++)
            {
                h = (h ^ row[j]) * 0xff51afd7ed558ccdULL;
                h ^= h >> 33;
            }
        }
    }
    return h;
}

// Keeps the digest of what `element`'s storage holds now, when verifying.
static void version_digest(Tape *t, const ArithElement *element)
{
    if (!verify_state)
        return;
    VersionEntry *v = version_lookup(t, element->handle);
    if (v != NULL)
    {
        v->digest = digest_of(element);
        v->digested = 1;
    }
}

// Fills in which primes `element` holds and which domain it is in.
static void event_describe(ArithTraceEvent *e, const ArithElement *element)
{
    RNS_Polynomial p = arith_rns_polynomial(element);
    e->N = p->base->N;
    e->mask = p->rns_mask;
    e->base = p->base;
    e->domain = element->domain;
}

// A fresh RNS_Polynomial holding a copy of `element`'s value.
static void *copy_of(const ArithElement *element)
{
    RNS_Polynomial p = arith_rns_polynomial(element);
    RNS_Polynomial c = polynomial_new_RNS_polynomial(p->base->N, p->rns_mask, p->base);
    polynomial_copy_RNS_polynomial(c, p);
    return c;
}

// The version of an operand. An operand no event wrote is recorded first
// as an INPUT event carrying its copy, so it gets a version too; so is one
// whose storage no longer holds its version's value, when verifying, and
// that INPUT is flagged. An operand holding no value (empty domain) has no
// version.
static uint64_t version_of(Tape *t, const ArithElement *element)
{
    if (element == NULL || element->handle == NULL || element->domain == ARITH_DOMAIN_EMPTY)
        return 0;
    VersionEntry *v = version_lookup(t, element->handle);
    int untraced = 0;
    if (v != NULL)
    {
        if (!verify_state)
            return v->version;
        if (!v->digested)
        {
            // Verification started after this version was written: what the
            // storage holds now is the only value known for it.
            version_digest(t, element);
            return v->version;
        }
        if (v->digest == digest_of(element))
            return v->version;
        untraced = 1;
        t->untraced++;
    }
    ArithTraceEvent *e = event_append(t);
    e->kind = ARITH_TRACE_INPUT;
    e->untraced = untraced;
    event_describe(e, element);
    e->value = copy_of(element);
    version_store(t, element->handle, e->sequence);
    version_digest(t, element);
    return e->sequence;
}

// Releases everything an event owns.
static void event_free(ArithTraceEvent *e)
{
    if (e->value != NULL)
        free_RNS_polynomial(e->value);
    free(e->scalar);
    free(e->label);
    e->value = NULL;
    e->scalar = NULL;
    e->label = NULL;
}

// Opens an event: the operands' versions are resolved (and unknown ones
// copied) before the operation runs, since an accumulator, an element
// divided in place, or an output that is also an operand is overwritten by
// it; then the event is appended with everything but its output.
typedef struct
{
    uint64_t primes, log_base, level, dropped;
    int centered;
} DigitParameters;

static uint64_t begin(ArithTraceKind kind, ArithRing ring, const ArithElement *out,
                      const ArithElement *a, const ArithElement *b, uint64_t param, uint64_t param2,
                      double sigma, const ArithScalar *scalar, const DigitParameters *digit)
{
    if (ring->impl != ARITH_IMPL_RNS)
        return 0;
    Tape *t = tape();
    pthread_mutex_lock(&t->lock);
    uint64_t inputs[ARITH_TRACE_INPUTS] = {0, 0, 0};
    uint32_t n = 0;
    if (kind == ARITH_TRACE_MUL_ADDTO || kind == ARITH_TRACE_MUL_SUBTO ||
        kind == ARITH_TRACE_SCALE_ADDTO || kind == ARITH_TRACE_ROUND_DIVISION ||
        kind == ARITH_TRACE_ROUND_DIVISION_APPROXIMATE)
        inputs[n++] = version_of(t, out);
    if (a != NULL)
        inputs[n++] = version_of(t, a);
    if (b != NULL)
        inputs[n++] = version_of(t, b);

    // INPUT events appended by version_of are complete; this one stays open
    // until its end, and readers see nothing from it on.
    publish(t);
    ArithTraceEvent *e = event_append(t);
    t->open[t->length - 1] = 1;
    e->kind = kind;
    memcpy(e->inputs, inputs, sizeof(inputs));
    e->n_inputs = n;
    e->param = param;
    e->param2 = param2;
    e->sigma = sigma;
    if (digit != NULL)
    {
        e->primes = digit->primes;
        e->log_base = digit->log_base;
        e->level = digit->level;
        e->dropped = digit->dropped;
        e->centered = digit->centered;
    }
    e->secret = kind == ARITH_TRACE_SAMPLE_UNIFORM || kind == ARITH_TRACE_SAMPLE_GAUSSIAN ||
                kind == ARITH_TRACE_SAMPLE_UNIFORM_SEEDED;
    if (scalar != NULL)
    {
        // An RNS scalar is one factor per base row up to the ring's highest.
        const uint64_t rows = 64 - (uint64_t)__builtin_clzll(arith_rns_ring_mask(ring));
        e->scalar = (uint64_t *)safe_malloc(rows * sizeof(uint64_t));
        memcpy(e->scalar, scalar->handle, rows * sizeof(uint64_t));
        e->scalar_len = rows;
    }
    const uint64_t sequence = e->sequence;
    pthread_mutex_unlock(&t->lock);
    return sequence;
}

__attribute__((noinline, cold)) uint64_t arith_trace_begin(
    ArithTraceKind kind, ArithRing ring, const ArithElement *out, const ArithElement *a,
    const ArithElement *b, uint64_t param, uint64_t param2, double sigma, const ArithScalar *scalar)
{
    return begin(kind, ring, out, a, b, param, param2, sigma, scalar, NULL);
}

__attribute__((noinline, cold)) uint64_t arith_trace_begin_digit(
    ArithRing ring, const ArithElement *out, const ArithElement *a, uint64_t primes,
    uint64_t log_base, uint64_t level, uint64_t dropped, int centered)
{
    const DigitParameters digit = {primes, log_base, level, dropped, centered};
    return begin(ARITH_TRACE_DECOMPOSE_DIGIT, ring, out, a, NULL, 0, 0, 0.0, NULL, &digit);
}

// Closes an event: on success the output is copied in and becomes the
// value its storage holds; on failure the event keeps no value.
__attribute__((noinline, cold)) void arith_trace_end(uint64_t sequence, const ArithElement *out,
                                                     ArithStatus status)
{
    Tape *t = tape();
    if (t == NULL)
        return;
    pthread_mutex_lock(&t->lock);
    // An open event cannot have been released, so it is still on the tape.
    if (t->length > 0 && sequence >= t->events[0].sequence)
    {
        const uint64_t index = sequence - t->events[0].sequence;
        ArithTraceEvent *e = &t->events[index];
        event_describe(e, out);
        if (status == ARITH_OK)
        {
            if (e->kind != ARITH_TRACE_COPY && e->kind != ARITH_TRACE_ZERO)
                e->value = copy_of(out);
            version_store(t, out->handle, e->sequence);
            version_digest(t, out);
        }
        else
            e->failed = 1;
        t->open[index] = 0;
    }
    publish(t);
    pthread_mutex_unlock(&t->lock);
}

__attribute__((noinline, cold)) void arith_trace_converting(const ArithElement *element)
{
    Tape *t = tape();
    if (t == NULL || !verify_state)
        return;
    pthread_mutex_lock(&t->lock);
    // A known storage is checked against its version before the conversion
    // rewrites it; a changed one is recorded again, as before any operation.
    if (version_lookup(t, element->handle) != NULL)
    {
        version_of(t, element);
        publish(t);
    }
    pthread_mutex_unlock(&t->lock);
}

__attribute__((noinline, cold)) void arith_trace_converted(const ArithElement *element)
{
    Tape *t = tape();
    if (t == NULL || !verify_state)
        return;
    pthread_mutex_lock(&t->lock);
    version_digest(t, element);
    pthread_mutex_unlock(&t->lock);
}

void arith_trace_forget(const void *handle)
{
    Tape *t = tape();
    if (t == NULL)
        return;
    pthread_mutex_lock(&t->lock);
    version_forget(t, handle);
    pthread_mutex_unlock(&t->lock);
}

void arith_trace_mark(const char *label, int enter)
{
    Tape *t = tape();
    if (t == NULL)
        return;
    pthread_mutex_lock(&t->lock);
    ArithTraceEvent *e = event_append(t);
    e->kind = ARITH_TRACE_MARK;
    const size_t len = strlen(label) + 1;
    e->label = (char *)safe_malloc(len);
    memcpy(e->label, label, len);
    e->enter = enter;
    publish(t);
    pthread_mutex_unlock(&t->lock);
}

uint64_t arith_trace_version(const void *handle)
{
    Tape *t = tape();
    if (t == NULL)
        return 0;
    pthread_mutex_lock(&t->lock);
    const VersionEntry *v = version_lookup(t, handle);
    const uint64_t version = v == NULL ? 0 : v->version;
    pthread_mutex_unlock(&t->lock);
    return version;
}

// ---- control and reading ----------------------------------------------------

void arith_trace_enable(void)
{
    pthread_mutex_lock(&creation_lock);
    if (arith_trace_tape == NULL)
    {
        Tape *t = (Tape *)safe_malloc(sizeof(Tape));
        memset(t, 0, sizeof(*t));
        t->next_sequence = 1;
        pthread_mutex_init(&t->lock, NULL);
        pthread_cond_init(&t->changed, NULL);
        arith_trace_tape = t;
    }
    arith_trace_active = arith_trace_tape;
    pthread_mutex_unlock(&creation_lock);
}

void arith_trace_disable(void)
{
    Tape *t = tape();
    if (t == NULL)
        return;
    // Under the lock, so a producer waiting for room sees the change.
    pthread_mutex_lock(&t->lock);
    arith_trace_active = NULL;
    pthread_cond_broadcast(&t->changed);
    pthread_mutex_unlock(&t->lock);
}

int arith_trace_is_enabled(void) { return arith_trace_active != NULL; }

void arith_trace_clear(void)
{
    pthread_mutex_lock(&creation_lock);
    Tape *t = tape();
    if (t == NULL)
    {
        pthread_mutex_unlock(&creation_lock);
        return;
    }
    // Stopping the recording under the lock wakes a producer waiting for
    // room, and the wait here lets it finish its append before the tape
    // goes; an operation in flight otherwise is the caller's to avoid.
    pthread_mutex_lock(&t->lock);
    arith_trace_active = NULL;
    pthread_cond_broadcast(&t->changed);
    while (t->waiters > 0)
        pthread_cond_wait(&t->changed, &t->lock);
    pthread_mutex_unlock(&t->lock);
    arith_trace_tape = NULL;
    pthread_mutex_unlock(&creation_lock);
    // No tape is published any more, so the forget hook of each freed value
    // finds nothing to lock.
    for (uint64_t i = 0; i < t->length; i++)
        event_free(&t->events[i]);
    free(t->events);
    free(t->open);
    free(t->table);
    pthread_cond_destroy(&t->changed);
    pthread_mutex_destroy(&t->lock);
    free(t);
}

void arith_trace_set_limit(uint64_t limit)
{
    Tape *t = tape();
    if (t == NULL)
        return;
    pthread_mutex_lock(&t->lock);
    t->limit = limit;
    pthread_cond_broadcast(&t->changed);
    pthread_mutex_unlock(&t->lock);
}

uint64_t arith_trace_limit(void)
{
    Tape *t = tape();
    return t == NULL ? 0 : t->limit;
}

int arith_trace_wait(uint64_t count, double seconds)
{
    Tape *t = tape();
    if (t == NULL)
        return 0;
    struct timespec deadline;
    if (seconds >= 0)
    {
        clock_gettime(CLOCK_REALTIME, &deadline);
        const time_t whole = (time_t)seconds;
        deadline.tv_sec += whole;
        deadline.tv_nsec += (long)((seconds - (double)whole) * 1e9);
        if (deadline.tv_nsec >= 1000000000L)
        {
            deadline.tv_sec += 1;
            deadline.tv_nsec -= 1000000000L;
        }
    }
    pthread_mutex_lock(&t->lock);
    while (t->published < count && arith_trace_active != NULL)
    {
        if (seconds < 0)
            pthread_cond_wait(&t->changed, &t->lock);
        else if (pthread_cond_timedwait(&t->changed, &t->lock, &deadline) == ETIMEDOUT)
            break;
    }
    const int filled = t->published >= count;
    pthread_mutex_unlock(&t->lock);
    return filled;
}

void arith_trace_set_verify(int enabled) { verify_state = enabled != 0; }

int arith_trace_verify(void) { return verify_state; }

uint64_t arith_trace_untraced(void)
{
    Tape *t = tape();
    if (t == NULL)
        return 0;
    pthread_mutex_lock(&t->lock);
    const uint64_t n = t->untraced;
    pthread_mutex_unlock(&t->lock);
    return n;
}

uint64_t arith_trace_length(void)
{
    Tape *t = tape();
    if (t == NULL)
        return 0;
    pthread_mutex_lock(&t->lock);
    const uint64_t n = t->published;
    pthread_mutex_unlock(&t->lock);
    return n;
}

int arith_trace_read(uint64_t index, ArithTraceEvent *out)
{
    Tape *t = tape();
    if (t == NULL)
        return -1;
    pthread_mutex_lock(&t->lock);
    const int found = index < t->published;
    if (found)
        *out = t->events[index];
    pthread_mutex_unlock(&t->lock);
    return found ? 0 : -1;
}

void *arith_trace_take_value(uint64_t index)
{
    Tape *t = tape();
    if (t == NULL)
        return NULL;
    pthread_mutex_lock(&t->lock);
    void *value = NULL;
    if (index < t->published)
    {
        value = t->events[index].value;
        t->events[index].value = NULL;
    }
    pthread_mutex_unlock(&t->lock);
    return value;
}

void arith_trace_release(uint64_t count)
{
    Tape *t = tape();
    if (t == NULL)
        return;
    pthread_mutex_lock(&t->lock);
    if (count > t->published)
        count = t->published;
    // Freed after the lock is released: freeing a value runs the forget
    // hook, which takes the lock itself.
    ArithTraceEvent *released =
        (ArithTraceEvent *)safe_malloc((count ? count : 1) * sizeof(ArithTraceEvent));
    memcpy(released, t->events, count * sizeof(ArithTraceEvent));
    memmove(t->events, t->events + count, (t->length - count) * sizeof(ArithTraceEvent));
    memmove(t->open, t->open + count, t->length - count);
    t->length -= count;
    t->published -= count;
    pthread_cond_broadcast(&t->changed);
    pthread_mutex_unlock(&t->lock);
    for (uint64_t i = 0; i < count; i++)
        event_free(&released[i]);
    free(released);
}

uint64_t arith_trace_next_sequence(void)
{
    Tape *t = tape();
    if (t == NULL)
        return 1;
    pthread_mutex_lock(&t->lock);
    const uint64_t n = t->next_sequence;
    pthread_mutex_unlock(&t->lock);
    return n;
}
