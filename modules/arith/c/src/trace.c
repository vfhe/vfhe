// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
#include <arith_trace.h>

#include <alloc.h>
#include <arith.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

// The tape: a growable array of events, and a table from an element's
// storage to the version it holds. One lock covers both, taken for the whole
// of a record, copy included: events are then numbered in the order the
// operations finished, whichever threads ran them.

typedef struct
{
    const void *handle; // NULL marks an empty slot
    uint64_t version;
} Version;

typedef struct
{
    ArithTraceEvent *events;
    uint64_t length, capacity;
    uint64_t next_sequence;
    // Open addressing with linear probing; `used` counts the slots taken.
    Version *table;
    uint64_t slots, used;
    pthread_mutex_t lock;
} Tape;

void *arith_trace_active = NULL;
void *arith_trace_tape = NULL;
static pthread_mutex_t creation_lock = PTHREAD_MUTEX_INITIALIZER;

static Tape *tape(void) { return (Tape *)arith_trace_tape; }

// ---- the version table ------------------------------------------------------

static uint64_t slot_of(const Tape *t, const void *handle)
{
    uint64_t h = (uint64_t)(uintptr_t)handle;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    return h & (t->slots - 1);
}

static Version *find(const Tape *t, const void *handle)
{
    if (t->slots == 0)
        return NULL;
    for (uint64_t i = slot_of(t, handle);; i = (i + 1) & (t->slots - 1))
    {
        Version *v = &t->table[i];
        if (v->handle == NULL)
            return NULL;
        if (v->handle == handle)
            return v;
    }
}

static void insert(Tape *t, const void *handle, uint64_t version);

static void grow_table(Tape *t)
{
    Version *old = t->table;
    const uint64_t old_slots = t->slots;
    t->slots = old_slots ? 2 * old_slots : 1024;
    t->table = (Version *)safe_malloc(t->slots * sizeof(Version));
    memset(t->table, 0, t->slots * sizeof(Version));
    t->used = 0;
    for (uint64_t i = 0; i < old_slots; i++)
        if (old[i].handle != NULL)
            insert(t, old[i].handle, old[i].version);
    free(old);
}

static void insert(Tape *t, const void *handle, uint64_t version)
{
    if (2 * (t->used + 1) > t->slots)
        grow_table(t);
    for (uint64_t i = slot_of(t, handle);; i = (i + 1) & (t->slots - 1))
    {
        Version *v = &t->table[i];
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

// Backward-shift deletion keeps every probe chain unbroken without
// tombstones.
static void erase(Tape *t, const void *handle)
{
    if (t->slots == 0)
        return;
    uint64_t i = slot_of(t, handle);
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
        const uint64_t home = slot_of(t, t->table[j].handle);
        // Entry j may move into the hole if its home is not in (hole, j].
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

static ArithTraceEvent *append(Tape *t)
{
    if (t->length == t->capacity)
    {
        t->capacity = t->capacity ? 2 * t->capacity : 256;
        t->events = (ArithTraceEvent *)safe_realloc(t->events, t->capacity * sizeof(ArithTraceEvent));
    }
    ArithTraceEvent *e = &t->events[t->length++];
    memset(e, 0, sizeof(*e));
    e->sequence = t->next_sequence++;
    return e;
}

static void describe(ArithTraceEvent *e, const ArithElement *element)
{
    RNS_Polynomial p = arith_rns_polynomial(element);
    e->N = p->base->N;
    e->mask = p->rns_mask;
    e->base = p->base;
    e->domain = element->domain;
}

static void *copy_of(const ArithElement *element)
{
    RNS_Polynomial p = arith_rns_polynomial(element);
    RNS_Polynomial c = polynomial_new_RNS_polynomial(p->base->N, p->rns_mask, p->base);
    polynomial_copy_RNS_polynomial(c, p);
    return c;
}

// The version of an operand, recording it as an INPUT first if no event
// wrote it. An operand holding no value (empty domain) has no version.
static uint64_t version_of(Tape *t, const ArithElement *element)
{
    if (element == NULL || element->handle == NULL || element->domain == ARITH_DOMAIN_EMPTY)
        return 0;
    Version *v = find(t, element->handle);
    if (v != NULL)
        return v->version;
    ArithTraceEvent *e = append(t);
    e->kind = ARITH_TRACE_INPUT;
    describe(e, element);
    e->value = copy_of(element);
    insert(t, element->handle, e->sequence);
    return e->sequence;
}

static void free_event(ArithTraceEvent *e)
{
    if (e->value != NULL)
        free_RNS_polynomial(e->value);
    free(e->scalar);
    free(e->label);
    e->value = NULL;
    e->scalar = NULL;
    e->label = NULL;
}

__attribute__((noinline, cold)) uint64_t arith_trace_begin(ArithTraceKind kind, ArithRing ring,
                                                           const ArithElement *out,
                                                           const ArithElement *a,
                                                           const ArithElement *b, uint64_t param,
                                                           uint64_t param2, double sigma,
                                                           const ArithScalar *scalar)
{
    if (ring->impl != ARITH_IMPL_RNS)
        return 0;
    Tape *t = tape();
    pthread_mutex_lock(&t->lock);
    // The operands' versions before the operation: an accumulator and the
    // element of an in-place division are read before they are written, and
    // an operand may be the output itself.
    uint64_t inputs[ARITH_TRACE_INPUTS] = {0, 0, 0};
    uint32_t n = 0;
    if (kind == ARITH_TRACE_MUL_ADDTO || kind == ARITH_TRACE_MUL_SUBTO ||
        kind == ARITH_TRACE_SCALE_ADDTO || kind == ARITH_TRACE_ROUND_DIVISION)
        inputs[n++] = version_of(t, out);
    if (a != NULL)
        inputs[n++] = version_of(t, a);
    if (b != NULL)
        inputs[n++] = version_of(t, b);

    ArithTraceEvent *e = append(t);
    e->kind = kind;
    memcpy(e->inputs, inputs, sizeof(inputs));
    e->n_inputs = n;
    e->param = param;
    e->param2 = param2;
    e->sigma = sigma;
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

__attribute__((noinline, cold)) void arith_trace_end(uint64_t sequence, const ArithElement *out,
                                                     ArithStatus status)
{
    Tape *t = tape();
    if (t == NULL)
        return;
    pthread_mutex_lock(&t->lock);
    // The event is still on the tape unless a reader released it meanwhile.
    if (t->length > 0 && sequence >= t->events[0].sequence)
    {
        ArithTraceEvent *e = &t->events[sequence - t->events[0].sequence];
        if (status == ARITH_OK)
        {
            describe(e, out);
            if (e->kind != ARITH_TRACE_COPY && e->kind != ARITH_TRACE_ZERO)
                e->value = copy_of(out);
            insert(t, out->handle, e->sequence);
        }
    }
    pthread_mutex_unlock(&t->lock);
}

void arith_trace_forget(const void *handle)
{
    Tape *t = tape();
    if (t == NULL)
        return;
    pthread_mutex_lock(&t->lock);
    erase(t, handle);
    pthread_mutex_unlock(&t->lock);
}

void arith_trace_mark(const char *label, int enter)
{
    Tape *t = tape();
    if (t == NULL)
        return;
    pthread_mutex_lock(&t->lock);
    ArithTraceEvent *e = append(t);
    e->kind = ARITH_TRACE_MARK;
    const size_t len = strlen(label) + 1;
    e->label = (char *)safe_malloc(len);
    memcpy(e->label, label, len);
    e->enter = enter;
    pthread_mutex_unlock(&t->lock);
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
        arith_trace_tape = t;
    }
    arith_trace_active = arith_trace_tape;
    pthread_mutex_unlock(&creation_lock);
}

void arith_trace_disable(void) { arith_trace_active = NULL; }

int arith_trace_is_enabled(void) { return arith_trace_active != NULL; }

void arith_trace_clear(void)
{
    pthread_mutex_lock(&creation_lock);
    arith_trace_active = NULL;
    Tape *t = tape();
    arith_trace_tape = NULL;
    pthread_mutex_unlock(&creation_lock);
    if (t == NULL)
        return;
    // No tape is published any more, so the forget hook of each freed value
    // finds nothing to lock.
    for (uint64_t i = 0; i < t->length; i++)
        free_event(&t->events[i]);
    free(t->events);
    free(t->table);
    pthread_mutex_destroy(&t->lock);
    free(t);
}

uint64_t arith_trace_length(void)
{
    Tape *t = tape();
    if (t == NULL)
        return 0;
    pthread_mutex_lock(&t->lock);
    const uint64_t n = t->length;
    pthread_mutex_unlock(&t->lock);
    return n;
}

int arith_trace_read(uint64_t index, ArithTraceEvent *out)
{
    Tape *t = tape();
    if (t == NULL)
        return -1;
    pthread_mutex_lock(&t->lock);
    const int found = index < t->length;
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
    if (index < t->length)
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
    if (count > t->length)
        count = t->length;
    // Freed after the lock is released: freeing a value runs the forget
    // hook, which takes the lock itself.
    ArithTraceEvent *released = (ArithTraceEvent *)safe_malloc((count ? count : 1) * sizeof(ArithTraceEvent));
    memcpy(released, t->events, count * sizeof(ArithTraceEvent));
    memmove(t->events, t->events + count, (t->length - count) * sizeof(ArithTraceEvent));
    t->length -= count;
    pthread_mutex_unlock(&t->lock);
    for (uint64_t i = 0; i < count; i++)
        free_event(&released[i]);
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
