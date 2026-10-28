// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file arith_trace.h
 * @brief The execution tape: a record of every whole-element ring operation
 *        the library runs, with a copy of each output.
 *
 * While the tape is enabled, each operation of the generic interface
 * (arith_generic.h) over an RNS ring appends one event: its kind, the
 * versions of its operands, its parameters and a copy of its output. A
 * value is identified by a version, the sequence number of the event that
 * wrote it; an operand that no event wrote is recorded first as an INPUT
 * event carrying its copy, so every operand of every event is on the tape.
 * Domain conversions are not events: both representations hold one value.
 *
 * Disabled, the tape costs the operations one load and one branch. Enabled,
 * it costs a lock and a copy per operation, from whichever thread runs it.
 * Events are read and released in order; a reader that drains as the
 * computation runs bounds what the tape holds.
 */
#ifndef VFHE_ARITH_TRACE_H
#define VFHE_ARITH_TRACE_H

#include <arith_generic.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /** What an event records. The first group mirrors the generic
     *  interface's operations; INPUT is a value first met as an operand,
     *  MARK a label a caller placed on the tape. */
    typedef enum
    {
        ARITH_TRACE_INPUT = 1,
        ARITH_TRACE_COPY,
        ARITH_TRACE_ZERO,
        ARITH_TRACE_ADD,
        ARITH_TRACE_SUB,
        ARITH_TRACE_MUL,
        ARITH_TRACE_MUL_ADDTO,
        ARITH_TRACE_MUL_SUBTO,
        ARITH_TRACE_SCALE_INT,
        ARITH_TRACE_SCALE_ADDTO,
        ARITH_TRACE_SCALE_BY,
        ARITH_TRACE_PERMUTE,
        ARITH_TRACE_MUL_BY_MONOMIAL,
        ARITH_TRACE_SAMPLE_UNIFORM,
        ARITH_TRACE_SAMPLE_GAUSSIAN,
        ARITH_TRACE_SAMPLE_UNIFORM_SEEDED,
        ARITH_TRACE_FROM_INT_ARRAY,
        ARITH_TRACE_ROUND_DIVISION,
        ARITH_TRACE_MOD_REDUCE_LIFTED,
        ARITH_TRACE_MARK,
    } ArithTraceKind;

/** The most operands one event names. */
#define ARITH_TRACE_INPUTS 3

    /**
     * @brief One event of the tape.
     *
     * `sequence` numbers events from 1 in the order they were recorded and is
     * the version of the value the event wrote. `inputs` are the versions of
     * the `n_inputs` operands, in the operation's argument order; an
     * operation that accumulates into its output or works in place
     * (MUL_ADDTO, MUL_SUBTO, SCALE_ADDTO, ROUND_DIVISION) names the output's
     * previous version first. COPY names the value copied and carries no
     * payload; ZERO carries none either. `value` is the output's copy, an
     * RNS_Polynomial over the output's primes (`mask`, `base`, `N`) in
     * `domain`, until arith_trace_take_value() hands it over. `secret` marks
     * a sampler's output. A MARK carries `label` and whether it opens
     * (`enter`) or closes a region.
     *
     * The parameters, by kind: PERMUTE's generator and MUL_BY_MONOMIAL's
     * power in `param` (`param2` its minus_one); SCALE_INT's and
     * SCALE_ADDTO's factor in `param`; SCALE_BY's per-row factors in
     * `scalar`; SAMPLE_GAUSSIAN's width in `sigma`; FROM_INT_ARRAY's count
     * and SAMPLE_UNIFORM_SEEDED's stream in `param`. ROUND_DIVISION's output
     * is the input divided by the primes it no longer holds and rounded.
     */
    typedef struct
    {
        uint64_t sequence;
        ArithTraceKind kind;
        uint64_t inputs[ARITH_TRACE_INPUTS];
        uint32_t n_inputs;
        uint64_t N;
        uint64_t mask;
        void *base;
        ArithDomain domain;
        int secret;
        void *value;
        uint64_t param;
        uint64_t param2;
        double sigma;
        uint64_t *scalar;
        uint64_t scalar_len;
        char *label;
        int enter;
    } ArithTraceEvent;

    /**
     * @brief Starts recording. The tape is created on the first call and
     *        kept, with its events and the versions it knows, across
     *        arith_trace_disable() and arith_trace_enable(), until
     *        arith_trace_clear().
     */
    void arith_trace_enable(void);
    /** @brief Stops recording; the events stay readable. */
    void arith_trace_disable(void);
    /** @brief Whether operations are being recorded. */
    int arith_trace_is_enabled(void);
    /** @brief Drops every event, the values not taken, and the versions. */
    void arith_trace_clear(void);

    /** @brief How many events are on the tape. */
    uint64_t arith_trace_length(void);
    /**
     * @brief Copies event `index` (0 the oldest) into `out`.
     * @return 0, or -1 when there is no such event.
     */
    int arith_trace_read(uint64_t index, ArithTraceEvent *out);
    /**
     * @brief Hands over the value of event `index`, to release with
     *        free_RNS_polynomial(); the event keeps none.
     * @return The RNS_Polynomial, or NULL when the event has no value.
     */
    void *arith_trace_take_value(uint64_t index);
    /** @brief Drops the oldest `count` events, freeing what was not taken. */
    void arith_trace_release(uint64_t count);
    /** @brief The sequence number the next event will get. */
    uint64_t arith_trace_next_sequence(void);
    /** @brief Appends a MARK event with a copy of `label`. */
    void arith_trace_mark(const char *label, int enter);

    /**
     * @brief Non-NULL while recording. The generic interface reads it
     *        before each operation; nothing else should.
     */
    extern void *arith_trace_active;
    /** @brief Non-NULL while a tape exists, recording or not. */
    extern void *arith_trace_tape;

    /**
     * @brief Opens the event of one operation of the generic interface,
     *        before the operation runs: resolves the operands' versions,
     *        recording as INPUT any the tape does not know -- which must
     *        happen before an in-place operation overwrites one -- and
     *        appends the event. `out` is the element the operation will
     *        write, `a` and `b` its operands, either NULL when it has
     *        fewer; the rest as in the event. An operation over a ring that
     *        is not RNS is not recorded.
     * @return The event's sequence number, or 0 when nothing was recorded.
     */
    uint64_t arith_trace_begin(ArithTraceKind kind, ArithRing ring, const ArithElement *out,
                               const ArithElement *a, const ArithElement *b, uint64_t param,
                               uint64_t param2, double sigma, const ArithScalar *scalar);
    /**
     * @brief Closes the event `sequence` opened by arith_trace_begin() once
     *        the operation returned `status`: on success copies `out` into
     *        the event and makes the event its version; on failure the
     *        event stays with no value and its domain empty.
     */
    void arith_trace_end(uint64_t sequence, const ArithElement *out, ArithStatus status);
    /**
     * @brief Forgets the version of an element's storage about to be freed,
     *        so a later element at the same address starts unknown.
     */
    void arith_trace_forget(const void *handle);

/** The test the interface makes before recording. */
#define ARITH_TRACE_ON() __builtin_expect(arith_trace_active != NULL, 0)

#ifdef __cplusplus
}
#endif

#endif // VFHE_ARITH_TRACE_H
