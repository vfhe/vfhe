// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "gadget_rns.h"
#include "mlwe.h"
#include "rns_rows.h"
#include "util.h"

// Hybrid (GHS) key switches over RNS rings, many at a time.
//
// A key switch is a sequence of steps, and every step after the digit split
// works on each RNS row (each prime) on its own. So instead of handing one
// sample to each task, a batch runs every step as one parallel loop over the
// rows of all its samples: a batch of two samples over eight primes has
// sixteen tasks per step rather than two.
//
// The steps, and the tasks of each (one task per item in parentheses):
//
//   digits   (sample, row of the key ring, digit group): the digits of the
//            group, each lifted to that row, multiplied by their keys and
//            subtracted into the group's accumulator. The digits are split
//            into groups only when the rows alone are too few for the
//            threads (rns_rows_parts).
//   sum      (sample, component, row of the output): the groups'
//            accumulators added and taken back to the coefficient domain.
//            The ModDown needs the special primes' rows complete, so one
//            task per component does all of them and then prepares them.
//   ModDown  (sample, component, row of the output): division by the special
//            primes, plus the components that pass through. When a rescale
//            follows, one task per component does the rows it drops and
//            prepares them.
//   rescale  (sample, component, row of the result): division by the
//            dropped primes; then one task per component finishes it.
//   NTT      (sample, component, row of the result), when asked for.
//
// Each row goes through the same arithmetic, in the same order, as in
// mlwe_RNSc_GHS_hybrid_keyswitch, and the groups' accumulators are added
// modulo the prime, which is exact. So the results do not depend on the
// number of threads or of groups.
//
// When the batch is wide enough that one sample per task already keeps at
// least three quarters of the threads busy (each sample with the share of
// the threads vfhe_parallel_for lends it), it runs that way instead: every
// sample is one task, which goes through the same steps on its own. A ring
// below ROWS_MIN_N always runs that way.

// Row `row` of component k of `c`, as a view.
static RNS_Polynomial row_of(struct _RNS_Polynomial *storage, MLWE c, uint64_t k, uint64_t row)
{
    return polynomial_RNS_view(storage, mlwe_component_rows(c, k), 1ULL << row);
}

static RNSc_Polynomial component_of(MLWE c, uint64_t k)
{
    return (RNSc_Polynomial)mlwe_component_rows(c, k);
}

static uint64_t lowest_bit(uint64_t mask) { return mask & (~mask + 1); }

bool mlwe_is_rns_hybrid_keyswitch(RNSc_MLWE out, RNS_MLWE_KS_Key ksk)
{
    if (out->ring->impl != ARITH_IMPL_RNS || ksk->ring->impl != ARITH_IMPL_RNS)
        return false;
    const uint64_t key = arith_rns_ring_mask(ksk->ring), target = arith_rns_ring_mask(out->ring);
    return (key & target) == target && key != target;
}

static uint64_t key_rows(RNS_MLWE_KS_Key ksk)
{
    return (uint64_t)__builtin_popcountll(arith_rns_ring_mask(ksk->ring));
}

// Whether a batch of n samples, each of whose steps has `rows` row tasks,
// keeps at least three quarters of its threads busy as one task per sample.
static bool one_sample_per_task(uint64_t N, uint64_t n, uint64_t rows, uint64_t n_threads)
{
    if (N < ROWS_MIN_N)
        return true;
    const uint64_t share = vfhe_share_for(n_threads, n);
    const uint64_t busy = vfhe_threads_for(n_threads, n) * (share < rows ? share : rows);
    return 4 * busy >= 3 * vfhe_threads_for(n_threads, UINT64_MAX);
}

// The threads the steps of one sample may use when the sample is one task:
// its lent share, or none of its own below ROWS_MIN_N.
static uint64_t sample_task_threads(uint64_t N) { return N < ROWS_MIN_N ? 1 : 0; }

// ---- the key switch -------------------------------------------------------

typedef struct
{
    KeySwitch ks;
    // The digit products (absent if every component passes through), and an
    // accumulator over the key ring per digit group; group_acc[0] ends up
    // holding the whole sum.
    GadgetSum sum;
    bool has_sum;
    uint64_t groups;
    RNS_MLWE *group_acc;
    // Per output component, the input component added to it after the
    // ModDown, or NULL.
    const ArithElement **pass_through;
    // The primes of the key ring, of `out`'s ring, the special primes (key
    // ring but not `out`'s), the primes the rescale drops, and the primes of
    // the result.
    uint64_t key_mask, out_mask, special_mask, dropped_mask, result_mask;
} KeySwitchState;

static void init_state(KeySwitchState *st, const KeySwitch *ks, uint64_t batch_rows,
                       uint64_t n_threads)
{
    RNSc_MLWE in = ks->in, out = ks->out;
    RNS_MLWE_KS_Key ksk = ks->ksk;
    *st = (KeySwitchState){*ks, {0}, false, 1, NULL, NULL, 0, 0, 0, 0, 0};

    // A NULL key marks a component that keeps the target key: the k-th such
    // component is added to out->a[k] after the ModDown, and in->b to out->b.
    RNS_MLWE *keys[in->r ? in->r : 1];
    const ArithElement *switched[in->r ? in->r : 1];
    uint64_t terms = 0, kept = 0;
    st->pass_through = (const ArithElement **)safe_malloc((out->r + 1) * sizeof(ArithElement *));
    for (uint64_t k = 0; k <= out->r; k++)
        st->pass_through[k] = NULL;
    for (uint64_t i = 0; i < in->r; i++)
    {
        if (ksk->s[i] == NULL)
            st->pass_through[kept++] = &in->a[i];
        else
        {
            keys[terms] = ksk->s[i];
            switched[terms++] = &in->a[i];
        }
    }
    st->pass_through[out->r] = &in->b;

    st->key_mask = arith_rns_ring_mask(ksk->ring);
    if (terms > 0)
    {
        gadget_sum_init(&st->sum, keys, switched, terms, &ksk->gadget_params, 1);
        st->has_sum = true;
        st->key_mask = st->sum.key_mask;
        st->groups = rns_rows_parts(ksk->ring->N, batch_rows, st->sum.n_digits, n_threads);
    }
    st->out_mask = arith_rns_ring_mask(out->ring);
    st->special_mask = st->key_mask & ~st->out_mask;
    st->dropped_mask = ks->rescale_to ? st->out_mask & ~arith_rns_ring_mask(ks->rescale_to) : 0;
    st->result_mask = st->out_mask & ~st->dropped_mask;

    st->group_acc = (RNS_MLWE *)safe_malloc(st->groups * sizeof(RNS_MLWE));
    for (uint64_t g = 0; g < st->groups; g++)
    {
        st->group_acc[g] = mlwe_alloc_sample(ksk->ring, out->r);
        for (uint64_t k = 0; k <= out->r; k++)
            component_of(st->group_acc[g], k)->rns_mask = st->key_mask;
    }
}

static void free_state(KeySwitchState *st)
{
    for (uint64_t g = 0; g < st->groups; g++)
        free_mlwe_RNS_sample(st->group_acc[g]);
    free(st->group_acc);
    free(st->pass_through);
    if (st->has_sum)
        gadget_sum_free(&st->sum);
}

static void digits_task(void *ctx, uint64_t j, uint64_t row, uint64_t g)
{
    KeySwitchState *st = &((KeySwitchState *)ctx)[j];
    struct _RNS_Polynomial storage;
    for (uint64_t k = 0; k <= st->ks.out->r; k++)
        polynomial_RNS_zero(row_of(&storage, st->group_acc[g], k, row));
    if (st->has_sum)
        gadget_sum_row(&st->sum, st->group_acc[g], g, st->groups, row);
}

// Adds the groups' accumulators into group_acc[0] on one row and takes the
// row back to the coefficient domain.
static void sum_row(KeySwitchState *st, uint64_t k, uint64_t row)
{
    struct _RNS_Polynomial acc_storage, storage;
    RNS_Polynomial acc = row_of(&acc_storage, st->group_acc[0], k, row);
    for (uint64_t g = 1; g < st->groups; g++)
        polynomial_add_RNS_polynomial(acc, acc, row_of(&storage, st->group_acc[g], k, row));
    polynomial_RNS_to_RNSc((RNSc_Polynomial)acc, acc);
}

// A row of the output, or, for the lowest special prime, every special
// prime's row followed by the ModDown's preparation of them.
static void sum_task(void *ctx, uint64_t j, uint64_t row, uint64_t k)
{
    KeySwitchState *st = &((KeySwitchState *)ctx)[j];
    if (!(st->special_mask & (1ULL << row)))
    {
        sum_row(st, k, row);
        return;
    }
    for (uint64_t i = 0; i < 64; i++)
        if (st->special_mask & (1ULL << i))
            sum_row(st, k, i);
    polynomial_mod_down_RNSc_prepare(component_of(st->group_acc[0], k), st->special_mask);
}

static void mod_down_row(KeySwitchState *st, uint64_t k, uint64_t row)
{
    const ArithElement *pass_through = st->pass_through[k];
    polynomial_mod_down_RNSc_rows(component_of(st->ks.out, k), component_of(st->group_acc[0], k),
                                  pass_through ? (RNSc_Polynomial)arith_rns_polynomial(pass_through)
                                               : NULL,
                                  st->special_mask, 1ULL << row);
}

// A row of the result, or, for the lowest dropped prime, every dropped
// prime's row followed by the rescale's preparation of them.
static void mod_down_task(void *ctx, uint64_t j, uint64_t row, uint64_t k)
{
    KeySwitchState *st = &((KeySwitchState *)ctx)[j];
    if (!(st->dropped_mask & (1ULL << row)))
    {
        mod_down_row(st, k, row);
        return;
    }
    for (uint64_t i = 0; i < 64; i++)
        if (st->dropped_mask & (1ULL << i))
            mod_down_row(st, k, i);
    polynomial_round_division_RNSc_prepare(component_of(st->ks.out, k), st->dropped_mask);
}

static void rescale_task(void *ctx, uint64_t j, uint64_t row, uint64_t k)
{
    KeySwitchState *st = &((KeySwitchState *)ctx)[j];
    polynomial_round_division_RNSc_rows(component_of(st->ks.out, k), st->dropped_mask, 1ULL << row);
}

static void rescale_finish_task(void *ctx, uint64_t j, uint64_t row, uint64_t k)
{
    (void)row;
    KeySwitchState *st = &((KeySwitchState *)ctx)[j];
    polynomial_round_division_RNSc_finish(component_of(st->ks.out, k), st->dropped_mask);
}

static void ntt_task(void *ctx, uint64_t j, uint64_t row, uint64_t k)
{
    KeySwitchState *st = &((KeySwitchState *)ctx)[j];
    struct _RNS_Polynomial storage;
    RNS_Polynomial x = row_of(&storage, st->ks.out, k, row);
    polynomial_RNSc_to_RNS(x, (RNSc_Polynomial)x);
}

typedef enum
{
    STEP_DIGITS,
    STEP_SUM,
    STEP_MOD_DOWN,
    STEP_RESCALE,
    STEP_RESCALE_FINISH,
    STEP_NTT
} Step;

// The tasks of one step for sample `st`: one per row in `mask` and part below
// `parts` (the digit groups for the digits, the components otherwise). A
// single row in the mask stands for a task per component.
static void step_tasks(const KeySwitchState *st, Step step, uint64_t *mask, uint64_t *parts)
{
    *parts = st->ks.out->r + 1;
    switch (step)
    {
    case STEP_DIGITS:
        *mask = st->key_mask;
        *parts = st->groups;
        break;
    case STEP_SUM:
        *mask = st->out_mask | lowest_bit(st->special_mask);
        break;
    case STEP_MOD_DOWN:
        *mask = st->result_mask | lowest_bit(st->dropped_mask);
        break;
    case STEP_RESCALE:
        *mask = st->dropped_mask ? st->result_mask : 0;
        break;
    case STEP_RESCALE_FINISH:
        *mask = st->dropped_mask ? lowest_bit(st->result_mask) : 0;
        break;
    case STEP_NTT:
        *mask = st->ks.to_mul_domain ? st->result_mask : 0;
        break;
    }
}

static void run_step(KeySwitchState *st, uint64_t n, Step step, uint64_t n_threads)
{
    static void (*const tasks[])(void *, uint64_t, uint64_t, uint64_t) = {
        digits_task, sum_task, mod_down_task, rescale_task, rescale_finish_task, ntt_task};
    uint64_t *masks = (uint64_t *)safe_malloc(2 * n * sizeof(uint64_t)), *parts = masks + n;
    for (uint64_t j = 0; j < n; j++)
        step_tasks(&st[j], step, &masks[j], &parts[j]);
    rns_rows_batch_for(n, masks, parts, tasks[step], st, n_threads);
    free(masks);
}

// The masks and domains each step leaves its outputs with are set between
// the steps: a task works on views, which never change the element's own.
static void run_steps(KeySwitchState *st, uint64_t n, uint64_t n_threads)
{
    run_step(st, n, STEP_DIGITS, n_threads);
    for (uint64_t j = 0; j < n; j++)
        for (uint64_t k = 0; st[j].has_sum && k <= st[j].ks.out->r; k++)
            component_of(st[j].group_acc[0], k)->rns_mask = gadget_sum_mask(&st[j].sum, k);
    run_step(st, n, STEP_SUM, n_threads);
    for (uint64_t j = 0; j < n; j++)
    {
        for (uint64_t k = 0; k <= st[j].ks.out->r; k++)
        {
            mlwe_component(st[j].group_acc[0], k)->domain = ARITH_DOMAIN_CANONICAL;
            component_of(st[j].ks.out, k)->rns_mask = st[j].out_mask;
            mlwe_component(st[j].ks.out, k)->domain = ARITH_DOMAIN_CANONICAL;
        }
    }
    run_step(st, n, STEP_MOD_DOWN, n_threads);
    run_step(st, n, STEP_RESCALE, n_threads);
    run_step(st, n, STEP_RESCALE_FINISH, n_threads);
    for (uint64_t j = 0; j < n; j++)
        if (st[j].dropped_mask)
            st[j].ks.out->ring = st[j].ks.rescale_to;
    run_step(st, n, STEP_NTT, n_threads);
    for (uint64_t j = 0; j < n; j++)
        for (uint64_t k = 0; st[j].ks.to_mul_domain && k <= st[j].ks.out->r; k++)
            mlwe_component(st[j].ks.out, k)->domain = ARITH_DOMAIN_MUL;
}

// The n key switches, each step one loop over all of them.
static void run_keyswitches(const KeySwitch *ks, uint64_t n, uint64_t n_threads)
{
    uint64_t batch_rows = 0;
    for (uint64_t j = 0; j < n; j++)
        batch_rows += key_rows(ks[j].ksk);
    KeySwitchState *st = (KeySwitchState *)safe_malloc((n ? n : 1) * sizeof(KeySwitchState));
    for (uint64_t j = 0; j < n; j++)
        init_state(&st[j], &ks[j], batch_rows, n_threads);
    run_steps(st, n, n_threads);
    for (uint64_t j = 0; j < n; j++)
        free_state(&st[j]);
    free(st);
}

static void keyswitch_sample_task(void *ctx, uint64_t j)
{
    KeySwitch *ks = &((KeySwitch *)ctx)[j];
    run_keyswitches(ks, 1, sample_task_threads(ks->out->ring->N));
}

void mlwe_rns_hybrid_keyswitch_batch(KeySwitch *ks, uint64_t n, uint64_t n_threads)
{
    if (n == 0)
        return;
    if (one_sample_per_task(ks[0].out->ring->N, n, key_rows(ks[0].ksk), n_threads))
        vfhe_parallel_for(n, n_threads, keyswitch_sample_task, ks);
    else
        run_keyswitches(ks, n, n_threads);
}

// ---- products ---------------------------------------------------------------
//
// Before the key switch: the operands to the mul domain, then the tensor
// product, a component and a row per task, each row taken back to the
// coefficient domain where it is computed.

typedef struct
{
    RNS_MLWE in1, in2;
    RNS_MLWE tensor; // rank R: the input of the key switch
    uint64_t mask;   // the operands' primes
    // Whether the tensor product goes by rows: every operand component is
    // over `mask` once in the mul domain. Otherwise it is mlwe_tensor_product.
    bool by_rows;
} Product;

typedef struct
{
    Product *p;
    KeySwitch *ks;
} ProductBatch;

// Part k < r + 1 is component k of in1, part r + 1 + k component k of in2.
static void operands_to_mul_task(void *ctx, uint64_t j, uint64_t row, uint64_t part)
{
    Product *p = &((Product *)ctx)[j];
    const uint64_t components = p->in1->r + 1;
    MLWE c = part < components ? p->in1 : p->in2;
    const uint64_t k = part % components;
    // A square has one operand, converted once.
    if ((part >= components && p->in2 == p->in1) ||
        mlwe_component(c, k)->domain == ARITH_DOMAIN_MUL ||
        !(component_of(c, k)->rns_mask & (1ULL << row)))
        return;
    struct _RNS_Polynomial storage;
    RNS_Polynomial x = row_of(&storage, c, k, row);
    polynomial_RNSc_to_RNS(x, (RNSc_Polynomial)x);
}

// Component k of the tensor product on one row, in mlwe_tensor_product's
// order -- the quadratic terms a_i a'_m (+ a_m a'_i) for i <= m, then the
// linear a_i b' + b a'_i, then b b' -- taken back to the coefficient domain.
static void tensor_task(void *ctx, uint64_t j, uint64_t row, uint64_t k)
{
    Product *p = &((Product *)ctx)[j];
    uint64_t i1, i2;
    const bool cross = mlwe_tensor_slot_factors(p->in1->r, k, &i1, &i2);
    struct _RNS_Polynomial o_storage, x_storage, y_storage;
    RNS_Polynomial o = row_of(&o_storage, p->tensor, k, row);
    polynomial_mul_RNS_polynomial(o, row_of(&x_storage, p->in1, i1, row),
                                  row_of(&y_storage, p->in2, i2, row));
    if (cross)
        polynomial_mul_addto_RNS_polynomial(o, row_of(&x_storage, p->in1, i2, row),
                                            row_of(&y_storage, p->in2, i1, row));
    polynomial_RNS_to_RNSc((RNSc_Polynomial)o, o);
}

static void run_products(Product *p, KeySwitch *ks, uint64_t n, uint64_t n_threads)
{
    uint64_t *masks = (uint64_t *)safe_malloc(2 * n * sizeof(uint64_t)), *parts = masks + n;
    for (uint64_t j = 0; j < n; j++)
    {
        masks[j] = 0;
        for (uint64_t k = 0; k <= p[j].in1->r; k++)
            masks[j] |= component_of(p[j].in1, k)->rns_mask | component_of(p[j].in2, k)->rns_mask;
        parts[j] = 2 * (p[j].in1->r + 1);
    }
    rns_rows_batch_for(n, masks, parts, operands_to_mul_task, p, n_threads);
    for (uint64_t j = 0; j < n; j++)
    {
        const uint64_t r = p[j].in1->r, R = p[j].tensor->r;
        p[j].by_rows = true;
        for (uint64_t k = 0; k <= r; k++)
        {
            mlwe_component(p[j].in1, k)->domain = ARITH_DOMAIN_MUL;
            mlwe_component(p[j].in2, k)->domain = ARITH_DOMAIN_MUL;
            p[j].by_rows = p[j].by_rows && component_of(p[j].in1, k)->rns_mask == p[j].mask &&
                           component_of(p[j].in2, k)->rns_mask == p[j].mask;
        }
        masks[j] = p[j].by_rows ? p[j].mask : 0;
        parts[j] = R + 1;
        if (!p[j].by_rows)
        {
            ArithElement tensor[R + 1];
            for (uint64_t k = 0; k <= R; k++)
                tensor[k] = *mlwe_component(p[j].tensor, k);
            mlwe_tensor_product(tensor, p[j].in1, p[j].in2);
            for (uint64_t k = 0; k <= R; k++)
                *mlwe_component(p[j].tensor, k) = tensor[k];
            mlwe_RNS_to_RNSc(p[j].tensor, p[j].tensor);
        }
    }
    rns_rows_batch_for(n, masks, parts, tensor_task, p, n_threads);
    for (uint64_t j = 0; j < n; j++)
    {
        for (uint64_t k = 0; p[j].by_rows && k <= p[j].tensor->r; k++)
        {
            component_of(p[j].tensor, k)->rns_mask = p[j].mask;
            mlwe_component(p[j].tensor, k)->domain = ARITH_DOMAIN_CANONICAL;
        }
    }
    free(masks);
    run_keyswitches(ks, n, n_threads);
}

static void product_sample_task(void *ctx, uint64_t j)
{
    ProductBatch *batch = (ProductBatch *)ctx;
    run_products(&batch->p[j], &batch->ks[j], 1, sample_task_threads(batch->p[j].in1->ring->N));
}

void mlwe_rns_multiply_batch(RNSc_MLWE *out, RNS_MLWE *in1, RNS_MLWE *in2, RNS_MLWE_KS_Key ksk,
                             ArithRing rescale_to, bool to_mul_domain, uint64_t n,
                             uint64_t n_threads)
{
    if (n == 0)
        return;
    Product *p = (Product *)safe_malloc(n * sizeof(Product));
    KeySwitch *ks = (KeySwitch *)safe_malloc(n * sizeof(KeySwitch));
    const uint64_t R = mlwe_extended_rank(in1[0]->r);
    for (uint64_t j = 0; j < n; j++)
    {
        p[j] = (Product){in1[j], in2[j], mlwe_alloc_sample(in1[j]->ring, R),
                         arith_rns_polynomial(&in1[j]->b)->rns_mask, false};
        ks[j] = (KeySwitch){out[j], p[j].tensor, ksk, rescale_to, to_mul_domain};
    }
    ProductBatch batch = {p, ks};
    if (one_sample_per_task(in1[0]->ring->N, n, key_rows(ksk), n_threads))
        vfhe_parallel_for(n, n_threads, product_sample_task, &batch);
    else
        run_products(p, ks, n, n_threads);
    for (uint64_t j = 0; j < n; j++)
        free_mlwe_RNS_sample(p[j].tensor);
    free(ks);
    free(p);
}

// ---- automorphisms ----------------------------------------------------------
//
// Before the key switch: an input in the mul domain is taken to the
// coefficient domain in `out` (which the key switch overwrites last), then
// permuted into `image`, both a component and a row per task.

typedef struct
{
    RNSc_MLWE in; // the input, or `out` once it holds the input in the coefficient domain
    RNSc_MLWE image;
    uint64_t gen;
    bool from_mul_domain;
} Automorphism;

typedef struct
{
    Automorphism *a;
    KeySwitch *ks;
} AutomorphismBatch;

// As mlwe_RNS_to_RNSc(out, in) on one row of component k: converted if it is
// in the mul domain, copied otherwise.
static void input_to_coeff_task(void *ctx, uint64_t j, uint64_t row, uint64_t k)
{
    const AutomorphismBatch *batch = (const AutomorphismBatch *)ctx;
    Automorphism *a = &batch->a[j];
    if (!a->from_mul_domain || !(component_of(a->in, k)->rns_mask & (1ULL << row)))
        return;
    struct _RNS_Polynomial in_storage, out_storage;
    RNS_Polynomial x = row_of(&in_storage, a->in, k, row);
    RNS_Polynomial y = row_of(&out_storage, batch->ks[j].out, k, row);
    if (mlwe_component(a->in, k)->domain == ARITH_DOMAIN_MUL)
        polynomial_RNS_to_RNSc((RNSc_Polynomial)y, x);
    else
        polynomial_copy_RNS_polynomial(y, x);
}

static void permute_task(void *ctx, uint64_t j, uint64_t row, uint64_t k)
{
    const AutomorphismBatch *batch = (const AutomorphismBatch *)ctx;
    Automorphism *a = &batch->a[j];
    RNSc_Polynomial x = component_of(a->in, k);
    if (x->rns_mask & (1ULL << row))
        polynomial_RNSc_permute_rows(component_of(a->image, k), x, a->gen, 1ULL << row);
}

static void run_automorphisms(Automorphism *a, KeySwitch *ks, uint64_t n, uint64_t n_threads)
{
    AutomorphismBatch batch = {a, ks};
    uint64_t *masks = (uint64_t *)safe_malloc(2 * n * sizeof(uint64_t)), *parts = masks + n;
    for (uint64_t j = 0; j < n; j++)
    {
        masks[j] = 0;
        parts[j] = a[j].in->r + 1;
        for (uint64_t k = 0; a[j].from_mul_domain && k <= a[j].in->r; k++)
            masks[j] |= component_of(a[j].in, k)->rns_mask;
    }
    rns_rows_batch_for(n, masks, parts, input_to_coeff_task, &batch, n_threads);
    for (uint64_t j = 0; j < n; j++)
    {
        if (a[j].from_mul_domain)
        {
            MLWE out = ks[j].out;
            for (uint64_t k = 0; k <= out->r; k++)
            {
                component_of(out, k)->rns_mask = component_of(a[j].in, k)->rns_mask;
                mlwe_component(out, k)->domain = ARITH_DOMAIN_CANONICAL;
            }
            a[j].in = out;
        }
        masks[j] = 0;
        for (uint64_t k = 0; k <= a[j].in->r; k++)
        {
            const uint64_t mask = component_of(a[j].in, k)->rns_mask;
            component_of(a[j].image, k)->rns_mask = mask;
            mlwe_component(a[j].image, k)->domain = ARITH_DOMAIN_CANONICAL;
            masks[j] |= mask;
        }
    }
    rns_rows_batch_for(n, masks, parts, permute_task, &batch, n_threads);
    free(masks);
    run_keyswitches(ks, n, n_threads);
}

static void automorphism_sample_task(void *ctx, uint64_t j)
{
    AutomorphismBatch *batch = (AutomorphismBatch *)ctx;
    run_automorphisms(&batch->a[j], &batch->ks[j], 1, sample_task_threads(batch->a[j].in->ring->N));
}

void mlwe_rns_automorphism_batch(RNSc_MLWE *out, RNSc_MLWE *in, const uint64_t *gens,
                                 RNS_MLWE_KS_Key *ksks, uint64_t n, uint64_t n_threads)
{
    if (n == 0)
        return;
    Automorphism *a = (Automorphism *)safe_malloc(n * sizeof(Automorphism));
    KeySwitch *ks = (KeySwitch *)safe_malloc(n * sizeof(KeySwitch));
    for (uint64_t j = 0; j < n; j++)
    {
        a[j] = (Automorphism){in[j], mlwe_alloc_sample(out[j]->ring, out[j]->r), gens[j],
                              mlwe_domain(in[j]) == ARITH_DOMAIN_MUL};
        ks[j] = (KeySwitch){out[j], a[j].image, ksks[j], NULL, false};
    }
    AutomorphismBatch batch = {a, ks};
    if (one_sample_per_task(out[0]->ring->N, n, key_rows(ksks[0]), n_threads))
        vfhe_parallel_for(n, n_threads, automorphism_sample_task, &batch);
    else
        run_automorphisms(a, ks, n, n_threads);
    for (uint64_t j = 0; j < n; j++)
        free_mlwe_RNS_sample(a[j].image);
    free(ks);
    free(a);
}
