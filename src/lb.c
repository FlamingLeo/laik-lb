// load balancing API / workflow example
#include <laik-internal.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

//////////////////////
// helper functions //
//////////////////////

// difference between the minimum and maximum of the times taken by each task
static double min_max_difference(double *weights, int64_t size, int gsize)
{
    assert(size > 0);
    double max = weights[size];
    double min = weights[size];
    for (int64_t i = size + 1; i < size + gsize; i++)
    {
        if (weights[i] > max)
            max = weights[i];
        if (weights[i] < min)
            min = weights[i];
    }
    return max - min;
}

// print times (elements in weight array starting at offset size) since last rebalance
static void print_times(double *weights, int64_t size, int gsize)
{
    printf("[LB] times in s since last rebalance: [");
    for (int64_t i = size; i < size + gsize; ++i)
    {
        printf("%.2f", weights[i]);
        if (i < size + gsize - 1)
            printf(", ");
    }
    printf("], max dt: %.2f\n", min_max_difference(weights, size, gsize));
}

// check if a number is a power of 2
static bool is_power_of_two(int64_t x)
{
    return (x != 0) && ((x & (x - 1)) == 0);
}

// safely allocate memory or panic on failure
static void *safe_malloc(size_t n)
{
    void *p = malloc(n);
    if (!p)
    {
        laik_panic("Could not allocate enough memory!");
        exit(EXIT_FAILURE);
    }
    return p;
}

// public: get algorithm string from enum
const char *laik_get_lb_algorithm_name(Laik_LBAlgorithm algo)
{
    switch (algo)
    {
    case LB_RCB:
        return "rcb";
    case LB_HILBERT:
        return "hilbert";
    case LB_MORTON:
        return "morton";
    default:
        return "unknown";
    }
}

///////////////////
// range weights //
///////////////////

// get range weight from index (1d)
static double get_idx_weight_1d(double *weights, int64_t idx)
{
    return weights[idx];
}

// get range weight from index (2d)
static double get_idx_weight_2d(double *weights, int64_t size_x, int64_t x, int64_t y)
{
    return weights[y * size_x + x];
}

///////////////////
// merging cells //
///////////////////

/* local helper structs for merging 2D blocks */
// a single 2d point defined by its coordinates
typedef struct
{
    uint32_t x, y;
} LBPoint;

// a 2d rectangle defined by its corner points (one could also use two point structs here, but this seems easier to work with)
typedef struct
{
    uint32_t x1, x2, y1, y2;
} LBRect;

// a horizontal run along a row (same y-axis)
// defined by the row we're on and the start and end x coordinates
typedef struct
{
    uint32_t y, x1, x2;
} LBHorizontalRun;

/* comparators for each scenario (points, runs, rectangles) */
// (y, x)
static int cmp_pt(const void *a, const void *b)
{
    const LBPoint *p = a, *q = b;
    if (p->y != q->y)
        return (p->y < q->y ? -1 : 1);
    return (p->x < q->x ? -1 : p->x > q->x);
}

// (y, x1, x2)
static int cmp_run(const void *a, const void *b)
{
    const LBHorizontalRun *r = a, *s = b;
    if (r->y != s->y)
        return (r->y < s->y ? -1 : 1);
    if (r->x1 != s->x1)
        return (r->x1 < s->x1 ? -1 : 1);
    return (r->x2 < s->x2 ? -1 : r->x2 > s->x2);
}

// (x1, x2)
static int cmp_rect(const void *a, const void *b)
{
    const LBRect *r = a, *s = b;
    if (r->x1 != s->x1)
        return (r->x1 < s->x1 ? -1 : 1);
    if (r->x2 != s->x2)
        return (r->x2 < s->x2 ? -1 : 1);
    return 0;
}

// helper function to merge a set of pixels into the minimal covering rectangles
//
// this works by traversing each row, building the horizontal runs from one end to the other,
// then "stitching" these vertically only if each point has another point above itself
//
// this is basically necessary when working with sfc algorithms, because having one range per index is insanely time-expensive
// the other way would be some sort of bounding-box algorithm over each region, which might be simpler, but sacrifices some precision
//
// this approach is still not perfect, running in O(n log n) due to sorts and some possibly overcomplicated logic towards the end
static void merge_cells_into_rects(const LBPoint *cells, size_t n, LBRect **out_rects, size_t *out_count)
{
    if (n == 0)
    {
        *out_rects = NULL;
        *out_count = 0;
        return;
    }

    // 1. sort points by (y,x)
    LBPoint *pts = (LBPoint *)safe_malloc(n * sizeof(LBPoint));
    memcpy(pts, cells, n * sizeof(LBPoint));
    qsort(pts, n, sizeof(LBPoint), cmp_pt);

    // 2. build horizontal runs
    LBHorizontalRun *runs = (LBHorizontalRun *)safe_malloc(n * sizeof(LBHorizontalRun));
    size_t nruns = 0;

    uint32_t run_y = pts[0].y;
    uint32_t run_x0 = pts[0].x;
    uint32_t prev_x = pts[0].x;

    // pass over sorted points
    for (size_t i = 1; i < n; ++i)
    {
        // if the row changes OR the next x is not exactly one past the previous (should always be the case?),
        // close the current run and start a new one
        if (pts[i].y != run_y || pts[i].x != prev_x + 1)
        {
            // end previous run
            runs[nruns++] = (LBHorizontalRun){run_y, run_x0, prev_x + 1};

            // start a new run
            run_y = pts[i].y;
            run_x0 = pts[i].x;
        }
        prev_x = pts[i].x;
    }

    // finish the last run and free point buffer
    runs[nruns++] = (LBHorizontalRun){run_y, run_x0, prev_x + 1};
    free(pts);

    // 3. sort runs by (y,x1,x2)
    qsort(runs, nruns, sizeof(LBHorizontalRun), cmp_run);

    // prepare active‐rect and output arrays for vertical stitching
    LBRect *active = (LBRect *)safe_malloc(nruns * sizeof(LBRect));      // set of rectangles to extend downwards
    LBRect *next_active = (LBRect *)safe_malloc(nruns * sizeof(LBRect)); // temp. buffer for next row's active set
    LBRect *output = (LBRect *)safe_malloc(nruns * sizeof(LBRect));      // accumulates finished rectangles
    size_t act_cnt = 0;
    size_t out_cnt = 0;

    // 4. perform vertical stitching
    size_t i = 0;
    while (i < nruns)
    {
        uint32_t y = runs[i].y;
        size_t start_i = i;

        // collect all runs at this y
        while (i < nruns && runs[i].y == y)
            ++i;
        size_t end_i = i;

        // sort active rects by (x1,x2) each row
        qsort(active, act_cnt, sizeof(LBRect), cmp_rect);

        // pointers into runs[start_i..end_i) and active[0..act_cnt)
        size_t r = start_i;
        size_t a = 0;
        size_t na = 0;

        // match or create
        while (r < end_i && a < act_cnt)
        {
            uint32_t rx1 = runs[r].x1;
            uint32_t rx2 = runs[r].x2;
            uint32_t ax1 = active[a].x1;
            uint32_t ax2 = active[a].x2;

            if (rx1 == ax1 && rx2 == ax2)
            {
                // extend active rectangle downward
                active[a].y2 = y + 1;
                next_active[na++] = active[a];
                ++r;
                ++a;
            }
            else if (ax1 < rx1 || (ax1 == rx1 && ax2 < rx2))
            {
                // active[a] has no matching run, so close it (move to output)
                output[out_cnt++] = active[a];
                ++a;
            }
            else
            {
                // go to new run at this row
                next_active[na++] = (LBRect){rx1, rx2, y, y + 1};
                ++r;
            }
        }

        // leftover active -> close
        while (a < act_cnt)
            output[out_cnt++] = active[a++];

        // leftover runs -> new rects
        while (r < end_i)
            next_active[na++] = (LBRect){runs[r].x1, runs[r].x2, y, y + 1}, ++r;

        // swap active & next_active
        memcpy(active, next_active, na * sizeof(LBRect));
        act_cnt = na;
    }

    // 5. close any still‐active rects (grown as far down as possible)
    for (size_t a = 0; a < act_cnt; ++a)
        output[out_cnt++] = active[a];

    // free memory and return out parameters
    free(runs);
    free(active);
    free(next_active);

    *out_rects = output;
    *out_count = out_cnt;
}

//////////////////////
// sfc partitioners //
//////////////////////

// TODO: make these work with domain sizes (sides) that are not powers of 2
//       also possibly make them work for non-square (rectangular, w != h) domains?
// TODO: 3d

// hilbert space-filling curve
// source: "Programming the Hilbert curve" by John Skilling. AIP Conference Proceedings 707, 381 (2004); https://doi.org/10.1063/1.1751381
static inline void hilbert_rotate(uint32_t s, uint32_t *x, uint32_t *y, uint32_t rx, uint32_t ry)
{
    // rotate as needed
    if (ry == 0)
    {
        if (rx == 1)
        {
            *x = s - 1 - *x;
            *y = s - 1 - *y;
        }

        // swap
        uint32_t t = *x;
        *x = *y;
        *y = t;
    }
}

// d -> (x,y) on a (2^b * 2^b) hilbert curve
static inline void hilbert_d2xy(int b, uint32_t d, uint32_t *x, uint32_t *y)
{
    *x = *y = 0;
    for (uint32_t s = 1; s < (1u << b); s <<= 1)
    {
        uint32_t rx = (d >> 1) & 1;
        uint32_t ry = (d ^ rx) & 1;
        hilbert_rotate(s, x, y, rx, ry);
        *x += s * rx;
        *y += s * ry;
        d >>= 2;
    }
}

// morton (z) space-filling curve
static inline uint32_t morton2D_compact(uint64_t x)
{
    x &= 0x5555555555555555ULL;
    x = (x ^ (x >> 1)) & 0x3333333333333333ULL;
    x = (x ^ (x >> 2)) & 0x0F0F0F0F0F0F0F0FULL;
    x = (x ^ (x >> 4)) & 0x00FF00FF00FF00FFULL;
    x = (x ^ (x >> 8)) & 0x0000FFFF0000FFFFULL;
    x = (x ^ (x >> 16)) & 0x00000000FFFFFFFFULL;
    return (uint32_t)x;
}

// d -> (x,y) on a (2^b * 2^b) morton curve
static inline void morton2D_d2xy(uint64_t d, uint32_t *x, uint32_t *y)
{
    *x = morton2D_compact(d);
    *y = morton2D_compact(d >> 1);
}

// ------------------------- //
// laik functions start here //
// ------------------------- //

void runMortonPartitioner(Laik_RangeReceiver *r, Laik_PartitionerParams *p)
{
    Laik_Instance *inst = p->space->inst;
    laik_svg_profiler_enter(inst, __func__);

    unsigned tidcount = p->group->size;
    int dims = p->space->dims;
    double *weights = (double *)p->partitioner->data;

    assert(dims == 2); // TODO: remove once 3d is supported

    Laik_Space *space = p->space;
    Laik_Range range = space->range;

    // validate square domain with side as a power of 2
    int64_t size_x = range.to.i[0] - range.from.i[0];
    int64_t size_y = range.to.i[1] - range.from.i[1];
    assert(size_x > 0 && size_y > 0 && size_x == size_y && is_power_of_two(size_x) && is_power_of_two(size_y));

    // compute total weight
    uint64_t N = size_x * size_y;
    double total_w = 0.0;
    for (uint64_t m = 0; m < N; ++m)
    {
        uint32_t x, y;
        morton2D_d2xy(m, &x, &y);
        total_w += get_idx_weight_2d(weights, size_x, (int64_t)x, (int64_t)y);
    }

    // define variables for distributing chunks evenly based on weight across all tasks
    double target = total_w / (double)tidcount;
    int task = 0;
    double sum = 0.0;

    // dynamic array for the current task's cells
    size_t cells_cap = 1024, cells_cnt = 0;
    LBPoint *cells = (LBPoint *)safe_malloc(cells_cap * sizeof *cells);

    laik_log(1, "size %ld, totalw %f, targetw %f for %d procs\n", N, total_w, target, tidcount);

    // scan morton curve and partition based on prefix sum
    for (uint64_t m = 0; m < N; ++m)
    {
        uint32_t x, y;
        morton2D_d2xy(m, &x, &y);
        sum += get_idx_weight_2d(weights, size_x, (int64_t)x, (int64_t)y);

        // store current point in task cell buffer and possibly grow dynamic task cell array
        if (cells_cnt == cells_cap)
        {
            cells_cap *= 2;
            cells = realloc(cells, cells_cap * sizeof *cells);
        }
        cells[cells_cnt++] = (LBPoint){x, y};

        // target reached -> merge task cells into larger rectangles and flush buffer
        if (sum >= target)
        {
            LBRect *rects;
            size_t nrects;
            merge_cells_into_rects(cells, cells_cnt, &rects, &nrects);

            // append each merged rect
            for (size_t i = 0; i < nrects; ++i)
            {
                Laik_Range big = {
                    .space = space,
                    .from = {{rects[i].x1, rects[i].y1, 0}},
                    .to = {{rects[i].x2, rects[i].y2, 0}}};

                laik_append_range(r, task, &big, 0, 0);
            }
            free(rects);

            // reset for next task
            task++;
            sum = 0.0;
            cells_cnt = 0;
        }
    }

    // flush and append leftover indices (final task)
    if (cells_cnt > 0)
    {
        LBRect *rects;
        size_t nrects;
        merge_cells_into_rects(cells, cells_cnt, &rects, &nrects);
        for (size_t i = 0; i < nrects; ++i)
        {
            Laik_Range big = {
                .space = space,
                .from = {{rects[i].x1, rects[i].y1, 0}},
                .to = {{rects[i].x2, rects[i].y2, 0}}};

            laik_append_range(r, task, &big, 0, 0);
        }
        free(rects);
    }

    free(cells);
    laik_svg_profiler_exit(inst, __func__);
}

void runHilbertPartitioner(Laik_RangeReceiver *r, Laik_PartitionerParams *p)
{
    Laik_Instance *inst = p->space->inst;
    laik_svg_profiler_enter(inst, __func__);

    unsigned tidcount = p->group->size;
    int dims = p->space->dims;
    double *weights = (double *)p->partitioner->data;

    assert(dims == 2); // TODO: remove once 3d is supported

    Laik_Space *space = p->space;
    Laik_Range range = space->range;

    // validate square domain with side as a power of 2
    int64_t size_x = range.to.i[0] - range.from.i[0];
    int64_t size_y = range.to.i[1] - range.from.i[1];
    assert(size_x > 0 && size_y > 0 && size_x == size_y && is_power_of_two(size_x) && is_power_of_two(size_y));

    // compute total weight
    int b = (int)log2(size_x); // side length 2^b
    uint64_t N = size_x * size_y;
    double total_w = 0.0;
    for (uint64_t m = 0; m < N; ++m)
    {
        uint32_t x, y;
        hilbert_d2xy(b, m, &x, &y);
        total_w += get_idx_weight_2d(weights, size_x, (int64_t)x, (int64_t)y);
    }

    // define variables for distributing chunks evenly based on weight across all tasks
    double target = total_w / (double)tidcount;
    int task = 0;
    double sum = 0.0;

    // dynamic array for the current task's cells
    size_t cells_cap = 1024, cells_cnt = 0;
    LBPoint *cells = (LBPoint *)safe_malloc(cells_cap * sizeof *cells);

    laik_log(1, "size %ld, totalw %f, targetw %f for %d procs\n", N, total_w, target, tidcount);

    // scan hilbert curve and partition based on prefix sum
    for (uint64_t m = 0; m < N; ++m)
    {
        uint32_t x, y;
        hilbert_d2xy(b, m, &x, &y);
        sum += get_idx_weight_2d(weights, size_x, (int64_t)x, (int64_t)y);

        // store current point in task cell buffer and possibly grow dynamic task cell array
        if (cells_cnt == cells_cap)
        {
            cells_cap *= 2;
            cells = realloc(cells, cells_cap * sizeof *cells);
        }
        cells[cells_cnt++] = (LBPoint){x, y};

        // target reached -> merge task cells into larger rectangles and flush buffer
        if (sum >= target)
        {
            LBRect *rects;
            size_t nrects;
            merge_cells_into_rects(cells, cells_cnt, &rects, &nrects);

            // append each merged rect
            for (size_t i = 0; i < nrects; ++i)
            {
                Laik_Range big = {
                    .space = space,
                    .from = {{rects[i].x1, rects[i].y1, 0}},
                    .to = {{rects[i].x2, rects[i].y2, 0}}};

                laik_append_range(r, task, &big, 0, 0);
            }
            free(rects);

            // reset for next task
            task++;
            sum = 0.0;
            cells_cnt = 0;
        }
    }

    // flush and append leftover indices (final task)
    if (cells_cnt > 0)
    {
        LBRect *rects;
        size_t nrects;
        merge_cells_into_rects(cells, cells_cnt, &rects, &nrects);
        for (size_t i = 0; i < nrects; ++i)
        {
            Laik_Range big = {
                .space = space,
                .from = {{rects[i].x1, rects[i].y1, 0}},
                .to = {{rects[i].x2, rects[i].y2, 0}}};

            laik_append_range(r, task, &big, 0, 0);
        }
        free(rects);
    }

    free(cells);
    laik_svg_profiler_exit(inst, __func__);
}

Laik_Partitioner *laik_new_morton_partitioner(double *weights)
{
    return laik_new_partitioner("morton", runMortonPartitioner, (void *)weights, 0);
}

Laik_Partitioner *laik_new_hilbert_partitioner(double *weights)
{
    return laik_new_partitioner("hilbert", runHilbertPartitioner, (void *)weights, 0);
}

/////////////////////
// rcb partitioner //
/////////////////////

// TODO: 3d

// internal 1d rcb helper function; [fromTask - toTask)
//
// note: like the range weight function above, i've separated this into a 1d and 2d version
//       partially due to recursion, mainly due to differences in the algorithm to avoid cluttering the function with ifs
static void rcb_1d(Laik_RangeReceiver *r, Laik_Range *range, int fromTask, int toTask, double *weights)
{
    Laik_Instance *inst = r->params->space->inst;
    laik_svg_profiler_enter(inst, __func__);

    int64_t from = range->from.i[0];
    int64_t to = range->to.i[0];

    // if there's only one processor left, stop here
    int count = toTask - fromTask + 1;
    if (count == 1 || from > to)
    {
        laik_append_range(r, fromTask, range, 0, 0);
        laik_svg_profiler_exit(inst, __func__);
        return;
    }

    // calculate how many procs go left vs. right
    int lcount = count / 2;
    int rcount = count - lcount;
    int tmid = fromTask + lcount - 1;

    // calculate sum of weights in current range (naive)
    double totalW = 0.0;
    for (int64_t i = from; i < to; ++i)
        totalW += get_idx_weight_1d(weights, i);

    // calculate target weight of left child
    double ltarget = totalW * ((double)lcount / count);

    laik_log(1, "[rcb1d] [T%d-T%d) [%ld-%ld) count: %d, lcount: %d, rcount: %d, tmid: %d, totalW: %f, ltarget: %f\n", fromTask, toTask, from, to, count, lcount, rcount, tmid, totalW, ltarget);

    // find first index where prefix sum exceeds target weight
    double sum = 0.0;
    int64_t split = from;
    for (int64_t i = from; i < to; ++i)
    {
        sum += get_idx_weight_1d(weights, i);
        if (sum >= ltarget)
        {
            split = i;
            break;
        }
    }

    // cut and recurse
    Laik_Range r1 = *range, r2 = *range;
    r1.to.i[0] = split;
    r2.from.i[0] = split;
    rcb_1d(r, &r1, fromTask, tmid, weights);
    rcb_1d(r, &r2, tmid + 1, toTask, weights);
    laik_svg_profiler_exit(inst, __func__);
}

// internal 2d rcb helper function; [fromTask - toTask)
//
// primary changes are determining the split direction by checking which side is longest and how the weights are computed / accumulated
static void rcb_2d(Laik_RangeReceiver *r, Laik_Range *range, int fromTask, int toTask, double *weights)
{
    Laik_Instance *inst = r->params->space->inst;
    laik_svg_profiler_enter(inst, __func__);

    int64_t size_x = range->space->range.to.i[0] - range->space->range.from.i[0];
    int64_t from_x = range->from.i[0];
    int64_t from_y = range->from.i[1];
    int64_t to_x = range->to.i[0];
    int64_t to_y = range->to.i[1];

    // if there's only one processor left, stop here
    int count = toTask - fromTask + 1;
    if (count == 1)
    {
        laik_append_range(r, fromTask, range, 0, 0);
        laik_svg_profiler_exit(inst, __func__);
        return;
    }

    // choose split axis (by longest side of region)
    int64_t dx = to_x - from_x;
    int64_t dy = to_y - from_y;
    int axis = dy > dx; // 0: x larger -> split vertically  ; accumulate weights column-wise (left to right)
                        // 1: y larger -> split horizontally; accumulate weights row-wise (bottom to top)

    // return if width is 1
    int64_t width = axis ? dy : dx;
    if (width == 1)
    {
        laik_append_range(r, fromTask, range, 0, 0);
        laik_svg_profiler_exit(inst, __func__);
        return;
    }

    // calculate how many procs go left vs. right
    int lcount = count / 2;
    int rcount = count - lcount;
    int tmid = fromTask + lcount - 1;

    // calculate sum of weights (naive) and target weight
    double totalW = 0.0;
    for (int64_t i = from_x; i < to_x; ++i)
        for (int64_t j = from_y; j < to_y; ++j)
            totalW += get_idx_weight_2d(weights, size_x, i, j);

    double ltarget = totalW * ((double)lcount / count);

    laik_log(1, "[rcb2d] [T%d-T%d) [%ld, %ld] -> (%ld, %ld), count: %d, lcount: %d, rcount: %d, tmid: %d, totalW: %f, ltarget: %f, axis: %d\n", fromTask, toTask, from_x, from_y, to_x, to_y, count, lcount, rcount, tmid, totalW, ltarget, axis);

    // accumulate weights along splitting axis and find first index where prefix sum exceeds target weight
    double sum = 0.0;
    int64_t split_x = from_x;
    int64_t split_y = from_y;

    if (axis)
    {
        // horizontal split
        for (int64_t y = from_y; y < to_y; ++y)
        {
            // gather sum of all weights in current row (same y)
            for (int64_t x = from_x; x < to_x; ++x)
            {
                sum += get_idx_weight_2d(weights, size_x, x, y);
            }

            // check if sum exceeds weight target, otherwise continue
            if (sum >= ltarget)
            {
                split_y = y;
                laik_log(1, "[rcb2d] found horizontal split at y = %ld (sum: %f)\n", y, sum);
                break;
            }
        }
    }
    else
    {
        // vertical split
        for (int64_t x = from_x; x < to_x; ++x)
        {
            // gather sum of all weights in current column (same x)
            for (int64_t y = from_y; y < to_y; ++y)
            {
                sum += get_idx_weight_2d(weights, size_x, x, y);
            }

            // check if sum exceeds weight target, otherwise continue
            if (sum >= ltarget)
            {
                split_x = x;
                laik_log(1, "[rcb2d] found vertical split at x = %ld (sum: %f)\n", x, sum);
                break;
            }
        }
    }

    // cut and recurse
    Laik_Range r1 = *range, r2 = *range;
    if (axis)
    {
        r1.to.i[1] = split_y;
        r2.from.i[1] = split_y;
    }
    else
    {
        r1.to.i[0] = split_x;
        r2.from.i[0] = split_x;
    }
    laik_log(1, "[rcb2d] split_x: %ld, split_y : %ld, r1: [%ld, %ld] -> (%ld, %ld); r2: [%ld, %ld] -> (%ld, %ld)\n", split_x, split_y, r1.from.i[0], r1.from.i[1], r1.to.i[0], r1.to.i[1], r2.from.i[0], r2.from.i[1], r2.to.i[0], r2.to.i[1]);
    rcb_2d(r, &r1, fromTask, tmid, weights);
    rcb_2d(r, &r2, tmid + 1, toTask, weights);
    laik_svg_profiler_exit(inst, __func__);
}

// ------------------------- //
// laik functions start here //
// ------------------------- //

void runRCBPartitioner(Laik_RangeReceiver *r, Laik_PartitionerParams *p)
{
    Laik_Instance *inst = p->space->inst;
    laik_svg_profiler_enter(inst, __func__);

    unsigned tidcount = p->group->size;
    int dims = p->space->dims;
    double *weights = (double *)p->partitioner->data;

    Laik_Space *s = p->space;
    Laik_Range range = s->range;

    if (dims == 1)
        rcb_1d(r, &range, 0, tidcount - 1, weights);
    else if (dims == 2)
        rcb_2d(r, &range, 0, tidcount - 1, weights);

    laik_svg_profiler_exit(inst, __func__);
}

Laik_Partitioner *laik_new_rcb_partitioner(double *weights)
{
    return laik_new_partitioner("rcb", runRCBPartitioner, (void *)weights, 0);
}

////////////////////
// load balancing //
////////////////////

double *laik_lb_measure(Laik_Partitioning *p, double ttime)
{
    double *weights;

    Laik_Space *space = p->space;
    Laik_Group *group = p->group;
    Laik_Instance *inst = group->inst;
    laik_svg_profiler_enter(inst, __func__);

    // allocate weight array and zero-initialize
    // for t tasks, the final t elements of the array are the raw times taken by each task (starting from 0, one after the last weight)
    int dims = p->space->dims;
    int task = laik_myid(group);
    int gsize = group->size;
    int64_t size_x = space->range.to.i[0] - space->range.from.i[0];
    int64_t size_y = dims >= 2 ? (space->range.to.i[1] - space->range.from.i[1]) : 1;
    int64_t size_z = dims >= 3 ? (space->range.to.i[2] - space->range.from.i[2]) : 1;
    int64_t size = size_x * size_y * size_z;

    weights = (double *)safe_malloc(sizeof(double) * (size /* elements in index space */ + gsize /* number of tasks */));
    memset(weights, 0, sizeof(double) * (size + gsize));

    // store time taken by own task
    weights[size + task] = ttime;

    // calculate weight and fill array at own indices
    // 1. accumulate number of items
    // 2. get task weight for task i as time_taken(i) * c / nitems(i), where c is some constant
    int tnitems = 0;
    int c = 1000000;
    for (int r = 0; r < laik_my_rangecount(p); ++r)
    {
        if (dims == 1)
        {
            int64_t from, to;
            laik_my_range_1d(p, r, &from, &to);
            tnitems += to - from;
        }
        else if (dims == 2)
        {
            int64_t from_x, from_y, to_x, to_y;
            laik_my_range_2d(p, r, &from_x, &to_x, &from_y, &to_y);
            int64_t count = (to_x - from_x) * (to_y - from_y);
            tnitems += count;
        }
    };

    // broadcast weight to own indices
    double tweight = (ttime * (double)c) / (double)tnitems;
    for (int r = 0; r < laik_my_rangecount(p); ++r)
    {
        if (dims == 1)
        {
            int64_t from, to;
            laik_my_range_1d(p, r, &from, &to);
            for (int64_t i = from; i < to; ++i)
            {
                weights[i] = tweight;
            }
        }
        else if (dims == 2)
        {
            int64_t from_x, from_y, to_x, to_y;
            laik_my_range_2d(p, r, &from_x, &to_x, &from_y, &to_y);
            for (int64_t x = from_x; x < to_x; ++x)
            {
                for (int64_t y = from_y; y < to_y; ++y)
                {
                    weights[y * size_x + x] = tweight;
                }
            }
        }
    }

    laik_log(1, "took %fs, nitems: %d, weight %f\n", ttime, tnitems, tweight);

    // initialize laik space for aggregating weights
    Laik_Space *weightspace;
    Laik_Data *weightdata;
    Laik_Partitioning *weightpart1, *weightpart2;

    // use weights directly as input data
    weightspace = laik_new_space_1d(inst, size + gsize);
    weightdata = laik_new_data(weightspace, laik_Double);
    weightpart1 = laik_new_partitioning(laik_All, group, weightspace, NULL);
    laik_data_provide_memory(weightdata, weights, (size + gsize) * sizeof(double));
    laik_set_initial_partitioning(weightdata, weightpart1);

    // collect times into weights, shared among all tasks
    weightpart2 = laik_new_partitioning(laik_All, group, weightspace, NULL);
    laik_switchto_partitioning(weightdata, weightpart2, LAIK_DF_Preserve, LAIK_RO_Sum);

    // print time taken by each task
    if (task == 0)
    {
        print_times(weights, size, gsize);
    }
    laik_svg_profiler_exit(inst, __func__);
    return weights;
}

Laik_Partitioning *laik_lb_balance(Laik_LBState state, Laik_Partitioning *partitioning, Laik_LBAlgorithm algorithm /*, double threshold*/)
{
    static double time = 0;

    // when starting a new load balancing segment, start timer and do nothing else
    if (state == START_LB_SEGMENT)
    {
        time = laik_wtime();
        return NULL;
    }

    // otherwise, stop timer and perform load balancing
    Laik_Instance *inst = partitioning->group->inst;
    laik_svg_profiler_enter(inst, __func__);

    Laik_Space *space = partitioning->space;
    Laik_Group *group = partitioning->group;

    // collect weights associated for each task
    double *weights = laik_lb_measure(partitioning, laik_wtime() - time);

    // if the balancing function is called for the first time, return the partitioning unchanged
    if (!weights)
    {
        // might not even be visible?
        laik_svg_profiler_exit(inst, __func__);
        return partitioning;
    }

    // use task weights to create new partitioning
    assert(weights != NULL);
    Laik_Partitioner *nparter;

    // choose load balancing algorithm based on input
    switch (algorithm)
    {
    case LB_RCB:
        nparter = laik_new_rcb_partitioner(weights);
        break;
    case LB_MORTON:
        nparter = laik_new_morton_partitioner(weights);
        break;
    case LB_HILBERT:
        nparter = laik_new_hilbert_partitioner(weights);
        break;
    default:
        laik_panic("Unknown / unimplemented load balancing algorithm!");
        exit(EXIT_FAILURE);
    }

    // create new partitioning to return and free weight array
    Laik_Partitioning *npart = laik_new_partitioning(nparter, group, space, partitioning);
    free(weights);

    laik_svg_profiler_exit(inst, __func__);
    return npart;
}
