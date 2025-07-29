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
#define MAX_RECTS 65536 // TODO: temporary, remove this

// rectangle used for merging calculations
typedef struct
{
    int x, y; // bottom left
    int w, h; // width, height
} LBRect;

// list of rectangles
typedef struct
{
    LBRect rects[MAX_RECTS]; // TODO: make this a dynamic linked list for more flexibility
    int count;
} LBRectList;

// helper function to merge singular indices into large rectangles
//
// this function takes as input a "map" of each index in the original index space to its corresponding task id
// i.e. the value at coordinate y * size_x + x is the task to which this index belongs to
//
// for each task T:
// |  create an empty list L_T of rectangles (ranges) for T
// |  repeat until we've exhausted this task's cells:
// |  |  find the bottom-leftmost cell belonging to T
// |  |  from there, measure how far right we can go (along the x axis) until we reach the edge / a cell which doesn't belong to T anymore (max width)
// |  |  extend this rectangle upwards, each time updating (shrinking) the weight if necessary to still encompass all cells that belong to T
// |  |  save rectangle with maximum area and add to list L_T
// |  |  mark cells as used (-1)
// |  done
// done
static void merge_rects(int *grid1D, int width, int height, LBRectList *out, int tidcount)
{
#define IDX(x, y) ((y) * width + (x))
    for (int tid = 0; tid < tidcount; ++tid)
    {
        LBRectList *rl = &out[tid];
        rl->count = 0;

        // keep carving until no more cells are found for this task
        while (1)
        {
            int found = 0, x0 = 0, y0 = 0;

            // find bottom‑leftmost cell == tid
            for (int y = 0; y < height && !found; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    if (grid1D[IDX(x, y)] == tid)
                    {
                        x0 = x;
                        y0 = y;
                        found = 1;
                        break;
                    }
                }
            }

            // couldn't find anything for this task?
            if (!found)
                break;

            // scan right to get initial width
            int w = 0;
            while (x0 + w < width && grid1D[IDX(x0 + w, y0)] == tid)
                ++w;

            // extend upward to maximize area
            int best_area = w;
            int best_w = w; // remember the width at which best_area occurs
            int best_h = 1;
            int curr_w = w;

            for (int h = 2; y0 + h <= height; ++h)
            {
                // measure run of tids in row y0 + h − 1
                int w_h = 0;
                while (x0 + w_h < width && grid1D[IDX(x0 + w_h, y0 + h - 1)] == tid)
                    ++w_h;
                if (w_h == 0)
                    break;

                if (w_h < curr_w)
                    curr_w = w_h;
                int area = curr_w * h;
                if (area > best_area)
                {
                    best_area = area;
                    best_w = curr_w;
                    best_h = h;
                }
            }

            // record the rectangle using best_w and best_h
            rl->rects[rl->count++] = (LBRect){x0, y0, best_w, best_h};

            // mark covered cells “used” by setting them to -1
            for (int dy = 0; dy < best_h; ++dy)
            {
                for (int dx = 0; dx < best_w; ++dx)
                {
                    grid1D[IDX(x0 + dx, y0 + dy)] = -1;
                }
            }
        }
    }
}

//////////////////////
// sfc partitioners //
//////////////////////

// TODO: make this work with domain sizes (sides) that are not powers of 2
//       also possibly make it work for non-square (rectangular, w != h) domains?
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

// ------------------------- //
// laik functions start here //
// ------------------------- //

void runHilbertPartitioner(Laik_RangeReceiver *r, Laik_PartitionerParams *p)
{
    Laik_Instance *inst = p->space->inst;
    laik_svg_profiler_enter(inst, __func__);

    int tidcount = p->group->size;
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

    // allocate index-to-task mapping array
    int *idxGrid = (int *)safe_malloc(N * sizeof(int));

    laik_log(1, "size %ld, totalw %f, targetw %f for %d procs\n", N, total_w, target, tidcount);

    // scan hilbert curve and partition based on prefix sum
    for (uint64_t m = 0; m < N; ++m)
    {
        uint32_t x, y;
        hilbert_d2xy(b, m, &x, &y);
        sum += get_idx_weight_2d(weights, size_x, (int64_t)x, (int64_t)y);
        idxGrid[y * size_x + x] = task;

        // target reached -> merge task cells into larger rectangles and flush buffer
        if (sum >= target)
        {
            // reset for next task
            task++;
            sum = 0.0;
        }
    }

    // decompose (destroy) idxgrid into axis-aligned rectangles
    LBRectList out[tidcount];
    merge_rects(idxGrid, size_x, size_y, out, tidcount);
    free(idxGrid);

    for (int tid = 0; tid < tidcount; ++tid)
    {
        for (int i = 0; i < out[tid].count; ++i)
        {
            LBRect *re = &out[tid].rects[i];
            Laik_Range ra = {.space = space,
                             .from = {{re->x, re->y, 0}},
                             .to = {{(re->x) + (re->w), (re->y) + (re->h), 0}}};
            laik_append_range(r, tid, &ra, 0, 0);
        }
    }

    laik_svg_profiler_exit(inst, __func__);
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
