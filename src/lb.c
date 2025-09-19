// load balancing API / workflow example
#include <laik-internal.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define IDX2D(x, y) ((y) * width + (x))
#define IDX3D(x, y, z) ((x) + width * ((y) + height * (z)))

// helper lookup table for fast log2 for powers of 2
static const int log2_tab[64] = {
    63, 0, 58, 1, 59, 47, 53, 2,
    60, 39, 48, 27, 54, 33, 42, 3,
    61, 51, 37, 40, 49, 18, 28, 20,
    55, 30, 34, 11, 43, 14, 22, 4,
    62, 57, 46, 52, 38, 26, 32, 41,
    50, 36, 17, 19, 29, 10, 13, 21,
    56, 45, 25, 31, 35, 16, 9, 12,
    44, 24, 15, 8, 23, 7, 6, 5};

// helper struct for SFC partitioner: algorithm + weights
typedef struct
{
    double *weights;
    Laik_LBAlgorithm algo;
} LB_SFC_Data;

///////////////////////
// space weight defs //
///////////////////////

// helper struct for storing weight arrays for an individual space
//
// this is needed to not keep freeing and (re)allocating weight memory on every call
// while simultaneously supporting load balancing for multiple spaces (partitionings) in one program
typedef struct LB_SpaceWeightList
{
    int id;                          // space id
    double *weights;                 // weight array corresponding to that space (count not needed, since it's just the total number of elements in that space)
    double *prev_weights;            // weights from last run, e.g. for smoothing
    Laik_Space *weightspace;         // laik space associated with weight array, used in weight initialization
    Laik_Data *weightdata;           // laik data container, used in weight initialization
    Laik_Partitioning *weightpart;   // laik space partitioning, used in weight initialization
    struct LB_SpaceWeightList *next; // next element in list
} LB_SpaceWeightList;

static LB_SpaceWeightList *swlist = NULL; // space weight list (space id + associated weight array)
static double *ext_weights = NULL;        // custom weight array
static Laik_Timer timer;                  // timer used for load balancing (zero initialized)
static bool do_print_times = true;        // print out times? (default: yes)

///////////////////////////
// load balancing tuning //
///////////////////////////

// load balancing start / stop parameters
static int p_stop = 3;       // stopping patience
static int p_start = 3;      // starting patience
static double t_stop = 0.05; // stop load balancing when relative imbalance is UNDER this threshold for p_stop consecutive times
static double t_start = 0.1; // restart load balancing when relative imbalance is OVER this threshold for p_start consecutive times

// do NOT change these!
static bool stopped = false; // is load balancing active right now?
static int p_stopctr = 0;    // consecutive counter for potential stopping
static int p_startctr = 0;   // consecutive counter for potential restarting
static int segment = 0;      // load balancing segment, for debugging purposes

// EMA parameters
static bool do_smoothing = false; // do EMA?
static double alpha_mul = 0.15;   // smoothing factor in multiplicative domain (0..1)
static double rmin = 0.7;         // minimum allowed multiplicative change per step
static double rmax = 1.25;        // maximum allowed multiplicative change per step

/////////////
// utility //
/////////////

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

// calculate the difference between the minimum and maximum of the times taken by each task and the mean
//
// this is used for determining whether to start or stop load balancing
static void min_max_mean(double *times, int gsize, double *maxdt, double *mean)
{
    double min = times[0], max = times[0], sum = times[0];
    for (int64_t i = 1; i < gsize; i++)
    {
        if (times[i] > max)
            max = times[i];
        if (times[i] < min)
            min = times[i];
        sum += times[i];
    }
    if (maxdt)
        *maxdt = max - min;
    if (mean)
        *mean = sum / (double)gsize;
}

// print information since last rebalance
static void print_times(double *times, int gsize, double maxdt, double mean)
{
    printf("[LAIK-LB] times in s for this segment: [");
    for (int i = 0; i < gsize; ++i)
    {
        printf("%.2f", times[i]);
        if (i < gsize - 1)
            printf(", ");
    }
    printf("], max dt: %.2f, mean %.2f, rel. imbalance %.2f (stopped: %d)\n", maxdt, mean, maxdt / mean, stopped);
}

// check if a number is a power of 2
static inline bool is_power_of_two(int64_t x) { return (x != 0) && ((x & (x - 1)) == 0); }

// get the sign of a 64-bit integer
static inline int64_t sgn(int64_t num) { return (num > 0) - (num < 0); }

// compute log2 for a 64-bit integer power of 2 using bitops and a LUT (cross-platform, compiler-/intrinsic-independent)
// source: https://stackoverflow.com/questions/11376288/fast-computing-of-log2-for-64-bit-integers
static inline int log2_int(uint64_t value)
{
    assert(is_power_of_two(value));

    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value |= value >> 32;
    return log2_tab[((uint64_t)((value - (value >> 1)) * 0x07EDD5E59A4E28C2)) >> 58];
}

// 1d index for 2d array mapped to 1d
static inline int64_t idx2d(int64_t x, int64_t y, int64_t width) { return y * width + x; }

// 1d index for 3d array mapped to 1d
static inline int64_t idx3d_local(int64_t x, int64_t y, int64_t z, int64_t width, int64_t height) { return (z * height + y) * width + x; }

// difference in numbere of elements between two rangelists (how many indices differ in task ids between two rangelists?)
uint64_t laik_rangelist_diff_bytes(const Laik_RangeList *rl_from, const Laik_RangeList *rl_to)
{
    assert(rl_from != NULL);
    assert(rl_to != NULL);

    uint64_t bytes = 0;

    for (unsigned i = 0; i < rl_from->count; i++)
    {
        const Laik_TaskRange_Gen *tr1 = &rl_from->trange[i];
        const Laik_Range *r1 = &tr1->range;
        int task1 = tr1->task;

        for (unsigned j = 0; j < rl_to->count; j++)
        {
            const Laik_TaskRange_Gen *tr2 = &rl_to->trange[j];
            const Laik_Range *r2 = &tr2->range;
            int task2 = tr2->task;

            Laik_Range *inter = laik_range_intersect(r1, r2);
            if (!inter)
                continue;
            if (laik_range_isEmpty(inter))
                continue;

            if (task1 != task2)
            {
                uint64_t count = laik_range_size(inter);
                bytes += count;
            }
        }
    }

    return bytes;
}

/////////////////////
// incremental rcb //
/////////////////////

// helper struct for incremental RCB partitioner
// linked list type of second-from-bottom-layer ranges and their associated task pairs for incremental rcb
// note: this could be computed using the other partitioning but this is easier
typedef struct LB_RCB_SBL
{
    int from, to;     // the task pair
    Laik_Range range; // the second-layer range
    struct LB_RCB_SBL *next;
} LB_RCB_SBL;

// TODO (med): allow incremental rcb for multiple spaces / partitionings simultaneously
static LB_RCB_SBL *sbl_parents = NULL; // the linked list itself
static bool sbl_recompute = true;      // should the list be reinitalized (second layer info pushed)?

// add a range to the second-from-bottom-layer range linked list
static void rcb_sl_push(int from, int to, const Laik_Range *range)
{
    LB_RCB_SBL *n = (LB_RCB_SBL *)safe_malloc(sizeof(LB_RCB_SBL));
    n->from = from;
    n->to = to;
    n->range = *range; // struct copy
    n->next = NULL;

    if (!sbl_parents)
        sbl_parents = n;
    else
    {
        LB_RCB_SBL *cur = sbl_parents;
        while (cur->next)
            cur = cur->next;
        cur->next = n;
    }
}

// clear linked list
static void rcb_sl_clear()
{
    if (!sbl_parents)
        return;

    LB_RCB_SBL *cur = sbl_parents;
    while (cur)
    {
        LB_RCB_SBL *next = cur->next;
        free(cur);
        cur = next;
    }
    sbl_parents = NULL;
}

////////////////////////////////////////
// space weight linked list functions //
////////////////////////////////////////

// create and return a new space weight list node, where the weight array is NULL and must be allocated afterwards
static LB_SpaceWeightList *swlist_create_node(int id)
{
    LB_SpaceWeightList *node = (LB_SpaceWeightList *)safe_malloc(sizeof(LB_SpaceWeightList));
    node->id = id;
    node->weights = NULL;
    node->prev_weights = NULL;
    node->weightspace = NULL;
    node->weightdata = NULL;
    node->next = NULL;
    laik_log(1, "lb/swlist: new swlist node for space %d", id);
    return node;
}

// try to find a node (by space id), and if we don't find it, insert it at the back
static LB_SpaceWeightList *swlist_find_or_insert(LB_SpaceWeightList **head, int id)
{
    // create node if head is NULL and return it
    if (*head == NULL)
    {
        laik_log(1, "lb/swlist: initializing swlist");
        *head = swlist_create_node(id);
        return *head;
    }

    // otherwise, search for it...
    LB_SpaceWeightList *curr = *head;
    LB_SpaceWeightList *prev = NULL;

    while (curr)
    {
        if (curr->id == id)
        {
            laik_log(1, "lb/swlist: found swlist for space id %d", id);
            return curr;
        }
        prev = curr;
        curr = curr->next;
    }

    // not found, so append to end of swlist
    laik_log(1, "lb/swlist: no swlist found for space id %d, creating...", id);
    LB_SpaceWeightList *new_node = swlist_create_node(id);
    prev->next = new_node;

    return new_node;
}

// free swlist
static void swlist_free(LB_SpaceWeightList *head)
{
    LB_SpaceWeightList *temp;
    while (head)
    {
        temp = head->next;
        if (head->weights)
            free(head->weights);
        if (head->prev_weights)
            free(head->prev_weights);
        if (head->weightpart)
            laik_free_partitioning(head->weightpart);
        if (head->weightspace)
            laik_free_space(head->weightspace);
        if (head->weightdata)
            laik_free(head->weightdata); // not actually implemented behind the scenes but it'll be correct
        free(head);
        head = temp;
    }
}

/////////////////////////
// lb helper functions //
/////////////////////////

// public: get algorithm enum from string
// unknown algorithm -> fall back to rcb
Laik_LBAlgorithm laik_strtolb(const char *str)
{
    if (strcmp(str, "rcb") == 0)
        return LB_RCB;
    else if (strcmp(str, "rcbincr") == 0 || strcmp(str, "rcbiter") == 0)
        return LB_RCB_INCR;
    else if (strcmp(str, "hilbert") == 0)
        return LB_HILBERT;
    else if (strcmp(str, "gilbert") == 0)
        return LB_GILBERT;
    else
    {
        laik_log(LAIK_LL_Error, "lb/strtolb: unknown load balancing algorithm! defaulting to rcb...");
        return LB_RCB;
    }
}

// public: get algorithm string from enum
const char *laik_get_lb_algorithm_name(Laik_LBAlgorithm algo)
{
    switch (algo)
    {
    case LB_RCB:
        return "rcb";
    case LB_RCB_INCR:
        return "rcbincr";
    case LB_HILBERT:
        return "hilbert";
    case LB_GILBERT:
        return "gilbert";
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

// get range weight from index (3d)
static double get_idx_weight_3d(double *weights, int64_t size_x, int64_t size_y, int64_t x, int64_t y, int64_t z)
{
    return weights[x + size_x * (y + size_y * z)];
}

// note regarding 3d: looking at preexisting LAIK examples, the dimensions are:
//   x: width  (horizontal)
//   y: height (vertical)
//   z: depth  (slices)

///////////////////
// merging cells //
///////////////////

// helper functions to merge singular indices into large rectangles
//
// this function takes as input a "map" of each index in the original index space to its corresponding task id
//
// for each task T:
// |  create an empty list L_T of rectangles / cuboid (ranges) for T
// |  repeat until we've exhausted this task's cells:
// |  |  find the bottom-leftmost cell belonging to T
// |  |  from there, measure how far right we can go (along the x axis) until we reach the edge / a cell which doesn't belong to T anymore (max width)
// |  |  extend this rectangle upwards, each time updating (shrinking) the weight if necessary to still encompass all cells that belong to T
// |  |  save rectangle with maximum area and add to list L_T
// |  |  mark cells as used (-1)
// |  done
// done
//
// there's probably better ways of doing this, especially for hilbert curves (quadtrees), but this should be versatile enough to work with various algorithms

// merge rectangles (2d)
void merge_rects_then_add_ranges(int *grid1D, int64_t width, int64_t height, Laik_RangeReceiver *r, int tidcount)
{
    Laik_Instance *inst = r->list->space->inst;
    laik_svg_profiler_enter(inst, __func__);

    for (int64_t y = 0; y < height; ++y)
    {
        for (int64_t x = 0; x < width; ++x)
        {
            int64_t base = idx2d(x, y, width);
            int tid = grid1D[base];

            // skip already-used (-1) and out-of-range tids
            if (tid < 0 || tid >= tidcount)
                continue;

            // anchor at (x0,y0)
            int64_t x0 = x;
            int64_t y0 = y;

            // initial horizontal run (bounded by width)
            int64_t w = 0;
            int64_t scan_idx = base;
            while (x0 + w < width && grid1D[scan_idx + w] == tid)
                ++w;

            // grow upward to maximize area
            int64_t best_area = w;
            int64_t best_w = w;
            int64_t best_h = 1;
            int64_t curr_w = w;

            for (int64_t h = 2; y0 + h <= height; ++h)
            {
                int64_t yy = y0 + h - 1;
                int64_t row_idx = idx2d(x0, yy, width);

                // if the row start isn't the same tid, we cannot extend
                if (grid1D[row_idx] != tid)
                    break;

                // measure run in this row but cap at curr_w
                int64_t w_h = 0;
                while (w_h < curr_w && x0 + w_h < width && grid1D[row_idx + w_h] == tid)
                    ++w_h;

                if (w_h == 0)
                    break;

                if (w_h < curr_w)
                    curr_w = w_h;

                int64_t area = curr_w * h;
                if (area > best_area)
                {
                    best_area = area;
                    best_w = curr_w;
                    best_h = h;
                }
            }

            // append the chosen rectangle
            Laik_Range range = {
                .space = r->list->space,
                .from = {{x0, y0, 0}},
                .to = {{x0 + best_w, y0 + best_h, 0}}};
            laik_append_range(r, tid, &range, 0, 0);

            // mark covered cells used (-1)
            for (int64_t dy = 0; dy < best_h; ++dy)
            {
                int64_t row_base = idx2d(x0, y0 + dy, width);
                for (int64_t dx = 0; dx < best_w; ++dx)
                    grid1D[row_base + dx] = -1;
            }

            // advance x to skip cells we've just covered (now -1)
            // subtract 1 because the for loop will increment x
            x = x0 + best_w - 1;
        }
    }

    laik_svg_profiler_exit(inst, __func__);
}

// merge cuboids (3d, x: width, y: height, z: depth)
static void merge_cuboids_then_add_ranges(int *grid1D, int64_t width, int64_t height, int64_t depth, Laik_RangeReceiver *r, int tidcount)
{
    Laik_Instance *inst = r->list->space->inst;
    laik_svg_profiler_enter(inst, __func__);

    // temporary buffers reused across anchors
    int64_t *w_layer = NULL; // stores curr_w per (dz, hh-1): size rem_d * rem_h
    int64_t *min_w_h = NULL; // running minima across layers for each height hh
    size_t buf_capacity = 0; // capacity in elements

    for (int64_t z = 0; z < depth; ++z)
    {
        for (int64_t y = 0; y < height; ++y)
        {
            for (int64_t x = 0; x < width; ++x)
            {
                int64_t base = idx3d_local(x, y, z, width, height);
                int tid = grid1D[base];

                // skip already-used (-1) and out-of-range tids
                if (tid < 0 || tid >= tidcount)
                    continue;

                // anchor at (x,y,z)
                int64_t x0 = x, y0 = y, z0 = z;

                // remaining extents from anchor
                int64_t rem_h = height - y0;
                int64_t rem_d = depth - z0;
                if (rem_h <= 0 || rem_d <= 0)
                {
                    // should not happen!
                    grid1D[base] = -1;
                    continue;
                }

                size_t need = (size_t)rem_d * (size_t)rem_h;
                if (need > buf_capacity)
                {
                    // allocate at least depth*height to amortize allocations
                    size_t new_cap = (size_t)depth * (size_t)height;
                    if (new_cap < need)
                        new_cap = need;
                    int64_t *tmp = (int64_t *)realloc(w_layer, new_cap * sizeof(int64_t));
                    if (!tmp)
                    {
                        // allocation fails, fallback
                        goto fallback_single_cell;
                    }
                    w_layer = tmp;
                    int64_t *tmp2 = (int64_t *)realloc(min_w_h, new_cap * sizeof(int64_t));
                    if (!tmp2)
                    {
                        // allocation fails, fallback
                        free(w_layer);
                        w_layer = NULL;
                        goto fallback_single_cell;
                    }
                    min_w_h = tmp2;
                    buf_capacity = new_cap;
                }

                // build w_layer
                // for each dz (depth layer), and each hh (height from anchor), store the minimum contiguous run length in x (from x0) across rows up to hh
                for (int64_t dz = 0; dz < rem_d; ++dz)
                {
                    int64_t zcur = z0 + dz;
                    int64_t curr_w = INT64_MAX;
                    size_t base_offset = (size_t)dz * (size_t)rem_h;
                    for (int64_t hh = 1; hh <= rem_h; ++hh)
                    {
                        int64_t ycur = y0 + (hh - 1);
                        int64_t run = 0;
                        int64_t row_idx = idx3d_local(x0, ycur, zcur, width, height);

                        // cap scanning by curr_w and by bounds
                        while (run < curr_w && x0 + run < width && grid1D[row_idx + run] == tid)
                            ++run;

                        if (hh == 1)
                            curr_w = run;
                        else if (run < curr_w)
                            curr_w = run;

                        w_layer[base_offset + (hh - 1)] = curr_w;

                        if (curr_w == 0)
                        {
                            // remaining hh for this dz will be zero
                            for (int64_t hh2 = hh + 1; hh2 <= rem_h; ++hh2)
                                w_layer[base_offset + (hh2 - 1)] = 0;
                            break;
                        }
                    }
                }

                // search for best (w,h,d) maximizing volume efficiently:
                //   iterate d from 1 to rem_d, keep min_w_h[hh-1] = min across layers seen so far
                //   for each d compute the best h using the min_w_h array
                int64_t best_vol = 0;
                int64_t best_w = 0, best_h = 0, best_d = 0;

                // initialize min_w_h with INF (set from first layer)
                for (int64_t hh = 0; hh < rem_h; ++hh)
                    min_w_h[hh] = INT64_MAX;

                for (int64_t dcur = 1; dcur <= rem_d; ++dcur)
                {
                    size_t layer_off = (size_t)(dcur - 1) * (size_t)rem_h;
                    // update running minima for each height
                    for (int64_t hh = 1; hh <= rem_h; ++hh)
                    {
                        int64_t w_for_layer = w_layer[layer_off + (hh - 1)];
                        if (dcur == 1)
                            min_w_h[hh - 1] = w_for_layer;
                        else if (w_for_layer < min_w_h[hh - 1])
                            min_w_h[hh - 1] = w_for_layer;
                    }

                    // now find best h for this dcur
                    for (int64_t hh = 1; hh <= rem_h; ++hh)
                    {
                        int64_t mw = min_w_h[hh - 1];
                        if (mw == 0)
                            continue;
                        int64_t vol = mw * hh * dcur;
                        if (vol > best_vol)
                        {
                            best_vol = vol;
                            best_w = mw;
                            best_h = hh;
                            best_d = dcur;
                        }
                    }
                }

                // fallback to single cell if something went wrong
                if (best_vol == 0)
                {
                fallback_single_cell:
                    best_w = 1;
                    best_h = 1;
                    best_d = 1;
                }

                // append the chosen cuboid
                Laik_Range range = {
                    .space = r->list->space,
                    .from = {{x0, y0, z0}},
                    .to = {{x0 + best_w, y0 + best_h, z0 + best_d}}};
                laik_append_range(r, tid, &range, 0, 0);

                // mark covered cells used (-1)
                for (int64_t dz = 0; dz < best_d; ++dz)
                {
                    for (int64_t dy = 0; dy < best_h; ++dy)
                    {
                        int64_t row_base = idx3d_local(x0, y0 + dy, z0 + dz, width, height);
                        for (int64_t dx = 0; dx < best_w; ++dx)
                            grid1D[row_base + dx] = -1;
                    }
                }

                // advance x to skip the region we just covered
                x = x0 + best_w - 1;
            }
        }
    }

    free(w_layer);
    free(min_w_h);

    laik_svg_profiler_exit(inst, __func__);
}

//////////////////////
// sfc partitioners //
//////////////////////

// TODO (lo): consider allowing full 64-bit range for indices

/*            hilbert space filling curve (2D, 3D) using Skilling's bitwise method            */
/* source:  Programming the Hilbert curve, John Skilling, AIP Conf. Proc. 707, 381–387 (2004) */
/*                             http://doi.org/10.1063/1.1751381                               */

// helper: rotate indices based on current (sub)quadrant
//
// rotate or reflect the coordinates x,y in-place for the sub-square of size s
// this implements the standard hilbert rotation step used when decoding d to xy
static inline void hilbert_rotate(uint64_t s, uint64_t *x, uint64_t *y, uint64_t rx, uint64_t ry)
{
    // if ry is zero then the quadrant requires a rotation/reflection
    if (ry == 0)
    {
        // when rx is one, reflect the coordinates across the center of the s*s subsquare
        // reflection is performed by mapping v -> s-1-v for both axes
        if (rx == 1)
        {
            *x = s - 1 - *x;
            *y = s - 1 - *y;
        }

        // swap x and y to complete the 90 degree rotation for this subsquare
        uint64_t t = *x;
        *x = *y;
        *y = t;
    }
}

// d -> (x,y) on a (2^b * 2^b) 2d hilbert curve
// using uint64_t to cover complete idx range (for int64_t indices) and ensure well-defined bitops
//
// note: this assumes b is in the valid range [1, 32]
static inline void hilbert_d2xy(unsigned b, uint64_t d, uint64_t *x, uint64_t *y)
{
    // start with zeroed coordinates
    *x = *y = 0;

    // iterate over each bit-plane, s is the current subsquare side length
    // loop doubles s each iteration and stops when s reaches 2^b
    for (uint64_t s = 1; s < ((uint64_t)1 << b); s <<= 1)
    {
        // extract the two control bits for this level from d
        // rx is the more significant of the pair and determines x movement direction
        uint64_t rx = (d >> 1) & 1u;
        // ry is derived from d xor rx and determines y movement direction
        uint64_t ry = (d ^ rx) & 1u;

        // apply the local rotation/reflection for this sub-square
        hilbert_rotate(s, x, y, rx, ry);

        // add the contribution of this level to the coordinates
        // if rx or ry is 1 then the coordinate moves by s in that axis
        *x += s * rx;
        *y += s * ry;

        // consume the two bits we just used and move to the next level
        d >>= 2;
    }
}

// helper: skilling's transposetoaxes, specialized to 3d
//
// converts coordinates in "transposed" hilbert form to normal x,y,z axes
// this implements the in-place bit-manipulation part of skilling's algorithm
static inline void hilbert_transpose(uint64_t *xx, uint64_t *yy, uint64_t *zz, int b)
{
    // initial rotate/xor step that mixes the most-significant bits between z, y, x
    // this sets up the words so the subsequent loop can undo hilbert rotations/reflections
    uint64_t t = (*zz) >> 1;
    *zz ^= *yy;
    *yy ^= *xx;
    *xx ^= t;

    // iterate over bit-planes from the second-least-significant power-of-two up to 2^(b-1)
    // each iteration fixes one level of the transposition using masks
    for (uint64_t q = 2ULL; q != (1ull << b); q <<= 1)
    {
        // p is a mask of all lower bits below q (i.e. q-1)
        // it is used to select the lower i bits that must be conditionally swapped/xor-ed
        uint64_t p = q - 1ull;

        // if the current bit of z is set then flip the lower bits of x by p
        // this corresponds to one branch of the hilbert coordinate correction
        if ((*zz & q) != 0ull)
            *xx ^= p;
        else
        {
            // otherwise swap the lower bits between x and z using xor trick
            // t holds the bits that differ for xor-ing
            t = ((*xx ^ *zz) & p);
            *xx ^= t;
            *zz ^= t;
        }

        // similarly correct x/y depending on the current bit of y
        if ((*yy & q) != 0ull)
            *xx ^= p;
        else
        {
            // swap the lower bits between x and y when y's bit is zero
            t = ((*xx ^ *yy) & p);
            *xx ^= t;
            *yy ^= t;
        }

        // last step: if the current bit of x is set then flip the lower bits of x by p
        if ((*xx & q) != 0ull)
            *xx ^= p; // x xor x = 0
    }
}

// d -> (x,y,z) on a (2^b * 2^b * 2^b) 3d hilbert curve
// note: this assumes b is in the valid range [1, 21] (could be higher, but then we'd have to use a 128 bit d)
static inline void hilbert_d2xyz(int b, uint64_t d, uint64_t *x, uint64_t *y, uint64_t *z)
{
    assert((b > 0 && b <= 21) && "b not in valid range [1, 21]"); // 3 * b <= 64

    // prepare transposed coordinates (each holds b bits, one bit per level)
    uint64_t xx = 0ull, yy = 0ull, zz = 0ull;

    // extract the interleaved bits from d into xx, yy, zz
    // d is assumed to have bits laid out as (..., x_i, y_i, z_i, x_{i-1}, y_{i-1}, z_{i-1}, ...)
    // the loop pulls the 3 bits for each level i out and places them into the i-th position of xx/yy/zz
    for (int i = b - 1; i >= 0; --i)
    {
        xx |= (uint64_t)(((d >> (3 * i + 2)) & 1ull) << i);
        yy |= (uint64_t)(((d >> (3 * i + 1)) & 1ull) << i);
        zz |= (uint64_t)(((d >> (3 * i + 0)) & 1ull) << i);
    }

    // convert from transposed hilbert coordinates to axis-aligned coordinates
    // i.e. undoes hilbert's local rotations and reflections so xx,yy,zz become the final x,y,z
    hilbert_transpose(&xx, &yy, &zz, b);

    // return results
    *x = xx;
    *y = yy;
    *z = zz;
}

/*           gilbert (generalized Hilbert) (2D, 3D) by Jakub Červený           */
/*          original C port by abetusk, 64-bit modification by myself          */
/* source: https://github.com/jakubcerveny/gilbert/blob/master/ports/gilbert.c */

// (tail-)recursive helper function for 2d gilbert
//
// note:  there are currently NO overflow checks, as the code has been adapted 1:1 for 64 bit signed ints
//        be careful when using this with possibly large values!
//        you could replace the binary operations here with own safe functions or compiler builtins
//
// note²: this is just some extra functionality and is not actually intended to be part of the final lb extension
//        this function is also generally slower than the regular hilbert sfc, but this is only really meant to be used
//        when the space size is not a square with side length a power of 2
static inline uint64_t gilbert_d2xy_r(uint64_t dst_idx, uint64_t cur_idx,
                                      int64_t *xres, int64_t *yres,
                                      int64_t ax, int64_t ay,
                                      int64_t bx, int64_t by)
{
    uint64_t nxt_idx;
    int64_t w, h, x, y, dax, day, dbx, dby, di;
    int64_t ax2, ay2, bx2, by2, w2, h2;

    w = llabs(ax + ay), h = llabs(bx + by);
    x = *xres;
    y = *yres;
    dax = sgn(ax), day = sgn(ay); // unit major direction
    dbx = sgn(bx), dby = sgn(by); // unit orthogonal direction
    di = dst_idx - cur_idx;

    if (h == 1)
    {
        *xres = x + dax * di;
        *yres = y + day * di;
        return 0;
    }

    if (w == 1)
    {
        *xres = x + dbx * di;
        *yres = y + dby * di;
        return 0;
    }

    // halve (floored)
    ax2 = ax >> 1;
    ay2 = ay >> 1;
    bx2 = bx >> 1;
    by2 = by >> 1;
    w2 = llabs(ax2 + ay2);
    h2 = llabs(bx2 + by2);

    if ((2 * w) > (3 * h))
    {
        if ((w2 & 1) && (w > 2))
        {
            // prefer even steps
            ax2 += dax;
            ay2 += day;
        }

        // long case: split in two parts only
        nxt_idx = (uint64_t)(cur_idx + llabs((ax2 + ay2) * (bx + by)));
        if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx))
        {
            *xres = x;
            *yres = y;
            return gilbert_d2xy_r(dst_idx, cur_idx, xres, yres, ax2, ay2, bx, by);
        }
        cur_idx = nxt_idx;

        *xres = x + ax2;
        *yres = y + ay2;
        return gilbert_d2xy_r(dst_idx, cur_idx, xres, yres, ax - ax2, ay - ay2, bx, by);
    }

    if ((h2 & 1) && (h > 2))
    {
        // prefer even steps
        bx2 += dbx;
        by2 += dby;
    }

    // standard case: one step up, one long horizontal, one step down
    nxt_idx = (uint64_t)(cur_idx + llabs((bx2 + by2) * (ax2 + ay2)));
    if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx))
    {
        *xres = x;
        *yres = y;
        return gilbert_d2xy_r(dst_idx, cur_idx, xres, yres, bx2, by2, ax2, ay2);
    }
    cur_idx = nxt_idx;

    nxt_idx = (uint64_t)(cur_idx + llabs((ax + ay) * ((bx - bx2) + (by - by2))));
    if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx))
    {
        *xres = x + bx2;
        *yres = y + by2;
        return gilbert_d2xy_r(dst_idx, cur_idx, xres, yres, ax, ay, bx - bx2, by - by2);
    }
    cur_idx = nxt_idx;

    *xres = x + (ax - dax) + (bx2 - dbx);
    *yres = y + (ay - day) + (by2 - dby);
    return gilbert_d2xy_r(dst_idx, cur_idx,
                          xres, yres,
                          -bx2, -by2,
                          -(ax - ax2), -(ay - ay2));
}

// d -> (x,y) on a gilbert curve for an arbitrary 2D domain
static inline int64_t gilbert_d2xy(int64_t *x, int64_t *y, uint64_t idx, int64_t w, int64_t h)
{
    *x = 0;
    *y = 0;

    if (w >= h)
        return gilbert_d2xy_r(idx, 0, x, y, w, 0, 0, h);
    return gilbert_d2xy_r(idx, 0, x, y, 0, h, w, 0);
}

// (tail-)recursive helper function for 3d gilbert
//
// sames notes and caveats as 2d version (no overflow checks, slightly slower, not preferred for square / cubic domains with side length power of 2...)
static inline int64_t gilbert_d2xyz_r(uint64_t dst_idx, uint64_t cur_idx,
                                      int64_t *xres, int64_t *yres, int64_t *zres,
                                      int64_t ax, int64_t ay, int64_t az,
                                      int64_t bx, int64_t by, int64_t bz,
                                      int64_t cx, int64_t cy, int64_t cz)
{
    uint64_t nxt_idx;
    uint64_t _di;
    int64_t x, y, z;

    int64_t w, h, d;
    int64_t w2, h2, d2;

    int64_t dax, day, daz,
        dbx, dby, dbz,
        dcx, dcy, dcz;
    int64_t ax2, ay2, az2,
        bx2, by2, bz2,
        cx2, cy2, cz2;

    x = *xres;
    y = *yres;
    z = *zres;

    w = llabs(ax + ay + az);
    h = llabs(bx + by + bz);
    d = llabs(cx + cy + cz);

    dax = sgn(ax), day = sgn(ay), daz = sgn(az); // unit major direction "right"
    dbx = sgn(bx), dby = sgn(by), dbz = sgn(bz); // unit ortho direction "forward"
    dcx = sgn(cx), dcy = sgn(cy), dcz = sgn(cz); // unit ortho direction "up"

    _di = dst_idx - cur_idx;

    // trivial row/column fills
    if ((h == 1) && (d == 1))
    {
        *xres = x + dax * (int64_t)_di;
        *yres = y + day * (int64_t)_di;
        *zres = z + daz * (int64_t)_di;
        return 0;
    }

    if ((w == 1) && (d == 1))
    {
        *xres = x + dbx * (int64_t)_di;
        *yres = y + dby * (int64_t)_di;
        *zres = z + dbz * (int64_t)_di;
        return 0;
    }

    if ((w == 1) && (h == 1))
    {
        *xres = x + dcx * (int64_t)_di;
        *yres = y + dcy * (int64_t)_di;
        *zres = z + dcz * (int64_t)_di;
        return 0;
    }

    ax2 = ax >> 1;
    ay2 = ay >> 1;
    az2 = az >> 1;

    bx2 = bx >> 1;
    by2 = by >> 1;
    bz2 = bz >> 1;

    cx2 = cx >> 1;
    cy2 = cy >> 1;
    cz2 = cz >> 1;

    w2 = llabs(ax2 + ay2 + az2);
    h2 = llabs(bx2 + by2 + bz2);
    d2 = llabs(cx2 + cy2 + cz2);

    // prefer even steps
    if ((w2 & 1) && (w > 2))
    {
        ax2 += dax;
        ay2 += day;
        az2 += daz;
    }
    if ((h2 & 1) && (h > 2))
    {
        bx2 += dbx;
        by2 += dby;
        bz2 += dbz;
    }
    if ((d2 & 1) && (d > 2))
    {
        cx2 += dcx;
        cy2 += dcy;
        cz2 += dcz;
    }

    // wide case, split in w only
    if (((2 * w) > (3 * h)) && ((2 * w) > (3 * d)))
    {
        nxt_idx = (uint64_t)(cur_idx + llabs((ax2 + ay2 + az2) * (bx + by + bz) * (cx + cy + cz)));
        if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx))
        {
            *xres = x;
            *yres = y;
            *zres = z;
            return gilbert_d2xyz_r(dst_idx, cur_idx,
                                   xres, yres, zres,
                                   ax2, ay2, az2,
                                   bx, by, bz,
                                   cx, cy, cz);
        }
        cur_idx = nxt_idx;

        *xres = x + ax2;
        *yres = y + ay2;
        *zres = z + az2;
        return gilbert_d2xyz_r(dst_idx, cur_idx,
                               xres, yres, zres,
                               ax - ax2, ay - ay2, az - az2,
                               bx, by, bz,
                               cx, cy, cz);
    }

    // do not split in d
    else if ((3 * h) > (4 * d))
    {
        nxt_idx = (uint64_t)(cur_idx + llabs((bx2 + by2 + bz2) * (cx + cy + cz) * (ax2 + ay2 + az2)));
        if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx))
        {
            *xres = x;
            *yres = y;
            *zres = z;
            return gilbert_d2xyz_r(dst_idx, cur_idx,
                                   xres, yres, zres,
                                   bx2, by2, bz2,
                                   cx, cy, cz,
                                   ax2, ay2, az2);
        }
        cur_idx = nxt_idx;

        nxt_idx = (uint64_t)(cur_idx + llabs((ax + ay + az) * ((bx - bx2) + (by - by2) + (bz - bz2)) * (cx + cy + cz)));
        if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx))
        {
            *xres = x + bx2;
            *yres = y + by2;
            *zres = z + bz2;
            return gilbert_d2xyz_r(dst_idx, cur_idx,
                                   xres, yres, zres,
                                   ax, ay, az,
                                   bx - bx2, by - by2, bz - bz2,
                                   cx, cy, cz);
        }
        cur_idx = nxt_idx;

        *xres = x + (ax - dax) + (bx2 - dbx);
        *yres = y + (ay - day) + (by2 - dby);
        *zres = z + (az - daz) + (bz2 - dbz);

        return gilbert_d2xyz_r(dst_idx, cur_idx,
                               xres, yres, zres,
                               -bx2, -by2, -bz2,
                               cx, cy, cz,
                               -(ax - ax2), -(ay - ay2), -(az - az2));
    }

    // do not split in h
    else if ((3 * d) > (4 * h))
    {
        nxt_idx = (uint64_t)(cur_idx + llabs((cx2 + cy2 + cz2) * (ax2 + ay2 + az2) * (bx + by + bz)));
        if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx))
        {
            *xres = x;
            *yres = y;
            *zres = z;
            return gilbert_d2xyz_r(dst_idx, cur_idx,
                                   xres, yres, zres,
                                   cx2, cy2, cz2,
                                   ax2, ay2, az2,
                                   bx, by, bz);
        }
        cur_idx = nxt_idx;

        nxt_idx = (uint64_t)(cur_idx + llabs((ax + ay + az) * (bx + by + bz) * ((cx - cx2) + (cy - cy2) + (cz - cz2))));
        if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx))
        {
            *xres = x + cx2;
            *yres = y + cy2;
            *zres = z + cz2;
            return gilbert_d2xyz_r(dst_idx, cur_idx,
                                   xres, yres, zres,
                                   ax, ay, az,
                                   bx, by, bz,
                                   cx - cx2, cy - cy2, cz - cz2);
        }
        cur_idx = nxt_idx;

        *xres = x + (ax - dax) + (cx2 - dcx);
        *yres = y + (ay - day) + (cy2 - dcy);
        *zres = z + (az - daz) + (cz2 - dcz);

        return gilbert_d2xyz_r(dst_idx, cur_idx,
                               xres, yres, zres,
                               -cx2, -cy2, -cz2,
                               -(ax - ax2), -(ay - ay2), -(az - az2),
                               bx, by, bz);
    }

    // regular case, split in all w/h/d
    nxt_idx = (uint64_t)(cur_idx + llabs((bx2 + by2 + bz2) * (cx2 + cy2 + cz2) * (ax2 + ay2 + az2)));
    if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx))
    {
        *xres = x;
        *yres = y;
        *zres = z;
        return gilbert_d2xyz_r(dst_idx, cur_idx,
                               xres, yres, zres,
                               bx2, by2, bz2,
                               cx2, cy2, cz2,
                               ax2, ay2, az2);
    }
    cur_idx = nxt_idx;

    nxt_idx = (uint64_t)(cur_idx + llabs((cx + cy + cz) * (ax2 + ay2 + az2) * ((bx - bx2) + (by - by2) + (bz - bz2))));
    if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx))
    {
        *xres = x + bx2;
        *yres = y + by2;
        *zres = z + bz2;
        return gilbert_d2xyz_r(dst_idx, cur_idx,
                               xres, yres, zres,
                               cx, cy, cz,
                               ax2, ay2, az2,
                               bx - bx2, by - by2, bz - bz2);
    }
    cur_idx = nxt_idx;

    nxt_idx = (uint64_t)(cur_idx + llabs((ax + ay + az) * (-bx2 - by2 - bz2) * (-(cx - cx2) - (cy - cy2) - (cz - cz2))));
    if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx))
    {
        *xres = x + (bx2 - dbx) + (cx - dcx);
        *yres = y + (by2 - dby) + (cy - dcy);
        *zres = z + (bz2 - dbz) + (cz - dcz);
        return gilbert_d2xyz_r(dst_idx, cur_idx,
                               xres, yres, zres,
                               ax, ay, az,
                               -bx2, -by2, -bz2,
                               -(cx - cx2), -(cy - cy2), -(cz - cz2));
    }
    cur_idx = nxt_idx;

    nxt_idx = (uint64_t)(cur_idx + llabs((-cx - cy - cz) * (-(ax - ax2) - (ay - ay2) - (az - az2)) * ((bx - bx2) + (by - by2) + (bz - bz2))));
    if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx))
    {
        *xres = x + (ax - dax) + bx2 + (cx - dcx);
        *yres = y + (ay - day) + by2 + (cy - dcy);
        *zres = z + (az - daz) + bz2 + (cz - dcz);
        return gilbert_d2xyz_r(dst_idx, cur_idx,
                               xres, yres, zres,
                               -cx, -cy, -cz,
                               -(ax - ax2), -(ay - ay2), -(az - az2),
                               bx - bx2, by - by2, bz - bz2);
    }
    cur_idx = nxt_idx;

    *xres = x + (ax - dax) + (bx2 - dbx);
    *yres = y + (ay - day) + (by2 - dby);
    *zres = z + (az - daz) + (bz2 - dbz);
    return gilbert_d2xyz_r(dst_idx, cur_idx,
                           xres, yres, zres,
                           -bx2, -by2, -bz2,
                           cx2, cy2, cz2,
                           -(ax - ax2), -(ay - ay2), -(az - az2));
}

// d -> (x,y,z) on a gilbert curve for an arbitrary 3d domain
// note: no overflow checks!
static inline int64_t gilbert_d2xyz(int64_t *x, int64_t *y, int64_t *z, uint64_t idx, int64_t width, int64_t height, int64_t depth)
{
    *x = 0;
    *y = 0;
    *z = 0;

    if ((width >= height) && (width >= depth))
    {
        return gilbert_d2xyz_r(idx, 0,
                               x, y, z,
                               width, 0, 0,
                               0, height, 0,
                               0, 0, depth);
    }
    else if ((height >= width) && (height >= depth))
    {
        return gilbert_d2xyz_r(idx, 0,
                               x, y, z,
                               0, height, 0,
                               width, 0, 0,
                               0, 0, depth);
    }

    // depth >= width and depth >= height
    return gilbert_d2xyz_r(idx, 0,
                           x, y, z,
                           0, 0, depth,
                           width, 0, 0,
                           0, height, 0);
}

// ------------------------- //
// laik functions start here //
// ------------------------- //

// TODO (lo): verify types for both

// internal 2d sfc helper function
static void sfc_2d(Laik_RangeReceiver *r, Laik_PartitionerParams *p, double *weights, Laik_LBAlgorithm algo)
{
    Laik_Instance *inst = r->params->space->inst;
    laik_svg_profiler_enter(inst, __func__);

    int tidcount = p->group->size;

    Laik_Space *space = p->space;
    Laik_Range range = space->range;

    int64_t size_x = range.to.i[0] - range.from.i[0];
    int64_t size_y = range.to.i[1] - range.from.i[1];

    // validate square domain with side as a power of 2 for hilbert curve
    if (algo == LB_HILBERT)
        assert(size_x > 0 && size_y > 0 && size_x == size_y && is_power_of_two(size_x) && is_power_of_two(size_y) && "hilbert curve requires square domain with side length power of two");

    // compute total weight (d2xy not needed here, since traversal order doesn't matter)
    uint64_t N = (uint64_t)size_x * (uint64_t)size_y; // uint64_t to cover full range (incl. exactly 2^32)
    double total_w = 0.0;

    for (int64_t y = 0; y < size_y; ++y)
        for (int64_t x = 0; x < size_x; ++x)
            total_w += get_idx_weight_2d(weights, size_x, x, y);

    // define variables for distributing chunks evenly based on weight across all tasks
    double target = total_w / (double)tidcount;
    int task = 0;
    double sum = 0.0;

    // allocate index-to-task mapping array
    int *idxGrid = (int *)safe_malloc(N * sizeof(int));

    laik_log(1, "lb/sfc_2d: allocated %ld bytes (%f kB) for index grid\n", N * sizeof(int), .001 * N * sizeof(int));
    laik_log(1, "lb/sfc_2d: size %ld, totalw %f, targetw %f for %d procs\n", N, total_w, target, tidcount);

    // scan curve and partition based on prefix sum
    // repeated code, done to avoid constantly checking algorithm inside for loop
    if (algo == LB_HILBERT)
    {
        int b = log2_int(size_x);                                     // side length 2^b, valid domains: [2^1, 2^1] -> [2^32, 2^32] in increments of consecutive powers of 2
        assert((b > 0 && b <= 32) && "b not in valid range [1, 32]"); // 2 * b <= 64

        for (uint64_t m = 0; m < N; ++m)
        {
            uint64_t x, y;
            hilbert_d2xy(b, m, &x, &y);
            sum += get_idx_weight_2d(weights, size_x, (int64_t)x, (int64_t)y);
            idxGrid[y * size_x + x] = task;

            // target reached: move to next (?) task
            if (sum >= target)
            {
                laik_log(1, "lb/sfc_2d/hilbert: found split at [x:%ld, y:%ld] for task %d (sum: %f, target %f)\n", x, y, task, sum, target);

                // reset for next task (cap to last index)
                if (task < tidcount - 1)
                    task++;
                sum = 0.0;
            }
        }
    }
    else /* if (algo == LB_GILBERT) */
    {
        for (uint64_t m = 0; m < N; ++m)
        {
            int64_t x, y;
            gilbert_d2xy(&x, &y, m, size_x, size_y); // no overflow checks
            sum += get_idx_weight_2d(weights, size_x, x, y);
            idxGrid[y * size_x + x] = task;

            // target reached: move to next (?) task
            if (sum >= target)
            {
                laik_log(1, "lb/sfc_2d/gilbert: found split at [x:%ld, y:%ld] for task %d (sum: %f, target %f)\n", x, y, task, sum, target);

                // reset for next task (cap to last index)
                if (task < tidcount - 1)
                    task++;
                sum = 0.0;
            }
        }
    }

    // decompose (destroy) idxgrid into axis-aligned rectangles
    merge_rects_then_add_ranges(idxGrid, size_x, size_y, r, tidcount);

    // free remaining memory
    free(idxGrid);

    laik_svg_profiler_exit(inst, __func__);
}

// internal 3d sfc helper function
static void sfc_3d(Laik_RangeReceiver *r, Laik_PartitionerParams *p, double *weights, Laik_LBAlgorithm algo)
{
    Laik_Instance *inst = r->params->space->inst;
    laik_svg_profiler_enter(inst, __func__);

    int tidcount = p->group->size;

    Laik_Space *space = p->space;
    Laik_Range range = space->range;

    int64_t size_x = range.to.i[0] - range.from.i[0];
    int64_t size_y = range.to.i[1] - range.from.i[1];
    int64_t size_z = range.to.i[2] - range.from.i[2];

    // validate cubic domain with side as a power of 2
    if (algo == LB_HILBERT)
        assert(size_x > 0 && size_y > 0 && size_z > 0 && size_x == size_y && size_y == size_z && is_power_of_two(size_x) && is_power_of_two(size_y) && is_power_of_two(size_z) && "hilbert curve requires cube domain with side length power of two");

    // compute total weight (d2xy not needed here, since traversal order doesn't matter)
    uint64_t N = (uint64_t)size_x * (uint64_t)size_y * (uint64_t)size_z; // uint64_t to cover full range (incl. exactly 2^32)
    double total_w = 0.0;

    for (int64_t z = 0; z < size_z; ++z)
        for (int64_t y = 0; y < size_y; ++y)
            for (int64_t x = 0; x < size_x; ++x)
                total_w += get_idx_weight_3d(weights, size_x, size_y, x, y, z);

    // define variables for distributing chunks evenly based on weight across all tasks
    double target = total_w / (double)tidcount;
    int task = 0;
    double sum = 0.0;

    // allocate index-to-task mapping array
    int *idxGrid = (int *)safe_malloc(N * sizeof(int));

    laik_log(1, "lb/sfc_3d: allocated %ld bytes (%f kB) for index grid\n", N * sizeof(int), .001 * N * sizeof(int));
    laik_log(1, "lb/sfc_3d: size %ld, totalw %f, targetw %f for %d procs\n", N, total_w, target, tidcount);

    // scan hilbert curve and partition based on prefix sum
    if (algo == LB_HILBERT)
    {
        int b = log2_int(size_x);                                     // side length 2^b, valid domains: [2^1, 2^1] -> [2^21, 2^21] in increments of consecutive powers of 2
        assert((b > 0 && b <= 21) && "b not in valid range [1, 21]"); // 3 * b <= 64, floored

        for (uint64_t m = 0; m < N; ++m)
        {
            uint64_t x, y, z;
            hilbert_d2xyz(b, m, &x, &y, &z);
            sum += get_idx_weight_3d(weights, size_x, size_y, (int64_t)x, (int64_t)y, (int64_t)z);
            idxGrid[x + size_x * (y + size_y * z)] = task;

            // target reached: move to next (?) task
            if (sum >= target)
            {
                laik_log(1, "lb/sfc_3d/hilbert: found split at [x:%ld, y:%ld, z:%ld] for task %d (sum: %f, target %f)\n", x, y, z, task, sum, target);

                // reset for next task (cap to last index)
                if (task < tidcount - 1)
                    task++;
                sum = 0.0;
            }
        }
    }
    else /* if (algo == LB_GILBERT) */
    {
        for (uint64_t m = 0; m < N; ++m)
        {
            int64_t x, y, z;
            gilbert_d2xyz(&x, &y, &z, m, size_x, size_y, size_z);
            sum += get_idx_weight_3d(weights, size_x, size_y, x, y, z);
            idxGrid[x + size_x * (y + size_y * z)] = task;

            // target reached: move to next (?) task
            if (sum >= target)
            {
                laik_log(1, "lb/sfc_3d/gilbert: found split at [x:%ld, y:%ld, z:%ld] for task %d (sum: %f, target %f)\n", x, y, z, task, sum, target);

                // reset for next task (cap to last index)
                if (task < tidcount - 1)
                    task++;
                sum = 0.0;
            }
        }
    }

    // decompose (destroy) idxgrid into axis-aligned cuboids
    merge_cuboids_then_add_ranges(idxGrid, size_x, size_y, size_z, r, tidcount);

    // free remaining memory
    free(idxGrid);

    laik_svg_profiler_exit(inst, __func__);
}

// ------------------------- //
// laik functions start here //
// ------------------------- //

void runSFCPartitioner(Laik_RangeReceiver *r, Laik_PartitionerParams *p)
{
    Laik_Instance *inst = p->space->inst;
    laik_svg_profiler_enter(inst, __func__);

    int dims = p->space->dims;
    LB_SFC_Data *data = (LB_SFC_Data *)p->partitioner->data;
    double *weights = data->weights;
    Laik_LBAlgorithm algo = data->algo;
    free(data); // not needed anymore

    if (dims == 2)
        sfc_2d(r, p, weights, algo);
    else if (dims == 3)
        sfc_3d(r, p, weights, algo);
    else
    {
        laik_panic("Unsupported dimensions for space-filling curve partitioner (must be: 2 or 3)");
        exit(EXIT_FAILURE);
    }

    laik_svg_profiler_exit(inst, __func__);
}

Laik_Partitioner *laik_new_sfc_partitioner(double *weights, Laik_LBAlgorithm algo)
{
    assert((algo != LB_RCB) && (algo != LB_RCB_INCR) && "that's not a space-filling curve!");

    LB_SFC_Data *data = (LB_SFC_Data *)safe_malloc(sizeof(LB_SFC_Data));
    data->weights = weights;
    data->algo = algo;

    return laik_new_partitioner(laik_get_lb_algorithm_name(algo), runSFCPartitioner, (void *)data, 0);
}

/////////////////////
// rcb partitioner //
/////////////////////

// internal 1d rcb helper function; [fromTask - toTask)
//
// note: like the range weight function above, i've separated this into a 1d and 2d version
//       partially due to recursion, mainly due to differences in the algorithm to avoid cluttering the function with ifs
static void rcb_1d(Laik_RangeReceiver *r, Laik_Range *range, int fromTask, int toTask, double *weights, bool rec)
{
    Laik_Instance *inst = r->params->space->inst;
    laik_svg_profiler_enter(inst, __func__);

    int64_t from = range->from.i[0];
    int64_t to = range->to.i[0];

    // if there's only one processor left, stop here
    int count = toTask - fromTask + 1;
    if (count == 1 || from > to)
    {
        // for odd task numbers, we got from the previous step if this child has a brother
        if (rec && sbl_recompute)
            rcb_sl_push(fromTask, toTask, range);
        laik_append_range(r, fromTask, range, 0, 0);
        laik_svg_profiler_exit(inst, __func__);
        return;
    }

    // push second-from-bottom layer range to linked list for incremental rcb
    if (count == 2 && sbl_recompute)
        rcb_sl_push(fromTask, toTask, range);

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

    laik_log(1, "lb/rcb_1d: [T%d-T%d) [%ld-%ld) count: %d, lcount: %d, rcount: %d, tmid: %d, totalW: %f, ltarget: %f\n", fromTask, toTask, from, to, count, lcount, rcount, tmid, totalW, ltarget);

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
    rcb_1d(r, &r1, fromTask, tmid, weights, (count & 1));
    rcb_1d(r, &r2, tmid + 1, toTask, weights, (count & 1));
    laik_svg_profiler_exit(inst, __func__);
}

// internal 2d rcb helper function; [fromTask - toTask)
//
// primary changes are determining the split direction by checking which side is longest and how the weights are computed / accumulated
static void rcb_2d(Laik_RangeReceiver *r, Laik_Range *range, int fromTask, int toTask, double *weights, bool rec)
{
    Laik_Instance *inst = r->params->space->inst;
    laik_svg_profiler_enter(inst, __func__);

    // if there's only one processor left, stop here
    int count = toTask - fromTask + 1;
    if (count == 1)
    {
        // for odd task numbers, we got from the previous step if this child has a brother
        if (rec && sbl_recompute)
            rcb_sl_push(fromTask, toTask, range);
        laik_append_range(r, fromTask, range, 0, 0);
        laik_svg_profiler_exit(inst, __func__);
        return;
    }

    // push second-from-bottom layer range to linked list for incremental rcb
    if (count == 2 && sbl_recompute)
        rcb_sl_push(fromTask, toTask, range);

    int64_t size_x = range->space->range.to.i[0] - range->space->range.from.i[0];
    int64_t from_x = range->from.i[0];
    int64_t from_y = range->from.i[1];
    int64_t to_x = range->to.i[0];
    int64_t to_y = range->to.i[1];

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

    laik_log(1, "lb/rcb_2d: [T%d-T%d) [%ld, %ld] -> (%ld, %ld), count: %d, lcount: %d, rcount: %d, tmid: %d, totalW: %f, ltarget: %f, axis: %d\n", fromTask, toTask, from_x, from_y, to_x, to_y, count, lcount, rcount, tmid, totalW, ltarget, axis);

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
                sum += get_idx_weight_2d(weights, size_x, x, y);

            // check if sum exceeds weight target, otherwise continue
            if (sum >= ltarget)
            {
                split_y = y;
                laik_log(1, "lb/rcb_2d: found horizontal split at y = %ld (sum: %f)\n", y, sum);
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
                sum += get_idx_weight_2d(weights, size_x, x, y);

            // check if sum exceeds weight target, otherwise continue
            if (sum >= ltarget)
            {
                split_x = x;
                laik_log(1, "lb/rcb_2d: found vertical split at x = %ld (sum: %f)\n", x, sum);
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
    laik_log(1, "lb/rcb_2d: split_x: %ld, split_y : %ld, r1: [%ld, %ld] -> (%ld, %ld); r2: [%ld, %ld] -> (%ld, %ld)\n", split_x, split_y, r1.from.i[0], r1.from.i[1], r1.to.i[0], r1.to.i[1], r2.from.i[0], r2.from.i[1], r2.to.i[0], r2.to.i[1]);
    rcb_2d(r, &r1, fromTask, tmid, weights, (count & 1));
    rcb_2d(r, &r2, tmid + 1, toTask, weights, (count & 1));
    laik_svg_profiler_exit(inst, __func__);
}

// internal 3d rcb helper function
static void rcb_3d(Laik_RangeReceiver *r, Laik_Range *range, int fromTask, int toTask, double *weights, bool rec)
{
    Laik_Instance *inst = r->params->space->inst;
    laik_svg_profiler_enter(inst, __func__);

    // if there's only one processor left, stop here
    int count = toTask - fromTask + 1;
    if (count == 1)
    {
        // for odd task numbers, we got from the previous step if this child has a brother
        if (rec && sbl_recompute)
            rcb_sl_push(fromTask, toTask, range);
        laik_append_range(r, fromTask, range, 0, 0);
        laik_svg_profiler_exit(inst, __func__);
        return;
    }

    // push second-from-bottom layer range to linked list for incremental rcb
    if (count == 2 && sbl_recompute)
        rcb_sl_push(fromTask, toTask, range);

    int64_t size_x = range->space->range.to.i[0] - range->space->range.from.i[0];
    int64_t size_y = range->space->range.to.i[1] - range->space->range.from.i[1];
    int64_t from_x = range->from.i[0];
    int64_t from_y = range->from.i[1];
    int64_t from_z = range->from.i[2];
    int64_t to_x = range->to.i[0];
    int64_t to_y = range->to.i[1];
    int64_t to_z = range->to.i[2];

    // axis: 0 -> x, 1 -> y, 2 -> z
    int64_t dx = to_x - from_x;
    int64_t dy = to_y - from_y;
    int64_t dz = to_z - from_z;
    int axis = 0;
    if (dy > dx && dy >= dz)
        axis = 1;
    else if (dz > dx && dz > dy)
        axis = 2;

    // return if we cannot split along that axis
    int64_t length_along_axis = (axis == 0) ? dx : (axis == 1) ? dy
                                                               : dz;
    if (length_along_axis == 1)
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
    for (int64_t x = from_x; x < to_x; ++x)
        for (int64_t y = from_y; y < to_y; ++y)
            for (int64_t z = from_z; z < to_z; ++z)
                totalW += get_idx_weight_3d(weights, size_x, size_y, x, y, z);

    double ltarget = totalW * ((double)lcount / count);

    laik_log(1, "lb/rcb_3d: [T%d-T%d) [%ld,%ld,%ld] -> (%ld,%ld,%ld), count: %d, lcount: %d, rcount: %d, tmid: %d, totalW: %f, ltarget: %f, axis: %d\n",
             fromTask, toTask, from_x, from_y, from_z, to_x, to_y, to_z, count, lcount, rcount, tmid, totalW, ltarget, axis);

    // accumulate weights along splitting axis and find first index where prefix sum exceeds target weight
    double sum = 0.0;
    int64_t split_x = from_x;
    int64_t split_y = from_y;
    int64_t split_z = from_z;

    if (axis == 0)
    {
        // x longest: for each x, sum over all y,z
        for (int64_t x = from_x; x < to_x; ++x)
        {
            for (int64_t y = from_y; y < to_y; ++y)
                for (int64_t z = from_z; z < to_z; ++z)
                    sum += get_idx_weight_3d(weights, size_x, size_y, x, y, z);

            if (sum >= ltarget)
            {
                split_x = x;
                laik_log(1, "lb/rcb_3d: found x-split at x = %ld (sum: %f)\n", x, sum);
                break;
            }
        }
    }
    else if (axis == 1)
    {
        // y longest: for each y, sum over all x,z
        for (int64_t y = from_y; y < to_y; ++y)
        {
            for (int64_t x = from_x; x < to_x; ++x)
                for (int64_t z = from_z; z < to_z; ++z)
                    sum += get_idx_weight_3d(weights, size_x, size_y, x, y, z);

            if (sum >= ltarget)
            {
                split_y = y;
                laik_log(1, "lb/rcb_3d: found y-split at y = %ld (sum: %f)\n", y, sum);
                break;
            }
        }
    }
    else /* axis == 2 */
    {
        // z longest: for each z, sum over all x,y
        for (int64_t z = from_z; z < to_z; ++z)
        {
            for (int64_t x = from_x; x < to_x; ++x)
                for (int64_t y = from_y; y < to_y; ++y)
                    sum += get_idx_weight_3d(weights, size_x, size_y, x, y, z);

            if (sum >= ltarget)
            {
                split_z = z;
                laik_log(1, "lb/rcb_3d: found z-split at z = %ld (sum: %f)\n", z, sum);
                break;
            }
        }
    }

    // cut and recurse
    Laik_Range r1 = *range, r2 = *range;
    if (axis == 0)
    {
        r1.to.i[0] = split_x;
        r2.from.i[0] = split_x;
    }
    else if (axis == 1)
    {
        r1.to.i[1] = split_y;
        r2.from.i[1] = split_y;
    }
    else /* axis == 2 */
    {
        r1.to.i[2] = split_z;
        r2.from.i[2] = split_z;
    }

    laik_log(1, "lb/rcb_3d: split (x,y,z): %ld,%ld,%ld; r1: [%ld,%ld,%ld] -> (%ld,%ld,%ld); r2: [%ld,%ld,%ld] -> (%ld,%ld,%ld)\n",
             split_x, split_y, split_z,
             r1.from.i[0], r1.from.i[1], r1.from.i[2], r1.to.i[0], r1.to.i[1], r1.to.i[2],
             r2.from.i[0], r2.from.i[1], r2.from.i[2], r2.to.i[0], r2.to.i[1], r2.to.i[2]);

    rcb_3d(r, &r1, fromTask, tmid, weights, (count & 1));
    rcb_3d(r, &r2, tmid + 1, toTask, weights, (count & 1));

    laik_svg_profiler_exit(inst, __func__);
}

// ------------------------- //
// laik functions start here //
// ------------------------- //

void runRCBPartitioner(Laik_RangeReceiver *r, Laik_PartitionerParams *p)
{
    Laik_Instance *inst = p->space->inst;
    laik_svg_profiler_enter(inst, __func__);

    int tidcount = p->group->size;
    int dims = p->space->dims;
    double *weights = (double *)p->partitioner->data;

    Laik_Space *s = p->space;
    Laik_Range range = s->range;

    if (dims == 1)
        rcb_1d(r, &range, 0, tidcount - 1, weights, 0);
    else if (dims == 2)
        rcb_2d(r, &range, 0, tidcount - 1, weights, 0);
    else /* if (dims == 3) */
        rcb_3d(r, &range, 0, tidcount - 1, weights, 0);

    laik_svg_profiler_exit(inst, __func__);
}

void runIncrementalRCBPartitioner(Laik_RangeReceiver *r, Laik_PartitionerParams *p)
{
    Laik_Instance *inst = p->space->inst;
    laik_svg_profiler_enter(inst, __func__);

    int tidcount = p->group->size;
    int dims = p->space->dims;
    double *weights = (double *)p->partitioner->data;

    Laik_Space *s = p->space;
    Laik_Range range = s->range;
reset:
    // correction step
    if (!sbl_parents)
    {
        laik_log(1, "lb/rcb_incr: sbl_parents null, using normal rcb\n");

        if (dims == 1)
            rcb_1d(r, &range, 0, tidcount - 1, weights, 0);
        else if (dims == 2)
            rcb_2d(r, &range, 0, tidcount - 1, weights, 0);
        else /* if (dims == 3) */
            rcb_3d(r, &range, 0, tidcount - 1, weights, 0);

        sbl_recompute = false; // don't recompute, we should have second-bottom-level parent nodes now...
    }
    // otherwise, only move bottom-most boundaries
    else
    {
        laik_log(1, "lb/rcb_incr: sbl_parents found, using incremental rcb\n");
        LB_RCB_SBL *current = sbl_parents;
        while (current)
        {
            if (dims == 1)
                rcb_1d(r, &(current->range), current->from, current->to, weights, 0);
            else if (dims == 2)
                rcb_2d(r, &(current->range), current->from, current->to, weights, 0);
            else /* if (dims == 3) */
                rcb_3d(r, &(current->range), current->from, current->to, weights, 0);
            current = current->next;
        }

        // if we move too few bytes, do a global correction step
        uint64_t diff = laik_rangelist_diff_bytes(laik_partitioning_allranges(p->other), r->list);
        double reldiff = (double)diff / (double)laik_space_size(s);
        if (reldiff < 0.02 /* this should be configurable */)
        {
            laik_log(1, "lb/rcb_incr: reldiff too small - doing global redo: total: %ld, diff: %ld, rel: %f\n", laik_space_size(s), diff, reldiff);
            rcb_sl_clear();
            sbl_recompute = true;
            r->list->count = 0;
            goto reset;
        }
    }

    laik_svg_profiler_exit(inst, __func__);
}

Laik_Partitioner *laik_new_rcb_partitioner(double *weights)
{
    return laik_new_partitioner("rcb", runRCBPartitioner, (void *)weights, 0);
}

Laik_Partitioner *laik_new_incr_rcb_partitioner(double *weights)
{
    return laik_new_partitioner("rcb_incr", runIncrementalRCBPartitioner, (void *)weights, 0);
}

////////////////////
// load balancing //
////////////////////

// calculate the imbalance to determine whether or not load balancing should be done
// formula: (max - min) / mean
static double get_imbalance(Laik_Partitioning *p, double ttime)
{
    static int lastgsize = -1;
    static Laik_Space *tspace = NULL;
    static Laik_Data *tdata = NULL;
    static Laik_Partitioning *tpart = NULL;

    Laik_Group *group = p->group;
    Laik_Instance *inst = group->inst;
    laik_svg_profiler_enter(inst, __func__);

    int task = laik_myid(group);
    int gsize = group->size;

    // initialize group size on first run
    if (lastgsize == -1)
        lastgsize = gsize;

    // if group size changes, invalidate previous data
    if (gsize != lastgsize)
    {
        lastgsize = gsize;
        if (tpart)
            laik_free_partitioning(tpart);
        if (tspace)
            laik_free_space(tspace);
        if (tdata)
            laik_free(tdata);
        tspace = NULL;
        tdata = NULL;
        tpart = NULL;
    }

    // times will be stored in here
    double times[gsize];
    memset(times, 0, gsize * sizeof(double));

    // store own time
    times[task] = ttime;

    // aggregation: use weights directly as input data
    laik_log(1, "lb/get_imbalance: aggregating task times...\n");
    if (!tspace)
    {
        tspace = laik_new_space_1d(inst, gsize);
        tdata = laik_new_data(tspace, laik_Double);
        tpart = laik_new_partitioning(laik_All, group, tspace, NULL);
        laik_data_provide_memory(tdata, times, gsize * sizeof(double));
        laik_data_set_name(tdata, "lb-timedata");
        laik_set_initial_partitioning(tdata, tpart);
    }
    else
        laik_data_provide_memory(tdata, times, gsize * sizeof(double));

    // collect times into weights, shared among all tasks
    laik_switchto_partitioning(tdata, tpart, LAIK_DF_Preserve, LAIK_RO_Sum);

    laik_log_begin(1);
    laik_log_append("lb/get_imbalance: got times (task:time)");
    for (int i = 0; i < gsize; ++i)
        laik_log_append("   %d:%f", i, times[i]);
    laik_log_flush("\n");

    // calculate maximum time difference
    double maxdt, mean;
    min_max_mean(times, gsize, &maxdt, &mean);

    laik_log(1, "lb/get_imbalance: maxdt %f, mean %f\n", maxdt, mean);

    // print time taken by each task
    if (task == 0 && do_print_times)
        print_times(times, gsize, maxdt, mean);

    // return relative threshold ((max - min) / mean)
    laik_svg_profiler_exit(inst, __func__);
    return maxdt / mean;
}

// initialize the weight array for the current load balancing run
static double *init_weights(Laik_Partitioning *p, double ttime)
{
    Laik_Space *space = p->space;
    Laik_Group *group = p->group;
    Laik_Instance *inst = group->inst;
    laik_svg_profiler_enter(inst, __func__);

    // find associated entry for current space in swlist
    LB_SpaceWeightList *swl = swlist_find_or_insert(&swlist, p->space->id);

    // allocate weight array and zero-initialize
    // for t tasks, the final t elements of the array are the raw times taken by each task (starting from 0, one after the last weight)
    int dims = p->space->dims; // assume dims in {1,2,3}
    int64_t size_x = space->range.to.i[0] - space->range.from.i[0];
    int64_t size_y = dims >= 2 ? (space->range.to.i[1] - space->range.from.i[1]) : 1;
    int64_t size_z = dims == 3 ? (space->range.to.i[2] - space->range.from.i[2]) : 1;
    int64_t size = size_x * size_y * size_z;

    // if we didn't provide external weights, compute shared weights for indices, assuming identical workload per index
    if (!ext_weights)
    {
        if (!swl->weights)
            swl->weights = (double *)safe_malloc(size * sizeof(double));

        memset(swl->weights, 0, size * sizeof(double));

        laik_log(1, "lb/init_weights: initialized weight array of %ld bytes (%f kB) (dims: %d, size: %ldx%ldx%ld=%ld)\n", size * sizeof(double), .001 * size * sizeof(double), dims, size_x, size_y, size_z, size);

        // calculate weight and fill array at own indices
        // 1. accumulate number of items for own task
        // 2. get task weight for task i as time_taken(i) * c / nitems(i), where c is some constant
        //
        // TODO (lo): remove redundant laik_my_range... calls
        int64_t tnitems = 0;
        int c = 1000000;
        for (int r = 0; r < laik_my_rangecount(p); ++r)
        {
            if (dims == 1)
            {
                int64_t from, to;
                laik_my_range_1d(p, r, &from, &to);
                tnitems += to - from;
                laik_log(1, "lb/init_weights: [%ld]->(%ld), tnitems + %ld = %ld\n", from, to, to - from, tnitems);
            }
            else if (dims == 2)
            {
                int64_t from_x, from_y, to_x, to_y;
                laik_my_range_2d(p, r, &from_x, &to_x, &from_y, &to_y);
                int64_t count = (to_x - from_x) * (to_y - from_y);
                tnitems += count;
                laik_log(1, "lb/init_weights: [%ld,%ld]->(%ld,%ld), tnitems + %ld = %ld\n", from_x, from_y, to_x, to_y, count, tnitems);
            }
            else /* if (dims == 3) */
            {
                int64_t from_x, from_y, from_z, to_x, to_y, to_z;
                laik_my_range_3d(p, r, &from_x, &to_x, &from_y, &to_y, &from_z, &to_z);
                int64_t count = (to_x - from_x) * (to_y - from_y) * (to_z - from_z);
                tnitems += count;
                laik_log(1, "lb/init_weights: [%ld,%ld,%ld]->(%ld,%ld,%ld), tnitems + %ld = %ld\n", from_x, from_y, from_z, to_x, to_y, to_z, count, tnitems);
            }
        };

        // broadcast weight to own indices
        double tweight = (ttime * (double)c) / (double)tnitems;
        laik_log(1, "lb/init_weights: tweight = (%f * %d) / %ld = %f, broadcasting to own indices...\n", ttime, c, tnitems, tweight);
        for (int r = 0; r < laik_my_rangecount(p); ++r)
        {
            if (dims == 1)
            {
                int64_t from, to;
                laik_my_range_1d(p, r, &from, &to);
                for (int64_t i = from; i < to; ++i)
                    swl->weights[i] = tweight;
            }
            else if (dims == 2)
            {
                int64_t from_x, from_y, to_x, to_y;
                laik_my_range_2d(p, r, &from_x, &to_x, &from_y, &to_y);
                for (int64_t y = from_y; y < to_y; ++y)
                    for (int64_t x = from_x; x < to_x; ++x)
                        swl->weights[y * size_x + x] = tweight;
            }
            else /* if (dims == 3) */
            {
                int64_t from_x, from_y, from_z, to_x, to_y, to_z;
                laik_my_range_3d(p, r, &from_x, &to_x, &from_y, &to_y, &from_z, &to_z);
                for (int64_t z = from_z; z < to_z; ++z)
                    for (int64_t y = from_y; y < to_y; ++y)
                        for (int64_t x = from_x; x < to_x; ++x)
                            swl->weights[x + size_x * (y + size_y * z)] = tweight;
            }
        }
    }
    // otherwise, use externally provided weights
    // note: make sure externally dimensions match!
    else
    {
        laik_log(1, "lb/init_weights: using external weight array %p...\n", ext_weights);
        swl->weights = ext_weights;
    }

    // initialize laik space for aggregating weights if they haven't been initialized yet
    // we only check for weightspace since everything should be initialized at once
    if (!swl->weightspace)
    {
        laik_log(1, "lb/init_weights: initializing new LAIK containers for weights...\n");
        swl->weightspace = laik_new_space_1d(inst, size);
        swl->weightdata = laik_new_data(swl->weightspace, laik_Double);
        swl->weightpart = laik_new_partitioning(laik_All, group, swl->weightspace, NULL);
        laik_data_provide_memory(swl->weightdata, swl->weights, size * sizeof(double));
        laik_data_set_name(swl->weightdata, "weights");
        laik_set_initial_partitioning(swl->weightdata, swl->weightpart);
    }
    else
    {
        laik_log(1, "lb/init_weights: LAIK containers already initialized\n");
        laik_data_provide_memory(swl->weightdata, swl->weights, size * sizeof(double));
    }

    // use weights directly as input data
    laik_log(1, "lb/init_weights: aggregating weights...\n");

    // share weights across all tasks (sum, 0 for every index that doesn't belong to current task)
    laik_switchto_partitioning(swl->weightdata, swl->weightpart, LAIK_DF_Preserve, LAIK_RO_Sum);
    laik_svg_profiler_exit(inst, __func__);

    // optional: do EMA
    if (do_smoothing)
    {
        if (swl->prev_weights)
        {
            laik_log(1, "lb/init_weights: previous run found, using multiplicative EMA (alpha=%f) + ratio clamp [%f,%f]\n", alpha_mul, rmin, rmax);

            // smooth multiplicatively
            for (int64_t i = 0; i < size; ++i)
            {
                double cur = swl->weights[i];       // aggregated current weight (>= 0)
                double prev = swl->prev_weights[i]; // previous smoothed weight (>= 0)

                // remain zero if both zero
                if (cur <= 0.0 && prev <= 0.0)
                {
                    swl->weights[i] = 0.0;
                    swl->prev_weights[i] = 0.0;
                    continue;
                }

                // compute multiplicative update factor f that would map prev -> cur in one step
                // naive instantaneous ratio r_inst = cur / (prev + eps)
                double r_inst = cur / (prev + 1e-30 /* to prevent divison by zero*/);

                // apply multiplicative EMA on the ratio
                double r_smoothed = alpha_mul * r_inst + (1.0 - alpha_mul) * 1.0;

                // clamp ratio to avoid extreme jumps
                if (r_smoothed < rmin)
                    r_smoothed = rmin;
                if (r_smoothed > rmax)
                    r_smoothed = rmax;

                // new weight = prev * r_smoothed
                double neww = prev * r_smoothed;

                // if prev was zero but cur > 0 (hot new cell), we need to set a base
                if (prev <= 0.0 && cur > 0.0)
                {
                    // initialize from cur but still smooth gently (blend cur and tiny fraction of cur)
                    neww = alpha_mul * cur + (1.0 - alpha_mul) * (cur * 0.01);
                }

                // allow absolute minimum zero
                if (neww <= 0.0)
                    neww = 0.0;

                swl->weights[i] = neww;
                swl->prev_weights[i] = neww; // keep prev equal to smoothed value for next round
            }
        }
        else
        {
            // first-time initialization: copy aggregated weights into prev
            laik_log(1, "lb/init_weights: first (smoothed) run, saving weights for multiplicative EMA...\n");
            swl->prev_weights = (double *)safe_malloc(size * sizeof(double));
            memcpy(swl->prev_weights, swl->weights, size * sizeof(double));
        }
    }

    // done
    return swl->weights;
}

////////////////
// public api //
////////////////

// TODO (hi): consider metric where overhead for sending data outweighs possible gains by load balancing, and if this is even necessary / good
Laik_Partitioning *laik_lb_balance(Laik_LBState state, Laik_Partitioning *partitioning, Laik_LBAlgorithm algorithm)
{
    assert(t_stop < t_start && "stopping threshold should be under starting threshold");

    // handle starting, pausing or resuming timer (no stop, so we don't perform load balancing yet)
    // return NULL so we can't do anything
    if (state == START_LB_SEGMENT)
    {
        laik_log(1, "lb: starting new load balancing segment lb-%d (stopped? %d)\n", segment, stopped);
        laik_timer_start(&timer);
        return NULL;
    }

    if (state == PAUSE_LB_SEGMENT)
    {
        laik_log(1, "lb: pausing load balancing segment lb-%d (stopped? %d)\n", segment, stopped);
        laik_timer_pause(&timer);
        return NULL;
    }

    if (state == RESUME_LB_SEGMENT)
    {
        laik_log(1, "lb: resuming load balancing segment lb-%d (stopped? %d)\n", segment, stopped);
        laik_timer_resume(&timer);
        return NULL;
    }

    // otherwise, stop timer and perform load balancing checks and algorithm
    Laik_Instance *inst = partitioning->group->inst;
    laik_svg_profiler_enter(inst, __func__);

    laik_log_begin(1);
    laik_log_append("lb: stopping load balancing segment lb-%d based on partitioning %s using algorithm %s after ", segment, partitioning->name, laik_get_lb_algorithm_name(algorithm));

    // check relative imbalance to determine whether or not to perform load balancing
    double ttime = laik_timer_stop(&timer);
    laik_log_append("%f seconds (stopped? %d)", ttime, stopped);
    laik_log_flush("\n");

    double imbal = get_imbalance(partitioning, ttime);

    /* handle possibly consecutive runs over / under respective thresholds */
    if (imbal < t_stop)
    {
        // imbalance BELOW STOPPING threshold: increment consecutive stopping runs and reset starting runs
        p_stopctr++;
        p_startctr = 0;
        laik_log(1, "lb: relative imbalance %f < stopping threshold %f for %d consecutive runs (out of %d), reset start ctr!", imbal, t_stop, p_stopctr, p_stop);
    }
    else if (imbal > t_start)
    {
        // imbalance ABOVE STARTING threshold: increment consecutive starting runs and reset stopping runs
        p_startctr++;
        p_stopctr = 0;
        laik_log(1, "lb: relative imbalance %f > starting threshold %f for %d consecutive runs (out of %d), reset stop ctr!", imbal, t_start, p_startctr, p_start);
    }
    else
    {
        // imbalance BETWEEN start and stop thresholds: keep state
        p_stopctr = 0, p_startctr = 0;
        laik_log(1, "lb: relative imbalance %f, ]stop: %f, start: %f[, no consecutive runs", imbal, t_stop, t_start);
    }

    /* start / stop load balancing based on consecutive runs */
    if (!stopped && p_stopctr >= p_stop)
    {
        // stop load balancing for consecutive runs UNDER stopping threshold
        stopped = true;
        p_stopctr = 0, p_startctr = 0;
        laik_log(1, "lb: relative imbalance %f < stopping threshold %f for %d consecutive runs (out of %d), LOAD BALANCING DISABLED! (counters reset)", imbal, t_stop, p_stopctr, p_stop);
    }
    if (stopped && p_startctr >= p_start)
    {
        // restart load balancing for consecutive runs OVER starting thresholds
        stopped = false;
        p_stopctr = 0, p_startctr = 0;
        laik_log(1, "lb: relative imbalance %f > starting threshold %f for %d consecutive runs (out of %d), LOAD BALANCING REENABLED! (counters reset)", imbal, t_start, p_startctr, p_start);
    }

    // when not load balancing, return old partitioning unchanged (to be verified by caller!)
    if (stopped)
    {
        laik_log(1, "lb: load balancing segment lb-%d currently INACTIVE, relative imbalance %f (t_stop %f, t_start %f), returning partitioning %s unchanged...\n", segment, imbal, t_stop, t_start, partitioning->name);
        segment++;

        laik_svg_profiler_exit(inst, __func__);
        return partitioning;
    }

    // collect weights associated for each task
    double *weights = init_weights(partitioning, ttime);
    assert(weights != NULL);

    // use task weights to create new partitioning
    Laik_Partitioner *nparter;

    // choose load balancing algorithm based on input
    switch (algorithm)
    {
    case LB_RCB:
        nparter = laik_new_rcb_partitioner(weights);
        break;
    case LB_RCB_INCR:
        nparter = laik_new_incr_rcb_partitioner(weights);
        break;
    case LB_HILBERT:
    case LB_GILBERT:
        nparter = laik_new_sfc_partitioner(weights, algorithm);
        break;
    default:
        laik_panic("Unknown / unimplemented load balancing algorithm!");
        exit(EXIT_FAILURE);
    }

    // create new partitioning to return and free weight array
    laik_log(1, "lb: creating new partitioning for load balancing segment lb-%d using other=%s", segment, partitioning->name);
    Laik_Partitioning *npart = laik_new_partitioning(nparter, partitioning->group, partitioning->space, partitioning);

    laik_log(1, "lb: finished load balancing segment lb-%d, created new partitioning %s\n", segment, npart->name);
    segment++;

    laik_svg_profiler_exit(inst, __func__);
    return npart;
}

void laik_lb_switch_and_free(Laik_Partitioning **part, Laik_Partitioning **npart, Laik_Data *data, Laik_DataFlow flow)
{
    assert(part && npart && *part && *npart);

    if (*part == *npart)
        return;

    Laik_Instance *inst = (*part)->group->inst;
    laik_svg_profiler_enter(inst, __func__);

    laik_switchto_partitioning(data, *npart, flow, LAIK_RO_None);
    laik_free_partitioning(*part);
    *part = *npart;
    *npart = NULL;

    laik_svg_profiler_exit(inst, __func__);
}

void laik_lb_free() { swlist_free(swlist); }
void laik_lb_set_ext_weights(double *weights) { ext_weights = weights; }
void laik_lb_output(bool output) { do_print_times = output; }
bool laik_lb_is_stopped() { return stopped; }

void laik_lb_config_smoothing(bool smoothing, double am, double rmi, double rma)
{
    // check if we want to do smoothing in the first place and do nothing if we don't
    do_smoothing = smoothing;
    if (!do_smoothing)
    {
        laik_log(1, "lb/do_smoothing: smoothing DISABLED\n");
        return;
    }

    // only allow valid values (otherwise do nothing)
    if (am > 0.0 && am < 1.0)
        alpha_mul = am;
    if (rmi > 0.0)
        rmin = rmi;
    if (rma > 0.0)
        rmax = rma;

    laik_log(1, "lb/do_smoothing: smoothing ENABLED: alpha=%f, rmin=%f, rmax=%f\n", alpha_mul, rmin, rmax);
}

void laik_lb_config_thresholds(int pstop, int pstart, double tstop, double tstart)
{
    // only allow valid values (-1: do nothing)
    if (pstop > 0)
        p_stop = pstop;
    if (pstart > 0)
        p_start = pstart;
    if (tstop > 0.0)
        t_stop = tstop;
    if (tstart > 0.0)
        t_start = tstart;

    assert(t_stop < t_start); // also in main lb, just another sanity check here
    laik_log(1, "lb/config: configured start/stop parameters: p_stop=%d, p_start=%d, t_stop=%f, t_start=%f\n", p_stop, p_start, t_stop, t_start);
}
