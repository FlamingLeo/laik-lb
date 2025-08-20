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

//////////////////////////////
// generic helper functions //
//////////////////////////////

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

// print times (elements in weight array starting at offset size) since last rebalance
static void print_times(double *times, int gsize, double maxdt, double mean)
{
    printf("[LAIK-LB] times in s for this segment: [");
    for (int i = 0; i < gsize; ++i)
    {
        printf("%.2f", times[i]);
        if (i < gsize - 1)
            printf(", ");
    }
    printf("], max dt: %.2f, mean %.2f, rel. imbalance %.2f\n", maxdt, mean, maxdt / mean);
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

// public: get algorithm enum from string
// unknown algorithm -> fall back to rcb
Laik_LBAlgorithm laik_strtolb(const char *str)
{
    if (strcmp(str, "rcb") == 0)
        return LB_RCB;
    else if (strcmp(str, "hilbert") == 0)
        return LB_HILBERT;
    else if (strcmp(str, "gilbert") == 0)
        return LB_GILBERT;
    else
        return LB_RCB;
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
// TODO: preprocessing to avoid multiple grid scans

// merge rectangles (2d)
static void merge_rects(int *grid1D, int64_t width, int64_t height, Laik_RangeReceiver *r, int tidcount)
{
    Laik_Instance *inst = r->list->space->inst;
    laik_svg_profiler_enter(inst, __func__);

    for (int tid = 0; tid < tidcount; ++tid)
    {
        // keep carving until no more cells are found for this task
        for (;;)
        {
            bool found = false;
            int64_t x0 = 0, y0 = 0;

            // find bottom‑leftmost cell == tid
            for (int64_t y = 0; y < height && !found; ++y)
            {
                for (int64_t x = 0; x < width; ++x)
                {
                    if (grid1D[IDX2D(x, y)] == tid)
                    {
                        x0 = x;
                        y0 = y;
                        found = true;
                        break;
                    }
                }
            }

            // couldn't find anything for this task?
            if (!found)
                break;

            // scan right to get initial width
            int64_t w = 0;
            while (x0 + w < width && grid1D[IDX2D(x0 + w, y0)] == tid)
                ++w;

            // extend upward to maximize area
            int64_t best_area = w;
            int64_t best_w = w; // remember the width at which best_area occurs
            int64_t best_h = 1;
            int64_t curr_w = w;

            for (int64_t h = 2; y0 + h <= height; ++h)
            {
                // measure run of tids in row y0 + h − 1
                int64_t w_h = 0;
                while (x0 + w_h < width && grid1D[IDX2D(x0 + w_h, y0 + h - 1)] == tid)
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

            // record the rectangle using best_w and best_h
            laik_append_range(r, tid, &(Laik_Range){.space = r->list->space, .from = {{x0, y0, 0}}, .to = {{x0 + best_w, y0 + best_h, 0}}}, 0, 0);

            // mark covered cells “used” by setting them to -1
            for (int64_t dy = 0; dy < best_h; ++dy)
                for (int64_t dx = 0; dx < best_w; ++dx)
                    grid1D[IDX2D(x0 + dx, y0 + dy)] = -1;
        }
    }

    laik_svg_profiler_exit(inst, __func__);
}

// merge cuboids (3d, x: width, y: height, z: depth)
static void merge_cuboids(int *grid1D, int64_t width, int64_t height, int64_t depth, Laik_RangeReceiver *r, int tidcount)
{
    Laik_Instance *inst = r->list->space->inst;
    laik_svg_profiler_enter(inst, __func__);

    for (int tid = 0; tid < tidcount; ++tid)
    {
        for (;;)
        {
            bool found = false;
            int64_t x0 = 0, y0 = 0, z0 = 0;

            // find bottom-front-leftmost smallest cell == tid
            for (int64_t z = 0; z < depth && !found; ++z)
            {
                for (int64_t y = 0; y < height && !found; ++y)
                {
                    for (int64_t x = 0; x < width; ++x)
                    {
                        if (grid1D[IDX3D(x, y, z)] == tid)
                        {
                            x0 = x;
                            y0 = y;
                            z0 = z;
                            found = true;
                            break;
                        }
                    }
                }
            }

            // couldn't find anything for this task?
            if (!found)
                break;

            // dimensions remaining from the anchor
            int64_t rem_h = height - y0;
            int64_t rem_d = depth - z0;

            // precompute minimal run length
            int64_t *w_layer = (int64_t *)safe_malloc(sizeof(int64_t) * rem_d * rem_h);

            for (int64_t dz = 0; dz < rem_d; ++dz)
            {
                int64_t z = z0 + dz;
                int64_t curr_w = INT64_MAX;
                for (int64_t hh = 1; hh <= rem_h; ++hh)
                {
                    int64_t y = y0 + (hh - 1);

                    // compute run length in x at (x0, y, z)
                    int64_t run = 0;
                    while (x0 + run < width &&
                           grid1D[IDX3D(x0 + run, y, z)] == tid)
                        ++run;

                    if (hh == 1)
                        curr_w = run;
                    else if (run < curr_w)
                        curr_w = run;

                    // store curr_w (may be 0)
                    w_layer[dz * rem_h + (hh - 1)] = curr_w;

                    // early stop for this layer if curr_w == 0: further hh will be zero too
                    if (curr_w == 0)
                    {
                        // fill remaining hh for this layer with 0 if any
                        for (int64_t hh2 = hh + 1; hh2 <= rem_h; ++hh2)
                            w_layer[dz * rem_h + (hh2 - 1)] = 0;
                        break;
                    }
                }
            }

            // search for (w,h,d) maximizing volume
            int64_t best_vol = 0;
            int64_t best_w = 0, best_h = 0, best_d = 0;

            for (int64_t d = 1; d <= rem_d; ++d)
            {
                for (int64_t h = 1; h <= rem_h; ++h)
                {
                    int64_t min_w = INT64_MAX;
                    for (int64_t dz = 0; dz < d; ++dz)
                    {
                        int64_t w_for_layer = w_layer[dz * rem_h + (h - 1)];
                        if (w_for_layer < min_w)
                            min_w = w_for_layer;
                        if (min_w == 0)
                            break;
                    }
                    if (min_w == 0)
                        continue;
                    int64_t vol = min_w * h * d;
                    if (vol > best_vol)
                    {
                        best_vol = vol;
                        best_w = min_w;
                        best_h = h;
                        best_d = d;
                    }
                }
            }

            free(w_layer);

            // best_vol must be > 0 because at least the anchor cell exists
            // fallback: take the single anchor cell
            if (best_vol == 0)
            {
                best_w = 1;
                best_h = 1;
                best_d = 1;
            }

            // record the cuboid using best_w, best_h, best_d
            laik_append_range(r, tid, &(Laik_Range){.space = r->list->space, .from = {{x0, y0, z0}}, .to = {{
                                                                                                         x0 + best_w,
                                                                                                         y0 + best_h,
                                                                                                         z0 + best_d,
                                                                                                     }}},
                              0, 0);

            // mark covered cells used (set to -1)
            for (int64_t dz = 0; dz < best_d; ++dz)
                for (int64_t dy = 0; dy < best_h; ++dy)
                    for (int64_t dx = 0; dx < best_w; ++dx)
                        grid1D[IDX3D(x0 + dx, y0 + dy, z0 + dz)] = -1;
        }
    }

    laik_svg_profiler_exit(inst, __func__);
}

//////////////////////
// sfc partitioners //
//////////////////////

// TODO: consider allowing full 64-bit range for indices

/* hilbert space filling curve (2D, 3D) using Skilling's bitwise method */
/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= 2D =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= 3D =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
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
static uint64_t gilbert_d2xy_r(uint64_t dst_idx, uint64_t cur_idx,
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
static int64_t gilbert_d2xy(int64_t *x, int64_t *y, uint64_t idx, int64_t w, int64_t h)
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
int64_t gilbert_d2xyz_r(uint64_t dst_idx, uint64_t cur_idx,
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
int64_t gilbert_d2xyz(int64_t *x, int64_t *y, int64_t *z, uint64_t idx, int64_t width, int64_t height, int64_t depth)
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

// TODO: verify types for both

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
    merge_rects(idxGrid, size_x, size_y, r, tidcount);

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
    merge_cuboids(idxGrid, size_x, size_y, size_z, r, tidcount);

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
    assert((algo != LB_RCB) && "that's not a space-filling curve!");

    LB_SFC_Data *data = (LB_SFC_Data *)safe_malloc(sizeof(LB_SFC_Data));
    data->weights = weights;
    data->algo = algo;

    return laik_new_partitioner(laik_get_lb_algorithm_name(algo), runSFCPartitioner, (void *)data, 0);
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
    rcb_2d(r, &r1, fromTask, tmid, weights);
    rcb_2d(r, &r2, tmid + 1, toTask, weights);
    laik_svg_profiler_exit(inst, __func__);
}

// internal 3d rcb helper function
static void rcb_3d(Laik_RangeReceiver *r, Laik_Range *range, int fromTask, int toTask, double *weights)
{
    Laik_Instance *inst = r->params->space->inst;
    laik_svg_profiler_enter(inst, __func__);

    int64_t size_x = range->space->range.to.i[0] - range->space->range.from.i[0];
    int64_t size_y = range->space->range.to.i[1] - range->space->range.from.i[1];

    int64_t from_x = range->from.i[0];
    int64_t from_y = range->from.i[1];
    int64_t from_z = range->from.i[2];
    int64_t to_x = range->to.i[0];
    int64_t to_y = range->to.i[1];
    int64_t to_z = range->to.i[2];

    // if there's only one processor left, stop here
    int count = toTask - fromTask + 1;
    if (count == 1)
    {
        laik_append_range(r, fromTask, range, 0, 0);
        laik_svg_profiler_exit(inst, __func__);
        return;
    }

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

    rcb_3d(r, &r1, fromTask, tmid, weights);
    rcb_3d(r, &r2, tmid + 1, toTask, weights);

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
        rcb_1d(r, &range, 0, tidcount - 1, weights);
    else if (dims == 2)
        rcb_2d(r, &range, 0, tidcount - 1, weights);
    else /* if (dims == 3) */
        rcb_3d(r, &range, 0, tidcount - 1, weights);

    laik_svg_profiler_exit(inst, __func__);
}

Laik_Partitioner *laik_new_rcb_partitioner(double *weights)
{
    return laik_new_partitioner("rcb", runRCBPartitioner, (void *)weights, 0);
}

////////////////////
// load balancing //
////////////////////

// calculate the imbalance to determine whether or not load balancing should be done
//
// TODO: consider moving average to smooth out possible noise (EMA)?
// formula: (max - min) / mean
static double get_imbalance(Laik_Partitioning *p, double ttime)
{
    Laik_Group *group = p->group;
    Laik_Instance *inst = group->inst;
    laik_svg_profiler_enter(inst, __func__);

    int task = laik_myid(group);
    int gsize = group->size;

    // times will be stored in here
    double times[gsize];
    memset(times, 0, gsize * sizeof(double));

    // store own time
    times[task] = ttime;

    // aggregate times
    // initialize laik space for aggregating weights
    Laik_Space *tspace;
    Laik_Data *tdata;
    Laik_Partitioning *tpart;

    // use weights directly as input data
    laik_log(1, "lb/get_imbalance: aggregating task times...\n");
    tspace = laik_new_space_1d(inst, gsize);
    tdata = laik_new_data(tspace, laik_Double);
    tpart = laik_new_partitioning(laik_All, group, tspace, NULL);
    laik_data_provide_memory(tdata, times, gsize * sizeof(double));
    laik_set_initial_partitioning(tdata, tpart);

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
    if (task == 0)
        print_times(times, gsize, maxdt, mean);

    // free partitionings since we don't need them anymore
    laik_free_partitioning(tpart);
    laik_free_space(tspace);

    // return relative threshold ((max - min) / mean)
    laik_svg_profiler_exit(inst, __func__);
    return maxdt / mean;
}

// initialize the weight array for the current load balancing run
static double *init_weights(Laik_Partitioning *p, double ttime)
{
    double *weights;

    Laik_Space *space = p->space;
    Laik_Group *group = p->group;
    Laik_Instance *inst = group->inst;
    laik_svg_profiler_enter(inst, __func__);

    // allocate weight array and zero-initialize
    // for t tasks, the final t elements of the array are the raw times taken by each task (starting from 0, one after the last weight)
    int dims = p->space->dims; // assume dims in {1,2,3}
    int64_t size_x = space->range.to.i[0] - space->range.from.i[0];
    int64_t size_y = dims >= 2 ? (space->range.to.i[1] - space->range.from.i[1]) : 1;
    int64_t size_z = dims == 3 ? (space->range.to.i[2] - space->range.from.i[2]) : 1;
    int64_t size = size_x * size_y * size_z;

    weights = (double *)safe_malloc(size * sizeof(double));
    memset(weights, 0, size * sizeof(double));

    laik_log(1, "lb/init_weights: initialized weight array of %ld bytes (%f kB) (dims: %d, size: %ldx%ldx%ld=%ld)\n", size * sizeof(double), .001 * size * sizeof(double), dims, size_x, size_y, size_z, size);

    // calculate weight and fill array at own indices
    // 1. accumulate number of items
    // 2. get task weight for task i as time_taken(i) * c / nitems(i), where c is some constant
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
                weights[i] = tweight;
        }
        else if (dims == 2)
        {
            int64_t from_x, from_y, to_x, to_y;
            laik_my_range_2d(p, r, &from_x, &to_x, &from_y, &to_y);
            for (int64_t y = from_y; y < to_y; ++y)
                for (int64_t x = from_x; x < to_x; ++x)
                    weights[y * size_x + x] = tweight;
        }
        else /* if (dims == 3) */
        {
            int64_t from_x, from_y, from_z, to_x, to_y, to_z;
            laik_my_range_3d(p, r, &from_x, &to_x, &from_y, &to_y, &from_z, &to_z);
            for (int64_t z = from_z; z < to_z; ++z)
                for (int64_t y = from_y; y < to_y; ++y)
                    for (int64_t x = from_x; x < to_x; ++x)
                        weights[x + size_x * (y + size_y * z)] = tweight;
        }
    }

    // initialize laik space for aggregating weights
    Laik_Space *weightspace;
    Laik_Data *weightdata;
    Laik_Partitioning *weightpart;

    // use weights directly as input data
    laik_log(1, "lb/init_weights: aggregating weights...\n");

    weightspace = laik_new_space_1d(inst, size);
    weightdata = laik_new_data(weightspace, laik_Double);
    weightpart = laik_new_partitioning(laik_All, group, weightspace, NULL);
    laik_data_provide_memory(weightdata, weights, size * sizeof(double));
    laik_set_initial_partitioning(weightdata, weightpart);

    // collect times into weights, shared among all tasks
    laik_switchto_partitioning(weightdata, weightpart, LAIK_DF_Preserve, LAIK_RO_Sum);

    // free partitionings since we don't need them anymore
    laik_free_partitioning(weightpart);
    laik_free_space(weightspace);

    laik_svg_profiler_exit(inst, __func__);
    return weights;
}

// TODO: consider metric where overhead for sending data outweighs possible gains by load balancing, and if this is even necessary / good
Laik_Partitioning *laik_lb_balance(Laik_LBState state, Laik_Partitioning *partitioning, Laik_LBAlgorithm algorithm)
{
    static double time = 0;
    static const int p_stop = 3;       // stopping patience
    static const int p_start = 3;      // starting patience
    static const double t_stop = 0.05; // stop load balancing when relative imbalance is UNDER this threshold for p_stop consecutive times
    static const double t_start = 0.1; // restart load balancing when relative imbalance is OVER this threshold for p_start consecutive times
    static bool stopped = false;
    static int p_stopctr = 0;
    static int p_startctr = 0;
    static int segment = 0; // load balancing segment, for debugging purposes

    assert(t_stop < t_start && "stopping threshold should be under starting threshold");

    // when starting a new load balancing segment, start timer and do nothing else
    if (state == START_LB_SEGMENT)
    {
        laik_log(1, "lb: starting new load balancing segment lb-%d (stopped? %d)\n", segment, stopped);
        time = laik_wtime();
        return NULL;
    }

    // otherwise, stop timer and perform load balancing checks and algorithm
    Laik_Instance *inst = partitioning->group->inst;
    laik_svg_profiler_enter(inst, __func__);

    laik_log_begin(1);
    laik_log_append("lb: stopping load balancing segment lb-%d based on partitioning %s using algorithm %s after ", segment, partitioning->name, laik_get_lb_algorithm_name(algorithm));

    // check relative imbalance to determine whether or not to perform load balancing
    double ttime = laik_wtime() - time;
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
    free(weights);

    laik_log(1, "lb: finished load balancing segment lb-%d, created new partitioning %s\n", segment, npart->name);
    segment++;

    laik_svg_profiler_exit(inst, __func__);
    return npart;
}
