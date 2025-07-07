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

// difference between minimum and maximum of a double array
static double min_max_difference(double *arr, size_t size)
{
    assert(size > 0);
    double max = arr[0];
    double min = arr[0];
    for (size_t i = 1; i < size; i++)
    {
        if (arr[i] > max)
            max = arr[i];
        if (arr[i] < min)
            min = arr[i];
    }
    return max - min;
}

// print times (elements in timearr) since last rebalance
static void print_times(double *timearr, size_t gsize, double maxdiff)
{
    printf("[LB] times in s since last rebalance: [");
    for (size_t i = 0; i < gsize; ++i)
    {
        printf("%.2f", timearr[i]);
        if (i < gsize - 1)
            printf(", ");
    }
    printf("], max dt: %.2f\n", maxdiff);
}

/////////////////////
// rcb partitioner //
/////////////////////

// TODO: make this incremental
double get_range_weight_from_index_1d(RangeWeight *rweights, unsigned rwcount, int idx)
{
    for (size_t i = 0; i < rwcount; ++i)
    {
        Laik_Range r = rweights[i].range;
        int64_t from = r.from.i[0];
        int64_t to = r.to.i[0];

        if (idx >= from && idx < to)
            return rweights[i].weight;
    }

    return 0.0;
}

double get_range_weight_from_index_2d(RangeWeight *rweights, unsigned rwcount, int x, int y)
{
    for (size_t i = 0; i < rwcount; ++i)
    {
        Laik_Range r = rweights[i].range;

        int64_t from_x = r.from.i[0];
        int64_t from_y = r.from.i[1];
        int64_t to_x = r.to.i[0];
        int64_t to_y = r.to.i[1];

        if (x >= from_x && y >= from_y && x < to_x && y < to_y)
            return rweights[i].weight;
    }

    return 0.0;
}

void get_range_weights(RangeWeight *rweights, Laik_RangeList *lr, double *timearr, int dims)
{
    if (dims == 1)
    {
        for (size_t i = 0; i < lr->count; ++i)
        {
            Laik_Range range = lr->trange[i].range;
            int rtask = lr->trange[i].task;
            int64_t rcount = range.to.i[0] - range.from.i[0];

            rweights[i].range = range;
            rweights[i].weight = (timearr[rtask] * 1000000) / rcount;

            laik_log(1, "[grw] task %d, i %ld, range %ld-%ld, weight %f\n", rtask, i, rweights[i].range.from.i[0], rweights[i].range.to.i[0], rweights[i].weight);
        }
    }
    else if (dims == 2)
    {
        for (size_t i = 0; i < lr->count; ++i)
        {
            Laik_Range range = lr->trange[i].range;

            int rtask = lr->trange[i].task;
            int64_t rcount_x = range.to.i[0] - range.from.i[0];
            int64_t rcount_y = range.to.i[1] - range.from.i[1];
            int64_t rcount = rcount_x * rcount_y;

            rweights[i].range = range;
            rweights[i].weight = (timearr[rtask] * 1000000) / rcount;

            laik_log(1, "[grw] task %d, i %ld, range [%ld,%ld] -> (%ld,%ld), rcount_x: %ld, r_count_y: %ld, weight %f\n", rtask, i, rweights[i].range.from.i[0], rweights[i].range.from.i[1], rweights[i].range.to.i[0], rweights[i].range.to.i[1], rcount_x, rcount_y, rweights[i].weight);
        }
    }
}

// internal 1d rcb helper function; [fromTask - toTask)
//
// note: like the range weight function above, i've separated this into a 1d and 2d version
//       partially due to recursion, mainly due to differences in the algorithm to avoid cluttering the function with ifs
static void rcb_1d(Laik_RangeReceiver *r, RangeWeight *rweights, unsigned rwcount, Laik_Range *range, int fromTask, int toTask)
{
    int64_t from = range->from.i[0];
    int64_t to = range->to.i[0];

    // if there's only one processor left, stop here
    int count = toTask - fromTask + 1;
    if (count == 1 || from > to)
    {
        laik_append_range(r, fromTask, range, 0, 0);
        return;
    }

    // calculate how many procs go left vs. right
    int lcount = count / 2;
    int rcount = count - lcount;
    int tmid = fromTask + lcount - 1;

    // calculate sum of weights in current range (naive)
    double totalW = 0.0;
    for (int64_t i = from; i < to; ++i)
        totalW += get_range_weight_from_index_1d(rweights, rwcount, i);

    // calculate target weight of left child
    double ltarget = totalW * ((double)lcount / count);

    laik_log(1, "[rcb1d] [T%d-T%d) [%ld-%ld) count: %d, lcount: %d, rcount: %d, tmid: %d, totalW: %f, ltarget: %f\n", fromTask, toTask, from, to, count, lcount, rcount, tmid, totalW, ltarget);

    // find first index where prefix sum exceeds target weight
    double sum = 0.0;
    int64_t split = from;
    for (int64_t i = from; i < to; ++i)
    {
        sum += get_range_weight_from_index_1d(rweights, rwcount, i);
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
    rcb_1d(r, rweights, rwcount, &r1, fromTask, tmid);
    rcb_1d(r, rweights, rwcount, &r2, tmid + 1, toTask);
}

// internal 2d rcb helper function; [fromTask - toTask)
//
// primary changes are determining the split direction by checking which side is longest and how the weights are computed / accumulated
static void rcb_2d(Laik_RangeReceiver *r, RangeWeight *rweights, unsigned rwcount, Laik_Range *range, int fromTask, int toTask)
{
    int64_t from_x = range->from.i[0];
    int64_t from_y = range->from.i[1];
    int64_t to_x = range->to.i[0];
    int64_t to_y = range->to.i[1];

    // if there's only one processor left, stop here
    int count = toTask - fromTask + 1;
    if (count == 1)
    {
        laik_append_range(r, fromTask, range, 0, 0);
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
            totalW += get_range_weight_from_index_2d(rweights, rwcount, i, j);

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
                sum += get_range_weight_from_index_2d(rweights, rwcount, x, y);
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
                sum += get_range_weight_from_index_2d(rweights, rwcount, x, y);
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
    rcb_2d(r, rweights, rwcount, &r1, fromTask, tmid);
    rcb_2d(r, rweights, rwcount, &r2, tmid + 1, toTask);
}

void runRCBPartitioner(Laik_RangeReceiver *r, Laik_PartitionerParams *p)
{
    RCBData *rcbd = (RCBData *)p->partitioner->data;
    assert(rcbd);

    RangeWeight *rweights = rcbd->rweights;
    unsigned rwcount = rcbd->count;
    unsigned tidcount = rcbd->tidcount;
    int dims = rcbd->dims;

    Laik_Space *s = p->space;
    Laik_Range range = s->range;

    if (dims == 1)
        rcb_1d(r, rweights, rwcount, &range, 0, tidcount - 1);
    else if (dims == 2)
        rcb_2d(r, rweights, rwcount, &range, 0, tidcount - 1);
}

Laik_Partitioner *laik_new_rcb_partitioner(RangeWeight *rweights, unsigned count, unsigned tidcount, int dims)
{
    RCBData *rcbd;
    rcbd = malloc(sizeof(RCBData));
    if (!rcbd)
        laik_panic("Could not allocate enough memory for RCB data!");

    rcbd->rweights = rweights;
    rcbd->count = count;
    rcbd->tidcount = tidcount;
    rcbd->dims = dims;

    return laik_new_partitioner("rcb", runRCBPartitioner, (void *)rcbd, 0);
}

////////////////////
// load balancing //
////////////////////

int laik_lb_measure(Laik_Group *group, double *timearr)
{
    static double time = 0;

    // on first call, just record time and exit function
    if (time == 0)
    {
        if (laik_myid(group) == 0)
            printf("[LB] first lb call, registering start time...\n");
        time = laik_wtime();
        return 1;
    }

    Laik_Instance *inst = group->inst;
    int gsize = laik_size(group);
    int task = laik_myid(group);
    memset(timearr, 0, sizeof(double) * gsize);

    // collect time of current task since last measurement and write to corresponding index in array
    //
    // e.g. for three processes p{0,1,2}:
    // p0: [t0,0,0], p1: [0,t1,0], p2: [0,0,t2]
    timearr[task] = laik_wtime() - time;

    // initialize laik space for aggregating times
    Laik_Space *timespace;
    Laik_Data *timedata;
    Laik_Partitioning *timepart1, *timepart2;

    // use timearr directly as input data
    timespace = laik_new_space_1d(inst, gsize);
    timedata = laik_new_data(timespace, laik_Double);
    timepart1 = laik_new_partitioning(laik_All, group, timespace, NULL);
    laik_data_provide_memory(timedata, timearr, gsize * sizeof(double));
    laik_set_initial_partitioning(timedata, timepart1);

    // collect times into timearr, shared among all tasks
    timepart2 = laik_new_partitioning(laik_All, group, timespace, NULL);
    laik_switchto_partitioning(timedata, timepart2, LAIK_DF_Preserve, LAIK_RO_Sum);

    // update time for next balancing call
    time = laik_wtime();
    return 0;
}

Laik_Partitioning *laik_lb_balance(Laik_Partitioning *partitioning /*, Laik_Partitioner *algorithm, double threshold*/)
{
    Laik_Space *space = partitioning->space;
    Laik_Group *group = partitioning->group;
    int task = laik_myid(group);
    int gsize = laik_size(group);
    int dims = partitioning->space->dims;

    // collect arrival times of all tasks upon calling the balancing function
    double timearr[gsize]; // collected times will be stored here; idx corresponds to task

    // if the balancing function is called for the first time, return the partitioning unchanged
    if (laik_lb_measure(group, timearr))
        return partitioning;

    // otherwise, continue with the balancing algorithm...
    double maxdiff = min_max_difference(timearr, gsize);
    if (task == 0)
        print_times(timearr, gsize, maxdiff);

    // calculate weights for each range
    Laik_RangeList *lr = laik_partitioning_myranges(partitioning);
    RangeWeight rweights[lr->count];
    get_range_weights(rweights, lr, timearr, dims);

    // use task weights to create new partitioning using RCB
    Laik_Partitioner *nparter = laik_new_rcb_partitioner(rweights, lr->count, gsize, dims);
    return laik_new_partitioning(nparter, group, space, partitioning);
}
