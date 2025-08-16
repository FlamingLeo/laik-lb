// testing laik functionality
#include <laik.h>
#include <laik-internal.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <stdint.h>
#include <stdarg.h>
#include <sys/types.h>
#include <sys/stat.h>

// visualize space by moving it to master process and printing out mapping contents
#define DEBUG

#ifdef DEBUG
#define PRINT(_data, _part, _base, _ysize, _ystride, _xsize, _text)           \
    pm = laik_new_partitioning(laik_Master, world, space, 0);                 \
    laik_switchto_partitioning(_data, pm, LAIK_DF_Preserve, LAIK_RO_None);    \
    laik_get_map_2d(_data, 0, (void **)&_base, &_ysize, &_ystride, &_xsize);  \
    if (laik_myid(world) == 0)                                                \
    {                                                                         \
        fprintf(fptrmaster, _text ":\n");                                     \
        for (uint64_t y = 0; y < _ysize; ++y)                                 \
        {                                                                     \
            for (uint64_t x = 0; x < _xsize; ++x)                             \
                fprintf(fptrmaster, "%4d ", _base[y * _ystride + x]);         \
            fprintf(fptrmaster, "\n");                                        \
        }                                                                     \
        fprintf(fptrmaster, "\n");                                            \
    }                                                                         \
    laik_switchto_partitioning(_data, _part, LAIK_DF_Preserve, LAIK_RO_None); \
    laik_get_map_2d(_data, 0, (void **)&_base, &_ysize, &_ystride, &_xsize);
#else
#define PRINT(_data, _part, _base, _ysize, _ystride, _xsize, _text) (void)0;
#endif

// c string buffer append function i took from an older project of mine
// this is done to nicely print and flush everything at once instead of broken up and between log info by just doing normal printfs
static int append_to_buf(char **buf, size_t *len, size_t *cap, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int needed = vsnprintf(NULL, 0, fmt, ap2);
    va_end(ap2);
    if (needed < 0)
    {
        va_end(ap);
        return -1;
    }
    size_t needed_sz = (size_t)needed;
    if (*len + needed_sz + 1 > *cap)
    {
        size_t newcap = (*cap == 0) ? (*len + needed_sz + 1) : *cap;
        while (newcap < *len + needed_sz + 1)
            newcap *= 2;
        char *tmp = (char *)realloc(*buf, newcap);
        if (!tmp)
        {
            va_end(ap);
            return -1;
        }
        *buf = tmp;
        *cap = newcap;
    }

    vsnprintf(*buf + *len, *cap - *len, fmt, ap);
    *len += needed_sz;
    (*buf)[*len] = '\0';
    va_end(ap);
    return 0;
}

int main(int argc, char *argv[])
{
    int64_t sdsize = 16;
    int64_t spsize = sdsize * sdsize;
    int lcount = 10;

    // initialization
    Laik_Instance *inst = laik_init(&argc, &argv);
    Laik_Group *world = laik_world(inst);
    int id = laik_myid(world);
    Laik_LBAlgorithm lbalg = 1;

    if (id == 0)
    {
        laik_log(1, "Running 2D example with %d iterations.\n", lcount);
        laik_log(1, "Space size %ldx%ld = %ld.\n", sdsize, sdsize, spsize);
        laik_log(1, "Using LB algorithm: %s\n", laik_get_lb_algorithm_name(lbalg));
    }

    Laik_Space *space = laik_new_space_2d(inst, sdsize, sdsize);
    Laik_Data *data = laik_new_data(space, laik_Int32);

    Laik_Partitioner *parter = laik_new_bisection_partitioner();
    Laik_Partitioning *pm;
    Laik_Partitioning *pWrite = laik_new_partitioning(parter, world, space, 0);
    laik_switchto_partitioning(data, pWrite, LAIK_DF_None, LAIK_RO_None);

    int iterations = (id + 1) * 10;
    double c = 0.25; // sleep time constant multiplier

    // debug logging
    int64_t from_x, from_y, to_x, to_y;
    laik_my_range_2d(pWrite, 0, &from_x, &to_x, &from_y, &to_y);
    int64_t count_x = to_x - from_x;
    int64_t count_y = to_y - from_y;

    // output log file
    struct stat st = {0};
    if (stat("logs", &st) == -1)
        mkdir("logs", 0700);

    FILE *fptr, *fptrmaster;
    char filename[MAX_FILENAME_LENGTH];
    sprintf(filename, "logs/logT%d.log", id);
    fptr = fopen(filename, "w");
    if (id == 0)
        fptrmaster = fopen("logs/logMaster.log", "w");

    // arbitrary data to check for bugs (xxyy)
    uint64_t ysizeW, ystrideW, xsizeW;
    int32_t *baseW, localSum = 0;
    laik_get_map_2d(data, 0, (void **)&baseW, &ysizeW, &ystrideW, &xsizeW);
    for (uint64_t y = 0; y < ysizeW; y++)
    {
        for (uint64_t x = 0; x < xsizeW; x++)
        {
            int32_t val = ((from_x + x) * 100 + (from_y + y));
            baseW[y * ystrideW + x] = val;
            localSum += val;
        }
    }

    // modified version of the 1d example, to be changed to something more "professional" later (e.g. n-body)
    for (int loop = 0; loop < lcount; ++loop)
    {
        if (id == 0)
            fprintf(fptrmaster, "loop iteration %d\n", loop);

        laik_log(1, "starting iteration %d\n", loop);
        laik_log(1, "starting load balance segment...\n");
        laik_lb_balance(START_LB_SEGMENT, 0, 0);
        for (int iter = 0; iter < iterations; ++iter)
        {
            uint64_t sleepdur = 0;
            for (int r = 0; r < laik_my_rangecount(pWrite); ++r)
            {
                int64_t from_x, from_y, to_x, to_y;
                laik_my_range_2d(pWrite, r, &from_x, &to_x, &from_y, &to_y);
                /*
                laik_get_map_2d(data, r, (void **)&baseW, &ysizeW, &ystrideW, &xsizeW);
                for (int64_t y = 0; y < ysizeW; ++y)
                    for (int64_t x = 0; x < xsizeW; ++x)
                        if (iter == 0)
                            baseW[y * ystrideW + x]++;
                */

                sleepdur += (to_x - from_x) * (to_y - from_y);
            }
            usleep((uint64_t)((double)sleepdur * c));
        }
        laik_log(1, "finished load balance segment... calculating new partitioning\n");

        // calculate and switch to new partitioning determined by load balancing algorithm
        Laik_Partitioning *npWrite = laik_lb_balance(STOP_LB_SEGMENT, pWrite, lbalg);
        if (pWrite == npWrite)
            continue;

        laik_log(1, "finished calculating new partitioning, switching...\n");
        laik_switchto_partitioning(data, npWrite, LAIK_DF_Preserve, LAIK_RO_None);
        laik_log(1, "finished switching to new partitioning\n");

        // free old partitioning
        laik_free_partitioning(pWrite);
        pWrite = npWrite;

        char *out = NULL;
        size_t out_len = 0, out_cap = 0;

        append_to_buf(&out, &out_len, &out_cap, "after loop %d\n", loop);

        // write range and map contents of each task
        for (int r = 0; r < laik_my_rangecount(pWrite); ++r)
        {
            laik_my_range_2d(pWrite, r, &from_x, &to_x, &from_y, &to_y);
            laik_get_map_2d(data, r, (void **)&baseW, &ysizeW, &ystrideW, &xsizeW);

            append_to_buf(&out, &out_len, &out_cap, "%d r%d: [%ld, %ld]->(%ld, %ld)\n", id, r, from_x, from_y, to_x, to_y);
            append_to_buf(&out, &out_len, &out_cap, "%d r%d: xsize %ld, ysize %ld, ystride %ld\n", id, r, xsizeW, ysizeW, ystrideW);
            append_to_buf(&out, &out_len, &out_cap, "%d r%d: values:\n", id, r);

            for (int64_t y = 0; y < ysizeW; ++y)
            {
                for (int64_t x = 0; x < xsizeW; ++x)
                {
                    append_to_buf(&out, &out_len, &out_cap, "%d ", baseW[y * ystrideW + x]);
                }
                append_to_buf(&out, &out_len, &out_cap, "\n");
            }
            append_to_buf(&out, &out_len, &out_cap, "\n");
        }
        append_to_buf(&out, &out_len, &out_cap, "done\n\n");

        if (out && out_len > 0)
            fwrite(out, 1, out_len, fptr);

        free(out);

        // print data
        laik_log(1, "printing output to file...\n");
        PRINT(data, pWrite, baseW, ysizeW, ystrideW, xsizeW, "baseW after switch (end)");
        laik_log(1, "finished loop iteration %d\n", loop);
    }

    // sum for testing
    Laik_Group *activeGroup = laik_data_get_group(data);
    Laik_Partitioning *pMaster = laik_new_partitioning(laik_Master, activeGroup, space, 0);
    laik_switchto_partitioning(data, pMaster, LAIK_DF_Preserve, LAIK_RO_None);

    if (laik_myid(activeGroup) == 0)
    {
        int32_t sum = 0;

        // n = 0, since we've moved data to master
        laik_get_map_2d(data, 0, (void **)&baseW, &ysizeW, &ystrideW, &xsizeW);
        for (uint64_t y = 0; y < ysizeW; y++)
            for (uint64_t x = 0; x < xsizeW; x++)
                sum += baseW[y * ystrideW + x];

        printf("Global value sum iterations: %d\n", sum);
    }

    // done
    laik_finalize(inst);

    return 0;
}

#if 0
// load balancing API / workflow example
#include <laik.h>
#include <laik-internal.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define FILENAME "lbviz/array_data.txt"
#define DO_VISUALIZATION

#ifdef DO_VISUALIZATION
#define EXPORT_TO_FILE(_id, _part)                              \
    if (_id == 0)                                               \
    {                                                           \
        Laik_RangeList *lr = laik_partitioning_myranges(_part); \
        export_to_file(lr);                                     \
    }
#define VISUALIZE(_id) \
    if (_id == 0)      \
        visualize();
#else
#define EXPORT_TO_FILE(_id, _part) (void)0
#define VISUALIZE(_id) (void)0
#endif

// export indices associated with tasks to newline-separated file for visualization purposes
//
// format:
// - 1d: (idx, task)\n
// - 2d: ((x,y), task)\n
//
// note: there's better, more efficient ways of storing and communicating this data to the python script
//       this is mainly here for debugging right now
// TODO: ideally call this + visualization script somehow after each iteration to view progress, avoiding I/O slowdown if possible (extra thread / proc?)
static void export_to_file(Laik_RangeList *lr)
{
    FILE *fp = fopen(FILENAME, "w");
    if (!fp)
        return;

    int dims = lr->space->dims;
    for (size_t i = 0; i < lr->count; ++i)
    {
        Laik_Range r = lr->trange[i].range;
        int task = lr->trange[i].task;

        if (dims == 1)
        {
            int64_t from = r.from.i[0];
            int64_t to = r.to.i[0];

            for (int64_t j = from; j < to; ++j)
                fprintf(fp, "(%ld,%d)\n", j, task);
        }
        else if (dims == 2)
        {
            int64_t from_x = r.from.i[0], from_y = r.from.i[1];
            int64_t to_x = r.to.i[0], to_y = r.to.i[1];

            for (int64_t x = from_x; x < to_x; ++x)
                for (int64_t y = from_y; y < to_y; ++y)
                    fprintf(fp, "((%ld,%ld),%d)\n", x, y, task);
        }
    }

    fclose(fp);
}

// call task visualization script from inside example directory automatically; no error checking for now
static void visualize()
{
    system("python3 lbviz/visualize.py");
}

// purge json data and images
static void remove_plots()
{
    system("../scripts/remove_plots.sh");
}

// plot program trace through external script
static void save_trace()
{
    system("python3 lbviz/trace.py");
}

// enable program trace visualization
static void enable_trace(int id, Laik_Instance *inst)
{
    char filename[MAX_FILENAME_LENGTH];
    sprintf(filename, "lbviz/lb-%d.json", id);
    laik_svg_enable_profiling(inst, filename);
}

////////////////////////
// iteration examples //
////////////////////////

// 1d example
int main_1d(int argc, char *argv[], int64_t spsize, int lcount)
{
    // initialize 1d index space of size spsize, each task getting the same slice size
    Laik_Instance *inst = laik_init(&argc, &argv);
    Laik_Group *world = laik_world(inst);
    int id = laik_myid(world);
    Laik_LBAlgorithm lbalg = LB_RCB;

    enable_trace(id, inst);
    laik_svg_profiler_enter(inst, __func__);

    if (id == 0)
    {
        printf("Running 1D example with %d iterations.\n", lcount);
        printf("Space size %ld.\n", spsize);
        printf("Using LB algorithm: %s\n", laik_get_lb_algorithm_name(lbalg));
    }

    Laik_Space *space = laik_new_space_1d(inst, spsize);
    Laik_Data *data = laik_new_data(space, laik_Int32);
    Laik_Partitioner *parter = laik_new_block_partitioner1();
    Laik_Partitioning *part = laik_new_partitioning(parter, world, space, 0);
    laik_switchto_partitioning(data, part, LAIK_DF_None, LAIK_RO_None);

    int iterations = (id + 1) * 10;

    // log range (debug)
    int64_t from, to;
    laik_my_range_1d(part, 0, &from, &to);
    laik_log(2, "[main1d] init. range [%ld, %ld) count %ld; performing %d iterations \n", from, to, to - from, iterations);

    // run the example lcount times to test load balancing (and stopping threshold)
    // maybe use laik's iteration functionality somewhere here?
    for (int i = 0; i < lcount; ++i)
    {
        // each task runs for a fixed task-specific number of iterations
        // for simplicity: 1 item == 1 microsecond
        laik_lb_balance(START_LB_SEGMENT, 0, 0);
        for (int j = 0; j < iterations; ++j)
        {
            uint64_t count;
            laik_get_map_1d(data, 0, NULL, &count);
            uint64_t sleepdur = count;
            usleep(sleepdur);
        }

        // calculate and switch to new partitioning determined by load balancing algorithm
        Laik_Partitioning *newpart = laik_lb_balance(STOP_LB_SEGMENT, part, lbalg);
        if (part == newpart)
            continue;

        laik_my_range_1d(newpart, 0, &from, &to);
        laik_log(2, "[main1d] new range [%ld, %ld) count %ld\n", from, to, to - from);
        laik_switchto_partitioning(data, newpart, LAIK_DF_None, LAIK_RO_None);

        // free old partitioning
        laik_free_partitioning(part);
        part = newpart;
    }

    // visualize task ranges
    EXPORT_TO_FILE(id, part);
    VISUALIZE(id);

    // done
    laik_svg_profiler_exit(inst, __func__);
    laik_svg_profiler_export_json(inst);
    laik_finalize(inst);
    if (id == 0)
        save_trace();
    return 0;
}

// 2d example
int main_2d(int argc, char *argv[], int64_t spsize, int lcount)
{
    // initialization
    int64_t sdsize = (int64_t)sqrt(spsize);
    Laik_Instance *inst = laik_init(&argc, &argv);
    Laik_Group *world = laik_world(inst);
    int id = laik_myid(world);
    Laik_LBAlgorithm lbalg = LB_HILBERT;

    enable_trace(id, inst);
    laik_svg_profiler_enter(inst, __func__);

    if (id == 0)
    {
        printf("Running 2D example with %d iterations.\n", lcount);
        printf("Space size %ldx%ld = %ld.\n", sdsize, sdsize, spsize);
        printf("Using LB algorithm: %s\n", laik_get_lb_algorithm_name(lbalg));
    }

    Laik_Space *space = laik_new_space_2d(inst, sdsize, sdsize);
    Laik_Data *data = laik_new_data(space, laik_Int32);
    Laik_Partitioner *parter = laik_new_bisection_partitioner();
    Laik_Partitioning *part = laik_new_partitioning(parter, world, space, 0);
    laik_switchto_partitioning(data, part, LAIK_DF_None, LAIK_RO_None);

    int iterations = (id + 1) * 10;
    double c = 0.25; // sleep time constant multiplier

    // debug logging
    int64_t from_x, from_y, to_x, to_y;
    laik_my_range_2d(part, 0, &from_x, &to_x, &from_y, &to_y);
    int64_t count_x = to_x - from_x;
    int64_t count_y = to_y - from_y;

    // modified version of the 1d example, to be changed to something more "professional" later (e.g. n-body)
    for (int loop = 0; loop < lcount; ++loop)
    {
        laik_log(1, "%d ranges\n", laik_my_rangecount(part));
        laik_lb_balance(START_LB_SEGMENT, 0, 0);
        for (int iter = 0; iter < iterations; ++iter)
        {
            uint64_t sleepdur = 0;
            for (int r = 0; r < laik_my_rangecount(part); ++r)
            {
                int64_t from_x, from_y, to_x, to_y;
                laik_my_range_2d(part, r, &from_x, &to_x, &from_y, &to_y);
                sleepdur += (to_x - from_x) * (to_y - from_y);
            }
            usleep((uint64_t)((double)sleepdur * c));
        }

        // calculate and switch to new partitioning determined by load balancing algorithm
        Laik_Partitioning *newpart = laik_lb_balance(STOP_LB_SEGMENT, part, lbalg);
        if (part == newpart)
            continue;

        laik_switchto_partitioning(data, newpart, LAIK_DF_Preserve, LAIK_RO_None);

        // free old partitioning
        laik_free_partitioning(part);
        part = newpart;
    }

    // visualize task ranges
    EXPORT_TO_FILE(id, part);
    VISUALIZE(id);

    // done
    laik_svg_profiler_exit(inst, __func__);
    laik_svg_profiler_export_json(inst);
    laik_finalize(inst);
    if (id == 0)
        save_trace();
    return 0;
}

// choose example and parameters based on input
//
// USAGE: <mpirun -n x> ./lb [example] <spacesize> <loopcount>
//   e.g:  mpirun -n 4  ./lb 1 250000 10
int main(int argc, char *argv[])
{
    // prepare program trace visualization
    remove_plots();

    // example must be specified
    if (argc < 2)
        exit(1);

    // example must be 1(d) or 2(d)
    int example = atoi(argv[1]);
    if (example != 1 && example != 2)
        exit(1);

    // optional space size and loop count arguments
    int64_t sidelen = 1024;
    int64_t spsize = sidelen * sidelen; // space size (sidelen^2)
    int lcount = 5;                     // loop count
    if (argc > 2)
        spsize = atoi(argv[2]);
    if (argc > 3)
        lcount = atoi(argv[3]);

    if (example == 1)
        main_1d(argc, argv, spsize, lcount);
    else if (example == 2)
        main_2d(argc, argv, spsize, lcount);
}
#endif