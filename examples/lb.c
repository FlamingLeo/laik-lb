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
int main_2d(int argc, char *argv[], int64_t sdsize, int lcount)
{
    // initialization
    Laik_Instance *inst = laik_init(&argc, &argv);
    Laik_Group *world = laik_world(inst);
    int id = laik_myid(world);
    Laik_LBAlgorithm lbalg = LB_HILBERT;

    enable_trace(id, inst);
    laik_svg_profiler_enter(inst, __func__);

    if (id == 0)
    {
        printf("Running 2D example with %d iterations.\n", lcount);
        printf("Space size %ldx%ld = %ld.\n", sdsize, sdsize, sdsize * sdsize);
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

// 3d example
int main_3d(int argc, char *argv[], int64_t sdsize, int lcount)
{
    Laik_Instance *inst = laik_init(&argc, &argv);
    Laik_Group *world = laik_world(inst);
    int id = laik_myid(world);
    Laik_LBAlgorithm lbalg = LB_HILBERT;

    enable_trace(id, inst);
    laik_svg_profiler_enter(inst, __func__);

    if (id == 0)
    {
        printf("Running 3D example with %d iterations.\n", lcount);
        printf("Space size %ldx%ldx%ld = %ld\n", sdsize, sdsize, sdsize, sdsize * sdsize * sdsize);
        printf("Using LB algorithm: %s\n", laik_get_lb_algorithm_name(lbalg));
    }

    Laik_Space *space = laik_new_space_3d(inst, sdsize, sdsize, sdsize);
    Laik_Data *data = laik_new_data(space, laik_Int32);
    Laik_Partitioner *parter = laik_new_bisection_partitioner();
    Laik_Partitioning *part = laik_new_partitioning(parter, world, space, 0);
    laik_switchto_partitioning(data, part, LAIK_DF_None, LAIK_RO_None);

    int iterations = (id + 1) * 10;
    double c = 0.25; // sleep time constant multiplier

    // debug logging for local ranges
    int64_t from_x, to_x, from_y, to_y, from_z, to_z;
    laik_my_range_3d(part, 0, &from_x, &to_x, &from_y, &to_y, &from_z, &to_z);
    int64_t count_x = to_x - from_x;
    int64_t count_y = to_y - from_y;
    int64_t count_z = to_z - from_z;

    for (int loop = 0; loop < lcount; ++loop)
    {
        laik_log(1, "%d ranges\n", laik_my_rangecount(part));
        laik_lb_balance(START_LB_SEGMENT, 0, 0);
        for (int iter = 0; iter < iterations; ++iter)
        {
            uint64_t sleepdur = 0;
            for (int r = 0; r < laik_my_rangecount(part); ++r)
            {
                int64_t fx, tx, fy, ty, fz, tz;
                laik_my_range_3d(part, r, &fx, &tx, &fy, &ty, &fz, &tz);
                int64_t dx = tx - fx;
                int64_t dy = ty - fy;
                int64_t dz = tz - fz;
                sleepdur += (uint64_t)(dx * dy * dz);
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
    if (example != 1 && example != 2 && example != 3)
        exit(1);

    // optional space size and loop count arguments
    int64_t sidelen = 64;
    int lcount = 10; // loop count
    if (argc > 2)
        sidelen = atoi(argv[2]);
    if (argc > 3)
        lcount = atoi(argv[3]);

    if (example == 1)
        main_1d(argc, argv, sidelen, lcount);
    else if (example == 2)
        main_2d(argc, argv, sidelen, lcount);
    else if (example == 3)
        main_3d(argc, argv, sidelen, lcount);
    else
    {
        fprintf(stderr, "Invalid example!\n");
        exit(EXIT_FAILURE);
    }
}