/**
 * Ping-pong + load balancing micro-benchmark.
 *
 * Used to test load balancing with switches inbetween calls (i.e. load balancing compute workload + communication).
 * Based on ping_pong.c
 */

#include <laik.h>
#include <laik-internal.h>
#include <assert.h>
#include <stdio.h>
#include <unistd.h>

// by default, process pairs exchanging data are close to each other by
// their process ID in the world group, ie. [0,1], [2,3] and so on.
// With use_spread = 1 (option "-s"), pairs are arranged such that the
// first half of processes exchanges data with the 2nd half instaed, ie.
// with 20 processes that is [0,10], [1,11] and so on
int use_spread = 0;

// custom partitioner: pairs can access pieces depending on phase 0/1
void runPairParter(Laik_RangeReceiver *r, Laik_PartitionerParams *p)
{
    laik_svg_profiler_enter(p->space->inst, __func__);

    int phase = *(int *)laik_partitioner_data(p->partitioner);
    assert((phase == 0) || (phase == 1));
    int pairs = laik_size(p->group) / 2;
    Laik_Space *space = p->space;
    int64_t size = laik_space_size(space);

    Laik_Range range;
    for (int p = 0; p < pairs; p++)
    {
        // array is split up in consecutive pieces among pairs
        laik_range_init_1d(&range, space,
                           size * p / pairs, size * (p + 1) / pairs);
        int proc = use_spread ? p + phase * pairs : 2 * p + phase;
        laik_append_range(r, proc, &range, 0, 0);
    }

    laik_svg_profiler_exit(p->space->inst, __func__);
}

// partitioner for aggregating times: access to data in 1st procs of pairs
void run1stInPairParter(Laik_RangeReceiver *r, Laik_PartitionerParams *p)
{
    laik_svg_profiler_enter(p->space->inst, __func__);

    Laik_Space *space = p->space;
    Laik_Range range;
    laik_range_init_1d(&range, space, 0, laik_space_size(space));
    int pairs = laik_size(p->group) / 2;
    for (int p = 0; p < pairs; p++)
    {
        int proc = use_spread ? p : 2 * p;
        laik_append_range(r, proc, &range, 0, 0);
    }

    laik_svg_profiler_exit(p->space->inst, __func__);
}

int main(int argc, char *argv[])
{
    laik_lbvis_remove_visdata();

    Laik_Instance *instance = laik_init(&argc, &argv);
    Laik_Group *world = laik_world(instance);

    // run parameters (defaults are set after parsing arguments)
    long size = 100000000;
    int iters = 0;

    laik_lbvis_enable_trace(laik_myid(world), instance);
    laik_svg_profiler_enter(instance, __func__);

    // parse command line arguments
    int arg = 1;
    while ((arg < argc) && (argv[arg][0] == '-'))
    {
        if (argv[arg][1] == 's')
            use_spread = 1;
        if (argv[arg][1] == 'h')
        {
            printf("Ping-pong micro-benchmark for LAIK\n"
                   "Usage: %s [options] [<size> [<iters>]]\n"
                   "\nArguments:\n"
                   " <size>  : number of double entries transfered (def: 100M)\n"
                   " <iters> : number of repetitions (def: 10)\n"
                   "\nOptions:\n"
                   " -s: arrange process pairs spread instead of close\n"
                   " -h: this help text\n",
                   argv[0]);
            exit(1);
        }
        arg++;
    }
    if (argc > arg)
        iters = atoi(argv[arg]);

    // set to defaults if not set by arguments
    if (iters == 0)
        iters = 20;

    int pairs = laik_size(world) / 2;
    if (pairs == 0)
    {
        printf("Error: need at least one process pair to run ping-pong\n");
        laik_finalize(instance);
        exit(1);
    }

    // print benchmark run parameters
    int myid = laik_myid(world);
    if (myid == 0)
    {
        double sizeMB = .000001 * sizeof(double) * size;
        printf("Do %d iterations, %d pairs (%s arrangement: 0/%d, %d/%d ...)\n",
               iters, pairs, use_spread ? "spread" : "close",
               use_spread ? pairs : 1, use_spread ? 1 : 2, use_spread ? 1 + pairs : 3);
        printf(" with %ld doubles (%.3f MB, per pair %.3f MB)\n",
               size, sizeMB, sizeMB / pairs);
    }

    // setup LAIK objects
    Laik_Space *space = laik_new_space_1d(instance, size);
    Laik_Data *array = laik_new_data(space, laik_Double);

    // CUSTOM: LB
    int sdsize = 1024;
    int lbevery = 3;
    Laik_LBAlgorithm lbalg = LB_HILBERT;
    Laik_Space *lbspace = laik_new_space_2d(instance, sdsize, sdsize);
    Laik_Data *lbdata = laik_new_data(lbspace, laik_Int32);
    Laik_Partitioner *parter = laik_new_bisection_partitioner();
    Laik_Partitioning *part = laik_new_partitioning(parter, world, lbspace, 0);
    laik_switchto_partitioning(lbdata, part, LAIK_DF_None, LAIK_RO_None);

    int iterations = (myid + 1) * 10;
    double c = (512.0 / (double)sdsize) * (512.0 / (double)sdsize) * 0.5;

    // run the ping-pong between pairs, using our custom partitioner
    Laik_Partitioner *pr0, *pr1;
    Laik_Partitioning *p0, *p1;
    int phase0 = 0, phase1 = 1;
    pr0 = laik_new_partitioner("even", runPairParter, &phase0, 0);
    pr1 = laik_new_partitioner("odd", runPairParter, &phase1, 0);
    p0 = laik_new_partitioning(pr0, world, space, 0);
    p1 = laik_new_partitioning(pr1, world, space, 0);

    // initialization by even procs
    laik_switchto_partitioning(array, p0, LAIK_DF_None, LAIK_RO_None);
    double *base;
    uint64_t count;
    laik_get_map_1d(array, 0, (void **)&base, &count);
    for (uint64_t i = 0; i < count; ++i)
        base[i] = (double)i;

    // ping pong
    double start_time, end_time;
    start_time = laik_wtime();
    laik_lb_balance(START_LB_SEGMENT, 0, 0);
    for (int it = 0; it < iters; ++it)
    {
        // ping
        laik_svg_profiler_enter(instance, "ping");
        laik_switchto_partitioning(array, p1, LAIK_DF_Preserve, LAIK_RO_None);
        laik_svg_profiler_exit(instance, "ping");

        // CUSTOM: LB
        // laik_lb_balance(it % lbevery == 0 ? START_LB_SEGMENT : RESUME_LB_SEGMENT, 0, 0);
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
        if ((it % lbevery == (lbevery - 1)))
        {
            Laik_Partitioning *newpart = laik_lb_balance(STOP_LB_SEGMENT, part, lbalg);
            laik_lb_balance(START_LB_SEGMENT, 0, 0);
            laik_lb_switch_and_free(&part, &newpart, lbdata, LAIK_DF_Preserve);
        }

        // pong
        laik_svg_profiler_enter(instance, "pong");
        laik_switchto_partitioning(array, p0, LAIK_DF_Preserve, LAIK_RO_None);
        laik_svg_profiler_exit(instance, "pong");
    }
    end_time = laik_wtime();
    double dt = end_time - start_time;

    if (pairs > 1)
    {
        // aggregate time from 1st processes in each pair on master
        Laik_Data *sum = laik_new_data_1d(instance, laik_Double, 1);
        Laik_Partitioner *pr = laik_new_partitioner("1st", run1stInPairParter, 0, 0);
        laik_switchto_new_partitioning(sum, world, pr, LAIK_DF_None, LAIK_RO_None);
        laik_get_map_1d(sum, 0, (void **)&base, 0);
        if (base) // all 1st processes of pairs
            base[0] = dt;
        laik_switchto_new_partitioning(sum, world, laik_Master,
                                       LAIK_DF_Preserve, LAIK_RO_Sum);
        laik_get_map_1d(sum, 0, (void **)&base, 0);
        if (base) // only master
            dt = base[0] / pairs;
    }

    if (myid == 0)
    {
        // statistics
        printf("Time: %.3lf s (average per iteration: %.3lf ms, per phase: %.3lf ms)\n",
               dt, dt * 1e3 / (double)iters, dt * 1e3 / (double)iters / 2);
        printf("GB/s: %lf\n",
               8.0 * 2 * iters * size / dt / 1.0e9);
    }

    laik_svg_profiler_exit(instance, __func__);
    laik_svg_profiler_export_json(instance);
    laik_finalize(instance);
    if (myid == 0)
        laik_lbvis_save_trace();
    return 0;
}
