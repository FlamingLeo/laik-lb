// smoothing test
#include <laik.h>
#include <laik-internal.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

// 2d example
int main_2d(int argc, char *argv[], int64_t sdsize, int lcount, Laik_LBAlgorithm lbalg, bool smoothing, bool suspend, bool intelligent)
{
    // initialization
    Laik_Instance *inst = laik_init(&argc, &argv);
    Laik_Group *world = laik_world(inst);
    int id = laik_myid(world);

    laik_lbvis_enable_trace(id, inst);
    laik_svg_profiler_enter(inst, __func__);

    // for basic smoothing only, start with smoothing immediately
    if (smoothing && !intelligent)
        laik_lb_config_smoothing(1, -1, -1, -1);

    if (id == 0)
    {
        printf("Running smoothing test example with %d iterations.\n", lcount);
        printf("Space size %ldx%ld = %ld.\n", sdsize, sdsize, sdsize * sdsize);
        printf("Using LB algorithm: %s\n", laik_get_lb_algorithm_name(lbalg));
        printf("Smoothing: %d, Suspend: %d, Intelligent: %d\n", smoothing, suspend, intelligent);
    }

    Laik_Space *space = laik_new_space_2d(inst, sdsize, sdsize);
    Laik_Data *data = laik_new_data(space, laik_Int32);
    Laik_Partitioner *parter = laik_new_bisection_partitioner();
    Laik_Partitioning *part = laik_new_partitioning(parter, world, space, 0);
    laik_switchto_partitioning(data, part, LAIK_DF_None, LAIK_RO_None);

    Laik_Timer t = {0};
    laik_timer_start(&t);

    // modified version of the 1d example, to be changed to something more "professional" later (e.g. n-body)
    for (int loop = 0; loop < lcount; ++loop)
    {
        laik_log(1, "%d ranges\n", laik_my_rangecount(part));
        laik_lb_balance(START_LB_SEGMENT, 0, 0);

        // for intelligent mode, only activate smoothing after a certain number of iterations
        if (smoothing && intelligent)
        {
            if (loop == 2)
                laik_lb_config_smoothing(1, -1, -1, -1);
            if (loop == 5)
                laik_lb_config_smoothing(1, 0.5, 0.001, 9999);
                laik_lb_config_thresholds(5,-1,-1,-1);
            if (loop == 7)
                laik_lb_config_smoothing(1, -1, -1, -1);
        }

        // suspend execution of first proc. only for 5th iter
        if (suspend)
            if (id == 0 && loop == 4)
                sleep(2);

        // cause some imbalance
        for (int r = 0; r < laik_my_rangecount(part); ++r)
        {
            uint64_t ysize, xsize;
            laik_get_map_2d(data, r, NULL, &ysize, NULL, &xsize);

            int64_t globFromX, globToX, globFromY, globToY;
            laik_my_range_2d(part, r, &globFromX, &globToX, &globFromY, &globToY);

            for (int64_t y = 0; y < ysize; ++y)
            {
                for (int64_t x = 0; x < xsize; ++x)
                {
                    int itercount = ((globFromX + x) + (globFromY + y)) * 4;
                    volatile double sink = 0.0;
                    for (int k = 0; k < itercount; ++k)
                        sink += 1;
                }
            }
        }

        // calculate and switch to new partitioning determined by load balancing algorithm
        Laik_Partitioning *newpart = laik_lb_balance(STOP_LB_SEGMENT, part, lbalg);
        laik_lb_switch_and_free(&part, &newpart, data, LAIK_DF_Preserve);
    }

    double time = laik_timer_stop(&t);
    if (id == 0)
        printf("Done. Time taken: %fs\n", time);

    // done
    laik_svg_profiler_exit(inst, __func__);
    laik_svg_profiler_export_json(inst);
    laik_finalize(inst);
    if (id == 0)
        laik_lbvis_save_trace();
    return 0;
}

// choose example and parameters based on input
//
// USAGE: <mpirun -n x> ./lb [example] <do_vis> <lb_algo> <spacesize> <loopcount>
int main(int argc, char *argv[])
{
    // prepare program trace visualization
    laik_lbvis_remove_visdata();

    // optional space size and loop count arguments
    int64_t sidelen = 0;
    Laik_LBAlgorithm algo = LB_RCB;
    bool smoothing = true, suspend = true, intelligent = true;
    int lcount = 10; // loop count, increase this to test thresholds

    int arg = 1;
    while ((argc > arg) && (argv[arg][0] == '-'))
    {
        // choose lb algorithm
        if (arg + 1 < argc && !strcmp(argv[arg], "-a"))
            algo = laik_strtolb(argv[++arg]);

        // choose side length
        if (arg + 1 < argc && !strcmp(argv[arg], "-s"))
            sidelen = atoi(argv[++arg]);

        // choose loop count
        if (arg + 1 < argc && !strcmp(argv[arg], "-l"))
            lcount = atoi(argv[++arg]);

        // do not use intelligent mode
        if (!strcmp(argv[arg], "-i"))
            intelligent = false;

        // do not apply smoothing
        if (!strcmp(argv[arg], "-S"))
            smoothing = false;

        // do not suspend execution
        if (!strcmp(argv[arg], "-p"))
            suspend = false;

        // help
        if (argv[arg][1] == 'h')
        {
            printf("Usage: %s [options]\n\n"
                   "Options:\n"
                   " -a : choose load balancing algorithm (default: rcb)\n"
                   " -s : choose side length (default: 1024)\n"
                   " -S : do not apply smoothing\n"
                   " -i : do not use intelligent smoothing\n"
                   " -p : do not suspend execution\n"
                   " -l : choose loop count (defaullt: 10)\n"
                   " -h : print this help text and exit with code 1\n",
                   argv[0]);
            exit(1);
        }
        arg++;
    }

    if (sidelen == 0)
        sidelen = 1024;
    main_2d(argc, argv, sidelen, lcount, algo, smoothing, suspend, intelligent);
}