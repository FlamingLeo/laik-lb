/* This file is part of the LAIK parallel container library.
 * Copyright (c) 2025 Flavius Schmidt
 *
 * LAIK is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 3.
 *
 * LAIK is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

 /*
 * Smoothing benchmark.
 */
#include <laik.h>
#include <laik-internal.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>


// smoothing params
static double am = -1.0;  // 0.15
static double rmi = -1.0; // 0.7
static double rma = -1.0; // 1.25

int do_it(int argc, char *argv[], int64_t sdsize, int lcount, Laik_LBAlgorithm lbalg, bool smoothing, bool suspend, bool intelligent)
{
    // initialization
    Laik_Instance *inst = laik_init(&argc, &argv);
    Laik_Group *world = laik_world(inst);
    int id = laik_myid(world);

    laik_lbvis_enable_trace(id, inst);
    laik_svg_profiler_enter(inst, __func__);

    // for basic smoothing only, start with smoothing immediately
    if (smoothing && !intelligent)
        laik_lb_config_smoothing(1, am, rmi, rma);

    if (id == 0)
    {
        printf("Running smoothing test example with %d iterations.\n", lcount);
        printf("Space size %ldx%ld = %ld.\n", sdsize, sdsize, sdsize * sdsize);
        printf("Using LB algorithm: %s\n", laik_get_lb_algorithm_name(lbalg));
        printf("Smoothing: %d, Suspend: %d, Intelligent: %d\n", smoothing, suspend, intelligent);
        printf("alpha: %f, rmin: %f, rmax: %f\n", am, rmi, rma);
    }

    Laik_Space *space = laik_new_space_2d(inst, sdsize, sdsize);
    Laik_Data *data = laik_new_data(space, laik_Int32);
    Laik_Partitioner *parter = laik_new_bisection_partitioner();
    Laik_Partitioning *part = laik_new_partitioning(parter, world, space, 0);
    laik_switchto_partitioning(data, part, LAIK_DF_None, LAIK_RO_None);

    Laik_Timer t = {0};
    laik_timer_start(&t);

    for (int loop = 0; loop < lcount; ++loop)
    {
        laik_log(1, "%d ranges\n", laik_my_rangecount(part));
        laik_lb_balance(START_LB_SEGMENT, 0, 0);

        // for intelligent mode, only activate smoothing after a certain number of iterations
        // for the future, this configuration should be done automatically
        if (smoothing && intelligent)
        {
            if (loop == 2)
                laik_lb_config_smoothing(1, am, rmi, rma);
            if (loop == 5)
            {
                laik_lb_config_smoothing(1, 0.99, 1e-30, HUGE_VAL);
                laik_lb_config_thresholds(5, -1, -1, -1);
            }
            if (loop == 8)
                laik_lb_config_smoothing(1, am, rmi, rma);
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
        Laik_LBDataStats before = {0};
        laik_lb_stats_store(&before, data);
        laik_lb_switch_and_free(&part, &newpart, data, LAIK_DF_Preserve);
        Laik_LBDataStats after = {0};
        laik_lb_stats_store(&after, data);
        laik_lb_print_diff(id, data, &after, &before);
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

    laik_lb_print_stats(id);
    return 0;
}

// choose example and parameters based on input
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

        // smoothing: alpha
        if (arg + 1 < argc && !strcmp(argv[arg], "-A"))
            am = atof(argv[++arg]);

        // smoothing: rmin
        if (arg + 1 < argc && !strcmp(argv[arg], "-r"))
            rmi = atof(argv[++arg]);

        // smoothing: rmax
        if (arg + 1 < argc && !strcmp(argv[arg], "-R"))
            rma = atof(argv[++arg]);

        // do not use intelligent mode
        if (!strcmp(argv[arg], "-i"))
            intelligent = false;

        // do not apply smoothing
        if (!strcmp(argv[arg], "-S"))
            smoothing = false;

        // do not suspend execution
        if (!strcmp(argv[arg], "-p"))
            suspend = false;

        // disable output
        if (!strcmp(argv[arg], "-O"))
            laik_lb_output(false);

        // help
        if (argv[arg][1] == 'h')
        {
            printf("Usage: %s [options]\n\n"
                   "Options:\n"
                   " -a : choose load balancing algorithm (default: rcb)\n"
                   " -A : choose alpha (default: 0.15)\n"
                   " -r : choose rmin (default: 0.7)\n"
                   " -R : choose rmax (default: 1.25)\n"
                   " -s : choose side length (default: 1024)\n"
                   " -S : do not apply smoothing\n"
                   " -i : do not use intelligent smoothing (if smoothing is off, intelligent mode has no effect)\n"
                   " -p : do not suspend execution\n"
                   " -l : choose loop count (defaullt: 10)\n"
                   " -O : disable output\n"
                   " -h : print this help text and exit with code 1\n",
                   argv[0]);
            exit(1);
        }
        arg++;
    }

    if (sidelen == 0)
        sidelen = 1024;
    do_it(argc, argv, sidelen, lcount, algo, smoothing, suspend, intelligent);
}