/* This file is part of the LAIK parallel container library.
 * Copyright (c) 2017-2019 Josef Weidendorfer
 * Extended with load balancing functionality 2025 by Flavius Schmidt.
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

/**
 * 3d Jacobi example (with load balancing).
 */

#include <laik.h>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define SIZE 64
#define MAXITER 100
#define RES_ITER 10
#define WL_LB_ITER 5
#define CSVNAME "array_data_jac3dlb.csv"

#define WORKLOAD

#ifdef WORKLOAD
#define DO_WORKLOAD(_iter)                                                                             \
    /* iter: current iteration inside loop */                                                          \
    /* pWrite: pointer to write partition */                                                           \
    /* r: current range index */                                                                       \
    if ((_iter == 0) || (iter < _iter))                                                                \
    {                                                                                                  \
        int64_t globFromX, globToX, globFromY, globToY, globFromZ, globToZ;                            \
        laik_my_range_3d(pWrite, r, &globFromX, &globToX, &globFromY, &globToY, &globFromZ, &globToZ); \
        int itercount = ((globFromX + x) + (globFromY + y) + (globFromZ + z)) * 150;                   \
        volatile double sink = 0.0; /* volatile to prevent optimizing out */                           \
        for (int k = 0; k < itercount; ++k)                                                            \
            sink += baseR[z * zstrideR + y * ystrideR + x] * 0.0 + k * 1e-9;                           \
    }

#define LOAD_BALANCE(_iter)                                                          \
    /* iter: current iteration inside loop */                                        \
    /* npWrite: pointer to recalculated write partition */                           \
    /* pWrite: pointer to old write partition, may be replaced */                    \
    /* npRead: pointer to new read partition based on new write partition borders */ \
    if ((_iter == 0) || (iter < _iter))                                              \
    {                                                                                \
        laik_timer_start(&lbm_timer);                                                \
        npWrite = laik_lb_balance(STOP_LB_SEGMENT, pWrite, algo);                    \
        npRead = laik_new_partitioning(prRead, world, space, npWrite);               \
        Laik_LBDataStats before_w = {0};                                             \
        Laik_LBDataStats before_r = {0};                                             \
        laik_lb_stats_store(&before_w, dWrite);                                      \
        laik_lb_stats_store(&before_r, dRead);                                       \
        laik_lb_switch_and_free(&pWrite, &npWrite, dWrite, LAIK_DF_Preserve);        \
        laik_lb_switch_and_free(&pRead, &npRead, dRead, LAIK_DF_None);               \
        Laik_LBDataStats after_w = {0};                                              \
        Laik_LBDataStats after_r = {0};                                              \
        laik_lb_stats_store(&after_w, dWrite);                                       \
        laik_lb_stats_store(&after_r, dRead);                                        \
        laik_lb_print_diff(myid, dWrite, &after_w, &before_w);                       \
        laik_lb_print_diff(myid, dRead, &after_r, &before_r);                        \
        lbm_time += laik_timer_stop(&lbm_timer);                                     \
    }
#else
#define DO_WORKLOAD(_iter) (void)0;
#define LOAD_BALANCE(_iter) (void)0;
#endif

// boundary values
static double loRowValue = -5.0, hiRowValue = 10.0;
static double loColValue = -10.0, hiColValue = 5.0;
static double loPlaneValue = -20.0, hiPlaneValue = 15.0;

void setBoundary(int size, Laik_Partitioning *pWrite, Laik_Data *dWrite, int rangeNo)
{
    double *baseW;
    uint64_t zsizeW, zstrideW, ysizeW, ystrideW, xsizeW;
    int64_t gx1, gx2, gy1, gy2, gz1, gz2;

    // global index ranges of the range of this process
    laik_my_range_3d(pWrite, rangeNo, &gx1, &gx2, &gy1, &gy2, &gz1, &gz2);

    // default mapping order for 3d:
    //   with z in [0;zsize[, y in [0;ysize[, x in [0;xsize[
    //   base[z][y][x] is at (base + z * zstride + y * ystride + x)
    laik_get_map_3d(dWrite, rangeNo, (void **)&baseW,
                    &zsizeW, &zstrideW, &ysizeW, &ystrideW, &xsizeW);

    // set fixed boundary values at the 6 faces
    if (gz1 == 0)
    {
        // front plane
        for (uint64_t y = 0; y < ysizeW; y++)
            for (uint64_t x = 0; x < xsizeW; x++)
                baseW[y * ystrideW + x] = loPlaneValue;
    }
    if (gz2 == size)
    {
        // back plane
        for (uint64_t y = 0; y < ysizeW; y++)
            for (uint64_t x = 0; x < xsizeW; x++)
                baseW[(zsizeW - 1) * zstrideW + y * ystrideW + x] = hiPlaneValue;
    }
    if (gy1 == 0)
    {
        // top plane (overwrites global front/back top edge)
        for (uint64_t z = 0; z < zsizeW; z++)
            for (uint64_t x = 0; x < xsizeW; x++)
                baseW[z * zstrideW + x] = loRowValue;
    }
    if (gy2 == size)
    {
        // bottom plane (overwrites global front/back bottom edge)
        for (uint64_t z = 0; z < zsizeW; z++)
            for (uint64_t x = 0; x < xsizeW; x++)
                baseW[z * zstrideW + (ysizeW - 1) * ystrideW + x] = hiRowValue;
    }
    if (gx1 == 0)
    {
        // left column, overwrites global left edges
        for (uint64_t z = 0; z < zsizeW; z++)
            for (uint64_t y = 0; y < ysizeW; y++)
                baseW[z * zstrideW + y * ystrideW] = loColValue;
    }
    if (gx2 == size)
    {
        // right column, overwrites global right edges
        for (uint64_t z = 0; z < zsizeW; z++)
            for (uint64_t y = 0; y < ysizeW; y++)
                baseW[z * zstrideW + y * ystrideW + (xsizeW - 1)] = hiColValue;
    }
}

//--------------------------------------------------------------
// main function
int main(int argc, char *argv[])
{
    laik_lbvis_remove_visdata();
    Laik_Instance *inst = laik_init(&argc, &argv);
    Laik_Group *world = laik_world(inst);

    int size = 0;
    int maxiter = 0;
    bool use_cornerhalo = true; // use halo partitioner including corners?
    bool do_profiling = false;
    bool do_visualization = false;
    bool do_sum = true;
    bool do_grid = false;
    bool do_lb = true;
    Laik_LBAlgorithm algo = LB_HILBERT;
    int myid = laik_myid(world);
    int xblocks = 0, yblocks = 0, zblocks = 0; // for grid partitioner

    int arg = 1;
    while ((argc > arg) && (argv[arg][0] == '-'))
    {
        if (argv[arg][1] == 'N')
            use_cornerhalo = false;
        if (argv[arg][1] == 'p')
            do_profiling = true;
        if (argv[arg][1] == 'v')
            do_visualization = true;
        if (argv[arg][1] == 's')
            do_sum = true;
        if (argv[arg][1] == 'g')
            do_grid = true;
        if (argv[arg][1] == 'x' && argc > arg + 1)
        {
            xblocks = atoi(argv[++arg]);
            do_grid = true;
        }
        if (argv[arg][1] == 'L')
            do_lb = false;
        if (argv[arg][1] == 'h')
        {
            if (myid == 0)
                printf("Usage: %s [options]\n\n"
                       "Options:\n"
                       " -a        : choose load balancing algorithm\n"
                       " -N        : use partitioner which does not include corners\n"
                       " -g        : use grid partitioning with automatic block size\n"
                       " -x <xgrid>: use grid partitioning with given x block length\n"
                       " -p        : export and visualize program trace as json files / collective svg'\n"
                       " -s        : print value sum at end (warning: sum done at master)\n"
                       " -v        : export and visualize partitioning borders at the end of the run\n"
                       " -L        : disable load balancing\n"
                       " -h        : print this help text and exit with code 1\n",
                       argv[0]);
            exit(1);
        }
        if (argv[arg][1] == 'a' && argc > arg + 1)
            algo = laik_strtolb(argv[++arg]);
        arg++;
    }
    /*
    if (argc > arg)
        size = atoi(argv[arg]);
    if (argc > arg)
        maxiter = atoi(argv[arg]);
    */

    if (size == 0)
        size = SIZE;
    if (maxiter == 0)
        maxiter = MAXITER;

    if (do_grid)
    {
        // find grid partitioning with less or equal blocks than processes
        int pcount = laik_size(world);
        int mind = 3 * pcount;
        int xmin = 1, xmax = pcount;
        if ((xblocks > 0) && (xblocks <= pcount))
            xmin = xmax = xblocks;
        for (int x = xmin; x <= xmax; x++)
            for (int y = 1; y <= pcount; y++)
            {
                int z = (int)(((double)pcount) / x / y);
                int pdiff = pcount - x * y * z;
                if ((z == 0) || (pdiff < 0))
                    continue;
                // minimize idle cores and diff in x/y/z
                int d = abs(y - x) + abs(z - x) + abs(z - y) + 2 * pdiff;
                if (mind <= d)
                    continue;
                mind = d;
                zblocks = z;
                yblocks = y;
                xblocks = x;
            }
    }

    if (laik_myid(world) == 0)
    {
        printf("%d x %d x %d cells (mem %.1f MB), running %d iterations with %d tasks using %s",
               size, size, size, .000016 * size * size * size,
               maxiter, laik_size(world), laik_get_lb_algorithm_name(algo));
        if (do_grid)
            printf(" (grid %d x %d x %d)", zblocks, yblocks, xblocks);
        if (!use_cornerhalo)
            printf(" (halo without corners)");
        printf("\nvisualization: %d, profiling: %d, sum: %d, load balancing: %d\n", do_visualization, do_profiling, do_sum, do_lb);
    }

    // start profiling interface
    if (do_profiling)
        laik_lbvis_enable_trace(myid, inst);

    double *baseR, *baseW, *sumPtr;
    uint64_t zsizeR, zstrideR, ysizeR, ystrideR, xsizeR;
    uint64_t zsizeW, zstrideW, ysizeW, ystrideW, xsizeW;
    int64_t gx1, gx2, gy1, gy2, gz1, gz2;
    int64_t x1, x2, y1, y2, z1, z2;

    // two 3d arrays for jacobi, using same space
    Laik_Space *space = laik_new_space_3d(inst, size, size, size);
    Laik_Data *data1 = laik_new_data(space, laik_Double);
    Laik_Data *data2 = laik_new_data(space, laik_Double);

    // we use two types of partitioners algorithms:
    // - prWrite: cells to update (disjunctive partitioning)
    // - prRead : extends partitionings by haloes, to read neighbor values
    Laik_Partitioner *prWrite, *prRead;
    prWrite = do_grid ? laik_new_grid_partitioner(xblocks, yblocks, zblocks) : laik_new_bisection_partitioner();
    prRead = use_cornerhalo ? laik_new_cornerhalo_partitioner(1) : laik_new_halo_partitioner(1);

    // run partitioners to get partitionings over 3d space and <world> group
    // data1/2 are then alternately accessed using pRead/pWrite
    Laik_Partitioning *pWrite, *pRead, *npWrite, *npRead;
    pWrite = laik_new_partitioning(prWrite, world, space, 0);
    pRead = laik_new_partitioning(prRead, world, space, pWrite);
    laik_partitioning_set_name(pWrite, "pWrite");
    laik_partitioning_set_name(pRead, "pRead");

    // for global sum, used for residuum: 1 double accessible by all
    Laik_Space *sp1 = laik_new_space_1d(inst, 1);
    Laik_Partitioning *pSum = laik_new_partitioning(laik_All, world, sp1, 0);
    Laik_Data *dSum = laik_new_data(sp1, laik_Double);
    laik_data_set_name(dSum, "sum");
    laik_switchto_partitioning(dSum, pSum, LAIK_DF_None, LAIK_RO_None);

    // start with writing (= initialization) data1
    Laik_Data *dWrite = data1;
    Laik_Data *dRead = data2;

    // distributed initialization
    laik_switchto_partitioning(dWrite, pWrite, LAIK_DF_None, LAIK_RO_None);
    laik_my_range_3d(pWrite, 0, &gx1, &gx2, &gy1, &gy2, &gz1, &gz2);

    // default mapping order for 3d:
    //   with z in [0;zsize[, y in [0;ysize[, x in [0;xsize[
    //   base[z][y][x] is at (base + z * zstride + y * ystride + x)
    laik_get_map_3d(dWrite, 0, (void **)&baseW,
                    &zsizeW, &zstrideW, &ysizeW, &ystrideW, &xsizeW);
    // arbitrary non-zero values based on global indexes to detect bugs
    for (uint64_t z = 0; z < zsizeW; z++)
        for (uint64_t y = 0; y < ysizeW; y++)
            for (uint64_t x = 0; x < xsizeW; x++)
                baseW[z * zstrideW + y * ystrideW + x] =
                    (double)((gx1 + x + gy1 + y + gz1 + z) & 6);

    setBoundary(size, pWrite, dWrite, 0);
    laik_log(2, "Init done\n");

    // for statistics (with LAIK_LOG=2)
    double t, t1 = laik_wtime(), t2 = t1;
    int last_iter = 0;
    int res_iters = 0; // iterations done with residuum calculation

    laik_svg_profiler_enter(inst, __func__);
    Laik_Timer timer = {0};
    Laik_Timer work_timer = {0};
    Laik_Timer switch_timer = {0};
    Laik_Timer lbm_timer = {0};
    double switch_time = 0.0;
    double lbm_time = 0.0;
    double work_time = 0.0;

    // begin iterations
    int iter = 0;
    laik_timer_start(&timer);
    for (; iter < maxiter; iter++)
    {
        laik_set_iteration(inst, iter + 1);

        // switch roles: data written before now is read
        if (dRead == data1)
        {
            dRead = data2;
            dWrite = data1;
        }
        else
        {
            dRead = data1;
            dWrite = data2;
        }

        if (iter < WL_LB_ITER)
            laik_timer_start(&switch_timer);

        //  switch to partitionings
        laik_switchto_partitioning(dRead, pRead, LAIK_DF_Preserve, LAIK_RO_None);
        laik_switchto_partitioning(dWrite, pWrite, LAIK_DF_None, LAIK_RO_None);

        if (iter < WL_LB_ITER)
            switch_time += laik_timer_stop(&switch_timer);

        if (iter < WL_LB_ITER)
            laik_timer_start(&work_timer);

        double vSum, vNew, diff, res = 0.0;
        double coeff = 1.0 / 6.0;
        bool resCond = ((iter % RES_ITER) == 0) && (iter >= RES_ITER);

        // set boundaries beforehand (taken from jac2d-lb)
        for (int r = 0; r < laik_my_rangecount(pWrite); ++r)
            setBoundary(size, pWrite, dWrite, r);

        // loop through all ranges (mappings; 1:1)
        for (int r = 0; r < laik_my_rangecount(pWrite); ++r)
        {
            laik_get_map_3d(dRead, r, (void **)&baseR,
                            &zsizeR, &zstrideR, &ysizeR, &ystrideR, &xsizeR);
            laik_get_map_3d(dWrite, r, (void **)&baseW,
                            &zsizeW, &zstrideW, &ysizeW, &ystrideW, &xsizeW);

            // determine local range for which to do 3d stencil, without global edges
            laik_my_range_3d(pWrite, r, &gx1, &gx2, &gy1, &gy2, &gz1, &gz2);

            z1 = (gz1 == 0) ? 1 : 0;
            y1 = (gy1 == 0) ? 1 : 0;
            x1 = (gx1 == 0) ? 1 : 0;
            z2 = (gz2 == size) ? (zsizeW - 1) : zsizeW;
            y2 = (gy2 == size) ? (ysizeW - 1) : ysizeW;
            x2 = (gx2 == size) ? (xsizeW - 1) : xsizeW;

            // relocate baseR to be able to use same indexing as with baseW
            if (gx1 > 0)
            {
                // ghost cells from left neighbor at x=0, move that to -1
                baseR++;
            }
            if (gy1 > 0)
            {
                // ghost cells from top neighbor at y=0, move that to -1
                baseR += ystrideR;
            }
            if (gz1 > 0)
            {
                // ghost cells from back neighbor at z=0, move that to -1
                baseR += zstrideR;
            }

            ///////////////
            // do jacobi //
            ///////////////

            // only start on first range to then accumulate extra time (_slightly_ noisy due to above code but insignificant)
            if (r == 0)
                laik_lb_balance(START_LB_SEGMENT, 0, 0);

            for (int64_t z = z1; z < z2; z++)
            {
                for (int64_t y = y1; y < y2; y++)
                {
                    for (int64_t x = x1; x < x2; x++)
                    {
                        vSum = baseR[(z - 1) * zstrideR + y * ystrideR + x] +
                               baseR[(z + 1) * zstrideR + y * ystrideR + x] +
                               baseR[z * zstrideR + (y - 1) * ystrideR + x] +
                               baseR[z * zstrideR + (y + 1) * ystrideR + x] +
                               baseR[z * zstrideR + y * ystrideR + (x - 1)] +
                               baseR[z * zstrideR + y * ystrideR + (x + 1)];
                        vNew = coeff * vSum;

                        // for calculating the residual, accumulate it for all ranges
                        if (resCond)
                        {
                            diff = baseR[z * zstrideR + y * ystrideR + x] - vNew;
                            res += diff * diff;
                        }

                        // do some artificial workload to simulate load
                        DO_WORKLOAD(WL_LB_ITER);
                        baseW[z * zstrideW + y * ystrideW + x] = vNew;
                    }
                }
            }
        } // end ranges / mappings loop

        if (iter < WL_LB_ITER)
            work_time += laik_timer_stop(&work_timer); // stop and accumulate

        if (do_lb)
            LOAD_BALANCE(WL_LB_ITER);

        // check for residuum on proper iteration
        if (resCond)
        {
            res_iters++;

            // calculate global residuum
            laik_switchto_flow(dSum, LAIK_DF_None, LAIK_RO_None);
            laik_get_map_1d(dSum, 0, (void **)&sumPtr, 0);
            *sumPtr = res;
            laik_switchto_flow(dSum, LAIK_DF_Preserve, LAIK_RO_Sum);
            laik_get_map_1d(dSum, 0, (void **)&sumPtr, 0);
            res = *sumPtr;

            if (iter > 0)
            {
                t = laik_wtime();
                // current iteration already done
                int diter = (iter + 1) - last_iter;
                double dt = t - t2;
                double gUpdates = 0.000000001 * size * size * size; // per iteration
                laik_log(2, "For %d iters: %.3fs, %.3f GF/s, %.3f GB/s",
                         diter, dt,
                         // 6 Flops per update in reg iters, with res 9 (once)
                         gUpdates * (9 + 6 * (diter - 1)) / dt,
                         // per update 48 bytes read + 8 byte written
                         gUpdates * diter * 56 / dt);
                last_iter = iter + 1;
                t2 = t;
            }

            if (laik_myid(world) == 0)
                printf("Residuum after %2d iters: %f\n", iter + 1, res);

            if (res < .001)
                break;
        } // end residual calculation
        laik_timer_start(&work_timer);
    } // end iterations
    double tfinal = laik_timer_stop(&timer);

    // statistics for all iterations and reductions
    // using work load in all tasks
    if (laik_log_shown(2))
    {
        t = laik_wtime();
        int diter = iter;
        double dt = t - t1;
        double gUpdates = 0.000000001 * size * size * size; // per iteration
        laik_log(2, "final for %d iters: %.3fs, %.3f GF/s, %.3f GB/s",
                 diter, dt,
                 // 6 Flops per update in reg iters, with res 4
                 gUpdates * (9 * res_iters + 6 * (diter - res_iters)) / dt,
                 // per update 48 bytes read + 8 byte written
                 gUpdates * diter * 56 / dt);
    }

    if (do_sum)
    {
        // for check at end: sum up all just written values
        Laik_Partitioning *pMaster;
        pMaster = laik_new_partitioning(laik_Master, world, space, 0);
        laik_switchto_partitioning(dWrite, pMaster, LAIK_DF_Preserve, LAIK_RO_None);

        if (laik_myid(world) == 0)
        {
            double sum = 0.0;
            laik_get_map_3d(dWrite, 0, (void **)&baseW,
                            &zsizeW, &zstrideW, &ysizeW, &ystrideW, &xsizeW);
            for (uint64_t z = 0; z < zsizeW; z++)
                for (uint64_t y = 0; y < ysizeW; y++)
                    for (uint64_t x = 0; x < xsizeW; x++)
                        sum += baseW[z * zstrideW + y * ystrideW + x];
            printf("Global value sum after %d iterations: %f\n",
                   iter, sum);
        }
    }

    if (myid == 0)
        printf("Done. Time taken: %fs\n", tfinal);

    if (do_visualization)
    {
        if (myid == 0)
        {
            Laik_RangeList *lr = laik_partitioning_myranges(pWrite);
            laik_lbvis_export_partitioning(CSVNAME, lr);
            laik_lbvis_visualize_partitioning(CSVNAME);
        }
    }

    laik_svg_profiler_exit(inst, __func__);
    laik_svg_profiler_export_json(inst);

    laik_finalize(inst);
    if (do_profiling && myid == 0)
        laik_lbvis_save_trace();

    laik_lb_print_stats(myid);

    // print individual times
    if (do_lb)
        printf("Task %d: work time = %fs, switch time = %fs, load balancer time = %fs\n", myid, work_time, switch_time, lbm_time);
    else
        printf("Task %d: work time = %fs, switch time = %fs, load balancer time = N/A\n", myid, work_time, switch_time);
    return 0;
}
