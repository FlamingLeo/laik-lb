/* This file is part of the LAIK parallel container library.
 * Copyright (c) 2017 Josef Weidendorfer
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
 * 2d Jacobi example (with load balancing).
 *
 * TODO: make this example work with sfc load balancing algorithms
 */

#include <laik.h>
#include <laik-internal.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <math.h>

#define PROFILING 1
#define SUM 1

#define SIZE 1024
#define MAXITER 100
#define RES_ITER 10
#define WL_LB_ITER 5
#define LB_ALGO LB_RCB

#define FILENAME "lbviz/array_data.txt"
// #define DO_VISUALIZATION

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

static void visualize()
{
    system("python3 lbviz/visualize.py");
}

static void remove_plots()
{
    system("../scripts/remove_plots.sh");
}

static void save_trace()
{
    system("python3 lbviz/trace.py");
}

#define WORKLOAD

#ifdef WORKLOAD
#define DO_WORKLOAD(_iter)                                                       \
    /* iter: current iteration inside loop */                                    \
    /* pWrite: pointer to write partition */                                     \
    if ((_iter == 0) || (iter < _iter))                                          \
    {                                                                            \
        int64_t globFromX, globToX, globFromY, globToY;                          \
        laik_my_range_2d(pWrite, 0, &globFromX, &globToX, &globFromY, &globToY); \
        int itercount = (globFromX + x) * (globFromY + y) / 5000;                \
        volatile double sink = 0.0; /* volatile to prevent optimizing out */     \
        for (int k = 0; k < itercount; ++k)                                      \
            sink += baseR[y * ystrideR + x] * 0.0 + k * 1e-9;                    \
    }

#define LOAD_BALANCE(_iter)                                                          \
    /* iter: current iteration inside loop */                                        \
    /* npWrite: pointer to recalculated write partition */                           \
    /* pWrite: pointer to old write partition, may be replaced */                    \
    /* npRead: pointer to new read partition based on new write partition borders */ \
    if ((_iter == 0) || (iter < _iter))                                              \
    {                                                                                \
        if ((npWrite = laik_lb_balance(STOP_LB_SEGMENT, pWrite, LB_ALGO)) == pWrite) \
            continue;                                                                \
                                                                                     \
        npRead = laik_new_partitioning(prRead, world, space, npWrite);               \
                                                                                     \
        laik_switchto_partitioning(dWrite, npWrite, LAIK_DF_Preserve, LAIK_RO_None); \
        laik_switchto_partitioning(dRead, npRead, LAIK_DF_None, LAIK_RO_None);       \
                                                                                     \
        laik_free_partitioning(pWrite);                                              \
        laik_free_partitioning(pRead);                                               \
        pWrite = npWrite;                                                            \
        pRead = npRead;                                                              \
    }
#else
#define DO_WORKLOAD(_iter) (void)0;
#define LOAD_BALANCE(_iter) (void)0;
#endif

// boundary values
double loRowValue = -5.0, hiRowValue = 10.0;
double loColValue = -10.0, hiColValue = 5.0;

// update boundary values
void setBoundary(int size, Laik_Partitioning *pWrite, Laik_Data *dWrite)
{
    double *baseW;
    uint64_t ysizeW, ystrideW, xsizeW;
    int64_t gx1, gx2, gy1, gy2;

    // global index ranges of the range of this process
    laik_my_range_2d(pWrite, 0, &gx1, &gx2, &gy1, &gy2);

    // default mapping order for 2d:
    //   with y in [0;ysize[, x in [0;xsize[
    //   base[y][x] is at (base + y * ystride + x)
    laik_get_map_2d(dWrite, 0, (void **)&baseW, &ysizeW, &ystrideW, &xsizeW);

    // set fixed boundary values at the 4 edges
    if (gy1 == 0)
    {
        // top row
        for (uint64_t x = 0; x < xsizeW; x++)
            baseW[x] = loRowValue;
    }
    if (gy2 == size)
    {
        // bottom row
        for (uint64_t x = 0; x < xsizeW; x++)
            baseW[(ysizeW - 1) * ystrideW + x] = hiRowValue;
    }
    if (gx1 == 0)
    {
        // left column, may overwrite global (0,0) and (0,size-1)
        for (uint64_t y = 0; y < ysizeW; y++)
            baseW[y * ystrideW] = loColValue;
    }
    if (gx2 == size)
    {
        // right column, may overwrite global (size-1,0) and (size-1,size-1)
        for (uint64_t y = 0; y < ysizeW; y++)
            baseW[y * ystrideW + xsizeW - 1] = hiColValue;
    }
}

int main(int argc, char *argv[])
{
    remove_plots();
    Laik_Instance *inst = laik_init(&argc, &argv);
    Laik_Group *world = laik_world(inst);

    int size = 0;
    int maxiter = 0;
    bool use_cornerhalo = true; // use halo partitioner including corners?
    bool do_profiling = PROFILING;
    bool do_sum = SUM;

    int arg = 1;
    int myid = laik_myid(world);

    if (argc > arg)
        size = atoi(argv[arg]);
    if (argc > arg + 1)
        maxiter = atoi(argv[arg + 1]);

    if (size == 0)
        size = SIZE; // size² entries
    if (maxiter == 0)
        maxiter = MAXITER;

    if (myid == 0)
    {
        printf("%d x %d cells (mem %.1f MB), running %d iterations with %d tasks",
               size, size, .000016 * size * size, maxiter, laik_size(world));
        if (!use_cornerhalo)
            printf(" (halo without corners)");
        printf("\n");
    }

#if PROFILING
    // start profiling interface
    if (do_profiling)
    {
        char filename[MAX_FILENAME_LENGTH];
        sprintf(filename, "lbviz/jac2d-lb-%d.json", myid);
        laik_svg_enable_profiling(inst, filename);
    }
#endif

    double *baseR, *baseW, *sumPtr;
    uint64_t ysizeR, ystrideR, xsizeR;
    uint64_t ysizeW, ystrideW, xsizeW;
    int64_t gx1, gx2, gy1, gy2;
    int64_t x1, x2, y1, y2;

    // two 2d arrays for jacobi, using same space
    Laik_Space *space = laik_new_space_2d(inst, size, size);
    Laik_Data *data1 = laik_new_data(space, laik_Double);
    Laik_Data *data2 = laik_new_data(space, laik_Double);

    // we use two types of partitioners algorithms:
    // - prWrite: cells to update (disjunctive partitioning)
    // - prRead : extends partitionings by haloes, to read neighbor values (!!)
    Laik_Partitioner *prWrite, *prRead;
    prWrite = laik_new_bisection_partitioner();
    prRead = use_cornerhalo ? laik_new_cornerhalo_partitioner(1) : laik_new_halo_partitioner(1);

    // run partitioners to get partitionings over 2d space and <world> group
    // data1/2 are then alternately accessed using pRead/pWrite
    Laik_Partitioning *pWrite, *pRead, *npWrite, *npRead;
    pWrite = laik_new_partitioning(prWrite, world, space, 0);
    pRead = laik_new_partitioning(prRead, world, space, pWrite);
    laik_partitioning_set_name(pWrite, "pWrite");
    laik_partitioning_set_name(pRead, "pRead");

    // for global sum, used for residuum: 1 double accessible by all
    Laik_Space *sp1 = laik_new_space_1d(inst, 1);
    Laik_Partitioning *sumP = laik_new_partitioning(laik_All, world, sp1, 0);
    Laik_Data *sumD = laik_new_data(sp1, laik_Double);
    laik_data_set_name(sumD, "sum");
    laik_switchto_partitioning(sumD, sumP, LAIK_DF_None, LAIK_RO_None);

    // start with writing (= initialization) data1
    Laik_Data *dWrite = data1;
    Laik_Data *dRead = data2;

    // distributed initialization
    laik_switchto_partitioning(dWrite, pWrite, LAIK_DF_None, LAIK_RO_None);
    laik_my_range_2d(pWrite, 0, &gx1, &gx2, &gy1, &gy2);

    // default mapping order for 2d:
    //   with y in [0;ysize], x in [0;xsize[
    //   base[y][x] is at (base + y * ystride + x)
    laik_get_map_2d(dWrite, 0, (void **)&baseW, &ysizeW, &ystrideW, &xsizeW);

    // arbitrary non-zero values based on global indexes to detect bugs
    for (uint64_t y = 0; y < ysizeW; y++)
        for (uint64_t x = 0; x < xsizeW; x++)
            baseW[y * ystrideW + x] = (double)((gx1 + x + gy1 + y) & 6);

    setBoundary(size, pWrite, dWrite);
    laik_log(2, "Init done\n");

    // for statistics (with LAIK_LOG=2)
    double _t, _t1 = laik_wtime(), _t2 = _t1;
    int _last_iter = 0;
    int _res_iters = 0; // iterations done with residuum calculation

    laik_svg_profiler_enter(inst, __func__);

    int iter = 0;
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

        laik_switchto_partitioning(dRead, pRead, LAIK_DF_Preserve, LAIK_RO_None);
        laik_switchto_partitioning(dWrite, pWrite, LAIK_DF_None, LAIK_RO_None);

        laik_get_map_2d(dRead, 0, (void **)&baseR, &ysizeR, &ystrideR, &xsizeR);
        laik_get_map_2d(dWrite, 0, (void **)&baseW, &ysizeW, &ystrideW, &xsizeW);

        setBoundary(size, pWrite, dWrite);

        // local range for which to do 2d stencil, without global edges
        laik_my_range_2d(pWrite, 0, &gx1, &gx2, &gy1, &gy2);
        y1 = (gy1 == 0) ? 1 : 0;
        x1 = (gx1 == 0) ? 1 : 0;
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

        ///////////////
        // do jacobi //
        ///////////////

        // check for residuum every RES_ITER iterations
        if (((iter % RES_ITER) == 0) && (iter >= RES_ITER))
        {
            laik_lb_balance(START_LB_SEGMENT, 0, 0);

            double newValue, diff, res;
            res = 0.0;
            for (int64_t y = y1; y < y2; y++)
            {
                for (int64_t x = x1; x < x2; x++)
                {
                    newValue = 0.25 * (baseR[(y - 1) * ystrideR + x] +
                                       baseR[y * ystrideR + x - 1] +
                                       baseR[y * ystrideR + x + 1] +
                                       baseR[(y + 1) * ystrideR + x]);
                    diff = baseR[y * ystrideR + x] - newValue;
                    res += diff * diff;
                    DO_WORKLOAD(WL_LB_ITER);
                    baseW[y * ystrideW + x] = newValue;
                }
            }
            _res_iters++;

            LOAD_BALANCE(WL_LB_ITER); // load balancing has to be performed before aggregated sum, otherwise the times are incorrect (because of the sum barrier)

            // calculate global residuum
            laik_switchto_flow(sumD, LAIK_DF_None, LAIK_RO_None);
            laik_get_map_1d(sumD, 0, (void **)&sumPtr, 0);
            *sumPtr = res;
            laik_switchto_flow(sumD, LAIK_DF_Preserve, LAIK_RO_Sum);
            laik_get_map_1d(sumD, 0, (void **)&sumPtr, 0);
            res = *sumPtr;

            if (iter > 0)
            {
                _t = laik_wtime();
                // current iteration already done
                int diter = (iter + 1) - _last_iter;
                double dt = _t - _t2;
                double gUpdates = 0.000000001 * size * size; // per iteration
                laik_log(2, "For %d iters: %.3fs, %.3f GF/s, %.3f GB/s",
                         diter, dt,
                         // 4 Flops per update in reg iters, with res 7 (once)
                         gUpdates * (7 + 4 * (diter - 1)) / dt,
                         // per update 32 bytes read + 8 byte written
                         gUpdates * diter * 40 / dt);
                _last_iter = iter + 1;
                _t2 = _t;
            }

            if (laik_myid(laik_data_get_group(sumD)) == 0)
            {
                printf("Residuum after %2d iters: %f\n", iter + 1, res);
            }

            if (res < .001)
                break;
        }
        else
        {
            laik_lb_balance(START_LB_SEGMENT, 0, 0);

            double newValue;
            for (int64_t y = y1; y < y2; y++)
            {
                for (int64_t x = x1; x < x2; x++)
                {
                    newValue = 0.25 * (baseR[(y - 1) * ystrideR + x] +
                                       baseR[y * ystrideR + x - 1] +
                                       baseR[y * ystrideR + x + 1] +
                                       baseR[(y + 1) * ystrideR + x]);
                    DO_WORKLOAD(WL_LB_ITER);
                    baseW[y * ystrideW + x] = newValue;
                }
            }
            
            LOAD_BALANCE(WL_LB_ITER);
        }
    }

    // statistics for all iterations and reductions
    // using work load in all tasks
    if (laik_log_shown(2))
    {
        _t = laik_wtime();
        int diter = iter;
        double dt = _t - _t1;
        double gUpdates = 0.000000001 * size * size; // per iteration
        laik_log(2, "For %d iters: %.3fs, %.3f GF/s, %.3f GB/s",
                 diter, dt,
                 // 2 Flops per update in reg iters, with res 5
                 gUpdates * (7 * _res_iters + 4 * (diter - _res_iters)) / dt,
                 // per update 32 bytes read + 8 byte written
                 gUpdates * diter * 40 / dt);
    }

    if (do_sum)
    {
        Laik_Group *activeGroup = laik_data_get_group(dWrite);

        // for check at end: sum up all just written values
        Laik_Partitioning *pMaster;
        pMaster = laik_new_partitioning(laik_Master, activeGroup, space, 0);
        laik_switchto_partitioning(dWrite, pMaster, LAIK_DF_Preserve, LAIK_RO_None);

        if (laik_myid(activeGroup) == 0)
        {
            double sum = 0.0;
            laik_get_map_2d(dWrite, 0, (void **)&baseW, &ysizeW, &ystrideW, &xsizeW);
            for (uint64_t y = 0; y < ysizeW; y++)
                for (uint64_t x = 0; x < xsizeW; x++)
                    sum += baseW[y * ystrideW + x];
            printf("Global value sum after %d iterations: %f\n",
                   iter, sum);
        }
    }

    EXPORT_TO_FILE(myid, pWrite);
    VISUALIZE(myid);

    laik_svg_profiler_exit(inst, __func__);
    laik_svg_profiler_export_json(inst);

    laik_finalize(inst);
    if (do_profiling && myid == 0)
    {
        save_trace();
    }
    return 0;
}
