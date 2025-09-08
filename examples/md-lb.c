/**
 * 2D Molecular Dynamics example for Load Balancing (parallel).
 * (Lennard-Jones force; collision of two particle cuboids, psuedo-reflective boundaries)
 *
 * WITH load balancing.
 *
 * Based on the MD practical course code (WiSe 2024/25) (not a 1:1 port!).
 * https://github.com/FlamingLeo/MolSim (w3t2, specifically)
 *
 * NOTE: Values will differ slightly from the serial version due to floating point imprecision.
 *       Tried using the Kahan summation algorithm but it didn't really do anything. Maybe use long doubles instead?
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "laik.h"

// simulation parameters
#define START_TIME 0.0
#define END_TIME 5.0 // def: 5.0 (feasable on home desktop)
#define DT 0.0005
#define HALF_DT2 (0.5 * (DT) * (DT))

#define DOMAIN_X 180.0
#define DOMAIN_Y 90.0

#define CUTOFF 3.0

// lennard-jones parameters, same for every particle here for simplicity
// technically, every particle could have a different one
#define MASS 1.0
#define EPSILON 5.0
#define SIGMA 1.0

// distance between particles in a cuboid
#define DISTANCE 1.1225

// cuboid A, stationary, large, bottom
#define A_POS_X 20.0
#define A_POS_Y 20.0
#define A_SIZE_X 100
#define A_SIZE_Y 20
#define A_VX 0.0
#define A_VY 0.0

// cuboid B, moving towards cuboid A, smaller, top
#define B_POS_X 70.0
#define B_POS_Y 60.0
#define B_SIZE_X 20
#define B_SIZE_Y 20
#define B_VX 0.0
#define B_VY -10.0

// precomputed lj constants
double sigma6;
double cutoff2;

// helper function to get local neighbor indices in read (dataW + cornerhalo) partition
// w_idx         : index in write partition
// w_rows, w_cols: write shape
// w_row0, w_col0: write origin (global)
// r_rows, r_cols: read shape
// r_row0, r_col0: read origin (global)
// out_buf[_len] : out array to write neighbor indices into
// returns number of elements written or -1 on error
static inline int64_t neighbors_in_read(int64_t w_idx, int64_t w_rows, int64_t w_cols,
                                        int64_t w_row0, int64_t w_col0, int64_t r_rows,
                                        int64_t r_cols, int64_t r_row0, int64_t r_col0,
                                        int64_t *out_buf, int64_t out_buf_len)
{
    if (w_rows <= 0 || w_cols <= 0 || r_rows <= 0 || r_cols <= 0)
        return -1;
    if (!out_buf)
        return -1;
    if (out_buf_len < 9)
        return -1;

    if (w_idx < 0 || w_idx >= w_rows * w_cols)
        return -1;

    // write index -> write-local row,col
    int64_t wr = w_idx / w_cols;
    int64_t wc = w_idx % w_cols;

    // write-local -> global coords
    int64_t g_r = w_row0 + wr;
    int64_t g_c = w_col0 + wc;

    // global -> read-local coords
    int64_t rr = g_r - r_row0;
    int64_t rc = g_c - r_col0;

    int64_t count = 0;
    for (int64_t dr = -1; dr <= 1; ++dr)
    {
        for (int64_t dc = -1; dc <= 1; ++dc)
        {
            int64_t nr = rr + dr;
            int64_t nc = rc + dc;
            if (nr < 0 || nr >= r_rows)
                continue;
            if (nc < 0 || nc >= r_cols)
                continue;
            out_buf[count++] = nr * r_cols + nc; // read-local 1D index
        }
    }
    return count;
}

static inline int64_t cindex(double px, double py, int64_t ncells_x, int64_t ncells_y)
{
    int64_t ix = (int64_t)(px / CUTOFF);
    int64_t iy = (int64_t)(py / CUTOFF);
    if (ix < 0)
        ix = 0;
    if (iy < 0)
        iy = 0;
    if (ix >= ncells_x)
        ix = ncells_x - 1;
    if (iy >= ncells_y)
        iy = ncells_y - 1;
    return ix + iy * ncells_x;
}

static inline void apply_reflective_bcs(double *baseX, double *baseY, double *baseVX, double *baseVY, int p)
{
    // flip in x direction
    if (baseX[p] < 0.0)
    {
        baseX[p] = -baseX[p];
        baseVX[p] = -baseVX[p];
    }
    else if (baseX[p] > DOMAIN_X)
    {
        baseX[p] = 2.0 * DOMAIN_X - baseX[p];
        baseVX[p] = -baseVX[p];
    }

    // flip in y direction
    if (baseY[p] < 0.0)
    {
        baseY[p] = -baseY[p];
        baseVY[p] = -baseVY[p];
    }
    else if (baseY[p] > DOMAIN_Y)
    {
        baseY[p] = 2.0 * DOMAIN_Y - baseY[p];
        baseVY[p] = -baseVY[p];
    }
}

///////////////////
// cli arguments //
///////////////////

// get strategy string
static const char *stratname(int strat)
{
    switch (strat)
    {
    case 0:
        return "cell time";
    case 1:
        return "particle interaction";
    }
    return "unknown";
}

// setup lb smoothing
//
// defaults:
//   smoothing off
//   alpha = 0.15;
//   rmin  = 0.70;
//   rmax  = 1.25;
static void setup_smoothing(const char *str)
{
    int smoothing = 0; // to be casted to bool
    double am = -1.0, rmi = -1.0, rma = -1.0;
    int rc = sscanf(str, "%d,%lf,%lf,%lf", &smoothing, &am, &rmi, &rma);
    if (rc == 4)
        laik_lb_config_smoothing((bool)smoothing, am, rmi, rma);
    else
    {
        fprintf(stderr, "could not parse smoothing string %s!\n", str);
        exit(EXIT_FAILURE);
    }
}

// setup lb start/stop thresholds
//
// defaults:
//   p_stop  = 3;
//   p_start = 3;
//   t_stop  = 0.05;
//   t_start = 0.10;
static void setup_thresholds(const char *str)
{
    int pstop = -1, pstart = -1;
    double tstop = -1.0, tstart = -1.0;
    int rc = sscanf(str, "%d,%d,%lf,%lf", &pstop, &pstart, &tstop, &tstart);
    if (rc == 4)
        laik_lb_config_thresholds(pstop, pstart, tstop, tstart);
    else
    {
        fprintf(stderr, "could not parse threshold string %s!\n", str);
        exit(EXIT_FAILURE);
    }
}

//////////
// main //
//////////

int main(int argc, char **argv)
{
    laik_lbvis_remove_visdata();

    Laik_Instance *inst = laik_init(&argc, &argv);
    Laik_Group *world = laik_world(inst);
    int myid = laik_myid(world);
    bool profiling = false, lboutput = false;
    int weightstrat = 0;                  // weight calculation strategy to use (0: time per cell, 1: particle proxy)
    int lbevery = 150;                    // do load balancing every __ iterations
    Laik_LBAlgorithm lbalgo = LB_GILBERT; // load balancing algorithm of choice

    // collect command line arguments
    //
    // options:
    // -a <algo>    : lb algorithm
    // -l           : show lb times
    // -n <count>   : do load balancing every n iterations
    // -p           : export trace data
    // -s <d,f,f,f> : configure lb smoothing
    // -t <d,d,f,f> : configure lb thresholds
    // -w <strat>   : choose weight strat
    for (int arg = 1; arg < argc; ++arg)
    {
        // profiling (svg)
        if (!strcmp(argv[arg], "-p"))
            profiling = true;

        // show lb times
        if (!strcmp(argv[arg], "-l"))
            lboutput = true;

        // choose (valid!) weight strat
        if (arg + 1 < argc && !strcmp(argv[arg], "-w"))
        {
            weightstrat = atoi(argv[++arg]);
            if (weightstrat < 0 || weightstrat > 1)
            {
                fprintf(stderr, "unknown weight strat!\n");
                exit(EXIT_FAILURE);
            }
        }

        // choose lb iteration count
        // clamped at 10, but this would still be too low (see: noise)
        if (arg + 1 < argc && !strcmp(argv[arg], "-n"))
        {
            lbevery = atoi(argv[++arg]);
            if (lbevery < 10)
                lbevery = 10;
        }

        // setup smoothing
        if (arg + 1 < argc && !strcmp(argv[arg], "-s"))
            setup_smoothing(argv[++arg]);

        // setup thresholds
        if (arg + 1 < argc && !strcmp(argv[arg], "-t"))
            setup_thresholds(argv[++arg]);

        // choose lb algorithm (unknown -> check lb.c)
        if (arg + 1 < argc && !strcmp(argv[arg], "-a"))
            lbalgo = laik_strtolb(argv[++arg]);
    }

    // enable svg trace
    if (profiling)
        laik_lbvis_enable_trace(myid, inst);

    laik_svg_profiler_enter(inst, __func__);

    // print simulation / program info...
    if (myid == 0)
    {
        printf("2D linked-cell Lennard-Jones cuboid collision\n");
        printf("domain: %g x %g, cutoff=%g, dt=%g\n", DOMAIN_X, DOMAIN_Y, CUTOFF, DT);
        printf("profiling: %d, lblog: %d\n", profiling, lboutput);
        printf("weightstrat: %s\n", stratname(weightstrat));
        printf("using %s with load balancing every %d iterations\n\n", laik_get_lb_algorithm_name(lbalgo), lbevery);
    }

    // log every _ iterations
    int print_every = 100;
    laik_lb_output(lboutput);

    // timers for measuring total integration loop time and per-cell time for LB weights
    Laik_Timer lbtimer = {0};
    Laik_Timer timer = {0};

    // create 1d space for all particles and data containers for each particle property
    int64_t countA = A_SIZE_X * A_SIZE_Y;
    int64_t countB = B_SIZE_X * B_SIZE_Y;
    int64_t nparticles = countA + countB;

    Laik_Space *particle_space = laik_new_space_1d(inst, nparticles);
    Laik_Data *data_x = laik_new_data(particle_space, laik_Double);      // pos x
    Laik_Data *data_y = laik_new_data(particle_space, laik_Double);      // pos y
    Laik_Data *data_vx = laik_new_data(particle_space, laik_Double);     // vel x
    Laik_Data *data_vy = laik_new_data(particle_space, laik_Double);     // vel y
    Laik_Data *data_ax = laik_new_data(particle_space, laik_Double);     // acc x
    Laik_Data *data_ay = laik_new_data(particle_space, laik_Double);     // acc y
    Laik_Data *data_ax_old = laik_new_data(particle_space, laik_Double); // acc x (old / prev)
    Laik_Data *data_ay_old = laik_new_data(particle_space, laik_Double); // acc x (old / prev)
    Laik_Data *data_next = laik_new_data(particle_space, laik_Int64);    // index of next particle in a cell (or -1)

    Laik_Partitioning *particle_space_partitioning = laik_new_partitioning(laik_new_block_partitioner1(), world, particle_space, 0); // block for loop calculations
    Laik_Partitioning *particle_space_partitioning_master = laik_new_partitioning(laik_Master, world, particle_space, 0);            // master for init (and aggregation?)
    Laik_Partitioning *particle_space_partitioning_all = laik_new_partitioning(laik_All, world, particle_space, 0);                  // all for aggregation

    laik_switchto_partitioning(data_x, particle_space_partitioning_master, LAIK_DF_None, LAIK_RO_None);
    laik_switchto_partitioning(data_y, particle_space_partitioning_master, LAIK_DF_None, LAIK_RO_None);
    laik_switchto_partitioning(data_vx, particle_space_partitioning_master, LAIK_DF_None, LAIK_RO_None);
    laik_switchto_partitioning(data_vy, particle_space_partitioning_master, LAIK_DF_None, LAIK_RO_None);
    laik_switchto_partitioning(data_ax, particle_space_partitioning_master, LAIK_DF_None, LAIK_RO_None);
    laik_switchto_partitioning(data_ay, particle_space_partitioning_master, LAIK_DF_None, LAIK_RO_None);

    // particle count and base pointers
    int64_t count;
    double *baseX, *baseY, *baseVX, *baseVY, *baseAX, *baseAY, *baseAXOld, *baseAYOld;
    int64_t *baseNext;

    // master initializes containers
    if (myid == 0)
    {
        int64_t idx = 0;

        laik_get_map_1d(data_x, 0, (void **)&baseX, &count);
        laik_get_map_1d(data_y, 0, (void **)&baseY, &count);
        laik_get_map_1d(data_vx, 0, (void **)&baseVX, &count);
        laik_get_map_1d(data_vy, 0, (void **)&baseVY, &count);
        laik_get_map_1d(data_ax, 0, (void **)&baseAX, &count);
        laik_get_map_1d(data_ay, 0, (void **)&baseAY, &count);

        // cuboid A
        for (int j = 0; j < A_SIZE_Y; ++j)
        {
            for (int i = 0; i < A_SIZE_X; ++i)
            {
                baseX[idx] = A_POS_X + i * DISTANCE;
                baseY[idx] = A_POS_Y + j * DISTANCE;
                baseVX[idx] = A_VX;
                baseVY[idx] = A_VY;
                baseAX[idx] = baseAY[idx] = 0.0;
                ++idx;
            }
        }

        // cuboid B
        for (int j = 0; j < B_SIZE_Y; ++j)
        {
            for (int i = 0; i < B_SIZE_X; ++i)
            {
                baseX[idx] = B_POS_X + i * DISTANCE;
                baseY[idx] = B_POS_Y + j * DISTANCE;
                baseVX[idx] = B_VX;
                baseVY[idx] = B_VY;
                baseAX[idx] = baseAY[idx] = 0.0;
                ++idx;
            }
        }

        // by now, every task should have the complete initial particle cuboid data
        assert(idx == nparticles);
    }

    // initialize cell grid
    // make sure that cutoff perfectly divides domain size in both dimensions
    assert(fabs(fmod(DOMAIN_X, CUTOFF)) < 1e-12);
    assert(fabs(fmod(DOMAIN_Y, CUTOFF)) < 1e-12);
    int64_t ncells_x = (int64_t)(DOMAIN_X / CUTOFF);
    int64_t ncells_y = (int64_t)(DOMAIN_Y / CUTOFF);
    assert(ncells_x != 0 && ncells_y != 0);
    int64_t ncells = ncells_x * ncells_y;
    Laik_Space *cell_space = laik_new_space_2d(inst, ncells_x, ncells_y);
    Laik_Data *data_head_w = laik_new_data(cell_space, laik_Int64); // index of first particle in cell (or -1)
    Laik_Data *data_head_r = laik_new_data(cell_space, laik_Int64);

    // custom weight array for load balancer
    // note: this will get freed by laik_lb_free (finalize) at the end!
    double *weights = (double *)malloc(ncells * sizeof(double));
    if (!weights)
    {
        fprintf(stderr, "Could not allocate memory for weight array!\n");
        exit(EXIT_FAILURE);
    }

    Laik_Partitioner *cell_partitioner_w = laik_new_bisection_partitioner();
    Laik_Partitioner *cell_partitioner_r = laik_new_cornerhalo_partitioner(1);
    Laik_Partitioning *cell_partitioning_master = laik_new_partitioning(laik_Master, world, cell_space, 0);
    Laik_Partitioning *cell_partitioning_w = laik_new_partitioning(cell_partitioner_w, world, cell_space, 0);
    Laik_Partitioning *cell_partitioning_r = laik_new_partitioning(cell_partitioner_r, world, cell_space, cell_partitioning_w);
    Laik_Partitioning *cpw_new = cell_partitioning_w, *cpr_new = cell_partitioning_r;

    int64_t *baseHeadW, *baseHeadR;

    // precompute lj constants
    sigma6 = pow(SIGMA, 6);
    cutoff2 = CUTOFF * CUTOFF;

    // calculate how many iteration steps we take
    long nsteps = (long)((END_TIME - START_TIME) / DT + 0.5);
    if (myid == 0)
    {
        printf("npart = %ld\n", nparticles);
        printf("Cells: %ld x %ld = %ld\n", ncells_x, ncells_y, ncells);
        printf("Running %ld steps (endTime=%g)\n\n", nsteps, END_TIME);
    }

    // vx, vy, axold, ayold always stay the same (block part)
    laik_switchto_partitioning(data_vx, particle_space_partitioning, LAIK_DF_Preserve, LAIK_RO_None);
    laik_switchto_partitioning(data_vy, particle_space_partitioning, LAIK_DF_Preserve, LAIK_RO_None);
    laik_switchto_partitioning(data_ax_old, particle_space_partitioning, LAIK_DF_None, LAIK_RO_None);
    laik_switchto_partitioning(data_ay_old, particle_space_partitioning, LAIK_DF_None, LAIK_RO_None);
    laik_get_map_1d(data_vx, 0, (void **)&baseVX, 0);
    laik_get_map_1d(data_vy, 0, (void **)&baseVY, 0);
    laik_get_map_1d(data_ax_old, 0, (void **)&baseAXOld, 0);
    laik_get_map_1d(data_ay_old, 0, (void **)&baseAYOld, 0);

    // kinetic energy for testing
    Laik_Space *kespace = laik_new_space_1d(inst, 1);
    Laik_Data *kedata = laik_new_data(kespace, laik_Double);
    Laik_Partitioning *kepart = laik_new_partitioning(laik_All, world, kespace, 0);
    laik_switchto_partitioning(kedata, kepart, LAIK_DF_None, LAIK_RO_None);

    ////////////////////////////////////
    // integration loop (X -> F -> V) //
    ////////////////////////////////////

    double t = START_TIME;
    laik_timer_start(&timer);
    laik_svg_profiler_enter(inst, "integration-loop");
    for (long step = 0; step < nsteps; ++step)
    {
        laik_svg_profiler_enter(inst, "x,y,ax,ay: master -> block");
        laik_switchto_partitioning(data_x, particle_space_partitioning, LAIK_DF_Preserve, LAIK_RO_None);
        laik_switchto_partitioning(data_y, particle_space_partitioning, LAIK_DF_Preserve, LAIK_RO_None);
        laik_switchto_partitioning(data_ax, particle_space_partitioning, LAIK_DF_Preserve, LAIK_RO_None);
        laik_switchto_partitioning(data_ay, particle_space_partitioning, LAIK_DF_Preserve, LAIK_RO_None);
        laik_svg_profiler_exit(inst, "x,y,ax,ay: master -> block");

        // count is all the same here (identical partitioning)
        laik_get_map_1d(data_x, 0, (void **)&baseX, &count);
        laik_get_map_1d(data_y, 0, (void **)&baseY, 0);
        laik_get_map_1d(data_ax, 0, (void **)&baseAX, 0);
        laik_get_map_1d(data_ay, 0, (void **)&baseAY, 0);

        // store old accels, calculate positions, handle reflective boundaries and zero forces before lj
        for (int64_t i = 0; i < count; ++i)
        {
            baseAXOld[i] = baseAX[i];
            baseAYOld[i] = baseAY[i];
            baseX[i] += baseVX[i] * DT + baseAX[i] * HALF_DT2;
            baseY[i] += baseVY[i] * DT + baseAY[i] * HALF_DT2;
            apply_reflective_bcs(baseX, baseY, baseVX, baseVY, i);
            baseAX[i] = 0.0;
            baseAY[i] = 0.0;
        }

        // distribute relevant data for acceleration calculation to all tasks
        laik_svg_profiler_enter(inst, "x,y,ax,ay: block -> all");
        laik_switchto_partitioning(data_x, particle_space_partitioning_all, LAIK_DF_Preserve, LAIK_RO_None);
        laik_switchto_partitioning(data_y, particle_space_partitioning_all, LAIK_DF_Preserve, LAIK_RO_None);
        laik_switchto_partitioning(data_ax, particle_space_partitioning_all, LAIK_DF_Preserve, LAIK_RO_None);
        laik_switchto_partitioning(data_ay, particle_space_partitioning_all, LAIK_DF_Preserve, LAIK_RO_None);
        laik_svg_profiler_exit(inst, "x,y,ax,ay: block -> all");

        // master (re-)initializes cell list
        laik_svg_profiler_enter(inst, "w,r,next: init");
        laik_switchto_partitioning(data_head_w, cell_partitioning_master, LAIK_DF_None, LAIK_RO_None);
        laik_switchto_partitioning(data_head_r, cell_partitioning_master, LAIK_DF_None, LAIK_RO_None);
        laik_switchto_partitioning(data_next, particle_space_partitioning_master, LAIK_DF_None, LAIK_RO_None);
        laik_svg_profiler_exit(inst, "w,r,next: init");

        if (myid == 0)
        {
            int64_t ysize, ystride, xsize;
            laik_get_map_2d(data_head_w, 0, (void **)&baseHeadW, &ysize, &ystride, &xsize);
            laik_get_map_2d(data_head_r, 0, (void **)&baseHeadR, 0, 0, 0); // size and stride should be the same
            laik_get_map_1d(data_next, 0, (void **)&baseNext, 0);          // maybe not needed? in case next base ptr doesn't change
            laik_get_map_1d(data_x, 0, (void **)&baseX, 0);
            laik_get_map_1d(data_y, 0, (void **)&baseY, 0);
            assert(xsize == ncells_x && ysize == ncells_y && ystride == ncells_x);

            // reset cell list by setting the head (first particle) of each cell to -1
            for (int64_t y = 0; y < ysize; ++y)
            {
                for (int64_t x = 0; x < xsize; ++x)
                {
                    baseHeadW[y * ystride + x] = -1;
                    baseHeadR[y * ystride + x] = -1;
                }
            }

            // then, for every particle, push it(s index) to the front of its corresponding cell
            for (int64_t p = 0; p < nparticles; ++p)
            {
                // push p to front of cell
                int64_t c = cindex(baseX[p], baseY[p], xsize, ysize);
                baseNext[p] = baseHeadW[c];
                baseHeadW[c] = p;
                baseHeadR[c] = p;
            }
        }

        // partition initialized cell list across all tasks
        laik_svg_profiler_enter(inst, "w,r,next: master -> ?/halo/all");
        if (cell_partitioning_w != cpw_new)
        {
            laik_lb_switch_and_free(&cell_partitioning_w, &cpw_new, data_head_w, LAIK_DF_Preserve);
            laik_lb_switch_and_free(&cell_partitioning_r, &cpr_new, data_head_r, LAIK_DF_Preserve);
            cpw_new = cell_partitioning_w;
            cpr_new = cell_partitioning_r;
        }
        else
        {
            laik_switchto_partitioning(data_head_w, cell_partitioning_w, LAIK_DF_Preserve, LAIK_DF_None);
            laik_switchto_partitioning(data_head_r, cell_partitioning_r, LAIK_DF_Preserve, LAIK_DF_None);
        }
        laik_switchto_partitioning(data_next, particle_space_partitioning_all, LAIK_DF_Preserve, LAIK_DF_None);
        laik_svg_profiler_exit(inst, "w,r,next: master -> ?/halo/all");

        // all   : x,y,ax,ay
        // bisect: cell head W/R (halo)
        laik_get_map_1d(data_x, 0, (void **)&baseX, 0);
        laik_get_map_1d(data_y, 0, (void **)&baseY, 0);
        laik_get_map_1d(data_ax, 0, (void **)&baseAX, 0);
        laik_get_map_1d(data_ay, 0, (void **)&baseAY, 0);
        laik_get_map_1d(data_next, 0, (void **)&baseNext, 0);

        ////////////////
        // force calc //
        ////////////////

        // initialize potential and weight arrays to 0
        double pot = 0.0;
        if (step % lbevery == 0)
            memset(weights, 0, ncells * sizeof(double));

        // start load balancing segment
        laik_lb_balance(step % lbevery == 0 ? START_LB_SEGMENT : RESUME_LB_SEGMENT, 0, 0);
        for (int r = 0; r < laik_my_rangecount(cell_partitioning_w); ++r)
        {
            int64_t ysizeW, ystrideW, xsizeW;
            int64_t ysizeR, ystrideR, xsizeR;
            int64_t fromXW, toXW, fromYW, toYW;
            int64_t fromXR, toXR, fromYR, toYR;
            laik_get_map_2d(data_head_w, r, (void **)&baseHeadW, &ysizeW, &ystrideW, &xsizeW);
            laik_get_map_2d(data_head_r, r, (void **)&baseHeadR, &ysizeR, &ystrideR, &xsizeR);
            laik_my_range_2d(cell_partitioning_w, r, &fromXW, &toXW, &fromYW, &toYW);
            laik_my_range_2d(cell_partitioning_r, r, &fromXR, &toXR, &fromYR, &toYR);

            // for all owned cells c... (write part.)
            for (int64_t cy = 0; cy < ysizeW; ++cy)
            {
                for (int64_t cx = 0; cx < xsizeW; ++cx)
                {
                    int64_t c = cx + cy * ystrideW;

                    int64_t pcount_c = 0;
                    int64_t pcount_n = 0;
                    laik_timer_start(&lbtimer);

                    // for all particles p in c (globally indexed)...
                    for (int p = baseHeadW[c]; p != -1; p = baseNext[p])
                    {
                        pcount_c++;
                        int64_t neighbors[9];
                        int64_t ncount = neighbors_in_read(c, ysizeW, ystrideW, fromYW, fromXW,
                                                           ysizeR, ystrideR, fromYR, fromXR,
                                                           neighbors, 9);
                        assert(ncount != -1);
                        // for all neighbor cells nc... (read part.)
                        for (int64_t nci = 0; nci < ncount; ++nci)
                        {
                            int64_t nc = neighbors[nci];

                            // for all particles q in nc (also globally indexed)...
                            for (int q = baseHeadR[nc]; q != -1; q = baseNext[q])
                            {
                                pcount_n++;
                                // avoid counting double (n3l)
                                if (q <= p)
                                    continue;

                                // do the formula
                                double dx = baseX[p] - baseX[q];
                                double dy = baseY[p] - baseY[q];
                                double r2 = dx * dx + dy * dy;

                                // same particle safety guard
                                if (r2 <= 1e-12)
                                    continue;

                                if (r2 <= cutoff2)
                                {
                                    double invr2 = 1.0 / r2;
                                    double invr6 = invr2 * invr2 * invr2; // (1/r^2)^3 = 1/r^6
                                    double sor6 = sigma6 * invr6;         // (sigma^6)/(r^6)
                                    double sor12 = sor6 * sor6;           // (sigma/r)^12

                                    // factor = 24*eps*(2*s12 - s6) * (1/r^2)
                                    double factor = 24.0 * EPSILON * (2.0 * sor12 - sor6) * invr2;
                                    double fx = factor * dx;
                                    double fy = factor * dy;

                                    baseAX[p] += fx / MASS;
                                    baseAY[p] += fy / MASS;
                                    baseAX[q] -= fx / MASS;
                                    baseAY[q] -= fy / MASS;

                                    double vp = 4.0 * EPSILON * (sor12 - sor6);
                                    pot += vp;
                                }
                            } // end q
                        } // end nc
                    } // end p
                    double taken = laik_timer_stop(&lbtimer);

                    int64_t global_x = fromXW + cx;
                    int64_t global_y = fromYW + cy;
                    int64_t global_idx = global_y * ncells_x + global_x;

                    // update cell weight based on chosen strategy
                    if (!weightstrat)
                    {
                        // time taken for this cell
                        weights[global_idx] += taken;
                    }
                    else
                    {
                        // remove neighbor cells counted for each current cell particle
                        if (pcount_c > 0)
                            pcount_n = (pcount_n / pcount_c) - pcount_c;

                        // weight based on intra- and inter-particle communication
                        // TODO: this probably would not work for CPUs with different speeds, consider taking time into account?
                        weights[global_idx] += (/* baseline */ 0.02) +
                                               (/* cost per particle */ 1.0 * pcount_c) +
                                               (/* inter-cell */ 0.40 * pcount_c * pcount_n) +
                                               (/* intra-cell */ 0.25 * pcount_c * pcount_c);
                    }

                    // get weight average across LB iterations
                    if (step % lbevery == (lbevery - 1))
                        weights[global_idx] /= lbevery;
                } // end cx
            } // end cy
        } // end ranges

        // stop load balancing, use custom weights and adjust partitioning pointers
        if ((step % lbevery == (lbevery - 1)))
        {
            laik_lb_set_ext_weights(weights);
            cpw_new = laik_lb_balance(STOP_LB_SEGMENT, cell_partitioning_w, lbalgo);
            if (cpw_new != cell_partitioning_w)
                cpr_new = laik_new_partitioning(cell_partitioner_r, world, cell_space, cpw_new);
        }
        else
            laik_lb_balance(PAUSE_LB_SEGMENT, 0, 0);

        // aggregate forces for velocity calc
        // NOTE: this is the part where tasks must wait for all other tasks to finish computing forces,
        //       since we need to aggregate x,y,ax,ay, which we used (all)
        //       this is basically the only real bottleneck, since everything else is balanced with even workloads
        laik_svg_profiler_enter(inst, "x,y: all -> master/block");
        laik_switchto_partitioning(data_x, particle_space_partitioning_master, LAIK_DF_Preserve, LAIK_RO_Max); // doesn't really do anything, just so ALL -> BLOCK in the next step will work
        laik_switchto_partitioning(data_y, particle_space_partitioning_master, LAIK_DF_Preserve, LAIK_RO_Max); // MAX because they all have the same values anyway
        laik_svg_profiler_exit(inst, "x,y: all -> master/block");

        laik_svg_profiler_enter(inst, "ax,ay: all -> block");
        laik_switchto_partitioning(data_ax, particle_space_partitioning, LAIK_DF_Preserve, LAIK_RO_Sum);
        laik_switchto_partitioning(data_ay, particle_space_partitioning, LAIK_DF_Preserve, LAIK_RO_Sum);
        laik_svg_profiler_exit(inst, "ax,ay: all -> block");

        laik_get_map_1d(data_ax, 0, (void **)&baseAX, &count); // count probably not needed here
        laik_get_map_1d(data_ay, 0, (void **)&baseAY, 0);

        // velocities (over particles)
        for (int64_t i = 0; i < count; ++i)
        {
            baseVX[i] += 0.5 * (baseAXOld[i] + baseAX[i]) * DT;
            baseVY[i] += 0.5 * (baseAYOld[i] + baseAY[i]) * DT;
        }

        t += DT;

        // follow intermediate state via kinetic energy (over particles)
        if ((step % print_every) == 0)
        {
            double ke = 0.0;
            for (int i = 0; i < count; ++i)
                ke += 0.5 * MASS * (baseVX[i] * baseVX[i] + baseVY[i] * baseVY[i]);
            double total = ke + pot;

            double *keBase;
            laik_switchto_flow(kedata, LAIK_DF_None, LAIK_RO_None);
            laik_get_map_1d(kedata, 0, (void **)&keBase, 0);
            *keBase = total;
            laik_switchto_flow(kedata, LAIK_DF_Preserve, LAIK_RO_Sum);
            laik_get_map_1d(kedata, 0, (void **)&keBase, 0);
            total = *keBase;

            if (myid == 0)
                printf("step %ld / %ld, t=%.4f, E=%.6f\n",
                       step, nsteps, t, total);
        }
    } // end loop

    laik_svg_profiler_exit(inst, "integration-loop");
    double tfinal = laik_timer_stop(&timer);

    if (myid == 0)
        printf("\nDone. Time taken: %fs\n", tfinal);

    laik_svg_profiler_exit(inst, __func__);
    laik_svg_profiler_export_json(inst);

    laik_finalize(inst);
    if (myid == 0 && profiling)
        laik_lbvis_save_trace();
    return 0;
}
