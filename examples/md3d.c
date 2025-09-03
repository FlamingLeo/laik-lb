/**
 * 3D Molecular Dynamics example for Load Balancing (parallel).
 * (Lennard-Jones force; collision of two particle cuboids, psuedo-reflective boundaries)
 *
 * 2D example converted to 3D, see credits from md.c if you're interested.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "laik.h"

// use the defines the user requested
// simulation parameters
#define START_TIME 0.0
#define END_TIME 1.0
#define DT 0.0005

#define DOMAIN_X 180.0
#define DOMAIN_Y 90.0
#define DOMAIN_Z 90.0

#define CUTOFF 3.0
#define CUTOFF2 ((CUTOFF) * (CUTOFF))

// lennard-jones parameters, same for every particle here for simplicity
#define MASS 1.0
#define EPSILON 5.0
#define SIGMA 1.0

// distance between particles in a cuboid
#define DISTANCE 1.1225

// x: width
// y: height
// z: depth

// cuboid A, stationary, large, bottom
#define A_POS_X 20.0
#define A_POS_Y 20.0
#define A_POS_Z 10.0
#define A_SIZE_X 100
#define A_SIZE_Y 20
#define A_SIZE_Z 10
#define A_VX 0.0
#define A_VY 0.0
#define A_VZ 0.0

// cuboid B, moving towards cuboid A, smaller, top
#define B_POS_X 70.0
#define B_POS_Y 60.0
#define B_POS_Z 10.0
#define B_SIZE_X 20
#define B_SIZE_Y 20
#define B_SIZE_Z 10
#define B_VX 0.0
#define B_VY -10.0
#define B_VZ 0.0

// precomputed lj constants
double sigma6;
double cutoff2;

// helper function to get local neighbor indices in read (dataW + cornerhalo) partition
// adapted to 3D
// Parameters:
//  w_idx         : index in write partition (1D index into write-local 3D block)
//  w_x, w_y, w_z : write shape (xsize, ysize, zsize)
//  w_x0, w_y0, w_z0 : write origin (global coords of the write block start)
//  r_x, r_y, r_z : read shape (xsize, ysize, zsize)
//  r_x0, r_y0, r_z0 : read origin (global coords of read block start)
//  out_buf[_len] : out array to write neighbor indices into (up to 27 entries)
// returns number of elements written or -1 on error
static inline int64_t neighbors_in_read_3d(int64_t w_idx,
                                           int64_t w_x, int64_t w_y, int64_t w_z,
                                           int64_t w_x0, int64_t w_y0, int64_t w_z0,
                                           int64_t r_x, int64_t r_y, int64_t r_z,
                                           int64_t r_x0, int64_t r_y0, int64_t r_z0,
                                           int64_t *out_buf, int64_t out_buf_len)
{
    if (w_x <= 0 || w_y <= 0 || w_z <= 0 || r_x <= 0 || r_y <= 0 || r_z <= 0)
        return -1;
    if (!out_buf)
        return -1;
    if (out_buf_len < 27)
        return -1;

    if (w_idx < 0 || w_idx >= w_x * w_y * w_z)
        return -1;

    // convert 1D index -> (wx, wy, wz)
    int64_t plane = w_x * w_y;
    int64_t wz = w_idx / plane;
    int64_t rem = w_idx % plane;
    int64_t wy = rem / w_x;
    int64_t wx = rem % w_x;

    // write-local -> global coords
    int64_t g_x = w_x0 + wx;
    int64_t g_y = w_y0 + wy;
    int64_t g_z = w_z0 + wz;

    // global -> read-local coords (relative to read origin)
    int64_t rr_x = g_x - r_x0;
    int64_t rr_y = g_y - r_y0;
    int64_t rr_z = g_z - r_z0;

    int64_t count = 0;
    for (int64_t dz = -1; dz <= 1; ++dz)
    {
        for (int64_t dy = -1; dy <= 1; ++dy)
        {
            for (int64_t dx = -1; dx <= 1; ++dx)
            {
                int64_t nx = rr_x + dx;
                int64_t ny = rr_y + dy;
                int64_t nz = rr_z + dz;
                if (nx < 0 || nx >= r_x)
                    continue;
                if (ny < 0 || ny >= r_y)
                    continue;
                if (nz < 0 || nz >= r_z)
                    continue;
                // linear index in read-local 1D layout (x fastest)
                int64_t idx = nx + ny * r_x + nz * (r_x * r_y);
                out_buf[count++] = idx;
            }
        }
    }
    return count;
}

static inline int64_t cindex3d(double px, double py, double pz, int64_t ncells_x, int64_t ncells_y, int64_t ncells_z)
{
    int64_t ix = (int64_t)(px / CUTOFF);
    int64_t iy = (int64_t)(py / CUTOFF);
    int64_t iz = (int64_t)(pz / CUTOFF);
    if (ix < 0)
        ix = 0;
    if (iy < 0)
        iy = 0;
    if (iz < 0)
        iz = 0;
    if (ix >= ncells_x)
        ix = ncells_x - 1;
    if (iy >= ncells_y)
        iy = ncells_y - 1;
    if (iz >= ncells_z)
        iz = ncells_z - 1;
    return ix + iy * ncells_x + iz * (ncells_x * ncells_y);
}

static inline void apply_reflective_bcs_3d(double *baseX, double *baseY, double *baseZ,
                                           double *baseVX, double *baseVY, double *baseVZ, int p)
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

    // flip in z direction
    if (baseZ[p] < 0.0)
    {
        baseZ[p] = -baseZ[p];
        baseVZ[p] = -baseVZ[p];
    }
    else if (baseZ[p] > DOMAIN_Z)
    {
        baseZ[p] = 2.0 * DOMAIN_Z - baseZ[p];
        baseVZ[p] = -baseVZ[p];
    }
}

//////////
// main //
//////////

int main(int argc, char **argv)
{
    Laik_Instance *inst = laik_init(&argc, &argv);
    Laik_Group *world = laik_world(inst);
    int myid = laik_myid(world);

    if (myid == 0)
    {
        printf("3D linked-cell Lennard-Jones cuboid collision\n");
        printf("domain: %g x %g x %g, cutoff=%g, dt=%g\n", DOMAIN_X, DOMAIN_Y, DOMAIN_Z, CUTOFF, DT);
    }

    // log every _ iterations
    int print_every = 100;

    // create 1d space for all particles and data containers for each particle property
    int64_t countA = A_SIZE_X * A_SIZE_Y * A_SIZE_Z;
    int64_t countB = B_SIZE_X * B_SIZE_Y * B_SIZE_Z;
    int64_t nparticles = countA + countB;

    Laik_Space *particle_space = laik_new_space_1d(inst, nparticles);
    Laik_Data *data_x = laik_new_data(particle_space, laik_Double);      // pos x
    Laik_Data *data_y = laik_new_data(particle_space, laik_Double);      // pos y
    Laik_Data *data_z = laik_new_data(particle_space, laik_Double);      // pos z
    Laik_Data *data_vx = laik_new_data(particle_space, laik_Double);     // vel x
    Laik_Data *data_vy = laik_new_data(particle_space, laik_Double);     // vel y
    Laik_Data *data_vz = laik_new_data(particle_space, laik_Double);     // vel z
    Laik_Data *data_ax = laik_new_data(particle_space, laik_Double);     // acc x
    Laik_Data *data_ay = laik_new_data(particle_space, laik_Double);     // acc y
    Laik_Data *data_az = laik_new_data(particle_space, laik_Double);     // acc z
    Laik_Data *data_ax_old = laik_new_data(particle_space, laik_Double); // acc x (old / prev)
    Laik_Data *data_ay_old = laik_new_data(particle_space, laik_Double); // acc y (old / prev)
    Laik_Data *data_az_old = laik_new_data(particle_space, laik_Double); // acc z (old / prev)
    Laik_Data *data_next = laik_new_data(particle_space, laik_Int64);    // index of next particle in a cell (or -1)

    Laik_Partitioning *particle_space_partitioning = laik_new_partitioning(laik_new_block_partitioner1(), world, particle_space, 0); // block for loop calculations
    Laik_Partitioning *particle_space_partitioning_master = laik_new_partitioning(laik_Master, world, particle_space, 0);            // master for init (and aggregation?)
    Laik_Partitioning *particle_space_partitioning_all = laik_new_partitioning(laik_All, world, particle_space, 0);                  // all for aggregation

    laik_switchto_partitioning(data_x, particle_space_partitioning_master, LAIK_DF_None, LAIK_RO_None);
    laik_switchto_partitioning(data_y, particle_space_partitioning_master, LAIK_DF_None, LAIK_RO_None);
    laik_switchto_partitioning(data_z, particle_space_partitioning_master, LAIK_DF_None, LAIK_RO_None);
    laik_switchto_partitioning(data_vx, particle_space_partitioning_master, LAIK_DF_None, LAIK_RO_None);
    laik_switchto_partitioning(data_vy, particle_space_partitioning_master, LAIK_DF_None, LAIK_RO_None);
    laik_switchto_partitioning(data_vz, particle_space_partitioning_master, LAIK_DF_None, LAIK_RO_None);
    laik_switchto_partitioning(data_ax, particle_space_partitioning_master, LAIK_DF_None, LAIK_RO_None);
    laik_switchto_partitioning(data_ay, particle_space_partitioning_master, LAIK_DF_None, LAIK_RO_None);
    laik_switchto_partitioning(data_az, particle_space_partitioning_master, LAIK_DF_None, LAIK_RO_None);

    // particle count and base pointers
    int64_t count;
    double *baseX, *baseY, *baseZ, *baseVX, *baseVY, *baseVZ, *baseAX, *baseAY, *baseAZ, *baseAXOld, *baseAYOld, *baseAZOld;
    int64_t *baseNext;

    // master initializes containers
    if (myid == 0)
    {
        int64_t idx = 0;

        laik_get_map_1d(data_x, 0, (void **)&baseX, &count);
        laik_get_map_1d(data_y, 0, (void **)&baseY, &count);
        laik_get_map_1d(data_z, 0, (void **)&baseZ, &count);
        laik_get_map_1d(data_vx, 0, (void **)&baseVX, &count);
        laik_get_map_1d(data_vy, 0, (void **)&baseVY, &count);
        laik_get_map_1d(data_vz, 0, (void **)&baseVZ, &count);
        laik_get_map_1d(data_ax, 0, (void **)&baseAX, &count);
        laik_get_map_1d(data_ay, 0, (void **)&baseAY, &count);
        laik_get_map_1d(data_az, 0, (void **)&baseAZ, &count);

        // cuboid A
        for (int kz = 0; kz < A_SIZE_Z; ++kz)
        {
            for (int j = 0; j < A_SIZE_Y; ++j)
            {
                for (int i = 0; i < A_SIZE_X; ++i)
                {
                    baseX[idx] = A_POS_X + i * DISTANCE;
                    baseY[idx] = A_POS_Y + j * DISTANCE;
                    baseZ[idx] = A_POS_Z + kz * DISTANCE;
                    baseVX[idx] = A_VX;
                    baseVY[idx] = A_VY;
                    baseVZ[idx] = A_VZ;
                    baseAX[idx] = baseAY[idx] = baseAZ[idx] = 0.0;
                    ++idx;
                }
            }
        }

        // cuboid B
        for (int kz = 0; kz < B_SIZE_Z; ++kz)
        {
            for (int j = 0; j < B_SIZE_Y; ++j)
            {
                for (int i = 0; i < B_SIZE_X; ++i)
                {
                    baseX[idx] = B_POS_X + i * DISTANCE;
                    baseY[idx] = B_POS_Y + j * DISTANCE;
                    baseZ[idx] = B_POS_Z + kz * DISTANCE;
                    baseVX[idx] = B_VX;
                    baseVY[idx] = B_VY;
                    baseVZ[idx] = B_VZ;
                    baseAX[idx] = baseAY[idx] = baseAZ[idx] = 0.0;
                    ++idx;
                }
            }
        }

        // by now, every task should have the complete initial particle cuboid data
        assert(idx == nparticles);
    }

    // initialize cell grid
    // make sure that cutoff perfectly divides domain size in all dimensions
    assert(fabs(fmod(DOMAIN_X, CUTOFF)) < 1e-12);
    assert(fabs(fmod(DOMAIN_Y, CUTOFF)) < 1e-12);
    assert(fabs(fmod(DOMAIN_Z, CUTOFF)) < 1e-12);
    int64_t ncells_x = (int64_t)(DOMAIN_X / CUTOFF);
    int64_t ncells_y = (int64_t)(DOMAIN_Y / CUTOFF);
    int64_t ncells_z = (int64_t)(DOMAIN_Z / CUTOFF);
    assert(ncells_x != 0 && ncells_y != 0 && ncells_z != 0);
    int64_t ncells = ncells_x * ncells_y * ncells_z;
    Laik_Space *cell_space = laik_new_space_3d(inst, ncells_x, ncells_y, ncells_z);
    Laik_Data *data_head_w = laik_new_data(cell_space, laik_Int64); // index of first particle in cell (or -1)
    Laik_Data *data_head_r = laik_new_data(cell_space, laik_Int64);

    Laik_Partitioner *cell_partitioner_w = laik_new_bisection_partitioner();
    Laik_Partitioner *cell_partitioner_r = laik_new_cornerhalo_partitioner(1);
    Laik_Partitioning *cell_partitioning_master = laik_new_partitioning(laik_Master, world, cell_space, 0);
    Laik_Partitioning *cell_partitioning_w = laik_new_partitioning(cell_partitioner_w, world, cell_space, 0);
    Laik_Partitioning *cell_partitioning_r = laik_new_partitioning(cell_partitioner_r, world, cell_space, cell_partitioning_w);

    int64_t *baseHeadW, *baseHeadR;

    // precompute lj constants
    sigma6 = pow(SIGMA, 6);
    cutoff2 = CUTOFF2;

    // calculate how many iteration steps we take
    long nsteps = (long)((END_TIME - START_TIME) / DT + 0.5);
    if (myid == 0)
    {
        printf("npart = %ld\n", nparticles);
        printf("Cells: %ld x %ld x %ld = %ld\n", ncells_x, ncells_y, ncells_z, ncells);
        printf("Running %ld steps (endTime=%g)\n", nsteps, END_TIME);
    }

    // vx, vy, vz, axold, ayold, azold always stay the same (block part)
    laik_switchto_partitioning(data_vx, particle_space_partitioning, LAIK_DF_Preserve, LAIK_RO_None);
    laik_switchto_partitioning(data_vy, particle_space_partitioning, LAIK_DF_Preserve, LAIK_RO_None);
    laik_switchto_partitioning(data_vz, particle_space_partitioning, LAIK_DF_Preserve, LAIK_RO_None);
    laik_switchto_partitioning(data_ax_old, particle_space_partitioning, LAIK_DF_None, LAIK_RO_None);
    laik_switchto_partitioning(data_ay_old, particle_space_partitioning, LAIK_DF_None, LAIK_RO_None);
    laik_switchto_partitioning(data_az_old, particle_space_partitioning, LAIK_DF_None, LAIK_RO_None);
    laik_get_map_1d(data_vx, 0, (void **)&baseVX, 0);
    laik_get_map_1d(data_vy, 0, (void **)&baseVY, 0);
    laik_get_map_1d(data_vz, 0, (void **)&baseVZ, 0);
    laik_get_map_1d(data_ax_old, 0, (void **)&baseAXOld, 0);
    laik_get_map_1d(data_ay_old, 0, (void **)&baseAYOld, 0);
    laik_get_map_1d(data_az_old, 0, (void **)&baseAZOld, 0);

    // kinetic energy for testing
    Laik_Space *kespace = laik_new_space_1d(inst, 1);
    Laik_Data *kedata = laik_new_data(kespace, laik_Double);
    Laik_Partitioning *kepart = laik_new_partitioning(laik_All, world, kespace, 0);
    laik_switchto_partitioning(kedata, kepart, LAIK_DF_None, LAIK_RO_None);

    ////////////////////////////////////
    // integration loop (X -> F -> V) //
    ////////////////////////////////////

    double t = START_TIME;
    for (long step = 0; step < nsteps; ++step)
    {
        laik_switchto_partitioning(data_x, particle_space_partitioning, LAIK_DF_Preserve, LAIK_RO_None);
        laik_switchto_partitioning(data_y, particle_space_partitioning, LAIK_DF_Preserve, LAIK_RO_None);
        laik_switchto_partitioning(data_z, particle_space_partitioning, LAIK_DF_Preserve, LAIK_RO_None);
        laik_switchto_partitioning(data_ax, particle_space_partitioning, LAIK_DF_Preserve, LAIK_RO_None);
        laik_switchto_partitioning(data_ay, particle_space_partitioning, LAIK_DF_Preserve, LAIK_RO_None);
        laik_switchto_partitioning(data_az, particle_space_partitioning, LAIK_DF_Preserve, LAIK_RO_None);

        // count is all the same here (identical partitioning)
        laik_get_map_1d(data_x, 0, (void **)&baseX, &count);
        laik_get_map_1d(data_y, 0, (void **)&baseY, 0);
        laik_get_map_1d(data_z, 0, (void **)&baseZ, 0);
        laik_get_map_1d(data_ax, 0, (void **)&baseAX, 0);
        laik_get_map_1d(data_ay, 0, (void **)&baseAY, 0);
        laik_get_map_1d(data_az, 0, (void **)&baseAZ, 0);

        // store old accels, calculate positions, handle reflective boundaries and zero forces before lj
        for (int64_t i = 0; i < count; ++i)
        {
            baseAXOld[i] = baseAX[i];
            baseAYOld[i] = baseAY[i];
            baseAZOld[i] = baseAZ[i];
            baseX[i] += baseVX[i] * DT + baseAX[i] * (0.5 * DT * DT);
            baseY[i] += baseVY[i] * DT + baseAY[i] * (0.5 * DT * DT);
            baseZ[i] += baseVZ[i] * DT + baseAZ[i] * (0.5 * DT * DT);
            apply_reflective_bcs_3d(baseX, baseY, baseZ, baseVX, baseVY, baseVZ, i);
            baseAX[i] = 0.0;
            baseAY[i] = 0.0;
            baseAZ[i] = 0.0;
        }

        // distribute relevant data for acceleration calculation to all tasks
        laik_switchto_partitioning(data_x, particle_space_partitioning_all, LAIK_DF_Preserve, LAIK_RO_None);
        laik_switchto_partitioning(data_y, particle_space_partitioning_all, LAIK_DF_Preserve, LAIK_RO_None);
        laik_switchto_partitioning(data_z, particle_space_partitioning_all, LAIK_DF_Preserve, LAIK_RO_None);
        laik_switchto_partitioning(data_ax, particle_space_partitioning_all, LAIK_DF_Preserve, LAIK_RO_None);
        laik_switchto_partitioning(data_ay, particle_space_partitioning_all, LAIK_DF_Preserve, LAIK_RO_None);
        laik_switchto_partitioning(data_az, particle_space_partitioning_all, LAIK_DF_Preserve, LAIK_RO_None);

        // master (re-)initializes cell list
        laik_switchto_partitioning(data_head_w, cell_partitioning_master, LAIK_DF_None, LAIK_RO_None);
        laik_switchto_partitioning(data_head_r, cell_partitioning_master, LAIK_DF_None, LAIK_RO_None);
        laik_switchto_partitioning(data_next, particle_space_partitioning_master, LAIK_DF_None, LAIK_RO_None);
        if (myid == 0)
        {
            uint64_t zsize, zstride, ysize, ystride, xsize;
            // get mapping info for heads (3D)
            laik_get_map_3d(data_head_w, 0, (void **)&baseHeadW, &zsize, &zstride, &ysize, &ystride, &xsize);
            laik_get_map_3d(data_head_r, 0, (void **)&baseHeadR, 0, 0, 0, 0, 0); // size and stride same
            laik_get_map_1d(data_next, 0, (void **)&baseNext, 0);
            laik_get_map_1d(data_x, 0, (void **)&baseX, 0);
            laik_get_map_1d(data_y, 0, (void **)&baseY, 0);
            laik_get_map_1d(data_z, 0, (void **)&baseZ, 0);
            assert(xsize == ncells_x && ysize == ncells_y && zsize == ncells_z);

            // reset cell list by setting the head (first particle) of each cell to -1
            for (int64_t kz = 0; kz < (int64_t)zsize; ++kz)
            {
                for (int64_t jy = 0; jy < (int64_t)ysize; ++jy)
                {
                    for (int64_t ix = 0; ix < (int64_t)xsize; ++ix)
                    {
                        int64_t c = ix + jy * ystride + kz * zstride;
                        baseHeadW[c] = -1;
                        baseHeadR[c] = -1;
                    }
                }
            }

            // then, for every particle, push its index to the front of its corresponding cell
            for (int64_t p = 0; p < nparticles; ++p)
            {
                // push p to front of cell
                int64_t c = cindex3d(baseX[p], baseY[p], baseZ[p], xsize, ysize, zsize);
                baseNext[p] = baseHeadW[c];
                baseHeadW[c] = p;
                baseHeadR[c] = p;
            }
        }

        // partition initialized cell list across all tasks
        laik_switchto_partitioning(data_head_w, cell_partitioning_w, LAIK_DF_Preserve, LAIK_DF_None);
        laik_switchto_partitioning(data_head_r, cell_partitioning_r, LAIK_DF_Preserve, LAIK_DF_None);
        laik_switchto_partitioning(data_next, particle_space_partitioning_all, LAIK_DF_Preserve, LAIK_DF_None);

        // all   : x,y,z,ax,ay,az
        // bisect: cell head W/R (halo)
        laik_get_map_1d(data_x, 0, (void **)&baseX, 0);
        laik_get_map_1d(data_y, 0, (void **)&baseY, 0);
        laik_get_map_1d(data_z, 0, (void **)&baseZ, 0);
        laik_get_map_1d(data_ax, 0, (void **)&baseAX, 0);
        laik_get_map_1d(data_ay, 0, (void **)&baseAY, 0);
        laik_get_map_1d(data_az, 0, (void **)&baseAZ, 0);
        laik_get_map_1d(data_next, 0, (void **)&baseNext, 0);

        uint64_t zsizeW, zstrideW, ysizeW, ystrideW, xsizeW;
        uint64_t zsizeR, zstrideR, ysizeR, ystrideR, xsizeR;
        int64_t fromXW, toXW, fromYW, toYW, fromZW, toZW;
        int64_t fromXR, toXR, fromYR, toYR, fromZR, toZR;
        laik_get_map_3d(data_head_w, 0, (void **)&baseHeadW, &zsizeW, &zstrideW, &ysizeW, &ystrideW, &xsizeW);
        laik_get_map_3d(data_head_r, 0, (void **)&baseHeadR, &zsizeR, &zstrideR, &ysizeR, &ystrideR, &xsizeR);
        laik_my_range_3d(cell_partitioning_w, 0, &fromXW, &toXW, &fromYW, &toYW, &fromZW, &toZW);
        laik_my_range_3d(cell_partitioning_r, 0, &fromXR, &toXR, &fromYR, &toYR, &fromZR, &toZR);

        ////////////////
        // force calc //
        ////////////////

        double pot = 0.0;
        // for all owned cells c... (write part.)
        for (int64_t cz = 0; cz < (int64_t)zsizeW; ++cz) // note: we iterate over mapping sizes; ordering preserved from 2D -> 3D usage
        {
            for (int64_t cy = 0; cy < (int64_t)ysizeW; ++cy)
            {
                for (int64_t cx = 0; cx < (int64_t)xsizeW; ++cx)
                {
                    // compute linear index according to mapping's lexicographic layout:
                    int64_t c = cx + cy * ystrideW + cz * zstrideW;

                    // for all particles p in c (globally indexed)...
                    for (int p = baseHeadW[c]; p != -1; p = baseNext[p])
                    {
                        int64_t neighbors[27];
                        int64_t ncount = neighbors_in_read_3d(c, xsizeW, ysizeW, zsizeW,
                                                              fromXW, fromYW, fromZW,
                                                              xsizeR, ysizeR, zsizeR,
                                                              fromXR, fromYR, fromZR,
                                                              neighbors, 27);
                        assert(ncount != -1);

                        // for all neighbor cells nc... (read part.)
                        for (int64_t nci = 0; nci < ncount; ++nci)
                        {
                            int64_t nc = neighbors[nci];

                            // for all particles q in nc (also globally indexed)...
                            for (int q = baseHeadR[nc]; q != -1; q = baseNext[q])
                            {
                                // avoid counting double (n3l)
                                if (q <= p)
                                    continue;

                                // do the formula
                                double dx = baseX[p] - baseX[q];
                                double dy = baseY[p] - baseY[q];
                                double dz = baseZ[p] - baseZ[q];
                                double r2 = dx * dx + dy * dy + dz * dz;

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
                                    double fz = factor * dz;

                                    baseAX[p] += fx / MASS;
                                    baseAY[p] += fy / MASS;
                                    baseAZ[p] += fz / MASS;
                                    baseAX[q] -= fx / MASS;
                                    baseAY[q] -= fy / MASS;
                                    baseAZ[q] -= fz / MASS;

                                    double vp = 4.0 * EPSILON * (sor12 - sor6);
                                    pot += vp;
                                }
                            }
                        }
                    }
                }
            }
        }

        // aggregate forces for velocity calc
        laik_switchto_partitioning(data_x, particle_space_partitioning_master, LAIK_DF_Preserve, LAIK_RO_Max); // doesn't really do anything, just so ALL -> BLOCK in the next step will work
        laik_switchto_partitioning(data_y, particle_space_partitioning_master, LAIK_DF_Preserve, LAIK_RO_Max);
        laik_switchto_partitioning(data_z, particle_space_partitioning_master, LAIK_DF_Preserve, LAIK_RO_Max);
        laik_switchto_partitioning(data_ax, particle_space_partitioning, LAIK_DF_Preserve, LAIK_RO_Sum);
        laik_switchto_partitioning(data_ay, particle_space_partitioning, LAIK_DF_Preserve, LAIK_RO_Sum);
        laik_switchto_partitioning(data_az, particle_space_partitioning, LAIK_DF_Preserve, LAIK_RO_Sum);

        laik_get_map_1d(data_ax, 0, (void **)&baseAX, &count); // count probably not needed here
        laik_get_map_1d(data_ay, 0, (void **)&baseAY, 0);
        laik_get_map_1d(data_az, 0, (void **)&baseAZ, 0);

        // velocities (over particles)
        for (int64_t i = 0; i < count; ++i)
        {
            baseVX[i] += 0.5 * (baseAXOld[i] + baseAX[i]) * DT;
            baseVY[i] += 0.5 * (baseAYOld[i] + baseAY[i]) * DT;
            baseVZ[i] += 0.5 * (baseAZOld[i] + baseAZ[i]) * DT;
        }

        t += DT;

        // follow intermediate state via kinetic energy (over particles)
        if ((step % print_every) == 0)
        {
            double ke = 0.0;
            for (int i = 0; i < count; ++i)
                ke += 0.5 * MASS * (baseVX[i] * baseVX[i] + baseVY[i] * baseVY[i] + baseVZ[i] * baseVZ[i]);
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
    }

    if (myid == 0)
        printf("Done.\n");
    laik_finalize(inst);
    return 0;
}
