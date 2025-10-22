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

/**
 * 3D Molecular Dynamics example for Load Balancing (sequential).
 * (Lennard-Jones force; collision of two particle cuboids, psuedo-reflective boundaries)
 *
 * Based on the MD practical course code (WiSe 2024/25) (not a 1:1 port!).
 * https://github.com/FlamingLeo/MolSim (w3t2, modified for 3D)
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <sys/time.h>

// simulation parameters
#define START_TIME 0.0
#define END_TIME 1.0
#define DT 0.0005

#define DOMAIN_X 128.0
#define DOMAIN_Y 128.0
#define DOMAIN_Z 128.0

#define CUTOFF 4.0
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
#define A_SIZE_X 70
#define A_SIZE_Y 20
#define A_SIZE_Z 10
#define A_VX 0.0
#define A_VY 0.0
#define A_VZ 0.0

// cuboid B, moving towards cuboid A, smaller, top
#define B_POS_X 50.0
#define B_POS_Y 60.0
#define B_POS_Z 10.0
#define B_SIZE_X 20
#define B_SIZE_Y 20
#define B_SIZE_Z 10
#define B_VX 0.0
#define B_VY -10.0
#define B_VZ 0.0

// particle metadata (num of particles; pos, vel, acc)
int npart = 0;
double *x, *y, *z, *vx, *vy, *vz, *ax, *ay, *az;
double *ax_old, *ay_old, *az_old;
int *next; // next[p] = index of the next particle after p in the same cell or -1 (if there is no next); length: npart

// linked cell structure
int ncell_x, ncell_y, ncell_z, ncell;
int *head; // head[cell] = index of first particle in that cell or -1; length: ncell

// lj constants
double sigma6, sigma12;
double cutoff2;

// utility
static inline void *safe_malloc(size_t n)
{
    void *p = malloc(n);
    if (!p)
    {
        fprintf(stderr, "malloc\n");
        exit(EXIT_FAILURE);
    }
    return p;
}

static inline double wtime()
{
    struct timeval tv;
    gettimeofday(&tv, 0);

    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

// initialize particles from two cuboids
static inline void init_particles()
{
    int countA = A_SIZE_X * A_SIZE_Y * A_SIZE_Z;
    int countB = B_SIZE_X * B_SIZE_Y * B_SIZE_Z;
    npart = countA + countB;

    x = (double *)safe_malloc(sizeof(double) * npart);
    y = (double *)safe_malloc(sizeof(double) * npart);
    z = (double *)safe_malloc(sizeof(double) * npart);
    vx = (double *)safe_malloc(sizeof(double) * npart);
    vy = (double *)safe_malloc(sizeof(double) * npart);
    vz = (double *)safe_malloc(sizeof(double) * npart);
    ax = (double *)safe_malloc(sizeof(double) * npart);
    ay = (double *)safe_malloc(sizeof(double) * npart);
    az = (double *)safe_malloc(sizeof(double) * npart);
    ax_old = (double *)safe_malloc(sizeof(double) * npart);
    ay_old = (double *)safe_malloc(sizeof(double) * npart);
    az_old = (double *)safe_malloc(sizeof(double) * npart);
    next = (int *)safe_malloc(sizeof(int) * npart);

    int idx = 0;
    // cuboid A
    for (int k = 0; k < A_SIZE_Z; ++k)
    {
        for (int j = 0; j < A_SIZE_Y; ++j)
        {
            for (int i = 0; i < A_SIZE_X; ++i)
            {
                x[idx] = A_POS_X + i * DISTANCE;
                y[idx] = A_POS_Y + j * DISTANCE;
                z[idx] = A_POS_Z + k * DISTANCE;
                vx[idx] = A_VX;
                vy[idx] = A_VY;
                vz[idx] = A_VZ;
                ax[idx] = ay[idx] = az[idx] = 0.0;
                next[idx] = -1;
                ++idx;
            }
        }
    }
    // cuboid B
    for (int k = 0; k < B_SIZE_Z; ++k)
    {
        for (int j = 0; j < B_SIZE_Y; ++j)
        {
            for (int i = 0; i < B_SIZE_X; ++i)
            {
                x[idx] = B_POS_X + i * DISTANCE;
                y[idx] = B_POS_Y + j * DISTANCE;
                z[idx] = B_POS_Z + k * DISTANCE;
                vx[idx] = B_VX;
                vy[idx] = B_VY;
                vz[idx] = B_VZ;
                ax[idx] = ay[idx] = az[idx] = 0.0;
                next[idx] = -1;
                ++idx;
            }
        }
    }
    assert(idx == npart);
}

// setup linked-cell grid
// IMPORTANT: we assume domain sizes are integer multiples of CUTOFF
//            if not, some nasty things might happen
static inline void init_cells()
{
    assert(fabs(fmod(DOMAIN_X, CUTOFF)) < 1e-12);
    assert(fabs(fmod(DOMAIN_Y, CUTOFF)) < 1e-12);
    assert(fabs(fmod(DOMAIN_Z, CUTOFF)) < 1e-12);

    ncell_x = (int)(DOMAIN_X / CUTOFF);
    if (ncell_x < 1)
        ncell_x = 1;
    ncell_y = (int)(DOMAIN_Y / CUTOFF);
    if (ncell_y < 1)
        ncell_y = 1;
    ncell_z = (int)(DOMAIN_Z / CUTOFF);
    if (ncell_z < 1)
        ncell_z = 1;
    ncell = ncell_x * ncell_y * ncell_z;
    head = (int *)safe_malloc(sizeof(int) * ncell);
}

// compute cell index from position (standard 3D to 1D mapping rule, (xyz) -> x+y*xsize+z*xsize*ysize)
static inline int cell_index_of(double px, double py, double pz)
{
    int ix = (int)(px / CUTOFF);
    int iy = (int)(py / CUTOFF);
    int iz = (int)(pz / CUTOFF);
    if (ix < 0)
        ix = 0;
    if (iy < 0)
        iy = 0;
    if (iz < 0)
        iz = 0;
    if (ix >= ncell_x)
        ix = ncell_x - 1;
    if (iy >= ncell_y)
        iy = ncell_y - 1;
    if (iz >= ncell_z)
        iz = ncell_z - 1;
    return ix + iy * ncell_x + iz * (ncell_x * ncell_y);
}

// (re-)build cell list from particle position
static inline void build_cell_list()
{
    for (int c = 0; c < ncell; ++c)
        head[c] = -1;
    for (int p = 0; p < npart; ++p)
    {
        // push p to front of cell
        int c = cell_index_of(x[p], y[p], z[p]);
        next[p] = head[c];
        head[c] = p;
    }
}

// compute forces with linked cells (3D), store accelerations in ax, ay, az and return potential energy
// see: https://en.wikipedia.org/wiki/Lennard-Jones_potential
static inline double compute_forces_cells()
{
    double potential = 0.0;

    // for all cells c...
    for (int cz = 0; cz < ncell_z; ++cz)
    {
        for (int cy = 0; cy < ncell_y; ++cy)
        {
            for (int cx = 0; cx < ncell_x; ++cx)
            {
                int c = cx + cy * ncell_x + cz * (ncell_x * ncell_y);
                // for all particles p in cell c...
                for (int p = head[c]; p != -1; p = next[p])
                {
                    // for all neighbor cells nc (including c)...
                    for (int dzc = -1; dzc <= 1; ++dzc)
                    {
                        int nz = cz + dzc;
                        if (nz < 0 || nz >= ncell_z)
                            continue;
                        for (int dyc = -1; dyc <= 1; ++dyc)
                        {
                            int ny = cy + dyc;
                            if (ny < 0 || ny >= ncell_y)
                                continue;
                            for (int dxc = -1; dxc <= 1; ++dxc)
                            {
                                int nx = cx + dxc;
                                if (nx < 0 || nx >= ncell_x)
                                    continue;
                                int nc = nx + ny * ncell_x + nz * (ncell_x * ncell_y);
                                // for all particles q in neighbor cell nc...
                                for (int q = head[nc]; q != -1; q = next[q])
                                {
                                    if (q <= p)
                                        continue; // avoid counting double (n3l)
                                    double dx = x[p] - x[q];
                                    double dy = y[p] - y[q];
                                    double dz = z[p] - z[q];
                                    double r2 = dx * dx + dy * dy + dz * dz;
                                    if (r2 <= 1e-12)
                                        continue; // same particle safety guard
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

                                        ax[p] += fx / MASS;
                                        ay[p] += fy / MASS;
                                        az[p] += fz / MASS;
                                        ax[q] -= fx / MASS;
                                        ay[q] -= fy / MASS;
                                        az[q] -= fz / MASS;

                                        double vp = 4.0 * EPSILON * (sor12 - sor6);
                                        potential += vp;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } // holy mother of nesting
    return potential;
}

// pseudo-reflective boundares (simplified), for when the particle exits the domain
// reflect position and invert velocity component
static inline void apply_reflective_bc(int p)
{
    // flip in x direction
    if (x[p] < 0.0)
    {
        x[p] = -x[p];
        vx[p] = -vx[p];
    }
    else if (x[p] > DOMAIN_X)
    {
        x[p] = 2.0 * DOMAIN_X - x[p];
        vx[p] = -vx[p];
    }

    // flip in y direction
    if (y[p] < 0.0)
    {
        y[p] = -y[p];
        vy[p] = -vy[p];
    }
    else if (y[p] > DOMAIN_Y)
    {
        y[p] = 2.0 * DOMAIN_Y - y[p];
        vy[p] = -vy[p];
    }

    // flip in z direction
    if (z[p] < 0.0)
    {
        z[p] = -z[p];
        vz[p] = -vz[p];
    }
    else if (z[p] > DOMAIN_Z)
    {
        z[p] = 2.0 * DOMAIN_Z - z[p];
        vz[p] = -vz[p];
    }
}

// compute kinetic energy (for verification / validation)
static inline double compute_kinetic()
{
    double ke = 0.0;
    for (int i = 0; i < npart; ++i)
        ke += 0.5 * MASS * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
    return ke;
}

// write VTK legacy ASCII POLYDATA file with points + velocity vectors as point data
// for debugging with ParaView
static inline void write_vtk_snapshot(long step)
{
    char fname[256];
    snprintf(fname, sizeof(fname), "vtk/md3d_particles_%06ld.vtk", step);
    FILE *f = fopen(fname, "w");
    if (!f)
    {
        fprintf(stderr, "Could not open '%s' for writing\n", fname);
        return;
    }

    fprintf(f, "# vtk DataFile Version 3.0\n");
    fprintf(f, "MD particles step %ld\n", step);
    fprintf(f, "ASCII\n");
    fprintf(f, "DATASET POLYDATA\n");

    // points (3D)
    fprintf(f, "POINTS %d float\n", npart);
    for (int i = 0; i < npart; ++i)
    {
        fprintf(f, "%g %g %g\n", x[i], y[i], z[i]);
    }

    // vertices (each point as its own vertex cell)
    fprintf(f, "VERTICES %d %d\n", npart, npart * 2);
    for (int i = 0; i < npart; ++i)
    {
        fprintf(f, "1 %d\n", i);
    }

    // point data (velocities as vectors)
    fprintf(f, "POINT_DATA %d\n", npart);
    fprintf(f, "VECTORS velocity float\n");
    for (int i = 0; i < npart; ++i)
    {
        fprintf(f, "%g %g %g\n", vx[i], vy[i], vz[i]);
    }

    // kinetic energy per particle
    fprintf(f, "SCALARS ke_per_particle float 1\n");
    fprintf(f, "LOOKUP_TABLE default\n");
    for (int i = 0; i < npart; ++i)
    {
        double ke = 0.5 * MASS * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
        fprintf(f, "%g\n", ke);
    }

    fclose(f);
}

//////////
// main //
//////////

int main(int argc, char **argv)
{
    bool output = !(argc > 1 && !strcmp(argv[1], "-o"));

    printf("3D linked-cell Lennard-Jones cuboid collision\n");
    printf("domain: %g x %g x %g, cutoff=%g, dt=%g\n", DOMAIN_X, DOMAIN_Y, DOMAIN_Z, CUTOFF, DT);
    printf("output: %d\n", output);

    // precompute lj constants
    sigma6 = pow(SIGMA, 6);
    sigma12 = sigma6 * sigma6;
    cutoff2 = CUTOFF * CUTOFF;

    // initialize containers
    init_particles();
    init_cells();

    printf("npart = %d\n", npart);
    printf("Cells: %d x %d x %d = %d\n", ncell_x, ncell_y, ncell_z, ncell);

    double pot;

    long nsteps = (long)((END_TIME - START_TIME) / DT + 0.5);
    printf("Running %ld steps (endTime=%g)\n", nsteps, END_TIME);

    // output parameters
    int print_every = 100;
    int write_every = 100;

    // integration loop (X -> F -> V)
    double t = START_TIME;
    double tbefore = wtime();
    for (long step = 0; step < nsteps; ++step)
    {
        // store old accelerations
        for (int i = 0; i < npart; ++i)
        {
            ax_old[i] = ax[i];
            ay_old[i] = ay[i];
            az_old[i] = az[i];
        }

        // calculate positions
        double half_dt2 = 0.5 * DT * DT;
        for (int i = 0; i < npart; ++i)
        {
            x[i] += vx[i] * DT + ax[i] * half_dt2;
            y[i] += vy[i] * DT + ay[i] * half_dt2;
            z[i] += vz[i] * DT + az[i] * half_dt2;
            apply_reflective_bc(i);
        }

        // zero forces before lj
        for (int i = 0; i < npart; ++i)
        {
            ax[i] = 0.0;
            ay[i] = 0.0;
            az[i] = 0.0;
        }

        // rebuild cells, recompute forces
        build_cell_list();
        pot = compute_forces_cells();

        // velocities
        for (int i = 0; i < npart; ++i)
        {
            vx[i] += 0.5 * (ax_old[i] + ax[i]) * DT;
            vy[i] += 0.5 * (ay_old[i] + ay[i]) * DT;
            vz[i] += 0.5 * (az_old[i] + az[i]) * DT;
        }

        t += DT;

        // log intermediate state
        if (output && (step % print_every) == 0)
        {
            double ke = compute_kinetic();
            double total = ke + pot;
            printf("step %ld / %ld, t=%.4f, KE=%.6f, PE=%.6f, E=%.6f\n",
                   step, nsteps, t, ke, pot, total);
        }

        // log state to vtk output
        if (output && (step % write_every) == 0)
        {
            write_vtk_snapshot(step);
        }
    }

    double ke_final = compute_kinetic();
    double total_final = ke_final + pot;
    double tafter = wtime();
    double ttotal = tafter - tbefore;
    printf("Done. Final KE=%.6f PE=%.6f Total=%.6f\nTime taken: %fs\n", ke_final, pot, total_final, ttotal);

    free(x);
    free(y);
    free(z);
    free(vx);
    free(vy);
    free(vz);
    free(ax);
    free(ay);
    free(az);
    free(ax_old);
    free(ay_old);
    free(az_old);
    free(next);
    free(head);

    return 0;
}