/**
 * 2D Molecular Dynamics example for Load Balancing (OpenMP, task-based)
 * (Lennard-Jones force; collision of two particle cuboids, pseudo-reflective boundaries)
 *
 * Parallelized with OpenMP for comparison against LAIK (yes, it's apples to oranges, but that's almost the point in a way...)
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>

// simulation parameters
#define START_TIME 0.0
#define END_TIME 5.0
#define DT 0.0005
#define HALF_DT2 (0.5 * (DT) * (DT))

#define DOMAIN_X 256.0
#define DOMAIN_Y 256.0

#define CUTOFF 4.0

// lennard-jones parameters, same for every particle here for simplicity
#define MASS 1.0
#define EPSILON 5.0
#define SIGMA 1.0

// distance between particles in a cuboid
#define DISTANCE 1.1225

// cuboid A, stationary, large, bottom
#define A_POS_X 20.0
#define A_POS_Y 70.0
#define A_SIZE_X 200
#define A_SIZE_Y 50
#define A_VX 0.0
#define A_VY 0.0

// cuboid B, moving towards cuboid A, smaller, top
#define B_POS_X 105.0
#define B_POS_Y 150.0
#define B_SIZE_X 50
#define B_SIZE_Y 50
#define B_VX 0.0
#define B_VY -10.0

// particle metadata (num of particles; pos, vel, acc)
int npart = 0;
double *x, *y, *vx, *vy, *ax, *ay;
double *ax_old, *ay_old;
int *next; // next[p] = index of the next particle after p in the same cell or -1 (if there is no next); length: npart

// linked cell structure
int ncell_x, ncell_y, ncell;
int *head; // head[cell] = index of first particle in that cell or -1; length: ncell

// thread buffers
double *ax_thread = NULL;
double *ay_thread = NULL;
int nthreads_for_alloc = 0;
int grainsize_mult = 4;

// precomputed lj constants
double sigma6;
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
    int countA = A_SIZE_X * A_SIZE_Y;
    int countB = B_SIZE_X * B_SIZE_Y;
    npart = countA + countB;

    x = (double *)safe_malloc(sizeof(double) * npart);
    y = (double *)safe_malloc(sizeof(double) * npart);
    vx = (double *)safe_malloc(sizeof(double) * npart);
    vy = (double *)safe_malloc(sizeof(double) * npart);
    ax = (double *)safe_malloc(sizeof(double) * npart);
    ay = (double *)safe_malloc(sizeof(double) * npart);
    ax_old = (double *)safe_malloc(sizeof(double) * npart);
    ay_old = (double *)safe_malloc(sizeof(double) * npart);
    next = (int *)safe_malloc(sizeof(int) * npart);

    int idx = 0;
    // cuboid A
    for (int j = 0; j < A_SIZE_Y; ++j)
    {
        for (int i = 0; i < A_SIZE_X; ++i)
        {
            x[idx] = A_POS_X + i * DISTANCE;
            y[idx] = A_POS_Y + j * DISTANCE;
            vx[idx] = A_VX;
            vy[idx] = A_VY;
            ax[idx] = ay[idx] = 0.0;
            next[idx] = -1;
            ++idx;
        }
    }
    // cuboid B
    for (int j = 0; j < B_SIZE_Y; ++j)
    {
        for (int i = 0; i < B_SIZE_X; ++i)
        {
            x[idx] = B_POS_X + i * DISTANCE;
            y[idx] = B_POS_Y + j * DISTANCE;
            vx[idx] = B_VX;
            vy[idx] = B_VY;
            ax[idx] = ay[idx] = 0.0;
            next[idx] = -1;
            ++idx;
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

    ncell_x = (int)(DOMAIN_X / CUTOFF);
    if (ncell_x < 1)
        ncell_x = 1;
    ncell_y = (int)(DOMAIN_Y / CUTOFF);
    if (ncell_y < 1)
        ncell_y = 1;
    ncell = ncell_x * ncell_y;
    head = (int *)safe_malloc(sizeof(int) * ncell);
}

// compute cell index from position
static inline int cell_index_of(double px, double py)
{
    int ix = (int)(px / CUTOFF);
    int iy = (int)(py / CUTOFF);
    if (ix < 0)
        ix = 0;
    if (iy < 0)
        iy = 0;
    if (ix >= ncell_x)
        ix = ncell_x - 1;
    if (iy >= ncell_y)
        iy = ncell_y - 1;
    return ix + iy * ncell_x;
}

// (re-)build cell list from particle position
// parallel-safe implementation using atomic capture for pushing into head[]
static inline void build_cell_list()
{
// initialize head[] in parallel
#pragma omp parallel for schedule(static)
    for (int c = 0; c < ncell; ++c)
        head[c] = -1;

// push particles into cells in parallel
#pragma omp parallel for schedule(static)
    for (int p = 0; p < npart; ++p)
    {
        int c = cell_index_of(x[p], y[p]);
        int old;
// capture old = head[c]; head[c] = p; done atomically for that head[c]
#pragma omp atomic capture
        {
            old = head[c];
            head[c] = p;
        }
        next[p] = old;
    }
}

// compute forces with linked cells, store accelerations in ax, ay and return potential energy
static inline double compute_forces_cells()
{
    double potential = 0.0;
    int nt = nthreads_for_alloc ? nthreads_for_alloc : 1;

    // zero per-thread accumulators (parallel)
#pragma omp parallel for schedule(static)
    for (size_t ii = 0; ii < (size_t)nt * npart; ++ii)
    {
        ax_thread[ii] = 0.0;
        ay_thread[ii] = 0.0;
    }

    // grainsize for taskloop
    int grainsize = ncell / (nt * grainsize_mult);
    if (grainsize < 1)
        grainsize = 1;

    // let each task process a contiguous subset of cells
#pragma omp parallel
    {
#pragma omp single nowait
        {
#pragma omp taskloop grainsize(grainsize)
            for (int cc = 0; cc < ncell; ++cc)
            {
                double pot_local = 0.0;
                int tid = omp_get_thread_num();
                double *ax_t = ax_thread + (size_t)tid * npart;
                double *ay_t = ay_thread + (size_t)tid * npart;

                int cx = cc % ncell_x;
                int cy = cc / ncell_x;

                for (int p = head[cc]; p != -1; p = next[p])
                {
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
                            int nc = nx + ny * ncell_x;

                            for (int q = head[nc]; q != -1; q = next[q])
                            {
                                if (q <= p)
                                    continue;

                                double dx = x[p] - x[q];
                                double dy = y[p] - y[q];
                                double r2 = dx * dx + dy * dy;
                                if (r2 <= 1e-12)
                                    continue;
                                if (r2 <= cutoff2)
                                {
                                    double invr2 = 1.0 / r2;
                                    double invr6 = invr2 * invr2 * invr2;
                                    double sor6 = sigma6 * invr6;
                                    double sor12 = sor6 * sor6;

                                    double factor = 24.0 * EPSILON * (2.0 * sor12 - sor6) * invr2;
                                    double fx = factor * dx;
                                    double fy = factor * dy;

                                    double fx_div_m = fx / MASS;
                                    double fy_div_m = fy / MASS;

                                    // accumulate into thread-local arrays
                                    ax_t[p] += fx_div_m;
                                    ay_t[p] += fy_div_m;

                                    ax_t[q] -= fx_div_m;
                                    ay_t[q] -= fy_div_m;

                                    pot_local += 4.0 * EPSILON * (sor12 - sor6);
                                }
                            } // q loop
                        } // dxc
                    } // dyc
                } // p loop

                if (pot_local != 0.0)
                {
#pragma omp atomic
                    potential += pot_local;
                }
            } // taskloop
#pragma omp taskwait
        } // single
    } // parallel

    // reduce per-thread accumulators into global ax/ay
#pragma omp parallel for schedule(static)
    for (int i = 0; i < npart; ++i)
    {
        double sumx = 0.0;
        double sumy = 0.0;
        for (int t = 0; t < nt; ++t)
        {
            sumx += ax_thread[(size_t)t * npart + i];
            sumy += ay_thread[(size_t)t * npart + i];
        }
        ax[i] += sumx;
        ay[i] += sumy;
    }

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
}

// compute kinetic energy (for verification / validation)
static inline double compute_kinetic()
{
    double ke = 0.0;
#pragma omp parallel for reduction(+ : ke) schedule(static)
    for (int i = 0; i < npart; ++i)
        ke += 0.5 * MASS * (vx[i] * vx[i] + vy[i] * vy[i]);
    return ke;
}

//////////
// main //
//////////

int main(int argc, char **argv)
{
    bool output = true;
    for (int arg = 1; arg < argc; ++arg)
    {
        // do general output (disable this for accurate profiling!)
        if (!strcmp(argv[arg], "-o"))
            output = false;

        // set grainsize multiplier
        if (arg + 1 < argc && !strcmp(argv[arg], "-g"))
            grainsize_mult = atoi(argv[++arg]);
    }

    printf("2D linked-cell Lennard-Jones cuboid collision (OpenMP, task-based)\n");
    printf("domain: %g x %g, cutoff=%g, dt=%g\n", DOMAIN_X, DOMAIN_Y, CUTOFF, DT);
    printf("max threads: %d, output: %d, grainsize multiplier: %d\n", omp_get_max_threads(), output, grainsize_mult);

    // precompute lj constants
    sigma6 = pow(SIGMA, 6);
    cutoff2 = CUTOFF * CUTOFF;

    // initialize containers
    init_particles();
    init_cells();

    printf("npart = %d\n", npart);
    printf("Cells: %d x %d = %d\n", ncell_x, ncell_y, ncell);

    // initialize thread buffers
    nthreads_for_alloc = omp_get_max_threads();
    if (nthreads_for_alloc < 1)
        nthreads_for_alloc = 1;
    ax_thread = (double *)safe_malloc(sizeof(double) * (size_t)nthreads_for_alloc * npart);
    ay_thread = (double *)safe_malloc(sizeof(double) * (size_t)nthreads_for_alloc * npart);

    printf("taskloop grainsize: %d\n", ncell / (nthreads_for_alloc * grainsize_mult));

    for (size_t i = 0; i < (size_t)nthreads_for_alloc * npart; ++i)
    {
        ax_thread[i] = 0.0;
        ay_thread[i] = 0.0;
    }

    // potential
    double pot = 0.0;

    long nsteps = (long)((END_TIME - START_TIME) / DT + 0.5);
    printf("Running %ld steps (endTime=%g)\n", nsteps, END_TIME);

    // output parameters
    int print_every = 100;

    // integration loop (X -> F -> V)
    double t = START_TIME;
    double before = wtime();

    for (long step = 0; step < nsteps; ++step)
    {
// store old accelerations
#pragma omp parallel for schedule(static)
        for (int i = 0; i < npart; ++i)
        {
            ax_old[i] = ax[i];
            ay_old[i] = ay[i];
        }

// calculate positions (and apply reflective BC)
#pragma omp parallel for schedule(static)
        for (int i = 0; i < npart; ++i)
        {
            x[i] += vx[i] * DT + ax[i] * HALF_DT2;
            y[i] += vy[i] * DT + ay[i] * HALF_DT2;
            apply_reflective_bc(i);
        }

// zero forces before lj
#pragma omp parallel for schedule(static)
        for (int i = 0; i < npart; ++i)
        {
            ax[i] = 0.0;
            ay[i] = 0.0;
        }

        // rebuild cells, recompute forces
        build_cell_list();
        pot = compute_forces_cells();

// velocities
#pragma omp parallel for schedule(static)
        for (int i = 0; i < npart; ++i)
        {
            vx[i] += 0.5 * (ax_old[i] + ax[i]) * DT;
            vy[i] += 0.5 * (ay_old[i] + ay[i]) * DT;
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
    }

    double ke_final = compute_kinetic();
    double total_final = ke_final + pot;
    double after = wtime();

    printf("Done. Final KE=%.6f PE=%.6f Total=%.6f\nTime taken: %fs\n", ke_final, pot, total_final, after - before);

    free(x);
    free(y);
    free(vx);
    free(vy);
    free(ax);
    free(ay);
    free(ax_old);
    free(ay_old);
    free(next);
    free(head);
    free(ax_thread);
    free(ay_thread);

    return 0;
}
