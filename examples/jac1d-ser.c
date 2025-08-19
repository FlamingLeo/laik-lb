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
 * 1d Jacobi example (Serial)
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/time.h>

// fixed boundary values
double loValue = -5.0, hiValue = 10.0;

double wtime()
{
    struct timeval tv;
    gettimeofday(&tv, 0);

    return tv.tv_sec+1e-6*tv.tv_usec;
}

int main(int argc, char* argv[])
{
    int ksize = 0;
    int maxiter = 0;

    // emulate LAIK log behavior
    bool stats = (getenv("LAIK_LOG") == 0) ? false : true;

    if (argc > 1) ksize = atoi(argv[1]);
    if (argc > 2) maxiter = atoi(argv[2]);

    if (ksize == 0) ksize = 10; // 10 * 1'000 = 10'000 entries
    if (maxiter == 0) maxiter = 50;

    printf("%d k cells (mem %.1f MB), running %d iterations\n", ksize, .016 * ksize, maxiter);

    double *baseR, *baseW;
    
    // memory layout is always the same for reading and writing
    uint64_t size = (uint64_t) ksize * 1000;
    double* data1 = malloc(sizeof(double) * size);
    double* data2 = malloc(sizeof(double) * size);

    // start with writing (= initialization) data1
    baseW = data1;
    baseR = data2;

    // arbitrary non-zero values based on (global) indexes to detect bugs
    for(uint64_t i = 1; i < size - 1; i++)
        baseW[i] = (double) (i & 6);

    // set fixed boundary values
    baseW[0]        = loValue;
    baseW[size - 1] = hiValue;

    if (stats)
        printf("Init done\n");

    // for statistics (with LAIK_LOG)
    double t, t1 = wtime(), t2 = t1;
    int last_iter = 0;
    int res_iters = 0; // iterations done with residuum calculation

    int iter = 0;
    for(; iter < maxiter; iter++) {

        // switch roles: data written before now is read
        if (baseR == data1) { baseR = data2; baseW = data1; }
        else                { baseR = data1; baseW = data2; }

        // write boundary values (not needed, just same as in LAIK version)
        baseW[0]        = loValue;
        baseW[size - 1] = hiValue;

        ///////////////
        // do jacobi //
        ///////////////

        // check for residuum every 10 iterations (3 Flops more per update)
        if ((iter % 10) == 0) {
            double newValue, diff, res;
            res = 0.0;
            for(int64_t i = 1; i < size - 1; i++) {
                newValue = 0.5 * (baseR[i-1] + baseR[i+1]);
                diff = baseR[i] - newValue;
                res += diff * diff;
                baseW[i] = newValue;
            }
            res_iters++;

            if ((iter > 0) && stats) {
                t = wtime();
                // current iteration already done
                int diter = (iter + 1) - last_iter;
                double dt = t - t2;
                double gUpdates = 0.000000001 * size; // per iteration
                printf("For %d iters: %.3fs, %.3f GF/s, %.3f GB/s\n",
                       diter, dt,
                       // 2 Flops per update in reg iters, with res 5 (once)
                       gUpdates * (5 + 2 * (diter-1)) / dt,
                       // per update 16 bytes read + 8 byte written
                       gUpdates * diter * 24 / dt);
                last_iter = iter + 1;
                t2 = t;
            }

            printf("Residuum after %2d iters: %f\n", iter+1, res);

            if (res < .001) break;
        }
        else {
            for(int64_t i = 1; i < size - 1; i++) {
                baseW[i] = 0.5 * (baseR[i-1] + baseR[i+1]);
            }
        }

    }

    // statistics for all iterations and reductions
    // using work load in all tasks
    if (stats) {
        t = wtime();
        int diter = iter;
        double dt = t - t1;
        double gUpdates = 0.000000001 * size; // per iteration
        printf("For %d iters: %.3fs, %.3f GF/s, %.3f GB/s",
               diter, dt,
               // 2 Flops per update in reg iters, with res 5
               gUpdates * (5 * res_iters + 2 * (diter - res_iters)) / dt,
               // per update 16 bytes read + 8 byte written
               gUpdates * diter * 24 / dt);
    }

    // for check at end: sum up all just written values
    double sum = 0.0;
    for(uint64_t i = 0; i < size; i++)
            sum += baseW[i];

    printf("Global value sum after %d iterations: %f\n",
           iter, sum);

    free(data1);
    free(data2);
    return 0;
}
