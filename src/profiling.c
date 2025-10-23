/*
 * This file is part of the LAIK library.
 * Copyright (c) 2017 Dai Yang, I10/TUM
 *
 * LAIK is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 3 or later.
 *
 * LAIK is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "laik-internal.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdarg.h>
#include <string.h>

#if 0
#define EARLY_RETURN() return
#else
#define EARLY_RETURN() (void)0
#endif

/**
 * Application controlled profiling
 *
 * Stop compute/LAIK times, and optionally write them
 * as one record (= one line) into a CSV file when requested.
 *
 * Currently, LAIK is implemented in such a way that we
 * expect most time to be spent within "laik_switchto".
 * This covers:
 * - triggering a partitioner run if not yet done for
 *   the partitioning we will switch to
 * - calculating the transition between old and new
 *   partitioning and data flow
 * - executing the transition with the help of a backend
 * If in the future, there are other relevant time spans
 * required by LAIK, instrumentation must be updated.
 *
 * We collect two LAIK-related time spans:
 * - how much time is spent in "laik_switchto"?
 *   Approximately, this should be the time spent in LAIK.
 *   Thus, it is called "LAIK total time"
 * - how much time is spent within the backend?
 *   This is part of LAIK total time, used for executing
 *   transitions and synchronizing the KV-store
 *
 * In addition, the application can specify time spans as
 * "user time", by surrounding code to measure with calls
 * laik_profile_user_start/laik_profile_user_stop.
 *
 * Functions are provided to get measured times.
 * In output-to-file mode, times are written out when
 * requested.
 *
 * API DISCUSSION:
 * - API suggests that we can profile per LAIK instance, but
 *   profiling can be active only for one instance?!
 * - ensure user time to be mutual exclusive to LAIK times
 * - how to make this usable for automatic load balancing?
 *   (connect absolute times (!) with a partitioning to modify)
 * - global user time instead of per-LAIK-instance user times
 * - control this from outside (environment variables)
 * - keep it usable also for production mode (too much
 *   instrumentation destroys measurements anyway, so
 *   better keep overhead low)
 * - enable automatic measurement output on iteration/phase change
 * - PAPI counters
 * - combine this with sampling to become more precise
 *   without higher overhead
*/

static Laik_Instance* laik_profinst = 0;
extern char* __progname;

// helper: write json output and escape whitespace characters
static void write_json_string(FILE *out, const char *s)
{
    fputc('"', out);
    if (!s) { fputc('"', out); return; }
    for (const unsigned char *p = (const unsigned char*)s; *p; ++p) {
        unsigned char c = *p;
        switch (c) {
            case '"':  fputs("\\\"", out); break;
            case '\\': fputs("\\\\", out); break;
            case '\b': fputs("\\b", out); break;
            case '\f': fputs("\\f", out); break;
            case '\n': fputs("\\n", out); break;
            case '\r': fputs("\\r", out); break;
            case '\t': fputs("\\t", out); break;
            default:
                if (c < 0x20) {
                    // control characters
                    fprintf(out, "\\u%04x", (int)c);
                } else {
                    fputc(c, out);
                }
        }
    }
    fputc('"', out);
}

// helper: free svg visualization state event list
static void free_event_list(Laik_Instance* i)
{
    Laik_Profiling_EventList *cur = i->profiling->head;
    while (cur) {
        Laik_Profiling_EventList *next = cur->next;
        free(cur->ev.name);
        free(cur);
        cur = next;
    }
}

// called by laik_init
Laik_Profiling_Controller* laik_init_profiling(void)
{
    Laik_Profiling_Controller* ctrl = (Laik_Profiling_Controller*) 
            calloc(1, sizeof(Laik_Profiling_Controller));

    return ctrl;
}

// time measurement functionality
double laik_realtime(){
    struct timeval tv;
    gettimeofday(&tv, 0);

    return tv.tv_sec+1e-6*tv.tv_usec;
}

double laik_cputime(){
    clock_t clk = clock();
    return (double)clk/CLOCKS_PER_SEC;
}

double laik_wtime()
{
  return laik_realtime();
}

// called by laik_finalize
void laik_free_profiling(Laik_Instance* i)
{
    free_event_list(i);
    if(i->profiling->profile_file) {
        fclose(i->profiling->profile_file);
    }
    free(i->profiling);
}

// start profiling measurement for given instance
// FIXME: why not globally enable/disable profiling?
void laik_enable_profiling(Laik_Instance* i)
{
    if (laik_profinst) {
        if (laik_profinst == i) return;
        free_event_list(laik_profinst);
        laik_profinst->profiling->do_profiling = false;
    }
    laik_profinst = i;
    if (!i) return;

    i->profiling->do_profiling = true;
    i->profiling->time_backend = 0.0;
    i->profiling->time_total = 0.0;
    i->profiling->time_user = 0.0;
    i->profiling->head = NULL;
    i->profiling->depth = 0;
}

// reset measured time spans and invalidate previous svg visualization event list
void laik_reset_profiling(Laik_Instance* i)
{
    if (laik_profinst) {
        if (laik_profinst == i) {
            if (i->profiling->do_profiling) {
                i->profiling->do_profiling = true;
                i->profiling->time_backend = 0.0;
                i->profiling->time_total = 0.0;
                i->profiling->time_user = 0.0;
                free_event_list(i);
                i->profiling->head = NULL;
                i->profiling->depth = 0;
            }
        }
    }
}

// start user-time measurement
void laik_profile_user_start(Laik_Instance* i)
{
    if (laik_profinst) {
        if (laik_profinst == i) {
            if (i->profiling->do_profiling) {
                i->profiling->timer_user = laik_wtime();
                i->profiling->user_timer_active = 1;
            }
        }
    }
}

// stop user-time measurement
void laik_profile_user_stop(Laik_Instance* i)
{
    if (laik_profinst) {
        if (laik_profinst == i) {
            if (i->profiling->do_profiling) {
                if (i->profiling->user_timer_active) {
                    i->profiling->time_user = laik_wtime() - i->profiling->timer_user;
                    i->profiling->timer_user = 0.0;
                    i->profiling->user_timer_active = 0;
                }
            }
        }
    }
}

// enable output-to-file mode for use of laik_writeout_profile()
void laik_enable_profiling_file(Laik_Instance* i, const char* filename)
{
    if (laik_profinst) {
        if (laik_profinst == i) return;
        laik_profinst->profiling->do_profiling = false;
    }

    laik_profinst = i;
    if (!i) return;

    i->profiling->do_profiling = true;
    i->profiling->time_backend = 0.0;
    i->profiling->time_total = 0.0;
    snprintf(i->profiling->filename, MAX_FILENAME_LENGTH, "t%s.%s", i->guid, filename);
    
    i->profiling->profile_file = fopen(filename, "a+");
    if (i->profiling->profile_file == NULL) {
        laik_log(LAIK_LL_Error, "Unable to start file based profiling");
    }

    fprintf((FILE*)i->profiling->profile_file, "======MEASUREMENT START AT: %lu======\n", 
            (unsigned long) time(NULL));

    fprintf((FILE*)i->profiling->profile_file, "======Application %s======\n", 
            __progname);

}

// get LAIK total time for LAIK instance for which profiling is enabled
double laik_get_total_time()
{
    if (!laik_profinst) return 0.0;

    return laik_profinst->profiling->time_total;
}

// get LAIK backend time for LAIK instance for which profiling is enabled
double laik_get_backend_time()
{
    if (!laik_profinst) return 0.0;

    return laik_profinst->profiling->time_backend;
}

// for output-to-file mode, write out meassured times
// This is done for the LAIK instance which currently is enabled.
// FIXME: why not reset timers? We never want same time span to appear in
//        multiple lines of the output file?!
void laik_writeout_profile()
{
    if (!laik_profinst) return;
    if (!laik_profinst->profiling->profile_file) return;
    //backend-id, phase, iteration, time_total, time_ackend, user_time
    fprintf( (FILE*)laik_profinst->profiling->profile_file,
             "%s, %d, %d, %f, %f, %f\n",
             laik_profinst->guid,
             laik_profinst->control->cur_phase,
             laik_profinst->control->cur_iteration,
             laik_profinst->profiling->time_total,
             laik_profinst->profiling->time_backend,
             laik_profinst->profiling->time_user
            );
}

// disable output-to-file mode, eventually closing yet open file before
void laik_close_profiling_file(Laik_Instance* i)
{
    if (i->profiling->profile_file != NULL) {
        if(!i->profiling->head) // assume that if there is no event list, we probably wrote a text file, not a json for svg viz.
                                // can be made more explicit using a flag
            fprintf((FILE*)i->profiling->profile_file, "======MEASUREMENT END AT: %lu======\n",
                    (unsigned long) time(NULL));
        fclose(i->profiling->profile_file);
        i->profiling->profile_file = NULL;
    }
}

// print arbitrary text to file in output-to-file mode
void laik_profile_printf(const char* msg, ...)
{
    if (laik_profinst->profiling->profile_file) {
        va_list args;
        va_start(args, msg);
        vfprintf((FILE*)laik_profinst->profiling->profile_file, msg, args);
        va_end(args);
    }
}

///////////////////////
// svg visualization //
///////////////////////

// enable output-to-file mode for svg visualization
void laik_svg_enable_profiling(Laik_Instance* i, const char* filename)
{
    EARLY_RETURN();

    if (laik_profinst) {
        if (laik_profinst == i) return;
        free_event_list(laik_profinst);
        laik_profinst->profiling->do_profiling = false;
    }
    laik_profinst = i;
    if (!i) return;

    i->profiling->do_profiling = true;
    i->profiling->time_backend = 0.0;
    i->profiling->time_total = 0.0;
    i->profiling->time_user = 0.0;
    i->profiling->head = NULL;
    i->profiling->depth = 0;

    i->profiling->profile_file = fopen(filename, "w");
    if (i->profiling->profile_file == NULL) {
        laik_log(LAIK_LL_Error, "Unable to start svg-visualization profiling");
    }
}

// enter new function
// call this (ideally) right before / at the start of the function to profile
void laik_svg_profiler_enter(Laik_Instance* i, const char *func_name)
{
    EARLY_RETURN();

    if (laik_profinst) {
        if (laik_profinst == i) {
            if (i->profiling->do_profiling) {
                // push one level deeper
                i->profiling->depth++;

                // we record the start time in a temporary stack by 
                // allocating a node and storing only start+name+depth for now
                Laik_Profiling_EventList *node = malloc(sizeof *node);
                node->ev.name                  = strdup(func_name);
                node->ev.start                 = laik_wtime();
                node->ev.end                   = -1.0;
                node->ev.depth                 = i->profiling->depth;
                node->next                     = i->profiling->head;
                i->profiling->head             = node;
            }
        }
    }
}

// exit function
// call this (ideally) right after / at the end of the function to profile
void laik_svg_profiler_exit(Laik_Instance* i, const char *func_name)
{
    EARLY_RETURN();

    if (laik_profinst) {
        if (laik_profinst == i) {
            if (i->profiling->do_profiling) {
                double finish = laik_wtime();

                // find the most recent incomplete event matching this name + depth combo
                for (Laik_Profiling_EventList *n = i->profiling->head; n; n = n->next) {
                    if (n->ev.end < 0 
                        && n->ev.depth == i->profiling->depth 
                        && strcmp(n->ev.name, func_name) == 0) {
                        n->ev.end = finish;
                        break;
                    }
                }

                // pop one level
                i->profiling->depth--;
            }
        }
    }
}

// push a marker event (zero-length) to the head list
void laik_svg_profiler_mark_iteration(Laik_Instance* i, int iter)
{
    if (!i) return;
    if (!i->profiling->do_profiling) return;

    Laik_Profiling_EventList *node = malloc(sizeof *node);
    if (!node) return;

    char buf[64];
    int n = snprintf(buf, sizeof(buf), "__iter__:%d", iter);
    if (n < 0) {
        free(node);
        return;
    }

    node->ev.name = strdup(buf);
    if (!node->ev.name) {
        free(node);
        return;
    }
    node->ev.start = laik_wtime();
    node->ev.end   = node->ev.start; // zero-length marker
    node->ev.depth = i->profiling->depth;
    node->next     = i->profiling->head;
    i->profiling->head = node;
}

// export current profiler event state to json-formatted file (or do nothing if profiling is disabled)
void laik_svg_profiler_export_json(Laik_Instance* i)
{
    EARLY_RETURN();

    if(!(i->profiling->do_profiling)) return;

    FILE* out = (FILE*) i->profiling->profile_file;
    if (!out) return;

    // collect closed events into a dynamic array of pointers
    size_t cap = 64;
    size_t len = 0;
    Laik_Profiling_EventList **arr = malloc(sizeof(*arr) * cap);
    if (!arr) return;

    for (Laik_Profiling_EventList *n = i->profiling->head; n; n = n->next) {
        if (n->ev.end < 0) continue; // skip incomplete
        if (len >= cap) {
            cap *= 2;
            Laik_Profiling_EventList **tmp = realloc(arr, sizeof(*arr) * cap);
            if (!tmp) break; // abort collecting
            arr = tmp;
        }
        arr[len++] = n;
    }

    // reverse array (could probably also update the python script to read json bottom-to-top, but this seems easier and isn't too slow)
    for (size_t i = 0; i < len / 2; ++i) {
        Laik_Profiling_EventList *tmp = arr[i];
        arr[i] = arr[len - 1 - i];
        arr[len - 1 - i] = tmp;
    }

    // write JSON array
    fprintf(out, "[\n");
    for (size_t idx = 0; idx < len; ++idx) {
        Laik_Profiling_EventList *n = arr[idx];
        fprintf(out, "  { ");
        fprintf(out, "\"name\": ");
        write_json_string(out, n->ev.name);
        fprintf(out, ", \"start\": %.6f, \"end\": %.6f, \"track\": %d ",
                n->ev.start, n->ev.end, laik_myid(i->world));
        fprintf(out, "}");
        if (idx + 1 < len) fprintf(out, ",\n");
    }
    fprintf(out, "\n]\n");

    // free temp array
    free(arr);
}
