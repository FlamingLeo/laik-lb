/**
 * Timer struct and functions for timing events.
 *
 * Used for load balancing calculations.
 */
#ifndef LAIK_TIMER_H
#define LAIK_TIMER_H

#include <stdbool.h>

// a timer used for timing workload intervals
typedef struct
{
    double starttime;
    double elapsed;
    bool running;
} Laik_Timer;

// start timer (reset elapsed, register start time)
void laik_timer_start(Laik_Timer *t);

// pause timer (add time taken so far to elapsed time)
// does nothing if the timer hasn't started yet
void laik_timer_pause(Laik_Timer *t);

// resume timer
// does nothing if the timer is already running
void laik_timer_resume(Laik_Timer *t);

// stop timer and obtain elapsed time
double laik_timer_stop(Laik_Timer *t);

#endif