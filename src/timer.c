#include "laik-internal.h"

void laik_timer_start(Laik_Timer *t)
{
    t->elapsed = 0.0;
    t->starttime = laik_wtime();
    t->running = true;
}

void laik_timer_pause(Laik_Timer *t)
{
    if (!t->running)
        return;
    t->elapsed += laik_wtime() - t->starttime;
    t->running = false;
}

void laik_timer_resume(Laik_Timer *t)
{
    if (t->running)
        return;
    t->starttime = laik_wtime();
    t->running = true;
}

double laik_timer_stop(Laik_Timer *t)
{
    if (t->running)
    {
        t->elapsed += laik_wtime() - t->starttime;
        t->running = false;
    }
    return t->elapsed;
}
