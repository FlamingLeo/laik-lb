#include <laik-internal.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <libgen.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

// note: this assumes the following files and directories exist in this structure:
//   examples/lbviz
//   examples/lbviz/visualize.py
//   examples/lbviz/trace.py
//   examples/lbviz/json
//   scripts/
//   scripts/remove_plots.sh
//
// note²: some functions here use unsafe C stdlib functions (e.g. strlen), be careful! these are mainly meant for debugging anyway...

// to keep my IDE from complaining...
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

static char project_root[PATH_MAX] = {0}; // absolute path to project root
static int project_root_cached = -1;      // -1: not run yet, 0: not found, 1: means found

// helper function to create a directory if it doesn't exist, otherwise does nothing
// returns the mkdir error code for error handling or 0 if it already exists, so nothing needs to be done
static int create_nonexistent_dir(const char *dirname)
{
    struct stat st = {0};
    if (stat(dirname, &st) == -1)
        return mkdir(dirname, 0700);
    return 0;
}

// helper function to initialize project root directory as starting point for all other locations, assuming FHS
// this is done to ensure that the programs can be executed from anywhere inside the project directory (examples, root, scripts, whatever...) and have visualization still work
//
// returns true if project root was found and written to project_root, false otherwise
static bool find_and_write_project_root()
{
    // avoid rerunning program if we already ran it before
    if (project_root_cached != -1)
        return project_root_cached == 1;

    // get current working directory
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof cwd) == NULL)
    {
        project_root_cached = 0;
        return false;
    }

    // work on a mutable copy of cwd
    char cur[PATH_MAX];
    if (strlen(cwd) >= sizeof cur)
    {
        project_root_cached = 0;
        return false;
    }
    strcpy(cur, cwd);

    for (;;)
    {
        // strip trailing slashes but keep root slash
        size_t len = strlen(cur);
        while (len > 1 && cur[len - 1] == '/')
        {
            cur[len - 1] = '\0';
            len--;
        }

        // build candidate path
        char candidate[PATH_MAX];
        int needed = snprintf(candidate, sizeof(candidate), "%s/Guidelines.md", cur);
        if (needed >= 0 && (size_t)needed < sizeof(candidate))
        {
            if (access(candidate, F_OK) == 0)
            {
                // found; write cur into static global
                if (strlen(cur) < sizeof project_root)
                    strcpy(project_root, cur);
                else
                {
                    project_root[0] = '\0';
                    project_root_cached = 0;
                    return false;
                }
                project_root_cached = 1;
                return true;
            }
        }

        // if cur is root, stop
        if (strcmp(cur, "/") == 0)
            break;

        // move to parent directory
        char *slash = strrchr(cur, '/');
        if (!slash)
            break;
        if (slash == cur)
            cur[1] = '\0';
        else
            *slash = '\0';
    }

    // not found
    project_root_cached = 0;
    return false;
}

// helper function to execute a shell command
static void run_command(const char *funcname, const char *fmt, ...)
{
    // check if project root was found and written to global variable
    if (!find_and_write_project_root())
    {
        fprintf(stderr, "%s: could not find project root directory, aborting...\n", funcname);
        return;
    }
    laik_log(1, "lb-vis/run_command: found project root (abs) %s\n", project_root);

    // allocate large enough buffer for full command
    va_list args;
    va_start(args, fmt);
    int needed = vsnprintf(NULL, 0, fmt, args);
    va_end(args);

    if (needed < 0)
    {
        perror("vsnprintf");
        return;
    }

    char cmdbuf[needed + 1];
    va_start(args, fmt);
    vsprintf(cmdbuf, fmt, args);
    va_end(args);

    // execute command
    laik_log(1, "lb-vis/run_command: executing cmd `%s`\n", cmdbuf);
    int ret = system(cmdbuf);
    if (ret == -1)
        perror("system failed");
}

////////////////
// public API //
////////////////

void laik_lbvis_export_partitioning(const char *filename, Laik_RangeList *lr)
{
    // check if project root was found and written to global variable
    if (!find_and_write_project_root())
    {
        fprintf(stderr, "%s: could not find project root directory, aborting...\n", __func__);
        return;
    }

    // allocate large enough buffer for absolute path to csv file to be generated
    int needed = snprintf(NULL, 0, "%s/examples/lbviz/%s", project_root, filename);
    char ff[needed + 1];
    sprintf(ff, "%s/examples/lbviz/%s", project_root, filename);

    // open or create file
    FILE *fp = fopen(ff, "w");
    if (!fp)
        return;

    // write header based on dimensions
    int dims = lr->space->dims;
    if (dims == 1)
        fprintf(fp, "from_x,to_x,task\n");
    else if (dims == 2)
        fprintf(fp, "from_x,to_x,from_y,to_y,task\n");
    else
        fprintf(fp, "from_x,to_x,from_y,to_y,from_z,to_z,task\n");

    // write each range (from incl., to excl.) alongside corresponding task to new line in output file
    for (size_t i = 0; i < lr->count; ++i)
    {
        Laik_Range r = lr->trange[i].range;
        int task = lr->trange[i].task;

        if (dims == 1)
        {
            int64_t from = r.from.i[0];
            int64_t to = r.to.i[0];
            fprintf(fp, "%ld,%ld,%d\n", from, to, task);
        }
        else if (dims == 2)
        {
            int64_t from_x = r.from.i[0], from_y = r.from.i[1];
            int64_t to_x = r.to.i[0], to_y = r.to.i[1];
            fprintf(fp, "%ld,%ld,%ld,%ld,%d\n", from_x, to_x, from_y, to_y, task);
        }
        else /* if (dims == 3) */
        {
            int64_t from_x = r.from.i[0], from_y = r.from.i[1], from_z = r.from.i[2];
            int64_t to_x = r.to.i[0], to_y = r.to.i[1], to_z = r.to.i[2];
            fprintf(fp, "%ld,%ld,%ld,%ld,%ld,%ld,%d\n", from_x, to_x, from_y, to_y, from_z, to_z, task);
        }
    }

    // cleanup
    fclose(fp);
}

void laik_lbvis_visualize_partitioning(const char *filename)
{
    run_command(__func__, "python3 %s/examples/lbviz/visualize.py %s/examples/lbviz/%s", project_root, project_root, filename);
}

void laik_lbvis_remove_visdata()
{
    run_command(__func__, "%s/scripts/remove_plots.sh", project_root);
}

void laik_lbvis_save_trace()
{
    run_command(__func__, "python3 %s/examples/lbviz/trace.py", project_root);
}

void laik_lbvis_enable_trace(int id, Laik_Instance *inst)
{
    if (!find_and_write_project_root())
    {
        fprintf(stderr, "%s: could not find project root directory, aborting...\n", __func__);
        return;
    }

    // create json directory if it doesn't exist already
    int needed = snprintf(NULL, 0, "%s/examples/lbviz/json/", project_root);
    if (needed < 0)
    {
        perror("snprintf");
        return;
    }
    char dirname[needed + 1];
    sprintf(dirname, "%s/examples/lbviz/json/", project_root);
    if (id == 0)
    {
        if (create_nonexistent_dir(dirname) != 0)
        {
            laik_panic("cannot create examples/lbviz/json directory");
            exit(EXIT_FAILURE);
        }
    }

    // create json output file for task
    needed = snprintf(NULL, 0, "%s/examples/lbviz/json/lb-%d.json", project_root, id);
    char filename[needed + 1];
    sprintf(filename, "%s/examples/lbviz/json/lb-%d.json", project_root, id);
    laik_svg_enable_profiling(inst, filename);
}