#!/usr/bin/env python3
import re
import subprocess
import csv
import os
import itertools
import time
import sys
from collections import defaultdict

# program config arguments
DEFAULT_PROG = "./md-lb"
PROGNAME = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROG

NTASKS_LIST = [2, 4, 8, 16, 32]
ALGOS = ["hilbert", "gilbert", "rcb", "rcbincr"]
FREQS = [125, 250, 500]

# directories
RESULTS_DIR = "results"
LOG_DIR = "logs"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# filenames
safe_prog = PROGNAME.replace("./", "").replace("/", "_")
TS_CSV = os.path.join(RESULTS_DIR, f"results_{safe_prog}_timeseries.csv")
SUM_CSV = os.path.join(RESULTS_DIR, f"results_{safe_prog}_summary.csv")

# patterns
RE_TASK_CONTAINER = re.compile(r"\[LAIK-LB\]\s+T(?P<task>\d+),\s*([^,]+),\s*(?P<seg>\d+):\s*(?P<metrics>.*)")
RE_MET_PAIR = re.compile(r"(?P<k>[a-zA-Z0-9_.]+)\s*(?:[:=])?\s*(?P<v>-?\d+\.?\d*(?:[eE][-+]?\d+)?)")
RE_SEG_TIMES = re.compile(
    r"\[LAIK-LB\]\s+times in s for this segment:\s*\[(?P<times>[\d\.,\s]+)\],\s*max dt:\s*(?P<max>[\d.]+),\s*mean\s*(?P<mean>[\d.]+),\s*rel\. imbalance\s*(?P<ri>[\d.]+)\s*\(stopped:\s*(?P<stopped>\d+)\)"
)
RE_STEP = re.compile(r"step\s+(?P<step>\d+)\s*/\s*\d+,\s*t=(?P<t>[0-9\.eE+-]+)")
RE_DONE = re.compile(r"^Done\.")
RE_SUMMARY_LINE = re.compile(r"\[LAIK-LB\]\s+T(?P<task>\d+):\s*(?P<metrics>.*)")
RE_TASK_SUM_TIME = re.compile(
    r"^Task\s+(?P<task>\d+):\s*work time\s*=\s*(?P<work>[0-9\.]+)s,\s*switch time\s*=\s*(?P<switch>[0-9\.]+)s,\s*load balancer time\s*=\s*(?P<lb>[0-9\.]+)s"
)
RE_ALLOC_LINE = re.compile(
    r"\[LAIK-LB\]\s+T(?P<task>\d+):\s*num\. allocs:\s*(?P<num_allocs>\d+),\s*bytes alloced:\s*(?P<bytes_alloced>\d+),\s*num\. frees:\s*(?P<num_frees>\d+),\s*bytes freed:\s*(?P<bytes_freed>\d+),\s*max concurrent:\s*(?P<max_concurrent>\d+)"
)


def parse_metric_pairs(s):
    d = {}
    for m in RE_MET_PAIR.finditer(s):
        k = m.group("k")
        v = m.group("v")
        if re.match(r"^-?\d+$", v):
            d[k] = int(v)
        else:
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = v
    return d


def parse_run_output(text, run_meta):
    timeseries_agg = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    timeseries_extra = {}

    summary_by_task = defaultdict(dict)

    last_step = None
    last_t = None
    done_seen = False

    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue

        # step line
        mstep = RE_STEP.search(ln)
        if mstep:
            last_step = int(mstep.group("step"))
            last_t = float(mstep.group("t"))
            continue

        # segment times line
        mseg = RE_SEG_TIMES.search(ln)
        if mseg:
            seg_times_str = mseg.group("times")
            times_list = [float(x) for x in re.split(r"\s*,\s*", seg_times_str.strip()) if x.strip()]
            seg_info = {
                "seg_times": times_list,
                "seg_max": float(mseg.group("max")),
                "seg_mean": float(mseg.group("mean")),
                "seg_rel_imbalance": float(mseg.group("ri")),
                "seg_stopped": int(mseg.group("stopped")),
                "step": last_step,
                "t": last_t,
            }
            timeseries_extra.setdefault(None, []).append(seg_info)
            continue

        # per-task container metrics
        mt = RE_TASK_CONTAINER.search(ln)
        if mt:
            task = int(mt.group("task"))
            seg = int(mt.group("seg"))
            metrics_str = mt.group("metrics")
            d = parse_metric_pairs(metrics_str)
            for k, v in d.items():
                timeseries_agg[seg][task][k] += v
            if None in timeseries_extra and timeseries_extra[None]:
                if seg not in timeseries_extra:
                    timeseries_extra[seg] = timeseries_extra[None].pop(0)
                else:
                    timeseries_extra[None].pop(0)
            continue

        # done
        if RE_DONE.search(ln):
            done_seen = True
            continue

        # summary
        msum = RE_SUMMARY_LINE.search(ln)
        if msum and done_seen:
            task = int(msum.group("task"))
            metrics_str = msum.group("metrics")
            d = parse_metric_pairs(metrics_str)
            for k, v in d.items():
                if k in summary_by_task[task]:
                    i = 1
                    while f"{k}_{i}" in summary_by_task[task]:
                        i += 1
                    summary_by_task[task][f"{k}_{i}"] = v
                else:
                    summary_by_task[task][k] = v
            continue

        # alloc line pattern
        malloc = RE_ALLOC_LINE.search(ln)
        if malloc and done_seen:
            task = int(malloc.group("task"))
            summary_by_task[task].update(
                {
                    "num_allocs": int(malloc.group("num_allocs")),
                    "bytes_alloced": int(malloc.group("bytes_alloced")),
                    "num_frees": int(malloc.group("num_frees")),
                    "bytes_freed": int(malloc.group("bytes_freed")),
                    "max_concurrent": int(malloc.group("max_concurrent")),
                }
            )
            continue

        # times for task
        mtasktime = RE_TASK_SUM_TIME.search(ln)
        if mtasktime and done_seen:
            task = int(mtasktime.group("task"))
            summary_by_task[task].update(
                {
                    "work_time_s": float(mtasktime.group("work")),
                    "switch_time_s": float(mtasktime.group("switch")),
                    "lb_time_s": float(mtasktime.group("lb")),
                }
            )
            continue

    # timeseries rows
    timeseries_rows = []
    for seg, tasks in sorted(timeseries_agg.items(), key=lambda x: x[0]):
        seg_info = timeseries_extra.get(seg, {})
        for task, metrics in sorted(tasks.items(), key=lambda x: x[0]):
            row = {
                "prog": run_meta["prog"],
                "ntasks": run_meta["ntasks"],
                "algo": run_meta["algo"],
                "freq": run_meta["freq"],
                "segment": seg,
                "task": task,
                "seg_stopped": seg_info.get("seg_stopped") if isinstance(seg_info, dict) else None,
            }
            for k, v in metrics.items():
                row[k] = v
            timeseries_rows.append(row)

    # summary rows
    summary_rows = []
    for task, d in sorted(summary_by_task.items(), key=lambda x: x[0]):
        row = {
            "prog": run_meta["prog"],
            "ntasks": run_meta["ntasks"],
            "algo": run_meta["algo"],
            "freq": run_meta["freq"],
            "task": task,
        }
        row.update(d)
        summary_rows.append(row)

    return timeseries_rows, summary_rows


def main():
    all_timeseries = []
    all_summary = []

    combos = list(itertools.product(NTASKS_LIST, ALGOS, FREQS))
    total = len(combos)
    print(f"Running {total} combos using program: {PROGNAME}\n")

    for idx, (ntasks, algo, freq) in enumerate(combos, start=1):
        cmd = ["mpirun", "-n", str(ntasks), PROGNAME, "-l", "-a", algo, "-n", str(freq)]
        meta = {"prog": PROGNAME, "ntasks": ntasks, "algo": algo, "freq": freq}
        print(f"[{idx}/{total}] Running: {' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout=600)
            out = proc.stdout
        except Exception as e:
            out = f"__ERROR__ {e}\n"
            print(f"Run failed: {e}")

        # store logs
        safe_name = f"run_{ntasks}_{algo}_{freq}.log"
        with open(os.path.join(LOG_DIR, safe_name), "w") as fh:
            fh.write(out)

        # parse output
        ts_rows, sum_rows = parse_run_output(out, meta)
        print(f"  parsed: {len(ts_rows)} timeseries rows, {len(sum_rows)} summary rows")
        all_timeseries.extend(ts_rows)
        all_summary.extend(sum_rows)

        time.sleep(0.2)

    # remove unwanted keys explicitly before writing csvs
    for r in all_timeseries:
        for k in ("step", "t", "seg_mean", "seg_rel_imbalance", "seg_times"):
            if k in r:
                del r[k]

    for r in all_summary:
        for k in ("work_time_s", "switch_time_s", "lb_time_s"):
            if k in r:
                del r[k]

    # csv writing helper
    def write_dicts_to_csv(path, rows, base_fields):
        keys = set()
        for r in rows:
            keys.update(r.keys())
        ordered = [k for k in base_fields if k in keys]
        others = sorted(k for k in keys if k not in ordered)
        header = ordered + others
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in header})
        print(f"Wrote {len(rows)} rows to {path}")

    # timeseries base fields
    ts_base = ["prog", "ntasks", "algo", "freq", "segment", "task", "seg_stopped"]
    write_dicts_to_csv(TS_CSV, all_timeseries, ts_base)

    # summary base fields
    sum_base = ["prog", "ntasks", "algo", "freq", "task", "mc", "mb", "fc", "fb", "bs", "br", "num_allocs", "bytes_alloced", "num_frees", "bytes_freed", "max_concurrent"]
    write_dicts_to_csv(SUM_CSV, all_summary, sum_base)

    print("\nAll done.")


if __name__ == "__main__":
    main()
