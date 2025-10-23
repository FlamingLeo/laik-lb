#!/usr/bin/env python3
import os
import subprocess
import csv
import re
from itertools import product
from collections import defaultdict

PROGRAMS = ["./md-lb", "./md3d-lb", "./jac2d-lb", "./jac3d-lb"]
NTASKS = [2, 4, 8, 16, 32, 64]
ALGOS = ["hilbert", "gilbert", "rcb", "rcbincr"]
FREQS_JAC = [1]
FREQS_MD = [125, 250, 500]
RESULTS_DIR = "results"

FIELDNAMES_TIMESERIES = ["ntasks", "algo", "freq", "segment", "task", "seg_stopped"]
FIELDNAMES_SUMMARY = ["ntasks", "algo", "freq", "task"]

RE_TASK_CONTAINER = re.compile(r"\[LAIK-LB\]\s+T(?P<task>\d+),\s*([^,]+),\s*(?P<seg>\d+):\s*(?P<metrics>.*)")
RE_MET_PAIR = re.compile(r"(?P<k>[a-zA-Z0-9_.]+)\s*(?:[:=])?\s*(?P<v>-?\d+\.?\d*(?:[eE][-+]?\d+)?)")
RE_SEG_TIMES = re.compile(r"\[LAIK-LB\]\s+times in s for this segment:\s*\[(?P<times>[\d\.,\s]+)\],\s*max dt:\s*(?P<max>[\d.]+),\s*mean\s*(?P<mean>[\d.]+),\s*rel\. imbalance\s*(?P<ri>[\d.]+)\s*\(stopped:\s*(?P<stopped>\d+)\)")
RE_DONE = re.compile(r"^Done\.")
RE_SUMMARY_LINE = re.compile(r"\[LAIK-LB\]\s+T(?P<task>\d+):\s*(?P<metrics>.*)")
RE_ALLOC_LINE = re.compile(r"\[LAIK-LB\]\s+T(?P<task>\d+):\s*num\. allocs:\s*(?P<num_allocs>\d+),\s*bytes alloced:\s*(?P<bytes_alloced>\d+),\s*num\. frees:\s*(?P<num_frees>\d+),\s*bytes freed:\s*(?P<bytes_freed>\d+),\s*max concurrent:\s*(?P<max_concurrent>\d+)")

os.makedirs(RESULTS_DIR, exist_ok=True)

def parse_value(v):
    if re.match(r"^-?\d+$", v):
        return int(v)
    try:
        return float(v)
    except ValueError:
        return v


def parse_metric_pairs(s):
    return {m.group("k"): parse_value(m.group("v")) for m in RE_MET_PAIR.finditer(s)}


total_runs = sum(
    len(NTASKS) * len(ALGOS) * len(FREQS_JAC if "jac" in prog else FREQS_MD)
    for prog in PROGRAMS
)
run_idx = 0

for prog in PROGRAMS:
    progname = os.path.basename(prog)
    freqs = FREQS_JAC if "jac" in progname else FREQS_MD    
    combos = list(product(NTASKS, ALGOS, freqs))
    
    print(f"Running {len(combos)} combinations for {prog}.")
    
    timeseries_agg = defaultdict(lambda: defaultdict(float))
    timeseries_extra = {}
    summary_by_combo_task = {}
    
    all_timeseries_keys = set()
    all_summary_keys = set()
    
    for ntasks, algo, freq in combos:
        run_idx += 1
        cmd = ["mpirun", "-n", str(ntasks), prog, "-l", "-a", algo, "-n", str(freq)]
        print(f"[{run_idx}/{total_runs}] {' '.join(cmd)}")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out = proc.stdout or ""
        
        done_seen = False
        pending_seg_info = None
        ts_count = 0
        sum_count = 0
        
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            
            # segment times
            mseg = RE_SEG_TIMES.search(line)
            if mseg:
                pending_seg_info = {
                    "seg_stopped": int(mseg.group("stopped")),
                }
                continue
            
            # container metrics
            mt = RE_TASK_CONTAINER.search(line)
            if mt:
                task = int(mt.group("task"))
                seg = int(mt.group("seg"))
                metrics = parse_metric_pairs(mt.group("metrics"))
                key = (ntasks, algo, freq, seg, task)
                
                # sum numeric metrics
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        timeseries_agg[key][k] += v
                        all_timeseries_keys.add(k)
                
                # pending segment info
                if pending_seg_info and key not in timeseries_extra:
                    timeseries_extra[key] = pending_seg_info
                    pending_seg_info = None
                
                ts_count += 1
                continue
            
            if RE_DONE.search(line):
                done_seen = True
                continue
            
            # summary lines
            if done_seen:
                msum = RE_SUMMARY_LINE.search(line)
                if msum:
                    task = int(msum.group("task"))
                    metrics = parse_metric_pairs(msum.group("metrics"))
                    
                    combo_key = (ntasks, algo, freq, task)
                    if combo_key not in summary_by_combo_task:
                        summary_by_combo_task[combo_key] = {}
                    
                    # duplicate keys
                    for k, v in metrics.items():
                        if k in summary_by_combo_task[combo_key]:
                            i = 1
                            while f"{k}_{i}" in summary_by_combo_task[combo_key]:
                                i += 1
                            summary_by_combo_task[combo_key][f"{k}_{i}"] = v
                            all_summary_keys.add(f"{k}_{i}")
                        else:
                            summary_by_combo_task[combo_key][k] = v
                            all_summary_keys.add(k)
                    sum_count += 1
                    continue
                
                # allocations
                malloc = RE_ALLOC_LINE.search(line)
                if malloc:
                    task = int(malloc.group("task"))
                    combo_key = (ntasks, algo, freq, task)
                    if combo_key not in summary_by_combo_task:
                        summary_by_combo_task[combo_key] = {}
                    
                    alloc_metrics = {
                        "num_allocs": int(malloc.group("num_allocs")),
                        "bytes_alloced": int(malloc.group("bytes_alloced")),
                        "num_frees": int(malloc.group("num_frees")),
                        "bytes_freed": int(malloc.group("bytes_freed")),
                        "max_concurrent": int(malloc.group("max_concurrent")),
                    }
                    summary_by_combo_task[combo_key].update(alloc_metrics)
                    all_summary_keys.update(alloc_metrics.keys())
                    continue
        
        print(f"  Parsed {ts_count} timeseries lines, {sum_count} summary lines.")
    
    timeseries_rows = []
    for (ntasks, algo, freq, seg, task), metrics in timeseries_agg.items():
        row = {
            "ntasks": ntasks,
            "algo": algo,
            "freq": freq,
            "segment": seg,
            "task": task,
        }
        # stopped?
        key = (ntasks, algo, freq, seg, task)
        if key in timeseries_extra:
            row["seg_stopped"] = timeseries_extra[key]["seg_stopped"]
        
        row.update(metrics)
        timeseries_rows.append(row)
    
    # cvt summary to rows
    summary_rows = []
    for (ntasks, algo, freq, task), metrics in summary_by_combo_task.items():
        row = {
            "ntasks": ntasks,
            "algo": algo,
            "freq": freq,
            "task": task,
        }
        row.update(metrics)
        summary_rows.append(row)
    
    # write timeseries csv
    ts_path = os.path.join(RESULTS_DIR, f"results_{progname}_timeseries.csv")
    ts_fieldnames = FIELDNAMES_TIMESERIES + sorted(all_timeseries_keys)
    
    with open(ts_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=ts_fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(timeseries_rows)
    
    print(f"Wrote {len(timeseries_rows)} rows to {ts_path}")
    
    # write summary csv
    sum_path = os.path.join(RESULTS_DIR, f"results_{progname}_summary.csv")
    sum_fieldnames = FIELDNAMES_SUMMARY + sorted(all_summary_keys)
    
    with open(sum_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=sum_fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summary_rows)
    
    print(f"Wrote {len(summary_rows)} rows to {sum_path}\n")

print("Done.")