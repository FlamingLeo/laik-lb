#!/usr/bin/env python3
import os
import subprocess
import csv
import re
from itertools import product
from collections import defaultdict

PROGRAMS = ["./md", "./md3d"]
NTASKS = [2, 4, 8, 16, 32, 64]
FREQS = [125, 250, 500]
RESULTS_DIR = "results"
FIELDNAMES = ["ntasks", "freq", "task", "segment"]
RE_TASK_CONTAINER = re.compile(r"\[LAIK-LB\]\s+T(?P<task>\d+),\s*(?P<data>[^,]+),\s*(?P<seg>\d+):\s*(?P<metrics>.*)")
RE_MET_PAIR = re.compile(r"(?P<k>[a-zA-Z0-9_.]+)\s*(?:[:=])?\s*(?P<v>-?\d+\.?\d*(?:[eE][-+]?\d+)?)")

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


total_runs = len(PROGRAMS) * len(NTASKS) * len(FREQS)
run_idx = 0

for prog in PROGRAMS:
    progname = os.path.basename(prog)
    out_csv = os.path.join(RESULTS_DIR, f"results_{progname}_metrics.csv")
    combos = list(product(NTASKS, FREQS))
    
    print(f"Running {len(combos)} combinations for {prog}.")
    
    all_metric_keys = set()
    aggregated_rows = defaultdict(lambda: defaultdict(int))
    
    for ntasks, freq in combos:
        run_idx += 1
        cmd = ["mpirun", "-n", str(ntasks), prog, "-l", "-n", str(freq)]
        
        print(f"[{run_idx}/{total_runs}] {' '.join(cmd)}")
        
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out = proc.stdout or ""
        
        parsed_count = 0
        for line in out.splitlines():
            match = RE_TASK_CONTAINER.search(line)
            if match:
                task = int(match.group("task"))
                seg = int(match.group("seg"))
                metrics = parse_metric_pairs(match.group("metrics"))
                key = (ntasks, freq, task, seg)
                
                # aggregate values of each data container
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        aggregated_rows[key][metric_name] += value # type: ignore
                    all_metric_keys.add(metric_name)
                
                parsed_count += 1
        
        print(f"  Parsed {parsed_count} metric rows.")
    
    all_rows = []
    for (ntasks, freq, task, seg), metrics in aggregated_rows.items():
        row = {
            "ntasks": ntasks,
            "freq": freq,
            "task": task,
            "segment": seg,
        }
        row.update(metrics)
        all_rows.append(row)
    
    final_fieldnames = FIELDNAMES + sorted(all_metric_keys)
    
    with open(out_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=final_fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"Finished {progname}. CSV written to: {out_csv} ({len(all_rows)} rows)\n")

print("Done.")