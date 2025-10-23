#!/usr/bin/env python3
import os
import subprocess
import re
import csv
from itertools import product
from pathlib import Path

PROGRAMS = ["./md-lb", "./md3d-lb", "./jac2d-lb", "./jac3d-lb"]
ALGOS = ["rcb", "rcbincr", "hilbert", "gilbert"]
NTASKS = [2, 4, 8, 16, 32, 64]
FREQS = [125, 250, 500]
FREQS_JAC = [1]
RESULTS_DIR = "results"
LOGS_DIR = "logs"
FIELDNAMES = ["ntasks", "freq", "algo", "segment", "max_dt", "mean", "rel_imbalance", "stopped"]

pattern = re.compile(r"\[LAIK-LB\].*max dt: ([0-9.]+), mean ([0-9.]+), rel. imbalance ([0-9.]+) \(stopped: (\d+)\)")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

for prog in PROGRAMS:
    progname = os.path.basename(prog)
    out_csv = os.path.join(RESULTS_DIR, f"results_progress_lb_{progname}.csv")
    
    freqs_for_prog = FREQS_JAC if progname.startswith("jac") else FREQS
    combos = list(product(NTASKS, freqs_for_prog, ALGOS))
    total_runs = len(combos)
    
    print(f"Running {total_runs} combinations for {prog}.")
    
    with open(out_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()
        
        for idx, (ntasks, freq, algo) in enumerate(combos, start=1):
            cmd = ["mpirun", "-n", str(ntasks), prog, "-a", algo, "-o", "-l"]
            print(f"[{idx}/{total_runs}] {' '.join(cmd)}")
            
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            out = proc.stdout or ""
            matches = list(pattern.finditer(out))
            
            if matches:
                for segment_counter, match in enumerate(matches, start=1):
                    max_dt, mean, rel_imbalance, stopped = match.groups()
                    writer.writerow({
                        "ntasks": ntasks,
                        "freq": freq,
                        "algo": algo,
                        "segment": segment_counter,
                        "max_dt": max_dt,
                        "mean": mean,
                        "rel_imbalance": rel_imbalance,
                        "stopped": stopped
                    })
                print(f"  Parsed {len(matches)} segments.")
            else:
                print(f"  No segments found.")
    
    print(f"Finished {progname}. CSV written to: {out_csv}\n")

print("Done.")