#!/usr/bin/env python3
import os
import subprocess
import re
import csv
from itertools import product
from pathlib import Path

PROGS = ["./lbsmooth"]
NTASKS_list = [2, 4, 8, 16, 32, 64]
ALGO_list = ["rcb", "hilbert"]
PARAM_SETS = [
    (0.25, 0.85, 1.2),
    (0.5, 0.6, 1.6),
    (0.05, 0.92, 1.08),
]
RESULTS_DIR = "results"
FIELDNAMES = ["ntasks", "algo", "A", "r", "R", "S", "i", "segment", "max_dt", "mean", "rel_imbalance", "stopped"]

pattern = re.compile(r"\[LAIK-LB\].*max dt: ([0-9.]+), mean ([0-9.]+), rel. imbalance ([0-9.]+) \(stopped: (\d+)\)")
os.makedirs(RESULTS_DIR, exist_ok=True)

def fn_label(x):
    return str(x).replace(".", "p")


for prog in PROGS:
    progname = os.path.basename(prog)
    out_csv = os.path.join(RESULTS_DIR, f"results_progress_lb_{progname}.csv")
    total_runs = len(NTASKS_list) * len(ALGO_list) * (1 + len(PARAM_SETS) * 2)
    
    print(f"Running {total_runs} combinations for {prog}.")
    
    with open(out_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()
        
        run_idx = 0
        for ntasks, algo in product(NTASKS_list, ALGO_list):
            # smoothing off
            run_idx += 1
            cmd = ["mpirun", "-n", str(ntasks), prog, "-a", algo, "-S"]
            param_label = "S"
            
            print(f"[{run_idx}/{total_runs}] {' '.join(cmd)}")
            
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            out = proc.stdout or ""
            matches = list(pattern.finditer(out))
            
            if matches:
                for segment_counter, match in enumerate(matches, start=1):
                    max_dt, mean, rel_imbalance, stopped = match.groups()
                    writer.writerow({
                        "ntasks": ntasks,
                        "algo": algo,
                        "A": "",
                        "r": "",
                        "R": "",
                        "S": True,
                        "i": False,
                        "segment": segment_counter,
                        "max_dt": max_dt,
                        "mean": mean,
                        "rel_imbalance": rel_imbalance,
                        "stopped": stopped
                    })
                print(f"  Parsed {len(matches)} segments.")
            else:
                print(f"  No segments found.")
            
            # with and without intelligent mode
            for (A, r, R) in PARAM_SETS:
                for i_flag in (False, True):
                    run_idx += 1
                    i_label = "i" if i_flag else "noi"
                    cmd = ["mpirun", "-n", str(ntasks), prog, "-a", algo, "-A", str(A), "-r", str(r), "-R", str(R)]
                    if i_flag:
                        cmd.append("-i")
                    
                    param_label = f"A{fn_label(A)}_r{fn_label(r)}_R{fn_label(R)}"
                    if i_flag:
                        param_label += "_i"
                    
                    print(f"[{run_idx}/{total_runs}] {' '.join(cmd)}")
                    
                    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    out = proc.stdout or ""
                    matches = list(pattern.finditer(out))
                    
                    if matches:
                        for segment_counter, match in enumerate(matches, start=1):
                            max_dt, mean, rel_imbalance, stopped = match.groups()
                            writer.writerow({
                                "ntasks": ntasks,
                                "algo": algo,
                                "A": A,
                                "r": r,
                                "R": R,
                                "S": False,
                                "i": i_flag,
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