#!/usr/bin/env python3
import os
import subprocess
import re
import csv
from itertools import product
from pathlib import Path

# program config
PROG = "./lbsmooth"
NTASKS = [2, 4, 8, 16, 32, 64]
ALGOS = ["hilbert", "rcb"]
RESULTS_DIR = "results"
FIELDNAMES = [
    "ntasks",
    "algo",
    "A",
    "r",
    "R",
    "i_flag",
    "S_flag",
    "total_time_s"
]

re_total = re.compile(r"Done\. Time taken:\s*([0-9]+(?:\.[0-9]+)?)s")

AR_COMBOS = [
    (0.25, 0.85, 1.20),
    (0.50, 0.60, 1.60),
    (0.05, 0.92, 1.08),
]

os.makedirs(RESULTS_DIR, exist_ok=True)
out_csv = os.path.join(RESULTS_DIR, "results_lbsmooth_times_lb.csv")

with open(out_csv, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
    writer.writeheader()
    total_runs = 0

    # NTASKS x ALGO x (AR) x i
    combos = list(product(NTASKS, ALGOS, AR_COMBOS, [False, True]))
    total_runs += len(combos)

    # NTASKS x ALGO
    s_runs = list(product(NTASKS, ALGOS))
    total_runs += len(s_runs)

    print(f"Total: {total_runs}\n")
    run_idx = 0

    # do combos with and without -i
    for ntasks, algo, (A_val, r_val, R_val), i_flag in combos:
        run_idx += 1
        cmd = ["mpirun", "-n", str(ntasks), PROG, "-O", "-a", algo, "-A", str(A_val), "-r", str(r_val), "-R", str(R_val)]
        if i_flag:
            cmd.append("-i")

        print(f"[{run_idx}/{total_runs}] {' '.join(cmd)}")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out = proc.stdout or ""
        m_total = re_total.search(out)
        total_time = m_total.group(1) if m_total else ""

        writer.writerow({
            "ntasks": ntasks,
            "algo": algo,
            "A": A_val,
            "r": r_val,
            "R": R_val,
            "i_flag": i_flag,
            "S_flag": False,
            "total_time_s": total_time
        })

        print(f"  total_time={total_time or 'N/A'}")

    # no smoothing
    for ntasks, algo in s_runs:
        run_idx += 1
        cmd = ["mpirun", "-n", str(ntasks), PROG, "-a", algo, "-S"]
        print(f"[{run_idx}/{total_runs}] {' '.join(cmd)}")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out = proc.stdout or ""
        m_total = re_total.search(out)
        total_time = m_total.group(1) if m_total else ""

        writer.writerow({
            "ntasks": ntasks,
            "algo": algo,
            "A": "",
            "r": "",
            "R": "",
            "i_flag": False,
            "S_flag": True,
            "total_time_s": total_time
        })

        print(f"  total_time={total_time or 'N/A'}")

print(f"\nDone. CSV written to: {out_csv}")
