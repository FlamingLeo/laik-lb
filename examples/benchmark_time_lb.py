#!/usr/bin/env python3
import os
import sys
import subprocess
import re
import csv
import itertools

# program config arguments
NTASKS = [2, 4, 8, 16, 32, 64]
ALGOS = ["hilbert", "gilbert", "rcb", "rcbincr"]
FREQS = [125, 250, 500]

RESULTS_DIR = "results"

# csv header
FIELDNAMES = [
    "ntasks",
    "algo",
    "freq",
    "total_time_s",
    "task_id",
    "work_time_s",
    "switch_time_s",
    "load_balancer_time_s",
    "returncode",
]

exe_arg = sys.argv[1] if len(sys.argv) >= 2 else "./md-lb"

# allow ./progname and progname, both valid
if os.path.sep in exe_arg or exe_arg.startswith("."):
    exe_path = exe_arg
else:
    exe_path = "./" + exe_arg

progname = os.path.basename(exe_arg)

# prepare output file
os.makedirs(RESULTS_DIR, exist_ok=True)
out_csv = os.path.join(RESULTS_DIR, f"results_{progname}_times_lb.csv")

# parsing regex
re_total = re.compile(r"Done\. Time taken:\s*([0-9]+(?:\.[0-9]+)?)s")
re_task = re.compile(
    r"Task\s+(\d+)\s*:\s*work\s*time\s*=\s*([0-9]+(?:\.[0-9]+)?)s\s*,?\s*switch\s*time\s*=\s*([0-9]+(?:\.[0-9]+)?)s\s*,?\s*load\s*balancer\s*time\s*=\s*([0-9]+(?:\.[0-9]+)?)s",
    flags=re.IGNORECASE,
)

# run and parse combos
combos = list(itertools.product(NTASKS, ALGOS, FREQS))
total_runs = len(combos)
print(f"Running {total_runs} combinations for {exe_path}.")
print(f"Results: {out_csv}")

with open(out_csv, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
    writer.writeheader()

    for idx, (ntasks, algo, freq) in enumerate(combos, start=1):
        cmd = ["mpirun", "-n", str(ntasks), exe_path, "-o", "-a", algo, "-n", str(freq)]
        print(f"[{idx}/{total_runs}] {' '.join(cmd)}")

        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except FileNotFoundError as e:
            print(f"  ERROR: failed to execute command: {e}")
            continue

        out = proc.stdout or ""
        rc = proc.returncode

        # parse total time
        m_total = re_total.search(out)
        total_time = m_total.group(1) if m_total else ""

        # parse task lines
        tasks = list(re_task.finditer(out))
        if tasks:
            for match in tasks:
                tid = match.group(1)
                w = match.group(2)
                s = match.group(3)
                lb = match.group(4)
                writer.writerow({
                    "ntasks": ntasks,
                    "algo": algo,
                    "freq": freq,
                    "total_time_s": total_time,
                    "task_id": tid,
                    "work_time_s": w,
                    "switch_time_s": s,
                    "load_balancer_time_s": lb,
                    "returncode": rc
                })
            print(f"  Parsed {len(tasks)} task lines. total_time={total_time or 'N/A'} rc={rc}")
        else:
           print(f"  No task lines found. total_time={total_time or 'N/A'} rc={rc}")

print("Done. CSV written to:", out_csv)
