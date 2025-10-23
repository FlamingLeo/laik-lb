#!/usr/bin/env python3
import os
import subprocess
import re
import csv
import itertools
from itertools import product
from pathlib import Path

NTASKS = [2, 4, 8, 16, 32, 64]
ALGOS = ["hilbert", "gilbert", "rcb", "rcbincr"]
FREQS = [125, 250, 500]
FREQS_JAC = [1]
RESULTS_DIR = "results"
FIELDNAMES = [
    "ntasks",
    "algo",
    "freq",
    "total_time_s",
    "task_id",
    "work_time_s",
    "switch_time_s",
    "load_balancer_time_s"
]
PROGRAMS = ["./md-lb", "./md3d-lb", "./jac2d-lb", "./jac3d-lb"]

re_total = re.compile(r"Done\. Time taken:\s*([0-9]+(?:\.[0-9]+)?)s")
re_task = re.compile(r"Task\s+(\d+)\s*:\s*work\s*time\s*=\s*([0-9]+(?:\.[0-9]+)?)s\s*,?\s*switch\s*time\s*=\s*([0-9]+(?:\.[0-9]+)?)s\s*,?\s*load\s*balancer\s*time\s*=\s*([0-9]+(?:\.[0-9]+)?)s", flags=re.IGNORECASE)

os.makedirs(RESULTS_DIR, exist_ok=True)

for prog in PROGRAMS:
    progname = os.path.basename(prog)
    out_csv = os.path.join(RESULTS_DIR, f"results_{progname}_times_lb.csv")
    freqs_for_prog = FREQS_JAC if progname.startswith("jac") else FREQS
    combos = list(product(NTASKS, ALGOS, freqs_for_prog))
    total_runs = len(combos)

    print(f"Running {total_runs} combinations for {prog}.")

    with open(out_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()

        for idx, (ntasks, algo, freq) in enumerate(combos, start=1):
            cmd = ["mpirun", "-n", str(ntasks), prog, "-o", "-a", algo, "-n", str(freq)]
            print(f"[{idx}/{total_runs}] {' '.join(cmd)}")
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            out = proc.stdout or ""

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
                        "load_balancer_time_s": lb
                    })
                print(f"  Parsed {len(tasks)} task lines. total_time={total_time or 'N/A'}")
            else:
               print(f"  No task lines found. total_time={total_time or 'N/A'}")

    print(f"Finished {progname}. CSV written to: {out_csv}\n")

print("Done.")
