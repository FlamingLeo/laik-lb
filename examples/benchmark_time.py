#!/usr/bin/env python3
import os
import subprocess
import re
import csv

NTASKS = [2, 4, 8, 16, 32, 64]
RESULTS_DIR = "results"
FIELDNAMES = [
    "ntasks",
    "total_time_s",
    "task_id",
    "work_time_s",
    "switch_time_s"
]
PROGRAMS = [
    ["./md"],
    ["./md3d"],
    ["./jac2d-lb", "-L"],
    ["./jac3d-lb", "-L"],
]

os.makedirs(RESULTS_DIR, exist_ok=True)
re_total = re.compile(r"Done\. Time taken:\s*([0-9]+(?:\.[0-9]+)?)s")
re_task = re.compile(r"Task\s+(\d+)\s*:\s*work\s*time\s*=\s*([0-9]+(?:\.[0-9]+)?)s\s*,?\s*switch\s*time\s*=\s*([0-9]+(?:\.[0-9]+)?)s\s*,?\s*load\s*balancer\s*time\s*=\s*N/A", flags=re.IGNORECASE)

for exe_cmd in PROGRAMS:
    exe_path = exe_cmd[0]
    progname = os.path.basename(exe_path)
    out_csv = os.path.join(RESULTS_DIR, f"results_{progname}_times.csv")

    print(f"\nRunning: {progname}")

    with open(out_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()

        total_runs = len(NTASKS)
        for idx, ntasks in enumerate(NTASKS, start=1):
            cmd = ["mpirun", "-n", str(ntasks)] + exe_cmd + ["-o"]
            print(f"[{idx}/{total_runs}] {' '.join(cmd)}")
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            out = proc.stdout or ""
            m_total = re_total.search(out)
            total_time = m_total.group(1) if m_total else ""

            # parse task lines
            tasks = list(re_task.finditer(out))
            if tasks:
                for match in tasks:
                    tid = match.group(1)
                    w = match.group(2)
                    s = match.group(3)
                    writer.writerow({
                        "ntasks": ntasks,
                        "total_time_s": total_time,
                        "task_id": tid,
                        "work_time_s": w,
                        "switch_time_s": s,
                    })
                print(f"  Parsed {len(tasks)} task lines. total_time={total_time or 'N/A'}")
            else:
                print(f"  No task lines found. total_time={total_time or 'N/A'}")

    print(f"Finished {progname}. CSV written to: {out_csv}\n")

print("Done.")
