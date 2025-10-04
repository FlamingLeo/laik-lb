#!/usr/bin/env python3
import subprocess
import sys
import re
import csv
from itertools import product
from pathlib import Path

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} PROG")
    sys.exit(1)

PROG = sys.argv[1]
PROG_NAME = Path(PROG).stem

NTASKS_list = [2, 4, 8, 16, 32, 64]
FREQ_list = [100, 250, 500, 1000]
ALGO_list = ["rcb", "rcbincr", "hilbert", "gilbert"]

pattern = re.compile(
    r"\[LAIK-LB\].*max dt: ([0-9.]+), mean ([0-9.]+), rel. imbalance ([0-9.]+) \(stopped: (\d+)\)"
)

results_dir = Path("results")
logs_dir = Path("logs")
results_dir.mkdir(exist_ok=True)
logs_dir.mkdir(exist_ok=True)

output_file = results_dir / f"results_progress_{PROG_NAME}.csv"

with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["NTASKS", "FREQ", "ALGO", "segment", "max_dt", "mean", "rel_imbalance", "stopped"])

    for NTASKS, FREQ, ALGO in product(NTASKS_list, FREQ_list, ALGO_list):
        print(f"Running NTASKS={NTASKS}, FREQ={FREQ}, ALGO={ALGO}...")
        cmd = ["mpirun", "-n", str(NTASKS), f"./{PROG}", "-n", str(FREQ), "-a", ALGO, "-o", "-l"]

        log_file = logs_dir / f"ntasks{NTASKS}_freq{FREQ}_algo{ALGO}.log"

        try:
            with open(log_file, "w") as lf:
                result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True, check=True)

            with open(log_file, "r") as lf:
                segment_counter = 1
                for line in lf:
                    match = pattern.search(line)
                    if match:
                        max_dt, mean, rel_imbalance, stopped = match.groups()
                        writer.writerow([NTASKS, FREQ, ALGO, segment_counter, max_dt, mean, rel_imbalance, stopped])
                        segment_counter += 1

        except subprocess.CalledProcessError as e:
            print(f"Error running {cmd}, check log {log_file}")
            continue

print(f"Done. Results written to {output_file}")
