import os
import subprocess
import itertools
import csv
import re
from statistics import mean

mpirun_cmd = "mpirun"
test_exe = "../test"
LAIK_LOG_VAL = "2"
repetitions = 3

ntasks_list = [2, 3, 4, 5]
tags = [0, 1]
lbalgos = ["hilbert"]
sidesizes = [128, 256, 512, 1024]
loopcounts = [5, 10, 15, 20]
out_csv = "laik_stats.csv"
raw_out_dir = "outputs"
os.makedirs(raw_out_dir, exist_ok=True)

# helper: convert (number string, unit) -> bytes
def to_bytes(num_str, unit):
    num = float(num_str)
    u = unit.upper().strip()
    if u in ("B",):
        return int(num)
    if u in ("KB", "KIB"):
        return int(num * 1024)
    if u in ("MB", "MIB"):
        return int(num * 1024 * 1024)
    # fallback
    return int(num)

# robust parse of a single output text: returns dict of relevant stats
def parse_output(text):
    stats = {
        "allocs": 0,
        "alloc_bytes": 0,
        "frees": 0,
        "free_bytes": 0,
        "copies": 0,
        "copy_bytes": 0,
        "msgs": 0,
        "msg_elems": 0,
        "msg_bytes": 0,
    }
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        # find data 'data-N' line anywhere on the line
        if re.search(r"data\s+'data-\d+'", line):
            # collect following lines that are prefixed by '..'
            block_lines = []
            j = i + 1
            while j < len(lines) and re.match(r'^\s*\.\.', lines[j]):
                block_lines.append(lines[j])
                j += 1
            block = "\n".join(block_lines)

            # data alloc: 13x, 1.0 KB (max 908 B), free: 8x, 352 B, init 0 B, copy 8 B
            m_alloc = re.search(r"data alloc:\s*(\d+)x,\s*([\d.]+)\s*([KM]?B|B)", block, re.I)
            if m_alloc:
                stats["allocs"] += int(m_alloc.group(1))
                stats["alloc_bytes"] += to_bytes(m_alloc.group(2), m_alloc.group(3))

            m_free = re.search(r"free:\s*(\d+)x,\s*([\d.]+)\s*([KM]?B|B)", block, re.I)
            if m_free:
                stats["frees"] += int(m_free.group(1))
                stats["free_bytes"] += to_bytes(m_free.group(2), m_free.group(3))

            # copy may appear multiple times; sum them
            for m in re.finditer(r"copy\s*([\d.]+)\s*([KM]?B|B)", block, re.I):
                stats["copies"] += 1
                stats["copy_bytes"] += to_bytes(m.group(1), m.group(2))

            # messages (recv/sent/reduce) — attempt to capture count, elems, bytes if present
            for m in re.finditer(r"msg\s+(?:recv|sent|reduce):\s*(\d+)x.*?([\d.]+)\s*Elems\s*=\s*([\d.]+)\s*([KM]?B|B)", block, re.I):
                stats["msgs"] += int(m.group(1))
                # elems may use 'K' (like 1.0 K Elems) — we keep as numeric-elems (approx)
                elems = float(m.group(2))
                stats["msg_elems"] += int(round(elems))
                stats["msg_bytes"] += to_bytes(m.group(3), m.group(4))

            i = j
        else:
            i += 1
    return stats

results = []
comb_count = (len(ntasks_list)*len(tags)*len(lbalgos)*len(sidesizes)*len(loopcounts))
comb_idx = 0

for ntasks, tag, algo, size, loops in itertools.product(ntasks_list, tags, lbalgos, sidesizes, loopcounts):
    comb_idx += 1
    print(f"[{comb_idx}/{comb_count}] ntasks={ntasks} tag={tag} algo={algo} size={size} loops={loops}")
    run_stats = []
    for rep in range(1, repetitions+1):
        fname = f"{raw_out_dir}/run_nt{ntasks}_tag{tag}_{algo}_s{size}_l{loops}_rep{rep}.log"
        cmd = [mpirun_cmd, "-n", str(ntasks), "-x", f"LAIK_LOG={LAIK_LOG_VAL}", test_exe, str(tag), algo, str(size), str(loops)]
        print("   running:", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        raw = proc.stdout + "\n" + proc.stderr

        # dump raw to file for inspection
        with open(fname, "w") as fh:
            fh.write(raw)
        if proc.returncode != 0:
            print(f"   WARNING: run returned code {proc.returncode} (log saved to {fname})")
        stats = parse_output(raw)
        run_stats.append(stats)

    # compute mean across repetitions for the keys present
    keys = run_stats[0].keys()
    avg = {k: mean([r[k] for r in run_stats]) for k in keys}
    row = {
        "ntasks": ntasks,
        "tag": tag,
        "algo": algo,
        "sidesize": size,
        "loopcount": loops,
        **avg
    }
    results.append(row)

# write csv
if results:
    fieldnames = list(results[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print("Wrote:", out_csv)
else:
    print("No results collected.")
