#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import re
import sys

NTASKS_LIST = [2, 4, 8, 16, 32, 64]
FREQ_LIST = [1]
ALGO_LIST = ["rcb", "rcbincr", "hilbert", "gilbert"]

# unit conversion
def bytes_from(value_float, unit_str):
    if unit_str is None:
        unit_str = ""
    u = unit_str.strip().lower().replace(" ", "")
    mul = 1
    if u.startswith("kb") or u == "k" or u == "kb/":
        mul = 10**3
    elif u.startswith("mb") or u == "m":
        mul = 10**6
    elif u.startswith("gb") or u == "g":
        mul = 10**9
    elif u in ("b", "bytes", ""):
        mul = 1
    else:
        if "k" in u:
            mul = 10**3
        elif "m" in u:
            mul = 10**6
        elif "g" in u:
            mul = 10**9
        else:
            mul = 1
    return int(round(float(value_float) * mul))

def elems_from(value_float, unit_str):
    if unit_str is None:
        unit_str = ""
    u = unit_str.strip().lower().replace(" ", "")
    mul = 1
    if u.startswith("k"):
        mul = 10**3
    elif u.startswith("m"):
        mul = 10**6
    else:
        # contains 'k' or 'm' anywhere
        if "k" in u:
            mul = 10**3
        elif "m" in u:
            mul = 10**6
        else:
            mul = 1
    return int(round(float(value_float) * mul))

# regex patterns
re_weights = re.compile(
    r"data\s+'weights':\s*(?P<switches>\d+)\s+switches\s*\(\s*(?P<without>\d+)\s+without actions\s*,\s*(?P<transitions>\d+)\s+transitions",
    re.IGNORECASE,
)

# extract msg type, msg_count, elems and bytes and avg inside parentheses
re_msgline = re.compile(
    r"msg\s+(?P<msgtype>[\w-]+)\s*:\s*(?P<msgcount>\d+)x\s*,\s*(?P<elems_num>[0-9]*\.?[0-9]+)\s*(?P<elems_unit>[KMkm]?\s*e?elems?|[KMkm])?\s*=\s*(?P<byte_num>[0-9]*\.?[0-9]+)\s*(?P<byte_unit>[KMkmgG]?\s*B|bytes|B)?\s*\(.*?avg\s*(?P<avg_num>[0-9]*\.?[0-9]+)\s*(?P<avg_unit>[KMkmgG]?\s*B|bytes|B)?",
    re.IGNORECASE,
)

# task id (LXX)
re_task = re.compile(r"-L(?P<task>\d+)\b")

def parse_weights_pair(line1, line2):
    m1 = re_weights.search(line1)
    if not m1:
        return None
    # try to find task id in line1
    task_m = re_task.search(line1)
    task_id = task_m.group("task") if task_m else ""

    m2 = re_msgline.search(line2)
    if not m2:
        # try to extract numbers more flexibly
        msgcount_m = re.search(r"(\d+)x", line2)
        msg_count = int(msgcount_m.group(1)) if msgcount_m else None
        # first float
        elems_m = re.search(r",\s*([0-9]*\.?[0-9]+)\s*([KMkm]?)\s*(?:KElems|KELEM|elems|Elems|E)?", line2)
        if elems_m:
            elems_num = elems_m.group(1)
            elems_unit = elems_m.group(2) or ""
        else:
            elems_num, elems_unit = None, ""
        # bytes after =
        bytes_m = re.search(r"=\s*([0-9]*\.?[0-9]+)\s*([KMkmgG]?)\s*(?:B|bytes)?", line2)
        if bytes_m:
            byte_num = bytes_m.group(1)
            byte_unit = bytes_m.group(2) or ""
        else:
            byte_num, byte_unit = None, ""
        # avg
        avg_m = re.search(r"avg\s*([0-9]*\.?[0-9]+)\s*([KMkmgG]?)\s*(?:B|bytes)?", line2)
        if avg_m:
            avg_num = avg_m.group(1)
            avg_unit = avg_m.group(2) or ""
        else:
            avg_num, avg_unit = None, ""
        # if any required value is missing, bail
        if msg_count is None or elems_num is None or byte_num is None:
            return None
        parsed = {
            "task": task_id,
            "switches": int(m1.group("switches")),
            "without_actions": int(m1.group("without")),
            "transitions": int(m1.group("transitions")),
            "msg_type": "reduce" if "reduce" in line2.lower() else m2.group("msgtype") if m2 else "",
            "msg_count": int(msg_count),
            "elems_raw": float(elems_num),
            "elems_unit": elems_unit or "",
            "bytes_raw": float(byte_num),
            "bytes_unit": byte_unit or "",
            "avg_raw": float(avg_num) if avg_num else None,
            "avg_unit": avg_unit or "",
        }
    else:
        parsed = {
            "task": task_id,
            "switches": int(m1.group("switches")),
            "without_actions": int(m1.group("without")),
            "transitions": int(m1.group("transitions")),
            "msg_type": m2.group("msgtype"),
            "msg_count": int(m2.group("msgcount")),
            "elems_raw": float(m2.group("elems_num")),
            "elems_unit": m2.group("elems_unit") or "",
            "bytes_raw": float(m2.group("byte_num")),
            "bytes_unit": m2.group("byte_unit") or "",
            "avg_raw": float(m2.group("avg_num")) if m2.group("avg_num") else None,
            "avg_unit": m2.group("avg_unit") or "",
        }

    # remove spaces and standardize unit strings
    parsed["elems_unit"] = (parsed["elems_unit"] or "").replace(" ", "")
    parsed["bytes_unit"] = (parsed["bytes_unit"] or "").replace(" ", "")
    parsed["avg_unit"] = (parsed["avg_unit"] or "").replace(" ", "")

    # convert
    parsed["elems"] = elems_from(parsed["elems_raw"], parsed["elems_unit"])
    parsed["bytes"] = bytes_from(parsed["bytes_raw"], parsed["bytes_unit"])
    parsed["avg_bytes"] = bytes_from(parsed["avg_raw"], parsed["avg_unit"]) if parsed["avg_raw"] is not None else None

    return parsed

def run_and_collect(prog_path, ntasks, freq, algo):
    cmd = ["mpirun", "-n", str(ntasks), prog_path, "-a", algo, "-o"]
    env = os.environ.copy()
    env["LAIK_LOG"] = "2"
    print(f"Running: LAIK_LOG=2 {' '.join(cmd)}")

    # run and capture stdout
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
    if proc.returncode != 0:
        # still return output so we can parse partial output, but warn
        print(f"Warning: command exited with status {proc.returncode}. Parsing whatever output exists.", file=sys.stderr)
    return proc.stdout

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prog")
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prog = args.prog
    prog_basename = os.path.basename(prog)
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    out_csv = os.path.join(outdir, f"results_weights_{prog_basename}.csv")

    # CSV header
    header = [
        "prog",
        "ntasks",
        "freq",
        "algo",
        "task",
        "switches",
        "without_actions",
        "transitions",
        "msg_type",
        "msg_count",
        "elems",
        "bytes", 
        "avg_bytes",
    ]

    rows = []

    if args.dry_run:
        print("Dry run: reading stdout from stdin (single capture). Use to test parsing.")
        data = sys.stdin.read()
        lines = [l.rstrip("\n") for l in data.splitlines()]
        for i, ln in enumerate(lines):
            if "data 'weights'" in ln:
                next_ln = lines[i + 1] if i + 1 < len(lines) else ""
                parsed = parse_weights_pair(ln, next_ln)
                if parsed:
                    parsed_row = {
                        "prog": prog_basename,
                        "ntasks": "",
                        "freq": "",
                        "algo": "",
                        "task": parsed.get("task", ""),
                        "switches": parsed["switches"],
                        "without_actions": parsed["without_actions"],
                        "transitions": parsed["transitions"],
                        "msg_type": parsed["msg_type"],
                        "msg_count": parsed["msg_count"],
                        "elems": parsed["elems"],
                        "bytes": parsed["bytes"],
                        "avg_bytes": parsed["avg_bytes"] if parsed["avg_bytes"] is not None else "",
                    }
                    rows.append(parsed_row)
        # write csv
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Wrote {len(rows)} rows to {out_csv}")
        return

    # sweep through combinations and run program
    total_runs = len(NTASKS_LIST) * len(FREQ_LIST) * len(ALGO_LIST)
    run_index = 0
    for ntasks in NTASKS_LIST:
        for freq in FREQ_LIST:
            for algo in ALGO_LIST:
                run_index += 1
                print(f"[{run_index}/{total_runs}] ntasks={ntasks} freq={freq} algo={algo}")
                try:
                    out = run_and_collect(prog, ntasks, freq, algo)
                except FileNotFoundError as e:
                    print(f"Error running mpirun or program: {e}", file=sys.stderr)
                    sys.exit(2)
                # parse output lines
                lines = [l.rstrip("\n") for l in out.splitlines()]
                found = 0
                for i, ln in enumerate(lines):
                    if "data 'weights'" in ln:
                        next_ln = lines[i + 1] if i + 1 < len(lines) else ""
                        parsed = parse_weights_pair(ln, next_ln)
                        if parsed:
                            found += 1
                            row = {
                                "prog": prog_basename,
                                "ntasks": ntasks,
                                "freq": freq,
                                "algo": algo,
                                "task": parsed.get("task", ""),
                                "switches": parsed["switches"],
                                "without_actions": parsed["without_actions"],
                                "transitions": parsed["transitions"],
                                "msg_type": parsed["msg_type"],
                                "msg_count": parsed["msg_count"],
                                "elems": parsed["elems"],
                                "bytes": parsed["bytes"],
                                "avg_bytes": parsed["avg_bytes"] if parsed["avg_bytes"] is not None else "",
                            }
                            rows.append(row)
                if found == 0:
                    print(f"  -> No 'weights' entries found for this run (ntasks={ntasks},freq={freq},algo={algo}).", file=sys.stderr)
                else:
                    print(f"  -> Found {found} 'weights' entries.")
    # write csv
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nDone. Wrote {len(rows)} rows to {out_csv}")

if __name__ == "__main__":
    main()
