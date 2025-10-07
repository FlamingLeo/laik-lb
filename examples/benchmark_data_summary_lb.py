#!/usr/bin/env python3
import re
import csv
import shlex
import subprocess
import argparse
import os
from collections import defaultdict
from datetime import datetime

NTASKS_LIST = [2, 4, 8, 16, 32, 64]
PROG_N_LIST = [1]
ALGORITHMS = ["rcb", "rcbincr", "hilbert", "gilbert"]
UNIT_MULT = {
    "B": 1,
    "KB": 10 ** 3,
    "MB": 10 ** 6,
    "GB": 10 ** 9,
}


def to_bytes(value_float, unit):
    unit = unit.upper()
    if unit not in UNIT_MULT:
        raise ValueError(f"unknown unit {unit}")
    return float(value_float) * UNIT_MULT[unit]


# regex helpers
rx_task = re.compile(r'-L(\d+)\b')
rx_summary_switches = re.compile(
    r"summary:\s*([0-9,]+)\s+switches\s*\(\s*([0-9,]+)\s+without actions\s*,\s*([0-9,]+)\s+transitions",
    re.IGNORECASE,
)
rx_data_alloc = re.compile(r"data alloc:\s*([0-9,]+)x\s*,\s*([0-9.]+)\s*(B|KB|MB|GB)", re.IGNORECASE)
rx_msg_sent = re.compile(r"msg sent:\s*([0-9,]+)x.*=\s*([0-9.]+)\s*(B|KB|MB|GB)", re.IGNORECASE)
rx_msg_recv = re.compile(r"recv:\s*([0-9,]+)x.*=\s*([0-9.]+)\s*(B|KB|MB|GB)", re.IGNORECASE)
rx_reduce = re.compile(r"reduce:\s*([0-9,]+)x.*=\s*([0-9.]+)\s*(B|KB|MB|GB)", re.IGNORECASE)


def parse_task_summary(task_lines):
    last_idx = None
    for i, ln in enumerate(task_lines):
        if "summary:" in ln:
            last_idx = i
    if last_idx is None:
        return None

    block_lines = task_lines[last_idx:]
    joined = "\n".join(block_lines)

    def iint(s):
        return int(s.replace(",", ""))

    result = {
        "switches": None,
        "without_actions": None,
        "transitions": None,
        "data_alloc_count": None,
        "data_alloc_bytes": None,
        "msg_sent_count": None,
        "msg_sent_bytes": None,
        "recv_count": None,
        "recv_bytes": None,
        "reduce_count": None,
        "reduce_bytes": None,
    }

    m = rx_summary_switches.search(joined)
    if m:
        result["switches"] = iint(m.group(1))
        result["without_actions"] = iint(m.group(2))
        result["transitions"] = iint(m.group(3))

    m = rx_data_alloc.search(joined)
    if m:
        result["data_alloc_count"] = iint(m.group(1))
        try:
            result["data_alloc_bytes"] = to_bytes(m.group(2), m.group(3))
        except Exception:
            result["data_alloc_bytes"] = None

    m = rx_msg_sent.search(joined)
    if m:
        result["msg_sent_count"] = iint(m.group(1))
        try:
            result["msg_sent_bytes"] = to_bytes(m.group(2), m.group(3))
        except Exception:
            result["msg_sent_bytes"] = None

    m = rx_msg_recv.search(joined)
    if m:
        result["recv_count"] = iint(m.group(1))
        try:
            result["recv_bytes"] = to_bytes(m.group(2), m.group(3))
        except Exception:
            result["recv_bytes"] = None

    m = rx_reduce.search(joined)
    if m:
        result["reduce_count"] = iint(m.group(1))
        try:
            result["reduce_bytes"] = to_bytes(m.group(2), m.group(3))
        except Exception:
            result["reduce_bytes"] = None

    return result


def run_and_capture(cmd, env=None, timeout=None):
    args = shlex.split(cmd)
    p = subprocess.run(args, env=env, capture_output=True, text=True, timeout=timeout)
    out = p.stdout + ("\n" if p.stdout and p.stderr else "") + p.stderr
    return out.splitlines(), p.returncode


def parse_combined_output(lines):
    lines_by_task = defaultdict(list)
    for ln in lines:
        m = rx_task.search(ln)
        if m:
            tid = int(m.group(1))
            lines_by_task[tid].append(ln)
    return lines_by_task


def extract_program_name(cmd_formatted, ntasks):
    toks = shlex.split(cmd_formatted)
    ntstr = str(ntasks)
    prog = None
    for i, t in enumerate(toks):
        if t == ntstr and i + 1 < len(toks):
            cand = toks[i + 1]
            if not cand.startswith("-"):
                prog = os.path.basename(cand)
                break
    if not prog:
        for t in reversed(toks):
            if not t.startswith("-") and t != 'mpirun' and t != 'mpiexec':
                prog = os.path.basename(t)
                break
    if not prog:
        prog = 'run'
    return prog


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("program", help="Base program to run (e.g. md3d)")
    args = parser.parse_args()

    logs_dir = "logs"
    results_dir = "results"

    ensure_dir(logs_dir)
    ensure_dir(results_dir)

    sample_cmd = f"mpirun -n {NTASKS_LIST[0]} ./{args.program} -a {ALGORITHMS[0]}"
    inferred_prog = extract_program_name(sample_cmd, NTASKS_LIST[0])
    out_csv = os.path.join(results_dir, f"results_data_lb_{inferred_prog}.csv")

    csv_fields = [
        "ntasks",
        "prog_n",
        "algorithm",
        "task_id",
        "timestamp",
        "switches",
        "without_actions",
        "transitions",
        "data_alloc_count",
        "data_alloc_bytes",
        "msg_sent_count",
        "msg_sent_bytes",
        "recv_count",
        "recv_bytes",
        "reduce_count",
        "reduce_bytes",
        "returncode",
        "logfile",
    ]

    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()

        for nt in NTASKS_LIST:
            for prog_n in PROG_N_LIST:
                for alg in ALGORITHMS:
                    cmd_formatted = f"mpirun -n {nt} ./{args.program} -n {prog_n} -a {alg}"

                    print(f"\n[{datetime.now().isoformat()}] Running: LAIK_LOG=2 {cmd_formatted}")
                    env = dict(**subprocess.os.environ)
                    env["LAIK_LOG"] = "2"

                    try:
                        lines, returncode = run_and_capture(cmd_formatted, env=env)
                    except subprocess.TimeoutExpired:
                        print(f"Command timed out for ntasks={nt} prog_n={prog_n} alg={alg}")
                        lines = [f"TIMEOUT for ntasks={nt} prog_n={prog_n} alg={alg}"]
                        returncode = -1

                    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
                    progname = extract_program_name(cmd_formatted, nt)
                    safe_alg = alg.replace("/", "_")
                    logfile = os.path.join(logs_dir, f"{progname}_nt{nt}_pn{prog_n}_a{safe_alg}_{ts}.log")
                    with open(logfile, "w") as lf:
                        lf.write("\n".join(lines))

                    lines_by_task = parse_combined_output(lines)

                    # write one CSV row per task id
                    for tid, tlines in lines_by_task.items():
                        s = parse_task_summary(tlines)
                        row = {
                            "ntasks": nt,
                            "prog_n": prog_n,
                            "algorithm": alg,
                            "task_id": tid,
                            "timestamp": datetime.now().isoformat(),
                            "switches": "" if not s or s.get("switches") is None else s["switches"],
                            "without_actions": "" if not s or s.get("without_actions") is None else s["without_actions"],
                            "transitions": "" if not s or s.get("transitions") is None else s["transitions"],
                            "data_alloc_count": "" if not s or s.get("data_alloc_count") is None else s["data_alloc_count"],
                            "data_alloc_bytes": "" if not s or s.get("data_alloc_bytes") is None else int(s["data_alloc_bytes"]),
                            "msg_sent_count": "" if not s or s.get("msg_sent_count") is None else s["msg_sent_count"],
                            "msg_sent_bytes": "" if not s or s.get("msg_sent_bytes") is None else int(s["msg_sent_bytes"]),
                            "recv_count": "" if not s or s.get("recv_count") is None else s["recv_count"],
                            "recv_bytes": "" if not s or s.get("recv_bytes") is None else int(s["recv_bytes"]),
                            "reduce_count": "" if not s or s.get("reduce_count") is None else s["reduce_count"],
                            "reduce_bytes": "" if not s or s.get("reduce_bytes") is None else int(s["reduce_bytes"]),
                            "returncode": returncode,
                            "logfile": logfile,
                        }
                        writer.writerow(row)

                    # print a compact summary for the run
                    task_ids = sorted(lines_by_task.keys())
                    print(f" ntasks={nt} prog_n={prog_n} alg={alg}: found task_ids={task_ids} (rows written={len(task_ids)})")

                    A = {
                        "switches": 0,
                        "transitions": 0,
                        "data_alloc_bytes": 0,
                        "msg_sent_bytes": 0,
                        "recv_bytes": 0,
                        "reduce_bytes": 0,
                        "num_with_summary": 0,
                    }
                    for tid, tlines in lines_by_task.items():
                        if tid == -1:   # skip unlabeled lines
                            continue
                        s = parse_task_summary(tlines)
                        if not s:
                            continue
                        A["num_with_summary"] += 1
                        if s.get("switches") is not None:
                            A["switches"] += s["switches"]
                        if s.get("transitions") is not None:
                            A["transitions"] += s["transitions"]
                        for k in ["data_alloc_bytes", "msg_sent_bytes", "recv_bytes", "reduce_bytes"]:
                            if s.get(k) is not None:
                                A[k] += s[k]
                    print(f"  tasks_with_summary={A['num_with_summary']}/{len(task_ids)}")
                    print(f"  total switches={A['switches']}, transitions={A['transitions']}")
                    print(f"  data_alloc_bytes={int(A['data_alloc_bytes'])} B, msg_sent_bytes={int(A['msg_sent_bytes'])} B")
                    print(f"  recv_bytes={int(A['recv_bytes'])} B, reduce_bytes={int(A['reduce_bytes'])} B")
                    print(f"  raw log -> {logfile} (returncode={returncode})")

    print(f"\nWrote CSV -> {out_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
