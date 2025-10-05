#!/usr/bin/env python3
# usage: python blocks.py --input-dir json --block_size 100 --out prof.svg

import argparse
import json
import os
import glob
import sys
from collections import defaultdict
import hashlib
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

ITER_MARKER_PREFIX = "__iter__:"

def load_events_from_file(path):
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a top-level JSON array in {path}")
    events = []
    for rec in data:
        try:
            name = rec["name"]
            start = float(rec["start"])
            end = float(rec["end"])
            track = int(rec.get("track", rec.get("tid", 0)))
        except Exception as e:
            raise ValueError(f"Invalid event record in {path}: {rec}") from e
        events.append({"name": name, "start": start, "end": end, "track": track})
    return events

def load_events_from_path(path, dedup=False):
    events = []
    files = []
    if os.path.isdir(path):
        # find json files (case-insensitive) and sort them
        patterns = [os.path.join(path, "*.json"), os.path.join(path, "*.JSON")]
        for pat in patterns:
            files.extend(glob.glob(pat))
        files = sorted(set(files))
        if not files:
            raise FileNotFoundError(f"No JSON files found in directory: {path}")
    elif os.path.isfile(path):
        files = [path]
    else:
        raise FileNotFoundError(f"Path not found: {path}")

    for fpath in files:
        loaded = load_events_from_file(fpath)
        events.extend(loaded)

    if dedup:
        seen = set()
        unique = []
        for e in events:
            key = (e["name"], round(float(e["start"]), 9), round(float(e["end"]), 9), int(e["track"]))
            # round timestamps to avoid floating scatter creating distinct keys
            if key not in seen:
                seen.add(key)
                unique.append(e)
        events = unique

    return events

def assign_events_to_iterations(events):
    # for each track, process events in ascending start time
    # when encountering a marker name "__iter__:N", set current_iter for that track to N
    # otherwise assign that event to the current_iter for that track (default 0 if none yet)
    events_sorted = sorted(events, key=lambda e: e["start"])
    per_track_iter_events = defaultdict(lambda: defaultdict(list))
    functions_set = set()
    current_iter_for_track = defaultdict(lambda: 0)
    max_iter = 0

    for e in events_sorted:
        name = e["name"]
        track = e["track"]
        if name.startswith(ITER_MARKER_PREFIX):
            # parse iter number if possible
            try:
                iter_num = int(name[len(ITER_MARKER_PREFIX):])
            except ValueError:
                # ignore malformed markers
                continue
            current_iter_for_track[track] = iter_num
            if iter_num > max_iter:
                max_iter = iter_num
            # store the marker itself (optional) - we don't need to accumulate its time
            per_track_iter_events[track][iter_num].append({"marker": True, "name": name, "start": e["start"], "end": e["end"]})
        else:
            iter_num = current_iter_for_track[track]
            per_track_iter_events[track][iter_num].append(e)
            functions_set.add(name)

    return per_track_iter_events, sorted(functions_set), max_iter

def build_times_array(per_track_iter_events, functions, max_iter):
    # identify track ids and map them to 0..n_tasks-1
    tracks = sorted(per_track_iter_events.keys())
    track_to_idx = {t: i for i, t in enumerate(tracks)}
    n_tasks = len(tracks)
    n_iters = max_iter + 1  # iterations assumed 0..max_iter inclusive
    n_funcs = len(functions)
    func_to_idx = {fn: i for i, fn in enumerate(functions)}

    times = np.zeros((n_tasks, n_iters, n_funcs), dtype=float)

    for track, iter_map in per_track_iter_events.items():
        t_idx = track_to_idx[track]
        for iter_num, evlist in iter_map.items():
            if iter_num < 0:
                continue
            if iter_num >= n_iters:
                # should not happen, but guard
                continue
            for e in evlist:
                if e.get("marker"):
                    continue
                name = e["name"]
                if name not in func_to_idx:
                    # skip unknown (shouldn't happen)
                    continue
                f_idx = func_to_idx[name]
                duration = max(0.0, float(e["end"]) - float(e["start"]))
                times[t_idx, iter_num, f_idx] += duration

    return times, tracks


def name_to_color_hex(name: str, saturation: float = 0.6, lightness: float = 0.5) -> str:
    # stable integer from name
    hsh = hashlib.sha1(name.encode("utf-8")).hexdigest()
    # take a slice and map into 0..359
    hue = int(hsh[:8], 16) % 360
    h = hue / 360.0
    r, g, b = colorsys.hls_to_rgb(h, lightness, saturation)
    return mcolors.to_hex((r, g, b))


def plot_blocks(times, function_names, block_size=100, out_path="blocks.svg"):
    n_tasks, n_iters, n_funcs = times.shape
    n_blocks = (n_iters + block_size - 1) // block_size

    # aggregate per block
    block_sums = np.zeros((n_tasks, n_blocks, n_funcs), dtype=float)
    block_totals = np.zeros((n_tasks, n_blocks), dtype=float)
    for t in range(n_tasks):
        for b in range(n_blocks):
            start = b * block_size
            end = min((b + 1) * block_size, n_iters)
            block_sums[t, b, :] = times[t, start:end, :].sum(axis=0)
            block_totals[t, b] = block_sums[t, b, :].sum()

    # consistent ordering across all tasks & blocks:
    # determine ordering from first block summed across all tasks (largest-first)
    first_block_all_tasks = block_sums[:, 0, :].sum(axis=0)
    order = np.argsort(-first_block_all_tasks)  # largest-first

    # create deterministic colors per function name (stable across inputs)
    func_colors = {i: name_to_color_hex(function_names[i]) for i in range(n_funcs)}

    # plotting
    fig, ax = plt.subplots(figsize=(12, 3 + n_tasks * 0.45))
    task_height = 0.8
    y_gap = 0.4
    yticks, yticklabels = [], []
    max_right = 0.0

    for task in range(n_tasks):
        y_bottom_lane = (n_tasks - 1 - task) * (task_height + y_gap)
        yticks.append(y_bottom_lane + task_height/2)
        yticklabels.append(f"{task}")

        x_offset = 0.0
        for b in range(n_blocks):
            width = block_totals[task, b]
            if width <= 0:
                width = 1e-12
            heights = (block_sums[task, b, order] / (block_totals[task, b] + 1e-30)) * task_height
            y_current = y_bottom_lane
            for idx_pos, func_idx in enumerate(order):
                h = heights[idx_pos]
                if h > 1e-12 and width > 0:
                    rect = Rectangle((x_offset, y_current), width, h, facecolor=func_colors[func_idx], edgecolor='none')
                    ax.add_patch(rect)
                y_current += h
            # vertical separator
            ax.add_line(plt.Line2D([x_offset + width, x_offset + width], [y_bottom_lane, y_bottom_lane + task_height], color='black', linewidth=0.3)) # type: ignore
            x_offset += width
        max_right = max(max_right, x_offset)

    # legend
    legend_order = order
    legend_patches = [Rectangle((0,0),1,1, facecolor=func_colors[i]) for i in legend_order]
    legend_labels = [function_names[i] for i in legend_order]
    ax.legend(legend_patches, legend_labels, bbox_to_anchor=(1.01, 1.0), loc='upper left', fontsize=8)

    ax.set_xlim(0, max_right * 1.01 if max_right > 0 else 1)
    ax.set_ylim(-0.2, n_tasks * (task_height + y_gap))
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("Time Offset (s)")
    ax.set_ylabel("Task", rotation=90, labelpad=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to: {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot profiling events (JSON) as stacked iteration blocks.")
    parser.add_argument("--input", "-i", help="Profiler JSON input file (single file) OR path to directory with many .json files (use --input-dir instead for clarity)")
    parser.add_argument("--input-dir", default='./json', help="Directory containing JSON files (alternative to --input)")
    parser.add_argument("--dedup", action="store_true", help="Deduplicate identical events across files")
    parser.add_argument("--block_size", type=int, default=100, help="Iterations per block")
    parser.add_argument("--out", default="blocks.svg", help="Output image path")
    args = parser.parse_args()

    path = None
    if args.input_dir:
        path = args.input_dir
    elif args.input:
        path = args.input
    else:
        print("Error: --input or --input-dir required for json mode", file=sys.stderr)
        sys.exit(2)

    events = load_events_from_path(path, dedup=args.dedup)
    per_track_iter_events, function_names, max_iter = assign_events_to_iterations(events)
    times, tracks = build_times_array(per_track_iter_events, function_names, max_iter)

    plot_blocks(times, function_names, block_size=args.block_size, out_path=args.out)
