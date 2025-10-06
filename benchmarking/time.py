#!/usr/bin/env python3
import argparse
import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

number_re = re.compile(r"(-?\d+(?:\.\d+)?)")


def parse_indexed_list(s):
    parts = [p for p in str(s).split(';') if p.strip()!='']
    d = {}
    for p in parts:
        if ':' in p:
            idx_part, val_part = p.split(':', 1)
            try:
                m_idx = re.search(r"\d+", idx_part)
                if not m_idx:
                    continue
                idx = int(m_idx.group(0))
            except Exception:
                continue
            m = number_re.search(val_part)
            if m:
                try:
                    d[idx] = float(m.group(1))
                except Exception:
                    d[idx] = float('nan')
            else:
                d[idx] = float('nan')
    if not d:
        return []
    return [d[i] for i in sorted(d.keys())]


def build_groups(df):
    # expect multiple rows per tasks (different attempts)
    # group by tasks
    groups = []
    for tasks, g in df.groupby('tasks'):
        # collect per-attempt totals and per-task arrays
        totals = g['time_s'].astype(float).values
        per_task_lists = list(g['per_task_times_list'].values)

        # pad arrays to same length
        max_len = 0
        for arr in per_task_lists:
            if arr is None:
                continue
            max_len = max(max_len, len(arr))
        if max_len==0:
            mean_times_per_task = np.array([])
        else:
            mats = np.vstack([np.pad(arr, (0, max_len-len(arr)), constant_values=np.nan) for arr in per_task_lists])
            mean_times_per_task = np.nanmean(mats, axis=0)

        # per-attempt scalar mean of per-task (truncated to declared tasks)
        per_attempt_pertask_mean = []
        per_attempt_cv = []
        for arr in per_task_lists:
            a = np.array(arr)
            if a.size==0:
                per_attempt_pertask_mean.append(np.nan)
                per_attempt_cv.append(np.nan)
                continue
            truncated = a[:int(tasks)] if a.size>=int(tasks) else a
            per_attempt_pertask_mean.append(float(np.nanmean(truncated)))

            # per-attempt CV (std/mean) across tasks in that attempt
            mean_a = float(np.nanmean(truncated)) if truncated.size>0 else float('nan')
            std_a = float(np.nanstd(truncated)) if truncated.size>0 else float('nan')
            per_attempt_cv.append(std_a / mean_a if mean_a!=0 and not math.isnan(mean_a) else float('nan'))

        groups.append({
            'tasks': int(tasks),
            'total_times': np.array(totals),
            'per_task_mean_array': mean_times_per_task,
            'per_attempt_pertask_mean': np.array(per_attempt_pertask_mean),
            'per_attempt_cv': np.array(per_attempt_cv)
        })
    summary = pd.DataFrame(groups).sort_values('tasks').reset_index(drop=True)
    return summary


def plot_total_and_pertask(summary, out_dir):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12,5))
    tasks = summary['tasks'].values

    total_medians = []
    pertask_medians = []
    total_mins = []
    total_maxs = []
    pertask_mins = []
    pertask_maxs = []

    for _, row in summary.iterrows():
        totals = row['total_times']
        pertask_scalars = row['per_attempt_pertask_mean']
        total_medians.append(np.nanmedian(totals))
        total_mins.append(np.nanmin(totals))
        total_maxs.append(np.nanmax(totals))
        pertask_medians.append(np.nanmedian(pertask_scalars))
        pertask_mins.append(np.nanmin(pertask_scalars))
        pertask_maxs.append(np.nanmax(pertask_scalars))

    ax.fill_between(tasks, total_mins, total_maxs, alpha=0.10)
    ax.plot(tasks, total_medians, marker='o', linestyle='-', label='total time (median)')

    ax.fill_between(tasks, pertask_mins, pertask_maxs, alpha=0.08)
    ax.plot(tasks, pertask_medians, marker='s', linestyle='--', label='eff. work mean per-task time (median)')

    ax.set_xlabel('Number of tasks')
    ax.set_ylabel('Seconds (total and mean per-task)')
    #ax.set_title('Mean total time and mean per-task time vs tasks')
    ax.grid(True)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    fig.tight_layout()
    path = os.path.join(out_dir, 'total_and_pertask_combined.svg')
    fig.savefig(path)
    print('Saved:', path)


def plot_cv(summary, out_dir):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12,5))
    tasks = summary['tasks'].values

    cv_medians = []
    cv_mins = []
    cv_maxs = []
    for _, row in summary.iterrows():
        cvs = row['per_attempt_cv']
        cv_medians.append(np.nanmedian(cvs))
        cv_mins.append(np.nanmin(cvs))
        cv_maxs.append(np.nanmax(cvs))

    ax.fill_between(tasks, cv_mins, cv_maxs, alpha=0.12)
    ax.plot(tasks, cv_medians, marker='o', linestyle='-', label='per-task work CV (median)')

    ax.set_xlabel('Number of tasks')
    ax.set_ylabel('Eff. work per-task CV (std/mean)')
    #ax.set_title('Per-task imbalance (CV) vs tasks')
    ax.grid(True)
    ax.legend(loc='lower right', frameon=True)
    fig.tight_layout()
    path = os.path.join(out_dir, 'per_task_cv_combined.svg')
    fig.savefig(path)
    print('Saved:', path)


def plot_small_multiples(summary, out_dir):
    plt.close('all')
    tasks_vals = summary['tasks'].tolist()
    n = len(tasks_vals)
    cols = min(3, n) if n>0 else 1
    rows = math.ceil(n/cols) if cols>0 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), squeeze=False)

    for idx, (_, row) in enumerate(summary.iterrows()):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        arr = np.array(row['per_task_mean_array'])
        n_tasks = int(row['tasks'])
        arr = arr[:n_tasks]
        x = np.arange(len(arr))
        ax.bar(x, arr)
        ax.set_xticks(x[::max(1, len(x)//10)])
        ax.set_xticklabels([str(t) for t in x[::max(1, len(x)//10)]], rotation=45, ha='right', fontsize=8)
        #ax.set_title(f'tasks={n_tasks}')
        ax.grid(axis='y', alpha=0.4)
        if c==0:
            ax.set_ylabel('Mean work time (s)')

    # hide unused axes
    for idx2 in range(n, rows*cols):
        r = idx2 // cols
        c = idx2 % cols
        axes[r][c].set_visible(False)

    fig.tight_layout()
    path = os.path.join(out_dir, 'per_task_small_multiples.svg')
    fig.savefig(path)
    print('Saved:', path)


def main():
    p = argparse.ArgumentParser(description='Plot non-LB CSV results with combined per-task and total lines.')
    p.add_argument('csv', help='Input CSV file (non-LB format)')
    p.add_argument('--outdir', default='./output', help='Output directory for PNGs')
    p.add_argument('--show', action='store_true', help='Show plots interactively')
    args = p.parse_args()

    out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    # ensure tasks is int and time_s numeric
    df['tasks'] = df['tasks'].astype(int)
    df['time_s'] = df['time_s'].astype(float)

    df['per_task_times_list'] = df['per_task_times_s'].apply(parse_indexed_list)
    df['per_task_pct_list'] = df['per_task_pct'].apply(parse_indexed_list)

    summary = build_groups(df)

    plot_total_and_pertask(summary, out_dir)
    plot_cv(summary, out_dir)
    plot_small_multiples(summary, out_dir)

    if args.show:
        plt.show()

if __name__ == '__main__':
    main()
