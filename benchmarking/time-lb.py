#!/usr/bin/env python3
import argparse
import os
import re
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
            # idx might contain non-digits if malformed
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


def build_summary(df):
    groups = []
    for (tasks, prog_n, algorithm), g in df.groupby(['tasks','prog_n','algorithm']):
        mean_time = float(g['time_s'].mean())
        max_len = max(len(x) for x in g['per_task_times_list'].values)
        times_arrays = np.vstack([np.pad(x, (0,max_len-len(x)), constant_values=np.nan) for x in g['per_task_times_list'].values])
        pct_arrays = np.vstack([np.pad(x, (0,max_len-len(x)), constant_values=np.nan) for x in g['per_task_pct_list'].values])
        mean_times_per_task = np.nanmean(times_arrays, axis=0)
        mean_pct_per_task = np.nanmean(pct_arrays, axis=0)
        truncated = mean_times_per_task[:int(tasks)] if len(mean_times_per_task)>0 else np.array([])
        per_task_mean_scalar = float(np.nanmean(truncated)) if truncated.size>0 else float('nan')
        groups.append({
            'tasks': int(tasks), 'prog_n': int(prog_n), 'algorithm': algorithm,
            'mean_time_s': mean_time,
            'mean_times_per_task': mean_times_per_task,
            'mean_pct_per_task': mean_pct_per_task,
            'per_task_mean': per_task_mean_scalar
        })
    summary = pd.DataFrame(groups).sort_values(['tasks','prog_n','algorithm']).reset_index(drop=True)
    return summary


def plot_total_and_pertask(summary, out_dir):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8,5))

    algs = summary['algorithm'].unique()
    for alg in algs:
        sel_alg = summary[summary['algorithm']==alg]

        # pivot total mean
        pivot_total = sel_alg.pivot(index='tasks', columns='prog_n', values='mean_time_s')
        tasks_sorted = list(pivot_total.index)
        vals_total = pivot_total.values
        min_vals_total = np.nanmin(vals_total, axis=1)
        max_vals_total = np.nanmax(vals_total, axis=1)
        median_vals_total = np.nanmedian(vals_total, axis=1)

        # shade for total mean
        ax.fill_between(tasks_sorted, min_vals_total, max_vals_total, alpha=0.10)
        ax.plot(tasks_sorted, median_vals_total, marker='o', linestyle='-', label=f'{alg} total (median across prog_n)')

        # pivot per-task mean scalar
        pivot_per_task = sel_alg.pivot(index='tasks', columns='prog_n', values='per_task_mean')
        vals_pt = pivot_per_task.values
        min_vals_pt = np.nanmin(vals_pt, axis=1)
        max_vals_pt = np.nanmax(vals_pt, axis=1)
        median_vals_pt = np.nanmedian(vals_pt, axis=1)

        # shade for per-task mean
        ax.fill_between(tasks_sorted, min_vals_pt, max_vals_pt, alpha=0.08)
        ax.plot(tasks_sorted, median_vals_pt, marker='s', linestyle='--', label=f'{alg} mean per-task (median)')

    ax.set_xlabel('Number of tasks')
    ax.set_ylabel('Seconds (total and mean per-task)')
    #ax.set_title('Mean total time and mean per-task time vs tasks')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, 'total_and_pertask_combined.svg')
    fig.savefig(path)
    print('Saved:', path)


def plot_cv(summary, out_dir):
    # per-task CV computed from mean_times_per_task array
    summary = summary.copy()
    summary['per_task_std'] = summary['mean_times_per_task'].apply(lambda arr: float(np.nanstd(arr)) if len(arr)>0 else float('nan'))
    summary['per_task_mean'] = summary['mean_times_per_task'].apply(lambda arr: float(np.nanmean(arr)) if len(arr)>0 else float('nan'))
    summary['cv'] = summary['per_task_std'] / summary['per_task_mean']

    fig, ax = plt.subplots(figsize=(8,5))
    for alg in summary['algorithm'].unique():
        sel_alg = summary[summary['algorithm']==alg]
        pivot = sel_alg.pivot(index='tasks', columns='prog_n', values='cv')
        tasks_sorted = list(pivot.index)
        vals = pivot.values
        min_vals = np.nanmin(vals, axis=1)
        max_vals = np.nanmax(vals, axis=1)
        median_vals = np.nanmedian(vals, axis=1)
        ax.fill_between(tasks_sorted, min_vals, max_vals, alpha=0.12)
        ax.plot(tasks_sorted, median_vals, marker='o', label=f'{alg} (median CV)')

    ax.set_xlabel('Number of tasks')
    ax.set_ylabel('Per-task CV (std/mean)')
    #ax.set_title('Per-task imbalance (CV) vs tasks')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, 'per_task_cv_combined.svg')
    fig.savefig(path)
    print('Saved:', path)


def plot_small_multiples(summary, out_dir):
    algos = sorted(summary['algorithm'].unique())
    prog_ns = sorted(summary['prog_n'].unique())
    rows = len(algos)
    cols = len(prog_ns)
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), sharey='row')
    if rows==1 and cols==1:
        axes = np.array([[axes]])
    elif rows==1:
        axes = axes.reshape(1,-1)
    elif cols==1:
        axes = axes.reshape(-1,1)

    for i, alg in enumerate(algos):
        for j, pn in enumerate(prog_ns):
            ax = axes[i,j]
            row = summary[(summary['algorithm']==alg)&(summary['prog_n']==pn)]
            if row.empty:
                ax.set_visible(False)
                continue
            arr = np.array(row.iloc[0]['mean_times_per_task'])
            n_tasks = int(row.iloc[0]['tasks'])
            arr = arr[:n_tasks]
            x = np.arange(n_tasks)
            ax.bar(x, arr)
            ax.set_xticks(x)
            ax.set_xticklabels([str(t) for t in x], rotation=0)
            #ax.set_title(f'{alg}  prog_n={pn}  tasks={n_tasks}')
            ax.grid(axis='y', alpha=0.4)
            if j==0:
                ax.set_ylabel('Mean work time (s)')

    fig.tight_layout()
    path = os.path.join(out_dir, 'per_task_small_multiples.svg')
    fig.savefig(path)
    print('Saved:', path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv')
    p.add_argument('--outdir', default='./output')
    p.add_argument('--show', action='store_true')
    args = p.parse_args()

    out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(args.csv, dtype={'tasks': int, 'prog_n': int, 'algorithm': str, 'attempt': int})
    df['per_task_times_list'] = df['per_task_times_s'].apply(parse_indexed_list)
    df['per_task_pct_list'] = df['per_task_pct'].apply(parse_indexed_list)

    summary = build_summary(df)

    plot_total_and_pertask(summary, out_dir)
    plot_cv(summary, out_dir)
    plot_small_multiples(summary, out_dir)

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
