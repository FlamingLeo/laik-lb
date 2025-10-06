#!/usr/bin/env python3
import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
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
    fig, ax = plt.subplots(figsize=(12,5))

    algs = summary['algorithm'].unique()
    for alg in algs:
        sel_alg = summary[summary['algorithm']==alg]

        # pivot total mean
        pivot_total = sel_alg.pivot(index='tasks', columns='prog_n', values='mean_time_s')
        if pivot_total.empty:
            continue
        tasks_sorted = list(pivot_total.index)
        vals_total = pivot_total.values
        min_vals_total = np.nanmin(vals_total, axis=1)
        max_vals_total = np.nanmax(vals_total, axis=1)
        median_vals_total = np.nanmedian(vals_total, axis=1)

        # shade for total mean
        ax.fill_between(tasks_sorted, min_vals_total, max_vals_total, alpha=0.10)
        ax.plot(tasks_sorted, median_vals_total, marker='o', linestyle='-', label=f'{alg} total (median)')

        # pivot per-task mean scalar
        pivot_per_task = sel_alg.pivot(index='tasks', columns='prog_n', values='per_task_mean')

        if pivot_per_task.empty or np.all(np.isnan(pivot_per_task.values)):
            # nothing to plot for per-task for this algorithm
            continue

        vals_pt = pivot_per_task.values
        min_vals_pt = np.nanmin(vals_pt, axis=1)
        max_vals_pt = np.nanmax(vals_pt, axis=1)
        median_vals_pt = np.nanmedian(vals_pt, axis=1)

        # shade for per-task mean
        ax.fill_between(tasks_sorted, min_vals_pt, max_vals_pt, alpha=0.08)
        ax.plot(tasks_sorted, median_vals_pt, marker='s', linestyle='--', label=f'{alg} eff. work mean per-task (median)')

    ax.set_xlabel('Number of tasks')
    ax.set_ylabel('Seconds (total and mean per-task)')
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0)
    fig.subplots_adjust(right=0.78)
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

    fig, ax = plt.subplots(figsize=(12,5))
    for alg in summary['algorithm'].unique():
        sel_alg = summary[summary['algorithm']==alg]
        pivot = sel_alg.pivot(index='tasks', columns='prog_n', values='cv')

        # if there's no per-task CV data (empty or all-NaN), skip plotting this algorithm entirely
        if pivot.empty or np.all(np.isnan(pivot.values)):
            continue

        tasks_sorted = list(pivot.index)
        vals = pivot.values
        min_vals = np.nanmin(vals, axis=1)
        max_vals = np.nanmax(vals, axis=1)
        median_vals = np.nanmedian(vals, axis=1)
        ax.fill_between(tasks_sorted, min_vals, max_vals, alpha=0.12)
        ax.plot(tasks_sorted, median_vals, marker='o', label=f'{alg} (median work CV)')

    ax.set_xlabel('Number of tasks')
    ax.set_ylabel('Eff. work per-task CV (std/mean)')
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0)
    fig.subplots_adjust(right=0.78)
    fig.tight_layout()
    path = os.path.join(out_dir, 'per_task_cv_combined.svg')
    fig.savefig(path)
    print('Saved:', path)


def plot_small_multiples(summary, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    algos = list(summary['algorithm'].unique())
    prog_ns_all = np.sort(summary['prog_n'].unique())
    norm = mpl.colors.Normalize(vmin=prog_ns_all.min(), vmax=prog_ns_all.max())
    cmap = mpl.cm.get_cmap('winter')

    rows = len(algos)
    fig, axes = plt.subplots(rows, 1, figsize=(10, 3*rows), sharex=True)
    if rows == 1:
        axes = [axes]

    for ax, alg in zip(axes, algos):
        sel_alg = summary[summary['algorithm'] == alg]
        tasks_sorted = np.sort(sel_alg['tasks'].unique())

        for pn in prog_ns_all:
            row = sel_alg[sel_alg['prog_n'] == pn]
            if row.empty:
                continue
            by_tasks = row.groupby('tasks')['mean_time_s'].median().reindex(tasks_sorted)
            ax.plot(tasks_sorted, by_tasks.values, color=cmap(norm(pn)), linewidth=1.2, alpha=0.7)

        pivot = sel_alg.pivot(index='tasks', columns='prog_n', values='mean_time_s')
        medians = np.nanmedian(pivot.values, axis=1) if pivot.shape[0] > 0 else np.array([])
        if medians.size > 0:
            ax.plot(pivot.index, medians, color='black', linewidth=2.2, label='median')

        ax.set_ylabel('Mean eff. work time (s)')
        ax.set_title(f'Algorithm: {alg}')
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel('Number of tasks')

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(np.linspace(prog_ns_all.min(), prog_ns_all.max(), 10))

    fig.subplots_adjust(right=0.88)

    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', pad=0.02, fraction=0.02)
    cbar.set_label('load balance frequency')

    path = os.path.join(out_dir, 'tasks_algorithm_combined_lines.svg')
    fig.savefig(path, bbox_inches='tight')
    print('Saved:', path)
    plt.close(fig)


def load_extra_threads_csv(path):
    df_e = pd.read_csv(path)

    # keep only rows that have a numeric threads/time_s
    df_e = df_e[~df_e['threads'].isna() & ~df_e['time_s'].isna()].copy()
    df_e['tasks'] = df_e['threads'].astype(int)
    df_e['prog_n'] = 0
    df_e['algorithm'] = df_e['variant'].astype(str)
    df_e['attempt'] = 1

    # create the expected columns for parsing
    df_e['per_task_times_s'] = ''
    df_e['per_task_pct'] = ''

    # select/rename relevant columns to match main CSV layout
    df_e = df_e[['tasks','prog_n','algorithm','attempt','time_s','per_task_times_s','per_task_pct']]
    return df_e


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv')
    p.add_argument('--extra', default=None)
    p.add_argument('--outdir', default='./output')
    p.add_argument('--show', action='store_true')
    args = p.parse_args()

    out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)

    # main CSV
    df = pd.read_csv(args.csv, dtype={'tasks': int, 'prog_n': int, 'algorithm': str, 'attempt': int})

    # OMP CSV
    if args.extra:
        df_extra = load_extra_threads_csv(args.extra)
        # make sure dtypes match
        df_extra = df_extra.astype({
            'tasks': int, 'prog_n': int, 'algorithm': str, 'attempt': int,
            'time_s': float, 'per_task_times_s': str, 'per_task_pct': str
        })
        df = pd.concat([df, df_extra], ignore_index=True, sort=False)
        print(f'Appended {len(df_extra)} rows from extra CSV: {args.extra}')

    # parse the indexed-list fields into Python lists
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
