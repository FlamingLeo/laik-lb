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

    # style extra vs main differently
    for (tasks, prog_n, algorithm, source), g in df.groupby(['tasks','prog_n','algorithm','source']):
        mean_time = float(g['time_s'].mean())

        # handle per-task arrays: some groups will have empty lists
        max_len = 0
        for x in g['per_task_times_list'].values:
            if len(x) > max_len:
                max_len = len(x)
        if max_len == 0:
            # create an empty array for consistency
            times_arrays = np.empty((len(g), 0))
            pct_arrays = np.empty((len(g), 0))
        else:
            times_arrays = np.vstack([np.pad(x, (0, max_len - len(x)), constant_values=np.nan) for x in g['per_task_times_list'].values])
            pct_arrays = np.vstack([np.pad(x, (0, max_len - len(x)), constant_values=np.nan) for x in g['per_task_pct_list'].values])

        mean_times_per_task = np.nanmean(times_arrays, axis=0) if times_arrays.size else np.array([])
        mean_pct_per_task = np.nanmean(pct_arrays, axis=0) if pct_arrays.size else np.array([])

        truncated = mean_times_per_task[:int(tasks)] if mean_times_per_task.size > 0 else np.array([])
        per_task_mean_scalar = float(np.nanmean(truncated)) if truncated.size > 0 and not np.all(np.isnan(truncated)) else float('nan')

        groups.append({
            'tasks': int(tasks), 'prog_n': int(prog_n), 'algorithm': algorithm, 'source': source,
            'mean_time_s': mean_time,
            'mean_times_per_task': mean_times_per_task,
            'mean_pct_per_task': mean_pct_per_task,
            'per_task_mean': per_task_mean_scalar
        })
    summary = pd.DataFrame(groups).sort_values(['tasks','prog_n','algorithm','source']).reset_index(drop=True)
    return summary


def plot_total_and_pertask(summary, out_dir):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12,5))

    algorithms = summary['algorithm'].unique()

    # style map per source
    style_map = {
        'main': {'linestyle': '-', 'marker': 'o', 'alpha': 0.95, 'linewidth': 1.8, 'markersize': 6},
        'extra': {'linestyle': '--', 'marker': 's', 'alpha': 1.0, 'linewidth': 2.2, 'markersize': 6}
    }

    for alg in algorithms:
        alg_summary = summary[summary['algorithm'] == alg]

        # get a color for this algorithm once and reuse for both sources
        color = None

        # iterate sources in deterministic order
        for source in ['main', 'extra']:
            sel_alg = alg_summary[alg_summary['source'] == source]
            if sel_alg.empty:
                continue

            # pivot total mean
            pivot_total = sel_alg.pivot(index='tasks', columns='prog_n', values='mean_time_s')
            if pivot_total.empty:
                continue
            tasks_sorted = list(pivot_total.index)
            vals_total = pivot_total.values
            min_vals_total = np.nanmin(vals_total, axis=1)
            max_vals_total = np.nanmax(vals_total, axis=1)
            median_vals_total = np.nanmedian(vals_total, axis=1)

            # choose color if unset
            if color is None:
                # draw a dummy invisible line to get the color cycle
                ln = ax.plot(tasks_sorted, median_vals_total, visible=False)
                color = ln[0].get_color()

            # shade for total mean
            alpha_shade = 0.10 if source == 'main' else 0.14
            ax.fill_between(tasks_sorted, min_vals_total, max_vals_total, alpha=alpha_shade, color=color)

            # plot total median line with style according to source
            st = style_map.get(source, style_map['main'])
            ax.plot(tasks_sorted, median_vals_total, marker=st['marker'], linestyle=st['linestyle'],
                    label=f'{alg} total (median)', linewidth=st['linewidth'],
                    markersize=st['markersize'], alpha=st['alpha'], color=color)

            # pivot per-task mean scalar
            pivot_per_task = sel_alg.pivot(index='tasks', columns='prog_n', values='per_task_mean')
            if pivot_per_task.empty or np.all(np.isnan(pivot_per_task.values)):
                # skip per-task plotting for this
                continue

            vals_pt = pivot_per_task.values
            min_vals_pt = np.nanmin(vals_pt, axis=1)
            max_vals_pt = np.nanmax(vals_pt, axis=1)
            median_vals_pt = np.nanmedian(vals_pt, axis=1)

            # shade and line for per-task mean
            alpha_shade_pt = 0.08 if source == 'main' else 0.10
            ax.fill_between(tasks_sorted, min_vals_pt, max_vals_pt, alpha=alpha_shade_pt, color=color)
            # use a different marker for per-task to make it distinct yet related
            per_task_marker = 'D' if source == 'main' else 'x'
            per_task_linestyle = (0, (5, 1)) if source == 'main' else (0, (1, 1))
            ax.plot(tasks_sorted, median_vals_pt, marker=per_task_marker, linestyle=per_task_linestyle,
                    label=f'{alg} eff. work mean per-task (median)',
                    linewidth=st['linewidth'] * 0.9, markersize=st['markersize'], alpha=st['alpha'], color=color)

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

    style_map = {
        'main': {'linestyle': '-', 'marker': 'o', 'alpha': 0.95, 'linewidth': 1.8, 'markersize': 6},
        'extra': {'linestyle': '--', 'marker': 's', 'alpha': 1.0, 'linewidth': 2.2, 'markersize': 6}
    }

    for alg in summary['algorithm'].unique():
        alg_summary = summary[summary['algorithm'] == alg]
        color = None
        for source in ['main', 'extra']:
            sel_alg = alg_summary[alg_summary['source'] == source]
            if sel_alg.empty:
                continue
            pivot = sel_alg.pivot(index='tasks', columns='prog_n', values='cv')
            if pivot.empty or np.all(np.isnan(pivot.values)):
                continue
            tasks_sorted = list(pivot.index)
            vals = pivot.values
            min_vals = np.nanmin(vals, axis=1)
            max_vals = np.nanmax(vals, axis=1)
            median_vals = np.nanmedian(vals, axis=1)

            # choose color if unset
            if color is None:
                ln = ax.plot(tasks_sorted, median_vals, visible=False)
                color = ln[0].get_color()

            st = style_map.get(source, style_map['main'])
            ax.fill_between(tasks_sorted, min_vals, max_vals, alpha=0.12 if source == 'main' else 0.14, color=color)
            ax.plot(tasks_sorted, median_vals, marker=st['marker'], linestyle=st['linestyle'],
                    label=f'{alg} (median work CV)', linewidth=st['linewidth'],
                    markersize=st['markersize'], alpha=st['alpha'], color=color)

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

    style_map = {
        'main': {'linestyle': '-', 'alpha': 0.85, 'linewidth': 1.6},
        'extra': {'linestyle': '--', 'alpha': 1.0, 'linewidth': 2.0}
    }

    for ax, alg in zip(axes, algos):
        sel_alg_all = summary[summary['algorithm'] == alg]
        tasks_sorted = np.sort(sel_alg_all['tasks'].unique())

        # plot each source with same color map over prog_n but different linestyle/linewidth
        for source in ['main', 'extra']:
            sel_alg = sel_alg_all[sel_alg_all['source'] == source]
            if sel_alg.empty:
                continue
            norm = mpl.colors.Normalize(vmin=prog_ns_all.min(), vmax=prog_ns_all.max())
            cmap = mpl.cm.get_cmap('winter')
            st = style_map.get(source, style_map['main'])
            for pn in prog_ns_all:
                row = sel_alg[sel_alg['prog_n'] == pn]
                if row.empty:
                    continue
                by_tasks = row.groupby('tasks')['mean_time_s'].median().reindex(tasks_sorted)
                color = cmap(norm(pn))
                ax.plot(tasks_sorted, by_tasks.values, color=color, linewidth=st['linewidth'], alpha=st['alpha'],
                        linestyle=st['linestyle'])

        # overall medians across prog_n (combining sources) drawn in black
        pivot = sel_alg_all.pivot(index='tasks', columns='prog_n', values='mean_time_s')
        medians = np.nanmedian(pivot.values, axis=1) if pivot.shape[0] > 0 else np.array([])
        if medians.size > 0:
            ax.plot(pivot.index, medians, color='black', linewidth=2.2, label='median')

        ax.set_ylabel('Mean eff. work time (s)')
        ax.set_title(f'Algorithm: {alg}')
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel('Number of tasks')

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=prog_ns_all.min(), vmax=prog_ns_all.max()))
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
    df_e['source'] = 'extra'

    # select/rename relevant columns to match main CSV layout
    df_e = df_e[['tasks','prog_n','algorithm','attempt','time_s','per_task_times_s','per_task_pct','source']]
    return df_e


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv')
    p.add_argument('--extra', help='Optional extra CSV (threads-based) to include', default=None)
    p.add_argument('--outdir', default='./output')
    p.add_argument('--show', action='store_true')
    args = p.parse_args()

    out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)

    # main CSV
    df = pd.read_csv(args.csv, dtype={'tasks': int, 'prog_n': int, 'algorithm': str, 'attempt': int})
    df['source'] = 'main'

    # omp CSV
    if args.extra:
        df_extra = load_extra_threads_csv(args.extra)
        
        # make sure dtypes match
        df_extra = df_extra.astype({
            'tasks': int, 'prog_n': int, 'algorithm': str, 'attempt': int,
            'time_s': float, 'per_task_times_s': str, 'per_task_pct': str, 'source': str
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
