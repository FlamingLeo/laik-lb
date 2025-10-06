#!/usr/bin/env python3
import argparse
import os
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def clean_df(df):
    df.columns = [c.strip().strip('*') for c in df.columns]
    if 'task' in df.columns:
        df['task'] = df['task'].astype(str).str.strip().str.lstrip('0').replace('', '0')
        df['task'] = pd.to_numeric(df['task'], errors='coerce').fillna(0).astype(int)
    for col in ['ntasks','freq','switches','msg_count','elems','bytes','avg_bytes','without_actions','transitions']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def make_total_switches_grid(df, outdir):
    agg = df.groupby(['ntasks','freq','algo'], as_index=False).agg(total_switches=('switches','sum'))
    ntasks_list = sorted(agg['ntasks'].unique())
    freq_list = sorted(agg['freq'].unique())
    algos = sorted(agg['algo'].unique())

    cmap = plt.get_cmap('tab10')
    color_map = {algo: cmap(i % 10) for i, algo in enumerate(algos)}

    nrows = max(1, len(ntasks_list))
    ncols = max(1, len(freq_list))
    fig_w = max(4*ncols, 6)
    fig_h = max(3*nrows, 4)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)

    for i, nt in enumerate(ntasks_list):
        for j, fr in enumerate(freq_list):
            ax = axes[i][j]
            subset = agg[(agg['ntasks']==nt) & (agg['freq']==fr)]
            heights = [subset[subset['algo']==a]['total_switches'].sum() if not subset[subset['algo']==a].empty else 0 for a in algos]
            x = np.arange(len(algos))
            bars = ax.bar(x, heights, color=[color_map[a] for a in algos])
            ax.set_xticks(x)
            ax.set_xticklabels(algos, rotation=45, ha='right')
            ax.set_ylabel("total switches")
            ax.set_title(f"ntasks={nt}, freq={fr}")
            for b in bars:
                h = b.get_height()
                if h > 0:
                    ax.text(b.get_x() + b.get_width()/2, h + max(1, 0.02*max(heights)), f"{int(h)}", ha='center', va='bottom', fontsize=8)

    handles = [plt.Rectangle((0,0),1,1, color=color_map[a]) for a in algos]
    fig.legend(handles, algos, loc='lower center', ncol=len(algos), bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0,0.05,1,1])
    fpath = os.path.join(outdir, "total_switches_grid_by_ntasks_freq.svg")
    fig.savefig(fpath, bbox_inches='tight')
    plt.close(fig)
    print("Saved:", fpath)

def make_bytes_comparisons(df, outdir, metric='bytes'):
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in data columns: {df.columns.tolist()}")

    agg = df.groupby(['ntasks','freq','algo'], as_index=False).agg(sum_metric=(metric,'sum'))
    pivot = agg.pivot_table(index=['ntasks','freq'], columns='algo', values='sum_metric', aggfunc='sum').fillna(0)
    group_labels = [f"n{int(idx[0])}-f{int(idx[1])}" for idx in pivot.index]
    algos = list(pivot.columns)
    n_groups = len(pivot)

    # grouped bar chart 
    plt.figure(figsize=(10,5))
    x = np.arange(n_groups)
    total_width = 0.8
    n_algos = len(algos)
    bar_width = total_width / max(1, n_algos)
    # color cycle from matplotlib
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)
    for i, algo in enumerate(algos):
        vals = pivot[algo].values
        positions = x - total_width/2 + i*bar_width + bar_width/2
        c = colors[i % len(colors)] if colors else None
        plt.bar(positions, vals, bar_width, label=algo, color=c)
        # annotate
        for xi, v in zip(positions, vals):
            if v > 0:
                plt.text(xi, v + max(1, 0.01*vals.max()), f"{int(v):,}", ha='center', va='bottom', fontsize=8)

    plt.xticks(x, group_labels, rotation=45, ha='right')
    plt.ylabel(f"SUM({metric})")
    plt.title(f"SUM({metric}) per (ntasks,freq) — grouped by algorithm")
    plt.legend(title="algo", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    f1 = os.path.join(outdir, f"{metric}_grouped_by_algo.svg")
    plt.savefig(f1, bbox_inches='tight')
    plt.close()
    print("Saved:", f1)

    # stacked bar chart
    plt.figure(figsize=(10,5))
    bottom = np.zeros(n_groups)
    for i, algo in enumerate(algos):
        vals = pivot[algo].values
        c = colors[i % len(colors)] if colors else None
        plt.bar(group_labels, vals, bottom=bottom, label=algo, color=c)
        bottom += vals
    for xi, tot in enumerate(bottom):
        if tot > 0:
            plt.text(xi, tot + max(1, 0.01*bottom.max()), f"{int(tot):,}", ha='center', va='bottom', fontsize=8)

    plt.ylabel(f"SUM({metric})")
    plt.title(f"SUM({metric}) per (ntasks,freq) — stacked by algorithm (composition)")
    plt.legend(title="algo", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    f2 = os.path.join(outdir, f"{metric}_stacked_by_algo.svg")
    plt.savefig(f2, bbox_inches='tight')
    plt.close()
    print("Saved:", f2)

    # proportional stacked bar chart (percentage)
    plt.figure(figsize=(10,5))
    row_sums = pivot.sum(axis=1).replace(0, np.nan)
    prop = pivot.div(row_sums, axis=0).fillna(0)
    bottom = np.zeros(n_groups)
    for i, algo in enumerate(algos):
        vals = prop[algo].values
        c = colors[i % len(colors)] if colors else None
        plt.bar(group_labels, vals, bottom=bottom, label=algo, color=c)
        bottom += vals

    # annotate percent for segments >= 5%
    for i, idx in enumerate(pivot.index):
        y = 0
        for algo in algos:
            val = prop.loc[idx, algo]
            if val >= 0.05:
                plt.text(i, y + val/2, f"{val*100:.0f}%", ha='center', va='center', fontsize=8, color='white')
            y += val

    plt.ylabel(f"Proportion of SUM({metric})")
    plt.title(f"Proportion of {metric} per algorithm within each (ntasks,freq) group")
    plt.legend(title="algo", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    f3 = os.path.join(outdir, f"{metric}_proportion_by_algo.svg")
    plt.savefig(f3, bbox_inches='tight')
    plt.close()
    print("Saved:", f3)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--outdir", default="output")
    p.add_argument("--metric", choices=['bytes','elems'], default='bytes')
    p.add_argument("--only-bytes-compare", action='store_true')
    args = p.parse_args()

    if not os.path.isfile(args.csv):
        raise SystemExit(f"CSV file not found: {args.csv}")

    os.makedirs(args.outdir, exist_ok=True)

    # read CSV
    df = pd.read_csv(args.csv)
    df = clean_df(df)

    # basic checks
    required = {'ntasks','freq','algo'}
    if not required.issubset(set(df.columns)):
        raise SystemExit(f"CSV must contain columns: {required}. Found: {df.columns.tolist()}")

    # plot
    if not args.only_bytes_compare:
        if 'switches' not in df.columns:
            print("Warning: 'switches' column missing; skipping switches grid.")
        else:
            make_total_switches_grid(df, args.outdir)

    # metric comparisons (bytes / elems)
    if args.metric not in df.columns:
        print(f"Warning: metric '{args.metric}' not found in CSV; skipping metric comparisons.")
    else:
        make_bytes_comparisons(df, args.outdir, metric=args.metric)

    print("All done. Output directory:", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()
