#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import warnings

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

warnings.filterwarnings("ignore", category=UserWarning)

ALGO_ORDER = ['no_lb', 'hilbert', 'gilbert', 'rcb', 'rcbincr']
NUMERIC_COLS = ["ntasks", "freq", "total_time_s", "task_id", 
                "work_time_s", "switch_time_s", "load_balancer_time_s"]


def darken(color, factor=0.75):
    rgb = np.array(mcolors.to_rgb(color)) * factor
    return tuple(rgb.clip(0, 1))


def load_data(lb_path, nolb_path=None):
    df_lb = pd.read_csv(lb_path)
    df_lb.columns = df_lb.columns.str.strip()
    
    # cvt numeric columns
    for col in NUMERIC_COLS:
        if col in df_lb.columns:
            df_lb[col] = pd.to_numeric(df_lb[col], errors="coerce")
    
    if 'algo' not in df_lb.columns:
        df_lb['algo'] = 'lb'
    
    # load no lb data if provided
    if nolb_path:
        df_nolb = pd.read_csv(nolb_path)
        df_nolb.columns = df_nolb.columns.str.strip()
        
        for col in NUMERIC_COLS:
            if col in df_nolb.columns:
                df_nolb[col] = pd.to_numeric(df_nolb[col], errors="coerce")
        
        if 'load_balancer_time_s' not in df_nolb.columns:
            df_nolb['load_balancer_time_s'] = 0.0
        if 'algo' not in df_nolb.columns:
            df_nolb['algo'] = 'no_lb'
        
        # duplicate data across frequencies
        if 'freq' not in df_nolb.columns:
            freqs = df_lb['freq'].dropna().unique()
            df_nolb = pd.concat([df_nolb.assign(freq=f) for f in freqs], ignore_index=True)
            df_nolb['freq'] = pd.to_numeric(df_nolb['freq'], errors='coerce')
        
        df = pd.concat([df_lb, df_nolb], ignore_index=True)
    else:
        df = df_lb
    
    # fill missing task_ids
    if 'task_id' not in df.columns:
        df['task_id'] = np.nan
    
    for (nt, algo, freq), group in df[df['task_id'].isna()].groupby(['ntasks', 'algo', 'freq']):
        try:
            n = int(nt)
            df.loc[group.index, 'task_id'] = range(min(n, len(group)))
        except:
            pass
    
    return df


def get_total_time(group):
    if 'total_time_s' in group.columns and group['total_time_s'].notna().any():
        return float(group['total_time_s'].dropna().iloc[0])
    
    time_sum = (group['work_time_s'].fillna(0) + 
                group['switch_time_s'].fillna(0) + 
                group['load_balancer_time_s'].fillna(0))
    return float(time_sum.max()) if len(time_sum) > 0 else 0.0


def plot_total_time(df, algos, colors, out_dir):
    grouped = df.groupby(['ntasks', 'freq', 'algo']).apply(get_total_time).reset_index(name='total_time_s')
    
    for freq in sorted(grouped['freq'].unique()):
        subset = grouped[grouped['freq'] == freq]
        pivot = subset.pivot(index='ntasks', columns='algo', values='total_time_s')
        
        # ensure all algorithms have columns
        for a in algos:
            if a not in pivot.columns:
                pivot[a] = 0.0
        pivot = pivot[algos].fillna(0).sort_index()
        
        ntasks = list(pivot.index)
        x = np.arange(len(ntasks))
        width = 0.8 / len(algos)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, algo in enumerate(algos):
            offset = (i - (len(algos) - 1) / 2) * width
            ax.bar(x + offset, pivot[algo], width=width, label=algo, color=colors[algo])
        
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(n)) for n in ntasks])
        ax.set_xlabel("Number of Processes")
        ax.set_ylabel("Time (s)")
        ax.legend(title="Algorithm", bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(axis='y', linestyle=':', linewidth=0.5)
        
        outpath = out_dir / f"total_time_freq_{int(freq)}.pdf"
        fig.tight_layout()
        fig.savefig(outpath, bbox_inches='tight')
        plt.close(fig)


def plot_per_process(df, algos, colors, out_dir):
    for freq in sorted(df['freq'].dropna().unique()):
        subset = df[df['freq'] == freq]
        ntasks_list = sorted(subset['ntasks'].dropna().unique())
        if not ntasks_list:
            continue
        
        ncols = min(3, len(ntasks_list))
        nrows = (len(ntasks_list) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        
        for idx, nt in enumerate(ntasks_list):
            ax = axes[idx // ncols][idx % ncols]
            block = subset[subset['ntasks'] == nt]
            
            algos_present = [a for a in algos if a in block['algo'].unique()]
            task_ids = sorted(block['task_id'].dropna().unique())
            if not task_ids:
                task_ids = list(range(int(nt)))
            
            n_algos = len(algos_present)
            n_tasks = len(task_ids)
            x_groups = np.arange(n_algos)
            bar_width = 0.8 / n_tasks
            
            # plot each task as separate bar
            for t_idx, tid in enumerate(task_ids):
                offset = (t_idx - (n_tasks - 1) / 2) * bar_width
                positions = x_groups + offset
                
                work, switch, lb = [], [], []
                for algo in algos_present:
                    row = block[(block['algo'] == algo) & (block['task_id'] == tid)]
                    if not row.empty:
                        r = row.iloc[0]
                        work.append(float(r.get('work_time_s', 0) or 0))
                        switch.append(float(r.get('switch_time_s', 0) or 0))
                        lb.append(float(r.get('load_balancer_time_s', 0) or 0))
                    else:
                        work.append(0)
                        switch.append(0)
                        lb.append(0)
                
                # stacked bars
                b1 = ax.bar(positions, work, bar_width, align='center')
                b2 = ax.bar(positions, switch, bar_width, bottom=work, align='center')
                b3 = ax.bar(positions, lb, bar_width, bottom=np.array(work) + np.array(switch), align='center')
                
                # clr by algorithm
                for i, algo in enumerate(algos_present):
                    base = colors[algo]
                    b1[i].set_color(base)
                    b2[i].set_color(darken(base, 0.75))
                    b3[i].set_color(darken(base, 0.55))
            
            ax.set_title(f"{int(nt)} Processes", fontsize=10)
            ax.set_xlabel("Algorithm")
            ax.set_ylabel("Time (s)")
            ax.set_xticks(x_groups)
            ax.set_xticklabels(algos_present, rotation=25)
            ax.grid(axis='y', linestyle=':', linewidth=0.5)
        
        # remove unused subplots
        for idx in range(len(ntasks_list), nrows * ncols):
            fig.delaxes(axes[idx // ncols][idx % ncols])
        
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        outpath = out_dir / f"perproc_grouped_by_algo_freq_{int(freq)}.pdf"
        fig.savefig(outpath, bbox_inches='tight')
        plt.close(fig)


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python plot_lb_visualizations.py ____times.csv [nolb____times.csv]")
    
    lb_csv = sys.argv[1]
    nolb_csv = sys.argv[2] if len(sys.argv) > 2 else None
    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)
    df = load_data(lb_csv, nolb_csv)
    
    # find algorithm order and colors
    algos_present = df['algo'].dropna().unique()
    algos = [a for a in ALGO_ORDER if a in algos_present] + [a for a in algos_present if a not in ALGO_ORDER]
    
    cmap = plt.get_cmap('tab10' if len(algos) <= 10 else 'tab20')
    colors = {algo: cmap.colors[i % len(cmap.colors)] for i, algo in enumerate(algos)}
    
    # draw plots
    plot_total_time(df, algos, colors, out_dir)
    plot_per_process(df, algos, colors, out_dir)
    
    print(f"\nDone.")


if __name__ == "__main__":
    main()