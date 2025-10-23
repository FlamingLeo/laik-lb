#!/usr/bin/env python3
import sys
from pathlib import Path
from collections import defaultdict
import math
import pandas as pd
import matplotlib.pyplot as plt

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

COLORS = {
    "no-lb": plt.get_cmap("tab10")(0),
    "hilbert": plt.get_cmap("tab10")(1),
    "gilbert": plt.get_cmap("tab10")(2),
    "rcb": plt.get_cmap("tab10")(3),
    "rcbincr": plt.get_cmap("tab10")(4),
}

LINESTYLES = {
    "rcb": (0, (6, 2)),
    "hilbert": (0, (2, 1)),
    "gilbert": (0, (4, 1)),
    "rcbincr": (0, (1, 1)),
    "no-lb": (0, (3, 1, 1, 1)),
}


def load_data(lb_path, metrics_path=None):
    df1 = pd.read_csv(lb_path)
    df1.columns = df1.columns.str.strip()
    
    if metrics_path and metrics_path.exists():
        df2 = pd.read_csv(metrics_path)
        df2.columns = df2.columns.str.strip()
        if 'algo' not in df2.columns:
            df2['algo'] = 'no-lb'
        else:
            df2['algo'] = df2['algo'].fillna('no-lb')
    else:
        df2 = pd.DataFrame()
    
    # cvt numeric columns
    numeric_cols = ['ntasks', 'freq', 'segment', 'task', 'seq_stopped', 'br', 'bs', 'fb', 'fc', 'mb', 'mc']
    for col in numeric_cols:
        if col in df1.columns:
            df1[col] = pd.to_numeric(df1[col], errors='coerce')
        if not df2.empty and col in df2.columns:
            df2[col] = pd.to_numeric(df2[col], errors='coerce')
    
    return pd.concat([df1, df2], ignore_index=True) if not df2.empty else df1


def split_runs(df):
    starts = df.index[df['segment'] == 1].tolist() or [0]
    runs = []
    
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(df)
        run_df = df.loc[start:end - 1].reset_index(drop=True)
        
        if run_df.empty:
            continue
        
        ntasks = int(run_df['ntasks'].dropna().iloc[0]) if not run_df['ntasks'].dropna().empty else None
        freq = int(run_df['freq'].dropna().iloc[0]) if not run_df['freq'].dropna().empty else None
        algo = run_df['algo'].dropna().iloc[0] if not run_df['algo'].dropna().empty else 'no-lb'
        
        runs.append({'ntasks': ntasks, 'freq': freq, 'algo': algo, 'df': run_df})
    
    return runs


def compute_series(run):
    g = run['df'].groupby('segment').agg({'bs': 'sum', 'br': 'sum'})
    s = (g['bs'].fillna(0) + g['br'].fillna(0)) / 2.0
    s.index = s.index.astype(int)
    return s.sort_index()


def aggregate_runs(runs):
    store = defaultdict(list)
    for run in runs:
        key = (run['ntasks'], run['freq'], run['algo'])
        series = compute_series(run)
        if not series.empty:
            store[key].append(series)
    
    # avg multiple runs
    avg_store = {}
    for key, series_list in store.items():
        if len(series_list) == 1:
            avg_store[key] = series_list[0]
        else:
            avg_store[key] = pd.concat(series_list, axis=1).mean(axis=1)
    
    return avg_store


def plot_by_freq(avg_store, out_dir, cols=3):
    freq_ntasks = defaultdict(set)
    for (ntasks, freq, algo) in avg_store.keys():
        if pd.notna(freq):
            freq_ntasks[freq].add(ntasks)

    cmap = plt.get_cmap("tab10")
    
    for freq in sorted(freq_ntasks.keys()):
        ntasks_list = sorted(x for x in freq_ntasks[freq] if pd.notna(x))
        if not ntasks_list:
            continue
        
        nrows = math.ceil(len(ntasks_list) / cols)
        fig, axes = plt.subplots(nrows, cols, figsize=(cols * 5, nrows * 4), sharey=True)
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        
        # hide unused subplots
        for ax in axes[len(ntasks_list):]:
            ax.set_visible(False)
        
        # get all algorithms for this frequency
        algos = sorted({k[2] for k in avg_store.keys() if k[1] == freq})
        color_map = {a: COLORS.get(a, cmap(i % cmap.N)) for i, a in enumerate(algos)}
        legend_lines = {}
        
        # plot by number of tasks
        for i, ntasks in enumerate(ntasks_list):
            ax = axes[i]
            algos_here = sorted([k[2] for k in avg_store.keys() if k[0] == ntasks and k[1] == freq])
            
            for algo in algos_here:
                series = avg_store.get((ntasks, freq, algo))
                if series is None or series.empty:
                    continue
                
                line, = ax.plot(series.index, series.values, label=algo,
                               color=color_map[algo],
                               linestyle=LINESTYLES.get(algo, (0, (1, 1))))
                
                if algo not in legend_lines:
                    legend_lines[algo] = line
            
            ax.set_title(f'{ntasks} Processes')
            ax.set_xlabel('Segment')
            ax.set_ylabel('Bytes Exchanged')
            ax.grid(True, linestyle=':', linewidth=0.5)
        
        # add single legend
        legend_order = [a for a in algos if a in legend_lines]
        fig.suptitle('Algorithm', fontsize=14, y=0.98)
        fig.legend([legend_lines[a] for a in legend_order], legend_order,
                  loc='upper center', ncol=len(legend_order),
                  bbox_to_anchor=(0.5, 0.94), fontsize='small', frameon=False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        outpath = out_dir / f'combined_freq{freq}_single_legend.pdf'
        fig.savefig(outpath, dpi=150)
        plt.close(fig)


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python plot_data_series.py results____timeseries.csv [results____metrics.csv]")
    
    lb_path = Path(sys.argv[1])
    metrics_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)
    df = load_data(lb_path, metrics_path)
    runs = split_runs(df)
    avg_store = aggregate_runs(runs)
    
    # draw plots
    plot_by_freq(avg_store, out_dir)
    
    print("Done.")


if __name__ == '__main__':
    main()