#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import numpy as np
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

ALGO_ORDER = ['hilbert', 'gilbert', 'rcb', 'rcbincr']
ALGO_COLORS = {
    'hilbert': plt.get_cmap('tab10')(1),
    'gilbert': plt.get_cmap('tab10')(2),
    'rcb': plt.get_cmap('tab10')(3),
    'rcbincr': plt.get_cmap('tab10')(4),
}
METRICS = [
    ('alloced', 'Bytes Allocated'),
    ('allocs', 'Number of Allocations'),
    ('concurrent', 'Max Concurrent Bytes Allocated'),
]


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python plot_data_summary.py ___summary.csv")

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)

    # load data
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # cvt numeric columns
    numeric_cols = ["ntasks", "freq", "task", "alloced", "allocs", "concurrent", "freed", "frees"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # verify
    group_cols = ["ntasks", "algo", "freq"]
    check_cols = ["alloced", "allocs", "concurrent", "freed", "frees"]
    inconsistent = []
    for name, group in df.groupby(group_cols):
        for col in check_cols:
            if col in group.columns and group[col].nunique(dropna=False) > 1:
                inconsistent.append((name, col))
    
    if inconsistent:
        print("WARNING: Found inconsistent values - using first row per group")

    # take first row per group
    summary = df.groupby(group_cols).first().reset_index()

    # get frequencies to plot
    freqs = sorted(summary["freq"].dropna().unique().astype(int))
    if not freqs:
        sys.exit("No frequency values found in data")

    saved = []
    for freq in freqs:
        subset = summary[summary["freq"] == freq]

        for metric, label in METRICS:
            # pivot data
            pivot = subset.pivot_table(index="ntasks", columns="algo", values=metric, aggfunc="first")
            pivot = pivot.reindex(columns=ALGO_ORDER).fillna(0).sort_index()
            
            if pivot.empty:
                continue

            ntasks = pivot.index.astype(int)
            algos = pivot.columns.tolist()
            x = np.arange(len(ntasks))
            width = min(0.8 / max(1, len(algos)), 0.18)

            # draw plot
            fig, ax = plt.subplots(figsize=(10, 5))
            for i, algo in enumerate(algos):
                offset = (i - (len(algos) - 1) / 2) * width
                ax.bar(x + offset, pivot[algo], width=width, 
                      label=algo, color=ALGO_COLORS.get(algo))

            ax.set_xticks(x)
            ax.set_xticklabels(ntasks)
            ax.set_xlabel("Number of Processes")
            ax.set_ylabel(label)
            ax.legend(title="Algorithm")
            ax.grid(axis="y", linestyle=":", linewidth=0.5)
            fig.tight_layout()

            outpath = out_dir / f"{metric}_freq{freq}.pdf"
            fig.savefig(outpath, dpi=200)
            plt.close(fig)
            saved.append(str(outpath))
            print(f"Saved: {outpath}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()