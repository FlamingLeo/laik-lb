#!/usr/bin/env python3
import os
import sys
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

LINESTYLES = {
    "rcb": (0, (6, 2)),
    "hilbert": (0, (2, 1)),
    "gilbert": (0, (4, 1)),
    "rcbincr": (0, (1, 1)),
}

COLORS = {"gilbert": 2, "hilbert": 1, "rcb": 3, "rcbincr": 4}


def assign_runs(segments):
    run_id, runs = -1, []
    for seg in segments.fillna(-999):
        if int(seg) == 1 if pd.notna(seg) else False:
            run_id += 1
        runs.append(run_id)
    return runs


def plot_freq(df, freq, outdir):
    data = df[df["freq"] == freq]
    if data.empty:
        raise ValueError(f"No data for freq={freq}")

    ntasks_list = sorted(data["ntasks"].unique())
    ncols = 3
    nrows = math.ceil(len(ntasks_list) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3*nrows), squeeze=False)
    axes = axes.flatten()
    algos = sorted(data["algo"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {a: cmap(COLORS.get(a, i % cmap.N)) for i, a in enumerate(algos)}
    legend_items = {}

    for i, ntasks in enumerate(ntasks_list):
        ax = axes[i]
        subset = data[data["ntasks"] == ntasks]
        
        ax.set_title(f"{int(ntasks)} Processes", fontsize=10)
        ax.set_xlabel("Segment")
        ax.set_ylabel("Relative Imbalance")
        ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.6)

        for algo in sorted(subset["algo"].unique()):
            algo_data = subset[subset["algo"] == algo]
            color = colors[algo]
            ls = LINESTYLES.get(algo, "-")
            
            n_runs = algo_data["run_id"].nunique()
            
            if n_runs == 1:
                # single run
                run = algo_data.sort_values("segment")
                line, = ax.plot(run["segment"], run["rel_imbalance"], 
                               linestyle=ls, linewidth=1.7, color=color, label=algo)
            else:
                # multiple runs
                for rid in algo_data["run_id"].unique():
                    run = algo_data[algo_data["run_id"] == rid].sort_values("segment")
                    ax.plot(run["segment"], run["rel_imbalance"], 
                           linestyle=ls, linewidth=1.0, alpha=0.22, color=color)
                
                mean = algo_data.groupby("segment")["rel_imbalance"].mean().reset_index()
                line, = ax.plot(mean["segment"], mean["rel_imbalance"],
                               linestyle=ls, linewidth=1.9, marker='o', markersize=4,
                               markeredgewidth=0.8, markerfacecolor='none',
                               color=color, label=f"{algo} (mean)")
            
            if algo not in legend_items:
                legend_items[algo] = line

    # hide unused subplots
    for j in range(len(ntasks_list), len(axes)):
        axes[j].axis("off")

    # add legend
    if legend_items:
        fig.legend(legend_items.values(), legend_items.keys(), 
                  title="Algorithm", loc="upper center", 
                  bbox_to_anchor=(0.5, 1.02), ncol=min(4, len(legend_items)), 
                  frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    outpath = os.path.join(outdir, f"rel_imbalance_freq{int(freq)}.pdf")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return outpath


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 plot_rel_imbalance_by_freq.py __progress___.csv")

    csv_path = sys.argv[1]
    if not os.path.isfile(csv_path):
        sys.exit(f"CSV not found: {csv_path}")

    # load data
    df = pd.read_csv(csv_path)
    required = {"ntasks", "freq", "algo", "segment", "rel_imbalance"}
    if missing := required - set(df.columns):
        sys.exit(f"Missing columns: {missing}")

    # cvt types
    df["segment"] = pd.to_numeric(df["segment"], errors="coerce").astype("Int64")
    df["ntasks"] = pd.to_numeric(df["ntasks"], errors="coerce").astype("Int64")
    df["freq"] = pd.to_numeric(df["freq"], errors="coerce").astype("Int64")
    df["rel_imbalance"] = pd.to_numeric(df["rel_imbalance"], errors="coerce")
    
    # assign run ids and filter invalid runs
    df["run_id"] = assign_runs(df["segment"])
    df = df[df["run_id"] != -1]

    # create output directory
    os.makedirs("plots", exist_ok=True)

    # generate plots
    saved = []
    for freq in sorted(df["freq"].unique()):
        try:
            path = plot_freq(df, freq, "plots")
            saved.append(path)
            print(f"Saved: {path}")
        except Exception as e:
            print(f"Skipping freq={freq}: {e}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()