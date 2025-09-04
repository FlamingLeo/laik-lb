#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

csv_file = "laik_stats.csv"
if not os.path.exists(csv_file):
    raise SystemExit(f"{csv_file} not found. Run collect_stats.py first.")

df = pd.read_csv(csv_file)
df["alloc_kb"] = df["alloc_bytes"] / 1024.0

algos = sorted(df["algo"].unique())
ntasks_list = sorted(df["ntasks"].unique())
loopcounts = sorted(df["loopcount"].unique())

def make_axes_grid(nrows, ncols, figsize=None):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=False)
    axes = np.atleast_2d(axes)
    return fig, axes

# 1) Allocation counts
fig, axes = make_axes_grid(len(algos), len(ntasks_list), figsize=(4*len(ntasks_list), 3*len(algos)))
for i, algo in enumerate(algos):
    for j, nt in enumerate(ntasks_list):
        ax = axes[i, j]
        subset = df[(df["algo"] == algo) & (df["ntasks"] == nt)]
        if subset.empty:
            ax.set_visible(False)
            continue
        # For each loopcount, plot tag=0 and tag=1
        for lc in loopcounts:
            sub_lc = subset[subset["loopcount"] == lc]
            for tag, sub_tag in sub_lc.groupby("tag"):
                sub_tag = sub_tag.sort_values("sidesize")
                ax.plot(sub_tag["sidesize"], sub_tag["allocs"],
                        marker="o", label=f"tag={tag}, loops={lc}")
        ax.set_title(f"{algo} | ntasks={nt}")
        if j == 0:
            ax.set_ylabel("allocs")
        if i == len(algos)-1:
            ax.set_xlabel("sidesize")
        ax.grid(True)
        ax.legend(fontsize=8)

plt.tight_layout()
plt.suptitle("Allocation counts: tag=0 vs tag=1 (loopcount curves)", y=1.02)
plt.savefig("alloc_counts_loopcount.png", dpi=300)
plt.show()

# 2) Allocation bytes (KB)
fig, axes = make_axes_grid(len(algos), len(ntasks_list), figsize=(4*len(ntasks_list), 3*len(algos)))
for i, algo in enumerate(algos):
    for j, nt in enumerate(ntasks_list):
        ax = axes[i, j]
        subset = df[(df["algo"] == algo) & (df["ntasks"] == nt)]
        if subset.empty:
            ax.set_visible(False)
            continue
        for lc in loopcounts:
            sub_lc = subset[subset["loopcount"] == lc]
            for tag, sub_tag in sub_lc.groupby("tag"):
                sub_tag = sub_tag.sort_values("sidesize")
                ax.plot(sub_tag["sidesize"], sub_tag["alloc_kb"],
                        marker="o", label=f"tag={tag}, loops={lc}")
        ax.set_title(f"{algo} | ntasks={nt}")
        if j == 0:
            ax.set_ylabel("alloc (KB)")
        if i == len(algos)-1:
            ax.set_xlabel("sidesize")
        ax.grid(True)
        ax.legend(fontsize=8)

plt.tight_layout()
plt.suptitle("Allocated bytes (KB): tag=0 vs tag=1 (loopcount curves)", y=1.02)
plt.savefig("alloc_bytes_loopcount.png", dpi=300)
plt.show()
