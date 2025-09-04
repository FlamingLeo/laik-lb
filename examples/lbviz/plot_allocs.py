import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

csv_file = "laik_stats.csv"
if not os.path.exists(csv_file):
    raise SystemExit(f"{csv_file} not found. Run collect_stats.py first.")

df = pd.read_csv(csv_file)

# convert bytes -> KB for readability
df["alloc_kb"] = df["alloc_bytes"] / 1024.0

# aggregate across loopcount 
agg = df.groupby(["ntasks", "tag", "algo", "sidesize"], as_index=False).agg(
    allocs=("allocs", "mean"),
    alloc_bytes=("alloc_bytes", "mean"),
    alloc_kb=("alloc_kb", "mean")
)

algos = sorted(agg["algo"].unique())
ntasks_list = sorted(agg["ntasks"].unique())

def make_axes_grid(nrows, ncols, figsize=None):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=False)
    axes = np.atleast_2d(axes)
    return fig, axes

# 1) Allocation counts
fig, axes = make_axes_grid(len(algos), len(ntasks_list), figsize=(4*len(ntasks_list), 3*len(algos)))
for i, algo in enumerate(algos):
    for j, nt in enumerate(ntasks_list):
        ax = axes[i, j]
        subset = agg[(agg["algo"] == algo) & (agg["ntasks"] == nt)]
        if subset.empty:
            ax.set_visible(False)
            continue
        for tag, sub in subset.groupby("tag"):
            sub = sub.sort_values("sidesize")
            ax.plot(sub["sidesize"], sub["allocs"], marker="o", label=f"tag={int(tag)}")
        ax.set_title(f"{algo} | ntasks={nt}")
        if j == 0:
            ax.set_ylabel("allocs")
        if i == len(algos) - 1:
            ax.set_xlabel("sidesize")
        ax.grid(True)
        ax.legend()

plt.tight_layout()
plt.suptitle("Allocation counts: tag=0 vs tag=1", y=1.02)
plt.savefig("alloc_counts.png", dpi=300)
plt.show()

# 2) Allocation bytes (KB)
fig, axes = make_axes_grid(len(algos), len(ntasks_list), figsize=(4*len(ntasks_list), 3*len(algos)))
for i, algo in enumerate(algos):
    for j, nt in enumerate(ntasks_list):
        ax = axes[i, j]
        subset = agg[(agg["algo"] == algo) & (agg["ntasks"] == nt)]
        if subset.empty:
            ax.set_visible(False)
            continue
        for tag, sub in subset.groupby("tag"):
            sub = sub.sort_values("sidesize")
            ax.plot(sub["sidesize"], sub["alloc_kb"], marker="o", label=f"tag={int(tag)}")
        ax.set_title(f"{algo} | ntasks={nt}")
        if j == 0:
            ax.set_ylabel("alloc (KB)")
        if i == len(algos) - 1:
            ax.set_xlabel("sidesize")
        ax.grid(True)
        ax.legend()

plt.tight_layout()
plt.suptitle("Allocated bytes (KB): tag=0 vs tag=1", y=1.02)
plt.savefig("alloc_bytes.png", dpi=300)
plt.show()

# 3) Percent change (tag1 vs tag0)
pivot = agg.pivot_table(index=["ntasks", "algo", "sidesize"], columns="tag", values="allocs").reset_index()
pivot = pivot.rename(columns={0: "allocs_tag0", 1: "allocs_tag1"})
pivot["allocs_abs_diff"] = pivot["allocs_tag1"] - pivot["allocs_tag0"]
pivot["allocs_pct_change"] = np.where(
    pivot["allocs_tag0"].abs() > 0,
    100.0 * (pivot["allocs_tag1"] - pivot["allocs_tag0"]) / pivot["allocs_tag0"],
    np.nan
)

fig, axes = make_axes_grid(len(algos), len(ntasks_list), figsize=(4*len(ntasks_list), 3*len(algos)))
for i, algo in enumerate(algos):
    for j, nt in enumerate(ntasks_list):
        ax = axes[i, j]
        sub = pivot[(pivot["algo"] == algo) & (pivot["ntasks"] == nt)]
        if sub.empty:
            ax.set_visible(False)
            continue
        ax.bar(sub["sidesize"].astype(str), sub["allocs_pct_change"])
        ax.set_title(f"{algo} | ntasks={nt}")
        if j == 0:
            ax.set_ylabel("% change (tag1 vs tag0)")
        ax.set_xticklabels(sub["sidesize"].astype(str), rotation=45)
        ax.grid(True)

plt.tight_layout()
plt.suptitle("Percent change in allocations: tag1 vs tag0 (avg across loopcount)", y=1.02)
plt.savefig("alloc_pct_change.png", dpi=300)
plt.show()
