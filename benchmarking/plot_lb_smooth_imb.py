#!/usr/bin/env python3
import os
import sys
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


def to_bool(x):
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    return s in ("true", "t", "1", "yes", "y")


def assign_runs(segments):
    run_id, runs = -1, []
    for seg in segments.fillna(-999):
        if int(seg) == 1 if pd.notna(seg) else False:
            run_id += 1
        runs.append(run_id)
    return runs


def fmt_label(x):
    if pd.isna(x):
        return "nan"
    try:
        xf = float(x)
        return str(int(xf)) if xf.is_integer() else f"{xf:.6f}".rstrip('0').rstrip('.')
    except:
        return str(x)


def plot_combo(df, ntasks, algo, A, r, R, outdir):
    subset = df[(df["ntasks"] == ntasks) & (df["algo"] == algo)]
    if subset.empty:
        return None

    s_true = subset[subset["S"] == True]
    s_false_i_false = subset[(subset["S"] == False) & (subset["i"] == False) &
                             (subset["A"] == A) & (subset["r"] == r) & (subset["R"] == R)]
    s_false_i_true = subset[(subset["S"] == False) & (subset["i"] == True) &
                            (subset["A"] == A) & (subset["r"] == r) & (subset["R"] == R)]

    if s_false_i_false.empty and s_false_i_true.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.set_title(fr"{ntasks} Processes, {algo}, $\alpha={A}$, $r_{{\min}}={r}$, $r_{{\max}}={R}$")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Relative Imbalance")
    ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.6)

    legend_items = {}

    def plot_series(data, label, key):
        if data.empty:
            return
        
        n_runs = data["run_id"].nunique()
        if n_runs == 1:
            # single run
            run = data.sort_values("segment")
            line, = ax.plot(run["segment"], run["rel_imbalance"],
                           linestyle="-", linewidth=1.6, label=label)
        else:
            # multiple runs
            for rid in data["run_id"].unique():
                run = data[data["run_id"] == rid].sort_values("segment")
                ax.plot(run["segment"], run["rel_imbalance"],
                       linestyle=":", linewidth=0.9, alpha=0.25)
            
            mean = data.groupby("segment")["rel_imbalance"].mean().reset_index().sort_values("segment")
            line, = ax.plot(mean["segment"], mean["rel_imbalance"],
                           linestyle="-", linewidth=1.8, label=label)
        
        if key not in legend_items:
            legend_items[key] = line

    plot_series(s_true, "S = True", "No Smoothing")
    plot_series(s_false_i_false, "S = False, i = False", "Smoothing, Normal")
    plot_series(s_false_i_true, "S = False, i = True", "Smoothing, Intelligent")

    if legend_items:
        ax.legend(legend_items.values(), legend_items.keys(), title="Type", loc="best")

    fname = f"NT{ntasks}_{algo}_A{fmt_label(A)}_r{fmt_label(r)}_R{fmt_label(R)}.pdf"
    outpath = os.path.join(outdir, fname)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python plot_lb_smooth_imb.py results_progress_lb_lbsmooth.csv")

    csv_path = sys.argv[1]
    if not os.path.isfile(csv_path):
        sys.exit(f"CSV not found: {csv_path}")

    outdir = "plots"
    os.makedirs(outdir, exist_ok=True)

    # load data
    df = pd.read_csv(csv_path)
    required = {"ntasks", "algo", "A", "r", "R", "S", "i", "segment", "rel_imbalance"}
    if missing := required - set(df.columns):
        sys.exit(f"Missing columns: {missing}")

    # cvt types
    df["ntasks"] = pd.to_numeric(df["ntasks"], errors="coerce").astype("Int64")
    df["segment"] = pd.to_numeric(df["segment"], errors="coerce").astype("Int64")
    df["rel_imbalance"] = pd.to_numeric(df["rel_imbalance"], errors="coerce")
    df["A"] = pd.to_numeric(df["A"], errors="coerce")
    df["r"] = pd.to_numeric(df["r"], errors="coerce")
    df["R"] = pd.to_numeric(df["R"], errors="coerce")
    df["S"] = df["S"].apply(to_bool)
    df["i"] = df["i"].apply(to_bool)

    # assign run ids
    df["run_id"] = assign_runs(df["segment"])
    df = df[df["run_id"] != -1]

    # get param combo
    combos = (df[df["S"] == False][["ntasks", "algo", "A", "r", "R"]]
              .drop_duplicates()
              .sort_values(["ntasks", "algo", "A", "r", "R"]))

    # draw plots
    for _, row in combos.iterrows():
        plot_combo(df, int(row["ntasks"]), row["algo"], row["A"], row["r"], row["R"], outdir)

    print("Done.")


if __name__ == "__main__":
    main()