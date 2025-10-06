#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

METRICS = [
    "data_alloc_count", "data_alloc_bytes",
    "msg_sent_count", "msg_sent_bytes",
    "recv_count", "recv_bytes",
    "reduce_count", "reduce_bytes"
]

BASE_COLUMNS = ["ntasks", "algorithm", "task_id", "switches"]


def safe_makedirs(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def ensure_numeric_df(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def savefig(fig, outpath):
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def line_plot_by_alg(df, ntasks_col, metric_col, outpath, ylabel=None, title=None, normalize_by_ntasks=False):
    fig, ax = plt.subplots(figsize=(12, 6))

    has_freq = "prog_n" in df.columns
    mean_all = df.groupby(["algorithm", ntasks_col])[metric_col].mean().reset_index()

    algs = sorted(mean_all["algorithm"].unique())
    markers = ["o", "s", "^", "x", "D", "v", ">", "<"]
    cmap = plt.get_cmap("tab10")
    alg_colors = {alg: cmap(i % 10) for i, alg in enumerate(algs)}

    for i, alg in enumerate(algs):
        sub = mean_all[mean_all["algorithm"] == alg].sort_values(ntasks_col)
        if sub.empty:
            continue
        x = sub[ntasks_col].values
        y = sub[metric_col].values
        if normalize_by_ntasks:
            y = y / np.maximum(sub[ntasks_col].values, 1)
        ax.plot(x, y, marker=markers[i % len(markers)], linewidth=2.5, markersize=6,
                label=f"{alg}", color=alg_colors[alg])

    if has_freq:
        mean_prog = df.groupby(["algorithm", ntasks_col, "prog_n"])[metric_col].mean().reset_index()
        for alg in algs:
            sub_alg = mean_prog[mean_prog["algorithm"] == alg]
            if sub_alg.empty:
                continue
            nts = sorted(sub_alg[ntasks_col].unique())
            x_vals = []
            min_vals = []
            max_vals = []
            for n in nts:
                vals = sub_alg[sub_alg[ntasks_col] == n][metric_col].values
                if vals.size == 0:
                    x_vals.append(n)
                    min_vals.append(np.nan)
                    max_vals.append(np.nan)
                else:
                    vmin = np.nanmin(vals)
                    vmax = np.nanmax(vals)
                    if normalize_by_ntasks:
                        vmin = vmin / max(n, 1)
                        vmax = vmax / max(n, 1)
                    x_vals.append(n)
                    min_vals.append(vmin)
                    max_vals.append(vmax)
            if len(x_vals) > 0:
                ax.fill_between(x_vals, min_vals, max_vals, color=alg_colors[alg], alpha=0.12, linewidth=0)

    alg_handles, alg_labels = ax.get_legend_handles_labels()
    ax.legend(alg_handles, alg_labels, title="algorithm",
            loc='upper left', bbox_to_anchor=(1.05, 1))

    ax.set_xlabel(ntasks_col)
    ax.set_ylabel(ylabel or metric_col)
    ax.grid(True, axis="y")
    #if title:
    #    ax.set_title(title)
    savefig(fig, outpath)


def boxplot_by_alg_and_prog(df, metric, outpath, ylabel=None, title=None):
    has_freq = "prog_n" in df.columns
    algs = sorted(df["algorithm"].unique())

    if not has_freq:
        alg_groups = []
        alg_names = []
        for alg, group in df.groupby("algorithm"):
            vals = group[metric].dropna().values
            if len(vals) > 0:
                alg_groups.append(vals)
                alg_names.append(alg)
        if len(alg_groups) == 0:
            print(f"[boxplot] no data for metric {metric}")
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(alg_groups, labels=alg_names, showfliers=True)
        ax.set_xlabel("algorithm")
        ax.set_ylabel(ylabel or metric)
        ax.grid(True, axis="y")
        #if title:
        #    ax.set_title(title)
        savefig(fig, outpath)
        return

    prog_vals = sorted(df["prog_n"].dropna().unique())
    n_algs = len(algs)
    n_prog = len(prog_vals)

    total_width = 0.8
    box_width = total_width / max(n_prog, 1)
    positions = []
    data_to_plot = []
    colors = []
    cmap = plt.get_cmap("tab10")
    prog_colors = {p: cmap(i % 10) for i, p in enumerate(prog_vals)}

    for i, alg in enumerate(algs):
        center = i
        for j, prog in enumerate(prog_vals):
            pos = center - total_width / 2 + j * box_width + box_width / 2
            positions.append(pos)
            vals = df[(df["algorithm"] == alg) & (df["prog_n"] == prog)][metric].dropna().values
            data_to_plot.append(vals)
            colors.append(prog_colors[prog])

    nonempty_idx = [idx for idx, d in enumerate(data_to_plot) if len(d) > 0]
    if len(nonempty_idx) == 0:
        print(f"[boxplot] no data for metric {metric}")
        return

    fig, ax = plt.subplots(figsize=(max(10, n_algs * 1.6), 6))
    bp = ax.boxplot([data_to_plot[i] for i in nonempty_idx],
                    positions=[positions[i] for i in nonempty_idx],
                    widths=box_width * 0.9,
                    patch_artist=True,
                    showfliers=True)

    for plot_idx, orig_idx in enumerate(nonempty_idx):
        color = colors[orig_idx]
        box = bp["boxes"][plot_idx]
        box.set_facecolor(color)
        box.set_alpha(0.75)

        bp["medians"][plot_idx].set(color="black", linewidth=1.0)

    xticks = []
    xticklabels = []
    for i, alg in enumerate(algs):
        alg_positions = []
        for j in range(n_prog):
            idx = i * n_prog + j
            if idx < len(positions) and len(data_to_plot[idx]) > 0:
                alg_positions.append(positions[idx])
        if alg_positions:
            xticks.append(np.mean(alg_positions))
            xticklabels.append(alg)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("algorithm")
    ax.set_ylabel(ylabel or metric)
    ax.grid(True, axis="y")
    #if title:
    #    ax.set_title(title)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=prog_colors[p], alpha=0.75) for p in prog_vals]
    labels = [f"n={int(p)}" for p in prog_vals]
    ax.legend(handles, labels, title="load balancing frequency", loc='center left', bbox_to_anchor=(1, 0.5))

    savefig(fig, outpath)


def correlation_heatmap(corr_df, outpath, title="Spearman correlation"):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_df.values, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(corr_df.columns)))
    ax.set_yticks(np.arange(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_df.index)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    #if title:
    #    ax.set_title(title)
    savefig(fig, outpath)


def kruskal_test_by_alg(df, metric):
    groups = [g[metric].dropna().values for name, g in df.groupby("algorithm")]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return (np.nan, np.nan)
    try:
        stat, p = stats.kruskal(*groups)
        return float(stat), float(p)
    except Exception:
        return (np.nan, np.nan)


def kruskal_per_ntasks(df, metric):
    results = []
    for n in sorted(df["ntasks"].dropna().unique()):
        sub = df[df["ntasks"] == n]
        groups = [g[metric].dropna().values for name, g in sub.groupby("algorithm")]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            try:
                stat, p = stats.kruskal(*groups)
            except Exception:
                stat, p = np.nan, np.nan
        else:
            stat, p = np.nan, np.nan
        results.append({"ntasks": int(n), "kw_stat": stat, "p_value": p})
    return pd.DataFrame(results)


def kruskal_per_ntasks_and_prog(df, metric):
    results = []
    if "prog_n" not in df.columns:
        return pd.DataFrame(results)
    for prog in sorted(df["prog_n"].dropna().unique()):
        sub_prog = df[df["prog_n"] == prog]
        for n in sorted(sub_prog["ntasks"].dropna().unique()):
            sub = sub_prog[sub_prog["ntasks"] == n]
            groups = [g[metric].dropna().values for name, g in sub.groupby("algorithm")]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) >= 2:
                try:
                    stat, p = stats.kruskal(*groups)
                except Exception:
                    stat, p = np.nan, np.nan
            else:
                stat, p = np.nan, np.nan
            results.append({"prog_n": int(prog), "ntasks": int(n), "kw_stat": stat, "p_value": p})
    return pd.DataFrame(results)


def run_analysis(csv_path, out_dir):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    plots_dir = out_dir / "plots"
    tables_dir = out_dir / "tables"
    safe_makedirs(plots_dir)
    safe_makedirs(tables_dir)

    print("Loading csv:", csv_path)
    df = pd.read_csv(csv_path)
    print("Shape:", df.shape)
    df.columns = [c.strip() for c in df.columns]

    for c in BASE_COLUMNS:
        if c not in df.columns:
            print(f"Warning: expected column '{c}' not found in csv.")

    ensure_numeric_df(df, ["ntasks", "switches", "prog_n"] + METRICS)
    df["algorithm"] = df["algorithm"].astype(str)

    present_metrics = [m for m in METRICS if m in df.columns]
    print("Metrics present:", present_metrics)

    has_freq = "prog_n" in df.columns

    summary_rows = []
    group_cols = ["algorithm", "ntasks"] + (["prog_n"] if has_freq else [])
    for keys, g in df.groupby(group_cols):
        if has_freq:
            alg, n, prog = keys
        else:
            alg, n = keys
            prog = None
        row = {"algorithm": alg, "ntasks": int(n)}
        if has_freq:
            row["prog_n"] = int(prog)
        row.update({
            "switches_mean": float(g["switches"].mean()) if "switches" in g else np.nan,
            "switches_median": float(g["switches"].median()) if "switches" in g else np.nan,
            "count": int(len(g))
        })
        for m in present_metrics:
            row[f"{m}_mean"] = float(g[m].mean()) if m in g else np.nan
            row[f"{m}_median"] = float(g[m].median()) if m in g else np.nan
            row[f"{m}_std"] = float(g[m].std()) if m in g else np.nan
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    if has_freq:
        summary_df = summary_df.sort_values(["ntasks", "prog_n", "algorithm"])
        summary_csv = tables_dir / "grouped_summary_algorithm_ntasks_prog_n.csv"
    else:
        summary_df = summary_df.sort_values(["ntasks", "algorithm"])
        summary_csv = tables_dir / "grouped_summary_algorithm_ntasks.csv"
    summary_df.to_csv(summary_csv, index=False)
    print("Saved grouped summary:", summary_csv)

    overall_kw_results = []
    all_plot_paths = []

    for m in present_metrics:
        print(f"\n--- Metric: {m} ---")

        p_mean = plots_dir / f"mean_{m}_vs_ntasks.svg"
        line_plot_by_alg(df, "ntasks", m, str(p_mean),
                                 ylabel=f"mean {m}", title=f"Mean {m} vs ntasks (by algorithm)")
        all_plot_paths.append(str(p_mean))
        print("Saved:", p_mean)

        p_mean_norm = plots_dir / f"mean_{m}_per_task_vs_ntasks.svg"
        line_plot_by_alg(df, "ntasks", m, str(p_mean_norm),
                                 ylabel=f"mean {m} per task",
                                 title=f"Mean {m}/ntasks vs ntasks (by algorithm) ",
                                 normalize_by_ntasks=True)
        all_plot_paths.append(str(p_mean_norm))
        print("Saved:", p_mean_norm)

        p_box = plots_dir / f"boxplot_{m}_by_algorithm.svg"
        boxplot_by_alg_and_prog(df, m, str(p_box), ylabel=m,
                                        title=f"Distribution of {m} by algorithm")
        all_plot_paths.append(str(p_box))
        print("Saved:", p_box)

        kw_stat, kw_p = kruskal_test_by_alg(df, m)
        overall_kw_results.append({"metric": m, "kw_stat": kw_stat, "kw_p": kw_p})
        print(f"Kruskal-Wallis (all ntasks/prog pooled) for {m}: stat={kw_stat}, p={kw_p}")

        pernt = kruskal_per_ntasks(df, m)
        pernt_csv = tables_dir / f"kruskal_per_ntasks_{m}.csv"
        pernt.to_csv(pernt_csv, index=False)
        print("Saved per-ntasks Kruskal results to", pernt_csv)

        if has_freq:
            perntprog = kruskal_per_ntasks_and_prog(df, m)
            perntprog_csv = tables_dir / f"kruskal_per_ntasks_per_prog_{m}.csv"
            perntprog.to_csv(perntprog_csv, index=False)
            print("Saved per-ntasks-per-prog Kruskal results to", perntprog_csv)
            sig = perntprog[perntprog["p_value"] < 0.05]
            if not sig.empty:
                print("(prog_n, ntasks) with p < 0.05 for metric", m, ":", sig[["prog_n", "ntasks"]].to_dict(orient="records"))

        sig_overall = pernt[pernt["p_value"] < 0.05]
        if not sig_overall.empty:
            print("Ntasks with p < 0.05 for metric", m, ":", sig_overall["ntasks"].tolist())

    overall_kw_df = pd.DataFrame(overall_kw_results)
    overall_kw_csv = tables_dir / "overall_kruskal_by_metric.csv"
    overall_kw_df.to_csv(overall_kw_csv, index=False)
    print("\nSaved overall Kruskal-Wallis results:", overall_kw_csv)

    corr_cols = [c for c in present_metrics if c in df.columns] + (["switches"] if "switches" in df.columns else [])
    if len(corr_cols) >= 2:
        corr_df = df[corr_cols].corr(method="spearman")
        corr_csv = tables_dir / "spearman_correlation_metrics_switches.csv"
        corr_df.to_csv(corr_csv)
        corr_svg = plots_dir / "spearman_correlation_metrics_switches.svg"
        correlation_heatmap(corr_df, str(corr_svg), title="Spearman correlation (metrics + switches)")
        all_plot_paths.append(str(corr_svg))
        print("Saved Spearman correlation csv and svg:", corr_csv, corr_svg)
    else:
        print("Not enough columns to compute correlation matrix.")

    if "switches" in df.columns:
        p_sw = plots_dir / "mean_switches_vs_ntasks.svg"
        line_plot_by_alg(df, "ntasks", "switches", str(p_sw),
                                 ylabel="mean switches", title="Mean switches vs ntasks (by algorithm)")
        all_plot_paths.append(str(p_sw))
        p_sw_box = plots_dir / "boxplot_switches_by_algorithm.svg"
        boxplot_by_alg_and_prog(df, "switches", str(p_sw_box),
                                        ylabel="switches", title="Distribution of switches by algorithm")
        all_plot_paths.append(str(p_sw_box))
        print("Saved switches plots.")

    print("\nDone. Outputs in:", out_dir)
    return out_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("outdir", nargs="?", default="./output")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    csv_path = args.csv
    out_dir = args.outdir
    if not os.path.exists(csv_path):
        print("ERROR: csv path does not exist:", csv_path)
        sys.exit(1)
    run_analysis(csv_path, out_dir)
