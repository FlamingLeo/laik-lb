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

BASE_COLUMNS = ["ntasks", "task_id", "timestamp", "switches"]

def safe_makedirs(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def ensure_numeric_df(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def savefig(fig, outpath):
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

def line_plot_by_ntasks(df, ntasks_col, metric_col, outpath, ylabel=None, title=None, normalize_by_ntasks=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_all = df.groupby([ntasks_col])[metric_col].mean().reset_index()
    if mean_all.empty:
        savefig(fig, outpath)
        return
    x = mean_all[ntasks_col].values
    y = mean_all[metric_col].values
    if normalize_by_ntasks:
        y = y / np.maximum(mean_all[ntasks_col].values, 1)
    ax.plot(x, y, marker="o", linewidth=2.5, markersize=6)
    ax.set_xlabel(ntasks_col)
    ax.set_ylabel(ylabel or metric_col)
    ax.grid(True, axis="y")
    savefig(fig, outpath)

def boxplot_by_ntasks(df, metric, outpath, ylabel=None, title=None):
    ntasks_vals = sorted(df["ntasks"].dropna().unique())
    groups = []
    labels = []
    for n in ntasks_vals:
        vals = df[df["ntasks"] == n][metric].dropna().values
        if len(vals) > 0:
            groups.append(vals)
            labels.append(str(n))
    if len(groups) == 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        savefig(fig, outpath)
        return
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 6))
    ax.boxplot(groups, labels=labels, showfliers=True)
    ax.set_xlabel("ntasks")
    ax.set_ylabel(ylabel or metric)
    ax.grid(True, axis="y")
    savefig(fig, outpath)

def correlation_heatmap(corr_df, outpath, title="Spearman correlation"):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_df.values, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(corr_df.columns)))
    ax.set_yticks(np.arange(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_df.index)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(fig, outpath)

def kruskal_test_by_task_id(df, metric):
    groups = [g[metric].dropna().values for name, g in df.groupby("task_id")]
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
        groups = [g[metric].dropna().values for name, g in sub.groupby("task_id")]
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

    ensure_numeric_df(df, ["ntasks", "switches"] + METRICS)
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        except Exception:
            pass

    present_metrics = [m for m in METRICS if m in df.columns]
    print("Metrics present:", present_metrics)

    summary_rows = []
    group_cols = ["ntasks", "task_id"]
    for keys, g in df.groupby(group_cols):
        n, tid = keys
        row = {"ntasks": int(n), "task_id": int(tid)}
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
    summary_df = summary_df.sort_values(["ntasks", "task_id"])
    summary_csv = tables_dir / "grouped_summary_ntasks_task_id.csv"
    summary_df.to_csv(summary_csv, index=False)
    print("Saved grouped summary:", summary_csv)

    overall_kw_results = []
    all_plot_paths = []

    for m in present_metrics:
        print(f"\n--- Metric: {m} ---")

        p_mean = plots_dir / f"mean_{m}_vs_ntasks.svg"
        line_plot_by_ntasks(df, "ntasks", m, str(p_mean),
                            ylabel=f"mean {m}", title=f"Mean {m} vs ntasks")
        all_plot_paths.append(str(p_mean))
        print("Saved:", p_mean)

        p_mean_norm = plots_dir / f"mean_{m}_per_task_vs_ntasks.svg"
        line_plot_by_ntasks(df, "ntasks", m, str(p_mean_norm),
                            ylabel=f"mean {m} per task",
                            title=f"Mean {m}/ntasks vs ntasks",
                            normalize_by_ntasks=True)
        all_plot_paths.append(str(p_mean_norm))
        print("Saved:", p_mean_norm)

        p_box = plots_dir / f"boxplot_{m}_by_ntasks.svg"
        boxplot_by_ntasks(df, m, str(p_box), ylabel=m, title=f"Distribution of {m} by ntasks")
        all_plot_paths.append(str(p_box))
        print("Saved:", p_box)

        kw_stat, kw_p = kruskal_test_by_task_id(df, m)
        overall_kw_results.append({"metric": m, "kw_stat": kw_stat, "kw_p": kw_p})
        print(f"Kruskal-Wallis (pooled across ntasks) comparing task_id groups for {m}: stat={kw_stat}, p={kw_p}")

        pernt = kruskal_per_ntasks(df, m)
        pernt_csv = tables_dir / f"kruskal_per_ntasks_{m}.csv"
        pernt.to_csv(pernt_csv, index=False)
        print("Saved per-ntasks Kruskal results to", pernt_csv)

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
        line_plot_by_ntasks(df, "ntasks", "switches", str(p_sw),
                            ylabel="mean switches", title="Mean switches vs ntasks")
        all_plot_paths.append(str(p_sw))
        p_sw_box = plots_dir / "boxplot_switches_by_ntasks.svg"
        boxplot_by_ntasks(df, "switches", str(p_sw_box), ylabel="switches", title="Distribution of switches by ntasks")
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
