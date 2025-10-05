#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description="Plot metrics from a CSV file")
parser.add_argument("csv_file", help="Path to the CSV file to read")
args = parser.parse_args()
df = pd.read_csv(args.csv_file)
df['NTASKS'] = df['NTASKS'].astype(int)
df['FREQ'] = df['FREQ'].astype(int)
df['segment'] = df['segment'].astype(int)
df['stopped'] = df['stopped'].astype(int)

for c in ['max_dt','mean','rel_imbalance']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df = df.sort_values(['NTASKS','FREQ','ALGO','segment']).reset_index(drop=True)
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
metrics = ['max_dt','mean','rel_imbalance']
groups = df.groupby(['NTASKS','FREQ','ALGO'])
plot_groups = {}

for (nt, fq, algo), g in groups:
    key = (nt, fq)
    label = f"{nt}-F{fq}-{algo}"
    plot_groups.setdefault(key, []).append((label, g))


for metric in metrics:
    for (nt, fq), group_items in sorted(plot_groups.items()):
        plt.figure(figsize=(10,4))
        ax = plt.gca()
        for label, g in group_items:
            ax.plot(g['segment'], g[metric], marker='o', label=label)
            stopped = g[g['stopped']==1]
            if not stopped.empty:
                ax.plot(stopped['segment'], stopped[metric], marker='x', linestyle='None')
        ax.set_xlabel('segment')
        ax.set_ylabel(metric)
        # ax.set_title(f'{metric} evolution (NTASKS={nt}, FREQ={fq})')
        ax.grid(True, linestyle=':', linewidth=0.5)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{metric}_NT{nt}_F{fq}.svg")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print("Saved plot:", out_path)

summary_csv = os.path.join(output_dir, "summary_head.csv")
df.head(200).to_csv(summary_csv, index=False)
print("Summary CSV saved:", summary_csv)
