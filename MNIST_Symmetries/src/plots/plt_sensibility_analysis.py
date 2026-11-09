import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-PATHresults', type=str, required=True)
parser.add_argument('-exp_pattern', type=str, default='exp_01_r*', help='Glob pattern for experiment folders')
parser.add_argument('-epoch',       type=int, default=599)
parser.add_argument('-output',      type=str, default=None, help='Output path for the figure')
args = parser.parse_args()

# =====================================================
# Load all runs

csv_files = sorted(glob.glob(
    os.path.join(args.PATHresults, args.exp_pattern, f'sensibility_analysis_{args.epoch}.csv')
))

if len(csv_files) == 0:
    raise FileNotFoundError(f'No sensibility_analysis_{args.epoch}.csv found matching pattern.')

print(f'Found {len(csv_files)} runs.')

dfs = [pd.read_csv(f) for f in csv_files]

# Compute dL = loss - loss_0, where loss_0 is loss at minimum thresholds (no compression)
for df in dfs:
    loss_0 = df.loc[
        (df['thr_fib_0'] == df['thr_fib_0'].min()) &
        (df['thr_fib_1'] == df['thr_fib_1'].min()) &
        (df['thr_fib_2'] == df['thr_fib_2'].min()), 'loss'
    ].values[0]
    df['dL'] = df['loss'] - loss_0

# =====================================================
# Marginal effects: mean dL and num_colors vs thr_i

thr_cols     = ['thr_fib_0', 'thr_fib_1', 'thr_fib_2']
color_cols   = ['num_colors_l1', 'num_colors_l2', 'num_colors_l3']
layer_labels = ['Layer 1', 'Layer 2', 'Layer 3']

colors    = plt.cm.tab10(np.linspace(0, 1, len(dfs)))
run_names = [os.path.basename(os.path.dirname(f)) for f in csv_files]

# =====================================================
# Plot 1: Loss and Colors vs epsilon (2x3)

fig1, axes1 = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

for ax, thr_col, label in zip(axes1, thr_cols, layer_labels):

    for df, color, name in zip(dfs, colors, run_names):
        marginal_dL = df.groupby(thr_col)['dL'].mean()
        ax.plot(marginal_dL.index, marginal_dL.values, color=color, linewidth=1.5, label=name)

    ax.set_xlim(0, 2)
    ax.set_xlabel(f'ε ({label})', fontsize=12)
    ax.set_title(label, fontsize=12)
    ax.grid(True, alpha=0.3)

axes1[0].set_ylabel('ΔLoss', fontsize=12)
axes1[-1].legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
fig1.suptitle('Sensitivity Analysis: ΔLoss vs Fiber Threshold (marginal effects)', fontsize=13)
fig1.tight_layout()

output_path = args.output if args.output is not None else \
    os.path.join(args.PATHresults, f'sensibility_analysis_loss.svg')
fig1.savefig(output_path, bbox_inches='tight')
print(f'Saved to {output_path}')

# =====================================================
# Plot 2: Num Colors (layer L) vs epsilon (layer L') — 3x3 grid

fig2, axes2 = plt.subplots(3, 3, figsize=(14, 12), sharey=True, sharex='col')

for row, (color_col, row_label) in enumerate(zip(color_cols, layer_labels)):
    for col, (thr_col, col_label) in enumerate(zip(thr_cols, layer_labels)):
        ax = axes2[row][col]

        for df, color, name in zip(dfs, colors, run_names):
            marginal = df.groupby(thr_col)[color_col].mean()
            ax.plot(marginal.index, marginal.values, color=color, linewidth=1.5, label=name)

        ax.set_xlim(0, 2)
        ax.set_ylim(0, 500)
        ax.grid(True, alpha=0.3)

        if row == 0:
            ax.set_title(f'ε Layer {col + 1}', fontsize=11)
        if col == 0:
            ax.set_ylabel(f'Colors L{row + 1}', fontsize=11)
        if row == 2:
            ax.set_xlabel(f'ε (Layer {col + 1})', fontsize=10)

axes2[0][-1].legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
fig2.suptitle('Num Colors (layer L) vs ε (layer L\')', fontsize=13)
fig2.tight_layout()

output_path2 = os.path.join(args.PATHresults, f'sensibility_analysis_colors_grid.svg')
fig2.savefig(output_path2, bbox_inches='tight')
print(f'Saved to {output_path2}')
