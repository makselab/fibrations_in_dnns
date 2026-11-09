import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-PATHresults', type=str, required=True)
parser.add_argument('-exp_pattern', type=str, default='exp_01_r*')
parser.add_argument('-epoch',       type=int, default=599)
parser.add_argument('-output',      type=str, default=None)
args = parser.parse_args()

csv_files = sorted(glob.glob(
    os.path.join(args.PATHresults, args.exp_pattern, f'sensibility_analysis_{args.epoch}.csv')
))

if len(csv_files) == 0:
    raise FileNotFoundError(f'No sensibility_analysis_{args.epoch}.csv found.')

print(f'Found {len(csv_files)} runs.')
dfs       = [pd.read_csv(f) for f in csv_files]
run_names = [os.path.basename(os.path.dirname(f)) for f in csv_files]

for df in dfs:
    loss_0 = df.loc[
        (df['thr_fib_0'] == df['thr_fib_0'].min()) &
        (df['thr_fib_1'] == df['thr_fib_1'].min()) &
        (df['thr_fib_2'] == df['thr_fib_2'].min()), 'loss'
    ].values[0]
    df['dL'] = df['loss'] - loss_0

thr_cols     = ['thr_fib_0', 'thr_fib_1', 'thr_fib_2']
layer_labels = ['Layer 1', 'Layer 2', 'Layer 3']

# =====================================================
# Sobol indices via law of total variance (valid for full factorial design)
#
# S1_i  = V(E[Y | X_i]) / V(Y)          — first-order (main effect)
# ST_i  = 1 - V(E[Y | X_{-i}]) / V(Y)  — total-order (main + all interactions)
# ST_i - S1_i captures interaction effects involving variable i

def sobol_indices(df, target='dL'):
    V_total = df[target].var(ddof=1)
    S1, ST = [], []
    for col in thr_cols:
        # First-order
        cond_mean = df.groupby(col)[target].mean()
        S1.append(cond_mean.var(ddof=1) / V_total)
        # Total-order
        other_cols = [c for c in thr_cols if c != col]
        cond_mean_neg = df.groupby(other_cols)[target].mean()
        ST.append(1 - cond_mean_neg.var(ddof=1) / V_total)
    return np.array(S1), np.array(ST)

all_S1 = []
all_ST = []
for df in dfs:
    S1, ST = sobol_indices(df)
    all_S1.append(S1)
    all_ST.append(ST)

all_S1 = np.array(all_S1)  # (n_runs, 3)
all_ST = np.array(all_ST)

# =====================================================
# Plot

fig, ax = plt.subplots(figsize=(8, 5))
x     = np.arange(len(layer_labels))
width = 0.35

if len(dfs) == 1:
    ax.bar(x - width/2, all_S1[0], width, label='First-order $S_1$', color='steelblue')
    ax.bar(x + width/2, all_ST[0], width, label='Total-order $S_T$',  color='coral')
else:
    ax.bar(x - width/2, all_S1.mean(0), width, yerr=all_S1.std(0),
           label='First-order $S_1$', color='steelblue', capsize=4)
    ax.bar(x + width/2, all_ST.mean(0), width, yerr=all_ST.std(0),
           label='Total-order $S_T$',  color='coral',     capsize=4)

ax.set_xticks(x)
ax.set_xticklabels(layer_labels, fontsize=12)
ax.set_ylabel('Sobol Index', fontsize=12)
ax.set_ylim(0, 1)
ax.axhline(1, color='gray', linewidth=0.8, linestyle='--')
ax.legend(fontsize=11)
ax.grid(True, axis='y', alpha=0.3)
ax.set_title('Sobol Sensitivity Indices — ΔLoss vs Fiber Thresholds', fontsize=13)
plt.tight_layout()

output_path = args.output or os.path.join(args.PATHresults, 'sobol_indices.svg')
plt.savefig(output_path, bbox_inches='tight')
print(f'Saved to {output_path}')

# =====================================================
# Print summary
print('\nMean Sobol indices across runs:')
print(f'{"Layer":<10} {"S1":>8} {"ST":>8} {"Interaction (ST-S1)":>20}')
for i, label in enumerate(layer_labels):
    s1 = all_S1[:, i].mean()
    st = all_ST[:, i].mean()
    print(f'{label:<10} {s1:>8.3f} {st:>8.3f} {st - s1:>20.3f}')
print(f'\nSum S1 = {all_S1.mean(0).sum():.3f}  (remainder = interactions)')
