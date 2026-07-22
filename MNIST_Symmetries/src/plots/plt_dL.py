import argparse

import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# Load args, paths.

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name', type=str, required=True, help='Exp Name')
parser.add_argument('-PATHresults', type=str, required=True, help='Results directory')

args = parser.parse_args()

results_folder = args.PATHresults + args.exp_name + '/'

# =====================================================
# Load data

df_approx = pd.read_csv(results_folder + 'pareto_frontier.csv')
df_eval   = pd.read_csv(results_folder + 'Ev_pareto_frontier_599.csv')

df_eval = df_eval.rename(columns={'opfib_0': 'opf_0', 'opfib_1': 'opf_1', 'opfib_2': 'opf_2'})

merge_keys = ['fib_0', 'fib_1', 'fib_2', 'reduction_pars_coll']
df = pd.merge(df_approx, df_eval[merge_keys + ['loss_coll']], on=merge_keys)

baseline_loss = df_eval['loss_coll'].iloc[0]
df['dL_real'] = df['loss_coll'] - baseline_loss

# =====================================================
# Plot

fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(df['dL'], df['dL_real'], alpha=0.7)
ax.plot([df['dL'].min(), df['dL'].max()],
        [df['dL'].min(), df['dL'].max()],
        '--', color='gray', lw=1, label='y = x')

ax.set_xlabel('dL Approximate')
ax.set_ylabel('dL Real')
ax.legend()
plt.tight_layout()

fig.savefig(results_folder + 'dL_approx_vs_real.svg', format='svg')
plt.close(fig)
