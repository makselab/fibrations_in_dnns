import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name',   type=str,   required=True,        help='Exp Name')
parser.add_argument('-PATHresults',type=str,   required=True,        help='Results directory')
parser.add_argument('-epoch',      type=int,   default=599,          help='Epoch')
parser.add_argument('-version',    type=str,   default='v2',         help='Version: v1 or v2')
parser.add_argument('-matrices',   type=str,   nargs='+',
                    default=['S', 'P', 'D', 'dL'],                   help='Matrices to plot')
args = parser.parse_args()

PATH = args.PATHresults + args.exp_name + '/'
df   = pd.read_csv(PATH + f'dL_distributions_{args.version}_{args.epoch}.csv')

percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
p_cols      = [f'p{p}' for p in percentiles]

num_layers  = df['layer'].nunique()
num_mats    = len(args.matrices)

fig, axes = plt.subplots(num_layers, num_mats, figsize=(4 * num_mats, 3 * num_layers), squeeze=False)

for row_idx, layer_idx in enumerate(sorted(df['layer'].unique())):
    for col_idx, mat in enumerate(args.matrices):
        ax   = axes[row_idx][col_idx]
        vals = df[(df['layer'] == layer_idx) & (df['matrix'] == mat)][p_cols].values[0]

        ax.plot(vals, [p / 100 for p in percentiles], marker='o', markersize=3)
        ax.set_title(f'Layer {layer_idx} — {mat}')
        ax.set_xlabel('Value')
        ax.set_ylabel('CDF')
        ax.grid(True, alpha=0.3)

fig.suptitle(f'{args.exp_name}  |  version {args.version}  |  epoch {args.epoch}', y=1.01)
fig.tight_layout()

out = PATH + f'plots/plt_dL_cdf_{args.version}_{args.epoch}'
fig.savefig(out + '.svg', format='svg', bbox_inches='tight')
fig.savefig(out + '.png', format='png', dpi=150, bbox_inches='tight')

plt.show()
