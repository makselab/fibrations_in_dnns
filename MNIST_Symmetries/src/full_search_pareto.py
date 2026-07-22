import argparse
import os

import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =====================================================
# Load args, paths.

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name', type=str, required=True, help='Exp Name')
parser.add_argument('-PATHresults', type=str, required=True, help='Results directory')

args = parser.parse_args()

results_folder = args.PATHresults + args.exp_name + '/'

# =====================================================

def pareto_frontier(df, col1="dL", col2="params", minimize=(True, True)):
    vals = df[[col1, col2]].to_numpy(dtype=np.float64)

    if not minimize[0]: vals[:, 0] = -vals[:, 0]
    if not minimize[1]: vals[:, 1] = -vals[:, 1]

    order = np.lexsort((vals[:, 1], vals[:, 0]))
    sorted_vals = vals[order]

    is_pareto = np.zeros(len(df), dtype=bool)
    min_col2 = np.inf
    for i, (_, v2) in enumerate(sorted_vals):
        if v2 <= min_col2:
            is_pareto[order[i]] = True
            min_col2 = v2

    return is_pareto

# =====================================================

csv_path = results_folder + 'Full_Grid_dL_vs_thr.csv'

if not os.path.exists(csv_path):
    txt_path = results_folder + 'Full_Grid_dL_vs_thr.txt'
    with open(txt_path, 'r') as f:
        url = next(line.strip() for line in f if line.strip().startswith('http'))
    print(f'Downloading Full_Grid_dL_vs_thr.csv from Zenodo...')
    response = requests.get(url)
    response.raise_for_status()
    with open(csv_path, 'wb') as f:
        f.write(response.content)
    print('Download complete.')

# =====================================================

data = pd.read_csv(csv_path)

pareto_mask = pareto_frontier(data, col1="dL", col2="reduction_pars_coll")
pareto_df = data[pareto_mask].sort_values("dL").reset_index(drop=True)

print(f"Puntos totales  : {len(data):,}")
print(f"Frontera Pareto : {pareto_mask.sum():,} puntos")

# =====================================================
# Plot

fig, ax = plt.subplots(1, 1)
ax.scatter(data['reduction_pars_coll'], data['dL'])
ax.plot(pareto_df['reduction_pars_coll'], pareto_df['dL'], '-', color='crimson', ms=4, lw=1.5, label='Pareto frontier')
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.35)
ax.set_xlabel('Compressed Size')
ax.set_ylabel('dL')
ax.legend()
plt.tight_layout()

fig.savefig(results_folder + 'Optimal_Pareto_dL.png', format='png')
plt.close(fig)

# =====================================================
# Save

pareto_df.to_csv(results_folder + 'pareto_frontier.csv', index=False)
