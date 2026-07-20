import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import interp1d

data1 = '/home/ovelarde/snap/snapd-desktop-integration/253/Documents/MNIST_Symmetries/results/Evaluation_exp_01_thr_599.csv'
data2 = '/home/ovelarde/snap/snapd-desktop-integration/253/Documents/MNIST_Symmetries/results/Collapse_exp_01_thr_599.csv'


df1 = pd.read_csv(data1)
df2 = pd.read_csv(data2)

df_merged = pd.merge(df1, df2, on='threshold', how='outer')

# ----------------------------------

reductions_nodes_common = np.linspace(0, 1, 100)
reductions_params_common = np.linspace(0, 1, 100)

curves_params = []
curves_nodes = []

for idx_abl in range(10):
	acc_abl_params = interp1d(df_merged['reduction_pars_abl_' + str(idx_abl)], df_merged['acc_abl_' + str(idx_abl)], bounds_error=False)(reductions_params_common)
	acc_abl_nodes = interp1d(df_merged['reduction_nodes'], df_merged['acc_abl_' + str(idx_abl)], bounds_error=False)(reductions_nodes_common)

	curves_params.append(acc_abl_params)
	curves_nodes.append(acc_abl_nodes)

curves_nodes = np.array(curves_nodes)
curves_params = np.array(curves_params)

average_curve_nodes = np.mean(curves_nodes, axis=0)
average_curve_params = np.mean(curves_params, axis=0)

# -------------------------------------------
fig, axs = plt.subplots(1,1,figsize=(10, 6))
axs.plot(df_merged['reduction_nodes'], df_merged['acc_coll'], color='red')
# axs.plot(df_merged['reduction_nodes'], df_merged['acc_abl'], color='blue')
axs.plot(reductions_nodes_common, average_curve_nodes, color='blue')
axs.plot(df_merged['reduction_nodes'], df_merged['acc_pruned'], color='yellow')

axs.set_xlabel('Percentage Reduction Nodes')
axs.set_ylabel('Performance')
axs.set_ylim([0,100])
axs.set_xlim([0,1])

fig.savefig('acc_vs_reduction_nodes.svg', format='svg')

# -------------------------------------------

fig, axs = plt.subplots(1,1,figsize=(10, 6))
axs.plot(df_merged['reduction_pars_coll'], df_merged['acc_coll'], color='red')
# axs.plot(df_merged['reduction_pars_abl'], df_merged['acc_abl'], color='blue')
axs.plot(reductions_params_common, average_curve_params, color='blue')
axs.plot(df_merged['reduction_pars_pruned'], df_merged['acc_pruned'], color='yellow')

axs.set_xlabel('Percentage Reduction Params')
axs.set_ylabel('Performance')
axs.set_ylim([0,100])
axs.set_xlim([0,1])

fig.savefig('acc_vs_reduction_params.svg', format='svg')