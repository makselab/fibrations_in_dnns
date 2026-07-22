import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np
from scipy.interpolate import interp1d

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name', type=str, required=True, help='Exp Name')
parser.add_argument('-PATHresults', type=str, required=True, help='Results directory')
args = parser.parse_args()

PATH_RES = args.PATHresults
filename_1 = PATH_RES + f'{args.exp_name}/compression_results_thr_599.csv'
data1 = pd.read_csv(filename_1)

filename_2 = PATH_RES + f'{args.exp_name}/Evaluation_CompressedModels_epoch_599.csv'
data2 = pd.read_csv(filename_2)

keys = ["thr_fib_0", "thr_fib_1", "thr_fib_2", "thr_opfib_0", "thr_opfib_1", "thr_opfib_2"]
data = pd.concat([data1, data2.drop(columns=keys)], axis=1)

fig, axs = plt.subplots(2,2,figsize=(14,14))
axs[0][0].set_xlim([0,1.0])
axs[0][1].set_xlim([0,1.0])
axs[1][0].set_xlim([0,1.0])
axs[1][1].set_xlim([0,1.0])

axs[0][0].set_ylim([0,100])
axs[0][0].set_ylabel('Accuracy')
axs[1][0].set_xlabel('Compressed Size')
axs[1][0].set_ylabel('Loss')
axs[1][1].set_xlabel('Compressed Size (Nodes)')


# axs[0][0].set_xlabel('Compressed Size')
# axs[0][0].set_ylabel('Accuracy')

# -----------------------------------------------------------
# ABLATION AVERAGE

reductions_nodes_common = np.linspace(0, 1, 100)
reductions_params_common = np.linspace(0, 1, 100)

acc_abl_params_mean = []
acc_abl_nodes_mean = []

loss_abl_params_mean = []
loss_abl_nodes_mean = []

for idx_abl in range(10):
	acc_abl_params = interp1d(data['reduction_pars_abl_' + str(idx_abl)], data['acc_abl_' + str(idx_abl)], bounds_error=False)(reductions_params_common)
	acc_abl_nodes = interp1d(data['reduction_nodes'], data['acc_abl_' + str(idx_abl)], bounds_error=False)(reductions_nodes_common)

	acc_abl_params_mean.append(acc_abl_params)
	acc_abl_nodes_mean.append(acc_abl_nodes)

	loss_abl_params = interp1d(data['reduction_pars_abl_' + str(idx_abl)], data['loss_abl_' + str(idx_abl)], bounds_error=False)(reductions_params_common)
	loss_abl_nodes = interp1d(data['reduction_nodes'], data['loss_abl_' + str(idx_abl)], bounds_error=False)(reductions_nodes_common)

	loss_abl_params_mean.append(loss_abl_params)
	loss_abl_nodes_mean.append(loss_abl_nodes)


acc_abl_nodes_mean = np.array(acc_abl_nodes_mean)
acc_abl_params_mean = np.array(acc_abl_params_mean)

acc_abl_nodes_mean = np.mean(acc_abl_nodes_mean, axis=0)
acc_abl_params_mean = np.mean(acc_abl_params_mean, axis=0)

loss_abl_nodes_mean = np.array(loss_abl_nodes_mean)
loss_abl_params_mean = np.array(loss_abl_params_mean)

loss_abl_nodes_mean = np.mean(loss_abl_nodes_mean, axis=0)
loss_abl_params_mean = np.mean(loss_abl_params_mean, axis=0)

data.sort_values("reduction_pars_coll", inplace=True)

for name, gp in data.groupby("thr_opfib_0"):
	axs[0][0].plot(gp['reduction_pars_coll'],gp['acc_coll'], label='Opf_Thr='+str(name))
	axs[1][0].plot(gp['reduction_pars_coll'],gp['loss_coll'])


data.sort_values("reduction_nodes", inplace=True)
axs[0][1].plot(data['reduction_nodes'],data['acc_pruned'], color='pink')
axs[0][1].plot(reductions_nodes_common, acc_abl_params_mean, color='blue')

axs[1][1].plot(data['reduction_nodes'],data['loss_pruned'], color='pink')
axs[1][1].plot(reductions_nodes_common, loss_abl_nodes_mean, color='blue')

for name, gp in data.groupby("thr_opfib_0"):
	axs[0][1].plot(gp['reduction_nodes'],gp['acc_coll'])
	axs[1][1].plot(gp['reduction_nodes'],gp['loss_coll'])

data.sort_values("reduction_pars_pruned", inplace=True)
axs[0][0].plot(data['reduction_pars_pruned'],data['acc_pruned'], color='pink', label = 'Random Pruning')
axs[0][0].plot(reductions_params_common, acc_abl_params_mean, color='blue', label = 'L2 Pruning')
axs[0][0].legend()
axs[1][0].plot(data['reduction_pars_pruned'],data['loss_pruned'], color='pink')
axs[1][0].plot(reductions_params_common, loss_abl_params_mean, color='blue')


fig.savefig(PATH_RES + args.exp_name + '/Optimal_Metrics_curve.svg',format='svg')

plt.show()