# '''
# Title: Synchronization
# Author: Osvaldo M Velarde
# Project: XXX
# '''

# ===================================================================
# ===================== MODULES & PATHS =============================

# import pandas as pd
# from pandas import read_csv, crosstab
# from numpy import zeros
# import torch
# eps_act = 0.05
# eps_error = 0.0001
# dataPATH = '/home/osvaldo/Documents/CCNY/Project_BreakingSymmetry/results/exp_' + exp_idx + '/synchronization/activity/'
# annot = False
# fibers_vs_time = {0:[],1:[],2:[],3:[],4:[]}

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import numpy as np

exp_idx = '03'
dataPATH = '/media/osvaldo/OMV5TB/BreakingSymmetry/results/exp_' + exp_idx + '/activations_errors/'
plotPATH = '/home/osvaldo/Documents/CCNY/Project_BreakingSymmetry/results/exp_' + exp_idx + '/plots/histograms/'

# ===================================================================
# ====================== IO & MODEL FILES ===========================

task_id = 0
num_tasks = 4
epochs  = [0,50,100,150,200,250]
num_epochs = len(epochs)
class_idx = 0
num_layers = 5
eps_act = 0.05
eps_error = 0.0001

# ===================================================================
# ======= ACTIVITY VS SIZE OF FIBERS IN GRAL CASE ===================

max_error = [0.001, 0.001, 0.003, 0.002, 0.001]
max_act = [0.6, 1 ,1.2, 1.6, 2.6] 

for task_idx in range(task_id, task_id + num_tasks):
	for ep_idx in epochs:

		fig = plt.figure(figsize=(8*num_layers,8))
		gs = gridspec.GridSpec(3, 3*num_layers)
		ax_main =[None for ii in range(num_layers)]
		ax_xDist=[None for ii in range(num_layers)]
		ax_yDist=[None for ii in range(num_layers)]

		for ll in range(num_layers):	
			ax_main[ll] = plt.subplot(gs[1:3, 3*ll:3*ll+2])
			ax_xDist[ll] = plt.subplot(gs[0, 3*ll:3*ll+2],sharex=ax_main[ll])
			ax_yDist[ll] = plt.subplot(gs[1:3, ll*3 + 2],sharey=ax_main[ll])
			ax_main[ll].set(xlabel="Activity")
			ax_main[ll].set(ylabel="Error")
			ax_main[ll].set(xlim=[0,max_act[ll]])
			ax_main[ll].set(ylim=[-max_error[ll],max_error[ll]])

		with open(dataPATH + 'task_idx_' + str(task_idx) + '_epoch_' + str(ep_idx) + '.pkl', 'rb') as f:
			activations = pickle.load(f)

		# L1 (batch x 32ch x 14 x 14),  L2 (batch x 64ch x 06 x 06),  L3 (batch x 512), L4 (batch x 128), L5 (batch x 128)
		mean_activations = [LL.mean(dim=0).detach() for LL in activations['activity'][class_idx]]
		mean_activations[0] = mean_activations[0].mean(dim=(1,2))
		mean_activations[1] = mean_activations[1].mean(dim=(1,2))

		mean_errors = {k: v.mean(dim=0).detach() for k, v in activations['errors'].items()}
		mean_errors[0] = mean_errors[0].mean(dim=(1,2))
		mean_errors[1] = mean_errors[1].mean(dim=(1,2))

		for ll in range(num_layers):
			mean_activations[ll] = mean_activations[ll].cpu().numpy()
			mean_errors[ll] = mean_errors[ll].cpu().numpy()

			ax_main[ll].scatter(mean_activations[ll], mean_errors[ll], marker='.', color='blue')

			act_bins = np.arange(min(mean_activations[ll]), max(mean_activations[ll]) + eps_act, eps_act)
			ax_xDist[ll].hist(mean_activations[ll],bins=act_bins,align='mid',alpha=0.3, color='blue', density=True)
			ax_xDist[ll].set(ylabel='county')

			error_bins = np.arange(min(mean_errors[ll]), max(mean_errors[ll]) + eps_error, eps_error)
			ax_yDist[ll].hist(mean_errors[ll],bins=error_bins,orientation='horizontal',align='mid',alpha=0.3, color='blue', density=True)
			ax_yDist[ll].set(xlabel='countx')

		fig.savefig(plotPATH + 'Task_' + str(task_idx) + '_Epoch_' + str(ep_idx) + '.svg', format='svg')
		plt.close(fig)

# ===================================================================