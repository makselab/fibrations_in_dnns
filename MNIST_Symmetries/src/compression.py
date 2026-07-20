# =====================================================
# MODULES

import argparse
import os
import copy
import random
from itertools import product

import torch
from torchvision.datasets import MNIST
import torch.nn.utils.prune as prune
from model import MLP

import pandas as pd

# =====================================================
# Load args, paths, device.

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name', type=str, required=True, help='Exp Name')
parser.add_argument('-PATHtrain', type=str, required=True, help='Training directory')
parser.add_argument('-PATHresults', type=str, required=True, help='Results directory')
parser.add_argument('-epoch', type=int, required=True, help='Epoch')

args = parser.parse_args()

dev = torch.device("cuda:0")

# =====================================================
model_filename = args.PATHtrain    + args.exp_name + '/checkpoints/model_batch_' + str(args.epoch) + '.pth'
net = torch.load(model_filename)
net.to(dev)
num_params = sum(p.numel() for p in net.parameters())
num_nodes = sum(net.dims[1:-1]) 
print('Num Params:', num_params)
print('Num Nodes:', num_nodes)

dataframe = []

for opfib_thrs in [1.0,0.7,0.5]:
	curve_filename = args.PATHresults  + args.exp_name + f'/optimal_curves/opfib_{opfib_thrs}.csv'
	dfs = pd.read_csv(curve_filename)
	dfs['thr_opf'] = opfib_thrs
	dataframe.append(dfs)

dataframe = pd.concat(dataframe, ignore_index=True)

# ==================================================================
# COLORING & COLLAPSE

data = []
clustering_method = {'name': 'agg_clustering', 'cfg': {'linkage': 'average'}}

for row in dataframe.itertuples():

	# Collaping by Symmetries --------------------------------------
	fib_thrs = torch.Tensor([row.thr_fib_0, row.thr_fib_1, row.thr_fib_2])
	opfib_thrs = torch.Tensor([row.thr_opf, row.thr_opf, row.thr_opf])

	net.covering_coloring(clustering_method,fib_thrs, opfib_thrs)
	num_colors = net.num_colors('covering')
	num_nodes_coll = sum(num_colors)

	net_coll, _ = net.collapse_version('covering')
	num_params_coll = sum(p.numel() for p in net_coll.parameters())

	name_coll = args.exp_name + '_epoch_' + str(args.epoch) + '_coll_' + f'thrs_{fib_thrs}_{opfib_thrs}'
	coll_folder = args.PATHtrain + name_coll + '/checkpoints/'
	if not os.path.exists(coll_folder): os.makedirs(coll_folder)
	torch.save(net_coll, coll_folder + 'model_batch_0.pth')

	# Ablation ----------------------------------------------------
	abl_results = []

	for idx_ablation in range(10):
		net_abl = net.ablation_version(num_nodes_coll)
		num_params_abl = sum(p.numel() for p in net_abl.parameters())

		name_abl = args.exp_name + '_epoch_' + str(args.epoch) + '_abl_' + str(idx_ablation) + f'_thrs_{fib_thrs}_{opfib_thrs}'
		abl_folder = args.PATHtrain + name_abl + '/checkpoints/'
		if not os.path.exists(abl_folder): os.makedirs(abl_folder)
		torch.save(net_abl , abl_folder + 'model_batch_0.pth')

		abl_results.append({'num_L1_abl_'+str(idx_ablation): net_abl.dims[1],
							'num_L2_abl_'+str(idx_ablation): net_abl.dims[2],
							'num_L3_abl_'+str(idx_ablation): net_abl.dims[3],
							'num_params_abl_'+str(idx_ablation): num_params_abl,
							'reduction_pars_abl_'+str(idx_ablation): num_params_abl/num_params})

	# Pruning -----------------------------------------------------
	amount = 1.*(num_nodes - num_nodes_coll)/(num_nodes)
	net_pruned =  net.pruning_version(amount)
	num_params_pruned = sum(p.numel() for p in net_pruned.parameters())

	name_pruned = args.exp_name + '_epoch_' + str(args.epoch) + '_pruned_thrs' + f'_{fib_thrs}_{opfib_thrs}'
	pruned_folder = args.PATHtrain + name_pruned + '/checkpoints/'
	if not os.path.exists(pruned_folder): os.makedirs(pruned_folder)
	torch.save(net_pruned, pruned_folder + 'model_batch_0.pth')

	# Saving -----------------------------------------------
	row_partial = {}
	for diccionario in abl_results:
		row_partial.update(diccionario)

	row = {
	'thr_fib_0': row.thr_fib_0, 
	'thr_fib_1': row.thr_fib_1, 
	'thr_fib_2': row.thr_fib_2,
	'thr_opfib_0': row.thr_opf, 
	'thr_opfib_1': row.thr_opf, 
	'thr_opfib_2': row.thr_opf,
	'num_nodes_coll': num_nodes_coll,
	'reduction_nodes': num_nodes_coll/num_nodes,
	'num_colors_l1': num_colors[0],
	'num_colors_l2': num_colors[1],
	'num_colors_l3': num_colors[2],
	'num_params_coll': num_params_coll,
	'reduction_pars_coll': num_params_coll/num_params,
	'num_L1_pruned': net_pruned.dims[1],
	'num_L2_pruned': net_pruned.dims[2],
	'num_L3_pruned': net_pruned.dims[3],
	'num_params_pruned': num_params_pruned,
	'reduction_pars_pruned': num_params_pruned/num_params,
	}

	row.update(row_partial)
	data.append(row)

df = pd.DataFrame(data)
results_filename = args.PATHresults + args.exp_name + f'/compression_results_thr_{args.epoch}.csv'
df.to_csv(results_filename, index=False)