# =====================================================
# MODULES

import argparse
import os
import copy
import re

import torch
from model import MLP

import pandas as pd

# =====================================================
# Load args, paths, device.

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name', type=str, required=True, help='Exp Name')
parser.add_argument('-PATHtrain', type=str, required=True, help='Training directory')
parser.add_argument('-PATHresults', type=str, required=True, help='Results directory')
parser.add_argument('-distance_thrs', type=float, nargs='+', required=True, help='Thresholds')
args = parser.parse_args()

dev = torch.device("cuda:0")

exp_folder = args.PATHtrain + args.exp_name + '/checkpoints'

# ==================================================================
# COLORING

clustering_method = {'name': 'agg_clustering', 'cfg': {'linkage': 'average'}}
distance_thrs = torch.Tensor(args.distance_thrs)

epochs = []
pattern = re.compile(r'model_batch_(\d+)\.pth')

for file in os.listdir(exp_folder):
    coincidencia = pattern.match(file)
    if coincidencia:
        epochs.append(int(coincidencia.group(1)))

epochs = sorted(epochs)
num_epochs = len(epochs)

data = []

for epoch in epochs:
	# Checkpoint ---------------------------------------------------
	
	filename = args.PATHtrain + args.exp_name + '/checkpoints/model_batch_' + str(epoch) + '.pth'
	net = torch.load(filename)
	net.to(dev)
	num_params = sum(p.numel() for p in net.parameters())
	num_nodes = sum(net.dims[1:-1])

	# Coloring -----------------------------------------------------

	net.covering_coloring(clustering_method, distance_thrs[:net.num_layers], distance_thrs[net.num_layers:])
	net_coll, _ = net.collapse_version('covering')
	num_params_coll = sum(p.numel() for p in net_coll.parameters())
	num_nodes_coll = sum(net_coll.dims[1:-1])

	# Saving -----------------------------------------------

	row = {
	'epoch': epoch,
	'num_nodes_coll': num_nodes_coll,
	'num_params_coll': num_params_coll,
	'reduction_pars_coll': num_params_coll/num_params,
	'reduction_nodes': num_nodes_coll/num_nodes
	}

	for idx_layer in range(net.num_layers):
		row[f'num_colors_l{idx_layer}'] = net_coll.dims[1+idx_layer]

	data.append(row)

df = pd.DataFrame(data)
results_filename = args.PATHresults + args.exp_name + f'/Symmetries_thrs_{args.distance_thrs}.csv'
df.to_csv(results_filename, index=False)