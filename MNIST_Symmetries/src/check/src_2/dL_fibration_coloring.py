# =====================================================
# MODULES

import argparse
import os
import copy
import random
import itertools

import torch
from torchvision.datasets import MNIST
import torch.nn.utils.prune as prune
from model import MLP

import pandas as pd

from coloring import fibration_linear

# =====================================================
# Load args, paths, device.

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name', type=str, required=True, help='Exp Name')
args = parser.parse_args()
dev = torch.device("cuda:0")

# =====================================================
# Model

# filename = '/home/osvaldo/Documents/CCNY/MNIST_Symmetries/train_dir/' + args.exp_name + '/gradients_training.pth'
# data = torch.load(filename)
# last_point = data[-1]
# grads = {p: torch.stack([x[p]['grad'] for x in data]).mean(dim=0) for p in last_point.keys()} #av

filename = '/home/osvaldo/Documents/CCNY/MNIST_Symmetries/train_dir/' + args.exp_name + '/gradients_test.pth'
data = torch.load(filename)
last_point = data[-1]
grads = {p: torch.stack([x[p]['grad'] for x in data]).mean(dim=0) for p in last_point.keys()} #av

hidden_size = last_point['fc1.weight']['param'].shape[0]
num_params = sum(v['param'].numel() for v in last_point.values())
print('Num Params:', num_params)

# ======================================================
# COLORING & COLLAPSE

num_thresholds_per_layer = 21
values_threshold_1= torch.linspace(0,1, num_thresholds_per_layer)
values_threshold_2= torch.linspace(0,1, num_thresholds_per_layer)
values_threshold_3= torch.linspace(0,1, num_thresholds_per_layer)
num_thresholds = num_thresholds_per_layer ** 3

data = []

for idx_th, (t1, t2, t3) in enumerate(itertools.product(values_threshold_1, values_threshold_2, values_threshold_3)):
	print(idx_th, num_thresholds)
	# print(t1,t2,t3)

	# COLORING ----------------------------------------
	colors = []

	colors_l1 = fibration_linear(weights=last_point['fc1.weight']['param'], 
	                          in_clusters=None, 
	                          threshold=t1.item(), 
	                          first_layer = True,
	                          bias = last_point['fc1.bias']['param'])
	num_colors_l1 = torch.unique(colors_l1).shape[0]

	colors.append(colors_l1)

	colors_l2 = fibration_linear(weights=last_point['fc2.weight']['param'], 
	                          in_clusters=colors_l1, 
	                          threshold=t2.item(), 
	                          first_layer = False,
	                          bias = last_point['fc2.bias']['param'])
	num_colors_l2 = torch.unique(colors_l2).shape[0]

	colors.append(colors_l2)

	colors_l3 = fibration_linear(weights=last_point['fc3.weight']['param'], 
	                          in_clusters=colors_l2, 
	                          threshold=t3.item(), 
	                          first_layer = False,
	                          bias = last_point['fc3.bias']['param'])
	num_colors_l3 = torch.unique(colors_l3).shape[0]

	colors.append(colors_l3)

	# COLLAPSE FORMULAR ----------------------------------------
	mtxs_partition = [torch.zeros(torch.unique(colors_layer).shape[0], hidden_size).scatter_(0, colors_layer.unsqueeze(0), 1).to(dev) for colors_layer in colors]
	szs_clusters = [torch.mm(mtx, torch.ones(hidden_size, 1).to(dev)) for mtx in mtxs_partition]

	Ws_coll = [(mtxs_partition[layer] @ last_point['fc' + str(layer+1) + '.weight']['param']) / szs_clusters[layer].view(-1, 1) for layer in range(3)] 
	Bs_coll = [(mtxs_partition[layer] @ last_point['fc' + str(layer+1) + '.bias']['param']) / szs_clusters[layer].view(-1,) for layer in range(3)] 

	# num_params_coll = sum(t.numel() for t in Ws_coll) + sum(t.numel() for t in Bs_coll) 

	# COLLAPSE -----------------------------------------------------
	net_coll = MLP(784, [num_colors_l1,num_colors_l2,num_colors_l3], 10)

	net_coll.fc1.weight.data = Ws_coll[0]
	net_coll.fc2.weight.data = Ws_coll[1] @ mtxs_partition[0].T
	net_coll.fc3.weight.data = Ws_coll[2] @ mtxs_partition[1].T
	net_coll.fc4.weight.data = last_point['fc4.weight']['param'] @ mtxs_partition[2].T

	net_coll.fc1.bias.data = Bs_coll[0]
	net_coll.fc2.bias.data = Bs_coll[1]
	net_coll.fc3.bias.data = Bs_coll[2]
	net_coll.fc4.bias.data = last_point['fc4.bias']['param']

	num_params_coll = sum(p.numel() for p in net_coll.parameters())

	coll_folder = '/media/osvaldo/OMV5TB/collapse_mnist/' +  args.exp_name + '_epoch_599_coll_thrs_' + str(t1.item()) + '_' + str(t2.item()) + '_' + str(t3.item()) + '/checkpoints/'

	if not os.path.exists(coll_folder): os.makedirs(coll_folder)
	torch.save(net_coll, coll_folder + 'model_batch_0.pth')

	# Approx W ----------------------------------------
	Ws_exp = [Ws_coll[layer][colors[layer]] for layer in range(3)]
	Bs_exp = [Bs_coll[layer][colors[layer]] for layer in range(3)]


	# dL ----------------------------------------------

	dL_W = [((Ws_exp[layer]-last_point['fc' + str(layer+1) + '.weight']['param']) * grads['fc' + str(layer+1) + '.weight']).sum(dim=1) for layer in range(3)]
	dL_b = [((Bs_exp[layer]-last_point['fc' + str(layer+1) + '.bias']['param']) * grads['fc' + str(layer+1) + '.bias']) for layer in range(3)]

	dL_layer = [((Ws_exp[layer]-last_point['fc' + str(layer+1) + '.weight']['param']) * grads['fc' + str(layer+1) + '.weight']).sum(dim=1) +\
				((Bs_exp[layer]-last_point['fc' + str(layer+1) + '.bias']['param']) * grads['fc' + str(layer+1) + '.bias']) \
				for layer in range(3)]

	dL_sum_layer = [dL_layer[layer].sum().item() for layer in range(3)]

	# Saving -----------------------------------------------

	row = {
	'thr1': t1.item(),
	'thr2': t2.item(),
	'thr3': t3.item(),
	'reduction_pars_coll': num_params_coll/num_params,
	'dL1': dL_sum_layer[0],
	'dL2': dL_sum_layer[1],
	'dL3': dL_sum_layer[2],
	}

	data.append(row)

	print(row)

df = pd.DataFrame(data)
results_filename = '/home/osvaldo/Documents/CCNY/MNIST_Symmetries/results/dL_vs_thr_' + args.exp_name + '_test.csv'
df.to_csv(results_filename, index=False)