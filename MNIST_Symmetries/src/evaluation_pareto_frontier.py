# =====================================================
# MODULES

import argparse
import os

import copy
import random
from itertools import product

import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from model import MLP

import pandas as pd

from torchvision.transforms import ToTensor
from torch.nn import CrossEntropyLoss

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

# =====================================================
# DATASET

batch_size = 100 
dataPATH = "/home/osvaldo/Documents/CCNY/MNIST_Symmetries/"
test_data = MNIST(root = dataPATH, train = False, transform = ToTensor())
test_gen = DataLoader(dataset = test_data, batch_size = batch_size, shuffle = False)

# =====================================================
# EVALUATION
dataframe = pd.read_csv(args.PATHresults + args.exp_name + '/pareto_frontier.csv')
loss_function = CrossEntropyLoss()
data = []

# ==================================================================
# COLORING & COLLAPSE

data = []
clustering_method = {'name': 'agg_clustering', 'cfg': {'linkage': 'average'}}

for row in dataframe.itertuples():
	print(row)

	# Collaping by Symmetries --------------------------------------
	fib_thrs = torch.Tensor([row.fib_0, row.fib_1, row.fib_2])
	opfib_thrs = torch.Tensor([row.opf_0, row.opf_1, row.opf_2])

	net.covering_coloring(clustering_method,fib_thrs, opfib_thrs)
	num_colors = net.num_colors('covering')
	num_nodes_coll = sum(num_colors)

	net_coll, _ = net.collapse_version('covering')
	net_coll.to(dev)
	num_params_coll = sum(p.numel() for p in net_coll.parameters())

	# Evaluation ----------------------------------------------------
	net_coll.eval()
	correct = 0
	total = 0
	loss_final = 0

	for images,labels in test_gen:
	    images = images.view(-1,784).to(dev)
	    labels = labels.to(dev)

	    out_1, out_2 = net_coll(images)
	    _, predicted = torch.max(out_2,1)
	    loss = loss_function(out_2, labels)
	    correct += (predicted == labels).sum()
	    total += labels.size(0)
	    loss_final += loss.item()

	acc_coll = (100*correct)/(total+1)
	loss_coll = loss_final/len(test_gen)

	# Saving -----------------------------------------------

	row = {
	'fib_0': row.fib_0, 
	'fib_1': row.fib_1, 
	'fib_2': row.fib_2,
	'opfib_0': row.opf_0, 
	'opfib_1': row.opf_1, 
	'opfib_2': row.opf_2,
	'num_nodes_coll': num_nodes_coll,
	'reduction_nodes': num_nodes_coll/num_nodes,
	'num_colors_l1': num_colors[0],
	'num_colors_l2': num_colors[1],
	'num_colors_l3': num_colors[2],
	'num_params_coll': num_params_coll,
	'reduction_pars_coll': num_params_coll/num_params,
	'acc_coll': acc_coll.item(),
	'loss_coll': loss_coll
	}

	data.append(row)

df = pd.DataFrame(data)
results_filename = args.PATHresults + args.exp_name + f'/Ev_pareto_frontier_{args.epoch}.csv'
df.to_csv(results_filename, index=False)