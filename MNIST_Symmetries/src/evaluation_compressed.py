# =====================================================
# MODULES

import argparse
import os

import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from model import MLP
from torch.nn import CrossEntropyLoss

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
# DATASET

batch_size = 100 
dataPATH = "/home/osvaldo/Documents/CCNY/MNIST_Symmetries/"
test_data = MNIST(root = dataPATH, train = False, transform = ToTensor())
test_gen = DataLoader(dataset = test_data, batch_size = batch_size, shuffle = False)

# =====================================================
# EVALUATION

dataframe = []

for opfib_thrs in [1.0,0.7,0.5]:
	curve_filename = args.PATHresults  + args.exp_name + f'/optimal_curves/opfib_{opfib_thrs}.csv'
	dfs = pd.read_csv(curve_filename)
	dfs['thr_opf'] = opfib_thrs
	dataframe.append(dfs)

dataframe = pd.concat(dataframe, ignore_index=True)

loss_function = CrossEntropyLoss()
data = []

for row in dataframe.itertuples():
	print(row)

	fib_thrs = torch.Tensor([row.thr_fib_0, row.thr_fib_1, row.thr_fib_2])
	opfib_thrs = torch.Tensor([row.thr_opf, row.thr_opf, row.thr_opf])

	# Evaluation Collapse ------------------------------------------
	name_coll = args.exp_name + '_epoch_' + str(args.epoch) + '_coll_thrs_' + f'{fib_thrs}_{opfib_thrs}'
	coll_folder = args.PATHtrain + name_coll + '/checkpoints/'
	net_coll = torch.load(coll_folder + 'model_batch_0.pth')
	net_coll.to(dev)

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


	# Evaluation Ablation ------------------------------------------
	abl_results = []

	for idx_ablation in range(10):
		name_abl = args.exp_name + '_epoch_' + str(args.epoch) + '_abl_' + str(idx_ablation) + f'_thrs_{fib_thrs}_{opfib_thrs}'
		abl_folder = args.PATHtrain + name_abl + '/checkpoints/'
		net_abl = torch.load(abl_folder + 'model_batch_0.pth')
		net_abl.to(dev)

		net_abl.eval()
		correct = 0
		total = 0
		loss_final = 0

		for images,labels in test_gen:
		    images = images.view(-1,784).to(dev)
		    labels = labels.to(dev)

		    out_1, out_2 = net_abl(images)
		    _, predicted = torch.max(out_2,1)
		    loss = loss_function(out_2, labels)
		    correct += (predicted == labels).sum()
		    total += labels.size(0)
		    loss_final += loss.item()

		acc_abl = (100*correct)/(total+1)
		loss_abl = loss_final/len(test_gen)

		abl_results.append({'acc_abl_'+str(idx_ablation): acc_abl.item(),
							'loss_abl_'+str(idx_ablation): loss_abl})
	

	# Evaluation Pruning ------------------------------------------
	name_pruned = args.exp_name + '_epoch_' + str(args.epoch) + '_pruned' + f'_thrs_{fib_thrs}_{opfib_thrs}'
	pruned_folder = args.PATHtrain + name_pruned + '/checkpoints/'
	net_pruned = torch.load(pruned_folder + 'model_batch_0.pth')
	net_pruned.to(dev)

	net_pruned.eval()
	correct = 0
	total = 0
	loss_final += loss.item()

	for images,labels in test_gen:
	    images = images.view(-1,784).to(dev)
	    labels = labels.to(dev)

	    out_1, out_2 = net_pruned(images)
	    _, predicted = torch.max(out_2,1)
	    correct += (predicted == labels).sum()
	    total += labels.size(0)
	    loss_final += loss.item()

	acc_pruned = (100*correct)/(total+1)
	loss_pruned = loss_final/len(test_gen)

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
	'acc_coll': acc_coll.item(),
	'acc_pruned': acc_pruned.item(),
	'loss_coll': loss_coll,
	'loss_pruned': loss_pruned
	}

	row.update(row_partial)
	data.append(row)

df = pd.DataFrame(data)
results_filename = args.PATHresults + args.exp_name + f'/Evaluation_CompressedModels_epoch_{args.epoch}.csv'
df.to_csv(results_filename, index=False)