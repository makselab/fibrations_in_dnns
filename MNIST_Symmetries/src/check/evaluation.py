# =====================================================
# MODULES

import argparse
import os

import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
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
# DATASET

batch_size = 100 
dataPATH = "/home/ovelarde/snap/snapd-desktop-integration/253/Documents/MNIST_Symmetries/"
test_data = MNIST(root = dataPATH, train = False, transform = ToTensor())
test_gen = DataLoader(dataset = test_data, batch_size = batch_size,shuffle = False)

# ==================================================================
# Evaluation

num_thresholds = 101
values_threshold= torch.linspace(0,1, num_thresholds)

data = []

for idx_th, thr in enumerate(values_threshold):
	print(thr)

	# Evaluation Collapse ------------------------------------------
	name_coll = args.exp_name + '_epoch_' + str(args.epoch) + '_coll_thr_' + str(thr.item())
	coll_folder = args.PATHtrain + name_coll + '/checkpoints/'
	net_coll = torch.load(coll_folder + 'model_batch_0.pth', weights_only=False)
	net_coll.to(dev)

	net_coll.eval()
	correct = 0
	total = 0

	for images,labels in test_gen:
	    images = images.view(-1,784).to(dev)
	    labels = labels.to(dev)

	    out_1, out_2 = net_coll(images)
	    _, predicted = torch.max(out_2,1)
	    correct += (predicted == labels).sum()
	    total += labels.size(0)

	acc_coll = (100*correct)/(total+1)
	

	# Evaluation Ablation ------------------------------------------

	abl_results = []

	for idx_ablation in range(10):
		name_abl = args.exp_name + '_epoch_' + str(args.epoch) + '_abl_' + str(idx_ablation) + '_thr_' + str(thr.item())
		abl_folder = args.PATHtrain + name_abl + '/checkpoints/'
		net_abl = torch.load(abl_folder + 'model_batch_0.pth', weights_only=False)
		net_abl.to(dev)

		net_abl.eval()
		correct = 0
		total = 0

		for images,labels in test_gen:
		    images = images.view(-1,784).to(dev)
		    labels = labels.to(dev)

		    out_1, out_2 = net_abl(images)
		    _, predicted = torch.max(out_2,1)
		    correct += (predicted == labels).sum()
		    total += labels.size(0)

		acc_abl = (100*correct)/(total+1)

		abl_results.append({'acc_abl_'+str(idx_ablation): acc_abl.item()})
	

	# Evaluation Pruning ------------------------------------------
	name_pruned = args.exp_name + '_epoch_' + str(args.epoch) + '_pruned_thr_' + str(thr.item())
	pruned_folder = args.PATHtrain + name_pruned + '/checkpoints/'
	net_pruned = torch.load(pruned_folder + 'model_batch_0.pth', weights_only=False)
	net_pruned.to(dev)

	net_pruned.eval()
	correct = 0
	total = 0

	for images,labels in test_gen:
	    images = images.view(-1,784).to(dev)
	    labels = labels.to(dev)

	    out_1, out_2 = net_pruned(images)
	    _, predicted = torch.max(out_2,1)
	    correct += (predicted == labels).sum()
	    total += labels.size(0)

	acc_pruned = (100*correct)/(total+1)

	# Saving -----------------------------------------------

	row_partial = {}
	for diccionario in abl_results:
		row_partial.update(diccionario)

	row = {
	'threshold': thr.item(),
	'acc_coll': acc_coll.item(),
	'acc_pruned': acc_pruned.item()
	}

	row.update(row_partial)
	data.append(row)

df = pd.DataFrame(data)
results_filename = args.PATHresults + 'Evaluation_' + args.exp_name + f'_thr_{args.epoch}.csv'
df.to_csv(results_filename, index=False)