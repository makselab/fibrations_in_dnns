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
import itertools

# =====================================================
# Load args, paths, device.

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name', type=str, required=True, help='Exp Name')

args = parser.parse_args()

dev = torch.device("cuda:0")

# =====================================================
# DATASET

batch_size = 100 
dataPATH = "/home/osvaldo/Documents/CCNY/MNIST_Symmetries/"
test_data = MNIST(root = dataPATH, train = False, transform = ToTensor())
test_gen = DataLoader(dataset = test_data, batch_size = batch_size,shuffle = False)

# ==================================================================
# Evaluation

num_thresholds_per_layer = 21
values_threshold_1= torch.linspace(0,1, num_thresholds_per_layer)
values_threshold_2= torch.linspace(0,1, num_thresholds_per_layer)
values_threshold_3= torch.linspace(0,1, num_thresholds_per_layer)
num_thresholds = num_thresholds_per_layer ** 3

data = []
loss_function = CrossEntropyLoss()

for idx_th, (t1, t2, t3) in enumerate(itertools.product(values_threshold_1, values_threshold_2, values_threshold_3)):
	print(idx_th, num_thresholds)


	# Evaluation Collapse ------------------------------------------
	coll_folder = '/media/osvaldo/OMV5TB/collapse_mnist/' +  args.exp_name + '_epoch_599_coll_thrs_' + str(t1.item()) + '_' + str(t2.item()) + '_' + str(t3.item()) + '/checkpoints/'

	# net_coll = torch.load(coll_folder + 'model_batch_0.pth', weights_only=False)
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
	    loss = loss_function(out_2, labels)

	    _, predicted = torch.max(out_2,1)
	    correct += (predicted == labels).sum()
	    total += labels.size(0)
	    loss_final += loss.item()

	acc_coll = (100*correct)/(total+1)
	loss_final = loss_final/len(test_gen)

	# Saving -----------------------------------------------

	row = {
	'thr1': t1.item(),
	'thr2': t2.item(),
	'thr3': t3.item(),
	'acc_coll': acc_coll.item(),
	'loss': loss_final}

	data.append(row)

df = pd.DataFrame(data)
results_filename = '/home/osvaldo/Documents/CCNY/MNIST_Symmetries/results/Evaluation_' + args.exp_name + '_thr_599.csv'
df.to_csv(results_filename, index=False)