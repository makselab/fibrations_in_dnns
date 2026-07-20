# =====================================================
# MODULES
 
import argparse
import os
import copy
import random
import itertools
 
import torch
from torchvision.datasets import MNIST
from model import MLP
 
import pandas as pd
 
from coloring_loss import global_fibration_coloring
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# =====================================================
# Load args, paths, device.
 
parser = argparse.ArgumentParser()
parser.add_argument('-exp_name', type=str, required=True, help='Exp Name')
args = parser.parse_args()
dev = torch.device("cuda:0")
 
# =====================================================
# Model
 
filename = '/home/osvaldo/Documents/CCNY/MNIST_Symmetries/train_dir/' + args.exp_name + '/gradients.pth'
data = torch.load(filename)
last_point = data[-1]
grads = {p: torch.stack([x[p]['grad'] for x in data]).mean(dim=0) for p in last_point.keys()} #av
 
hidden_size = last_point['fc1.weight']['param'].shape[0]
num_params = sum(v['param'].numel() for v in last_point.values())
print('Num Params:', num_params)

# =====================================================
# DATASET

batch_size = 100 
dataPATH = "/home/osvaldo/Documents/CCNY/MNIST_Symmetries/"
test_data = MNIST(root = dataPATH, train = False, transform = ToTensor())
test_gen = DataLoader(dataset = test_data, batch_size = batch_size,shuffle = False)
 
# ======================================================
# COLORING & COLLAPSE
 
num_budgets = 31
values_budget = torch.linspace(0.0, 0.6, num_budgets)
 
data = []
 
for idx_b, dL_budget in enumerate(values_budget):
	print('dL_budget:', dL_budget)
 
	# COLORING ----------------------------------------
	colors = global_fibration_coloring(
		layers_w=[last_point['fc1.weight']['param'], last_point['fc2.weight']['param'], last_point['fc3.weight']['param']],
		layers_g=[grads['fc1.weight'], grads['fc2.weight'], grads['fc3.weight']],
		layers_bias=[last_point['fc1.bias']['param'], last_point['fc2.bias']['param'], last_point['fc3.bias']['param']],
		layers_bias_grad=[grads['fc1.bias'], grads['fc2.bias'], grads['fc3.bias']],
		dL_budget=dL_budget)
 
	colors = [c.to('cpu') for c in colors]
	num_colors = [torch.unique(c).shape[0] for c in colors]
 
	print(num_colors)
 
 
	# COLLAPSE FORMULAR ----------------------------------------
	mtxs_partition = [torch.zeros(torch.unique(colors_layer).shape[0], hidden_size).scatter_(0, colors_layer.unsqueeze(0), 1).to(dev) for colors_layer in colors]
	szs_clusters = [torch.mm(mtx, torch.ones(hidden_size, 1).to(dev)) for mtx in mtxs_partition]
 
	Ws_coll = [(mtxs_partition[layer] @ last_point['fc' + str(layer+1) + '.weight']['param']) / szs_clusters[layer].view(-1, 1) for layer in range(3)] 
	Bs_coll = [(mtxs_partition[layer] @ last_point['fc' + str(layer+1) + '.bias']['param']) / szs_clusters[layer].view(-1,) for layer in range(3)] 
 
	# num_params_coll = sum(t.numel() for t in Ws_coll) + sum(t.numel() for t in Bs_coll) 
 
	# COLLAPSE -----------------------------------------------------
	net_coll = MLP(784, num_colors, 10)
 
	net_coll.fc1.weight.data = Ws_coll[0]
	net_coll.fc2.weight.data = Ws_coll[1] @ mtxs_partition[0].T
	net_coll.fc3.weight.data = Ws_coll[2] @ mtxs_partition[1].T
	net_coll.fc4.weight.data = last_point['fc4.weight']['param'] @ mtxs_partition[2].T
 
	net_coll.fc1.bias.data = Bs_coll[0]
	net_coll.fc2.bias.data = Bs_coll[1]
	net_coll.fc3.bias.data = Bs_coll[2]
	net_coll.fc4.bias.data = last_point['fc4.bias']['param']
 
	num_params_coll = sum(p.numel() for p in net_coll.parameters())
 
	coll_folder = '/media/osvaldo/OMV5TB/collapse_mnist_loss_coloring/' +  args.exp_name + '_epoch_599_coll_dLbudget_' + str(dL_budget.item()) +'/checkpoints/'
 
	if not os.path.exists(coll_folder): os.makedirs(coll_folder)
	torch.save(net_coll, coll_folder + 'model_batch_0.pth')

	# Evaluation Collapse ------------------------------------------
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

	# Approx W --------------------------------------------
	Ws_exp = [Ws_coll[layer][colors[layer]] for layer in range(3)]
	Bs_exp = [Bs_coll[layer][colors[layer]] for layer in range(3)]
 
	# dL --------------------------------------------------
 
	dL_W = [((Ws_exp[layer]-last_point['fc' + str(layer+1) + '.weight']['param']) * grads['fc' + str(layer+1) + '.weight']).sum(dim=1) for layer in range(3)]
	dL_b = [((Bs_exp[layer]-last_point['fc' + str(layer+1) + '.bias']['param']) * grads['fc' + str(layer+1) + '.bias']) for layer in range(3)]
 
	dL_layer = [((Ws_exp[layer]-last_point['fc' + str(layer+1) + '.weight']['param']) * grads['fc' + str(layer+1) + '.weight']).sum(dim=1) +\
				((Bs_exp[layer]-last_point['fc' + str(layer+1) + '.bias']['param']) * grads['fc' + str(layer+1) + '.bias']) \
				for layer in range(3)]
 
	dL_sum_layer = [dL_layer[layer].sum().item() for layer in range(3)]
 
	# Saving -----------------------------------------------
 
	row = {
	'dL_budget': dL_budget.item(),
	'reduction_pars_coll': num_params_coll/num_params,
	'dL1': dL_sum_layer[0],
	'dL2': dL_sum_layer[1],
	'dL3': dL_sum_layer[2],
	'acc': acc_coll.item()
	}
 
	data.append(row)
 
	print(row)
 
df = pd.DataFrame(data)
results_filename = '/home/osvaldo/Documents/CCNY/MNIST_Symmetries/results/loss_coloring_dL_vs_thr_' + args.exp_name + '.csv'
df.to_csv(results_filename, index=False)