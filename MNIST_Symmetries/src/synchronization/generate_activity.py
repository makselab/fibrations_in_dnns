# =====================================================
# MODULES

import argparse
import os

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# =====================================================
# Load args, paths, device.

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name', type=str, required=True, help='Exp Name')
parser.add_argument('-PATHtrain', type=str, required=True, help='Training directory')
parser.add_argument('-PATHresults', type=str, required=True, help='Results directory')
parser.add_argument('-PATHdata', type=str, required=True, help='Dataset directory')
parser.add_argument('-epoch', type=int, required=True, help='Epoch')
parser.add_argument('-num_random_inputs', type=int, default=200, help='Number of random input samples')

args = parser.parse_args()

dev = torch.device("cuda:0")

# =====================================================
# DATASET

batch_size = 100
test_data = MNIST(root=args.PATHdata, train=False, transform=ToTensor())
gen = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# =====================================================
# MODEL

model_filename = args.PATHtrain + args.exp_name + '/checkpoints/model_batch_' + str(args.epoch) + '.pth'
net = torch.load(model_filename)
net.to(dev)
net.eval()

input_size = net.dims[0]

results_folder = args.PATHresults + args.exp_name + '/synchronization/'
if not os.path.exists(results_folder): os.makedirs(results_folder)

# =====================================================
# ACTIVITY - TEST SET

activities = [[] for _ in range(net.num_layers)]
list_labels = []
list_predictions = []

for images, labels in gen:
	images = images.view(-1, input_size).to(dev)
	layer_activations, out = net(images)
	_, predicted = torch.max(out, 1)

	for idx_layer in range(net.num_layers):
		activities[idx_layer].append(layer_activations[idx_layer])

	list_labels.append(labels)
	list_predictions.append(predicted.cpu())

tensor_activities = [torch.cat(act, dim=0) for act in activities]
list_labels = torch.cat(list_labels)
list_predictions = torch.cat(list_predictions)

torch.save({'activity': tensor_activities, 'labels': list_labels, 'prediction': list_predictions},
	results_folder + 'activity_batch_' + str(args.epoch) + '.pth')

# =====================================================
# ACTIVITY - RANDOM INPUTS

images = torch.rand(args.num_random_inputs, input_size).to(dev)
random_activations, _ = net(images)

torch.save(random_activations, results_folder + 'activity_random_input_batch_' + str(args.epoch) + '.pth')
